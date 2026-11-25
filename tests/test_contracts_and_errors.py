"""End-to-end contract, safety, conformal forecasting, and LangGraph verification tests."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import duckdb
import pandas as pd
from fastapi.testclient import TestClient

from src.agents.flexibility_copilot import (
    FlexibilityCopilot,
    verify_citations,
)
from src.analytics.curtailment_detector import (
    detect_grid_regimes,
    summarize_dno_headroom,
)
from src.app.main import app
from src.evaluation.error_analysis import generate_error_analysis_report
from src.ingestion.grid_telemetry import (
    UK_DNO_REGIONS,
    ingest_grid_telemetry,
    pseudonymise_mpan,
)
from src.models.forecast_and_dispatch import (
    DispatchConstraints,
    enforce_thermal_and_soc_limits,
    train_and_evaluate_rl_dispatcher,
    train_conformal_forecasters,
)
from src.run_pipeline import generate_uk_grid_telemetry


def test_mpan_hmac_pseudonymisation_and_quarantine(tmp_path: Path) -> None:
    """Verify UK GDPR MPAN HMAC-SHA256 hashing, DuckDB persistence, and Parquet quarantine."""
    raw_mpan = "1000123456789"
    digest1 = pseudonymise_mpan(raw_mpan, secret="test-secret")
    digest2 = pseudonymise_mpan(raw_mpan, secret="test-secret")
    assert digest1 == digest2
    assert len(digest1) == 64
    assert raw_mpan not in digest1

    now = datetime(2026, 11, 20, 12, 0, tzinfo=timezone.utc)
    rows = [
        {
            "timestamp": now,
            "received_at": now + timedelta(minutes=2),
            "dno_region_id": 10,
            "grid_frequency_hz": 50.02,
            "wind_generation_mw": 1500.0,
            "solar_generation_mw": 250.0,
            "demand_mw": 2400.0,
            "substation_load_mw": 2200.0,
            "carbon_intensity_gco2_kwh": 140.0,
            "agile_import_pence": 12.5,
            "ev_battery_soc_pct": 55.0,
            "mpan": raw_mpan,
        },
        # Bad frequency (< 49.0 Hz)
        {
            "timestamp": now,
            "received_at": now + timedelta(minutes=2),
            "dno_region_id": 11,
            "grid_frequency_hz": 47.9,
            "wind_generation_mw": 1200.0,
            "solar_generation_mw": 200.0,
            "demand_mw": 2100.0,
            "substation_load_mw": 1900.0,
            "carbon_intensity_gco2_kwh": 150.0,
            "agile_import_pence": 14.0,
            "ev_battery_soc_pct": 50.0,
            "mpan": "1100123456789",
        },
        # Negative wind generation
        {
            "timestamp": now,
            "received_at": now + timedelta(minutes=2),
            "dno_region_id": 12,
            "grid_frequency_hz": 50.0,
            "wind_generation_mw": -40.0,
            "solar_generation_mw": 100.0,
            "demand_mw": 2800.0,
            "substation_load_mw": 2600.0,
            "carbon_intensity_gco2_kwh": 180.0,
            "agile_import_pence": 20.0,
            "ev_battery_soc_pct": 60.0,
            "mpan": "1200123456789",
        },
        # Invalid DNO region ID
        {
            "timestamp": now,
            "received_at": now + timedelta(minutes=2),
            "dno_region_id": 99,
            "grid_frequency_hz": 50.0,
            "wind_generation_mw": 900.0,
            "solar_generation_mw": 100.0,
            "demand_mw": 2000.0,
            "substation_load_mw": 1800.0,
            "carbon_intensity_gco2_kwh": 160.0,
            "agile_import_pence": 18.0,
            "ev_battery_soc_pct": 40.0,
            "mpan": "9900123456789",
        },
        # Stale telemetry (> 24h)
        {
            "timestamp": now - timedelta(hours=48),
            "received_at": now,
            "dno_region_id": 14,
            "grid_frequency_hz": 50.01,
            "wind_generation_mw": 800.0,
            "solar_generation_mw": 120.0,
            "demand_mw": 2200.0,
            "substation_load_mw": 1950.0,
            "carbon_intensity_gco2_kwh": 170.0,
            "agile_import_pence": 19.0,
            "ev_battery_soc_pct": 48.0,
            "mpan": "1400123456789",
        },
    ]

    db_path = tmp_path / "voltflex.duckdb"
    q_dir = tmp_path / "quarantine"
    res = ingest_grid_telemetry(pd.DataFrame(rows), db_path=db_path, quarantine_dir=q_dir, batch_id="unit")

    assert res.total_rows == 5
    assert res.valid_rows == 1
    assert res.quarantined_rows == 4
    assert "mpan" not in res.valid_df.columns
    assert "mpan_hash" in res.valid_df.columns
    assert res.quarantine_path is not None and res.quarantine_path.exists()

    q_df = pd.read_parquet(res.quarantine_path)
    assert len(q_df) == 4
    assert "quarantine_reason" in q_df.columns

    with duckdb.connect(str(db_path), read_only=True) as con:
        stored = con.execute("SELECT * FROM grid_telemetry").fetchdf()
    assert len(stored) == 1
    assert "mpan" not in stored.columns


def test_curtailment_and_14_dno_headroom_analytics() -> None:
    """Verify curtailment, plunge pricing, and 14-DNO substation headroom detection."""
    raw_df = generate_uk_grid_telemetry(days=4, corrupt_fraction=0.0)
    enriched = detect_grid_regimes(raw_df)
    summary = summarize_dno_headroom(enriched)

    assert len(summary) == len(UK_DNO_REGIONS) == 14
    assert {"Dunkelflaute Peak", "Wind Curtailment Plunge", "Normal"} >= set(enriched["grid_regime"].unique())
    assert (enriched["is_negative_agile_plunge"] == (enriched["agile_import_pence"] < 0.0)).all()


def test_conformal_forecasting_and_rl_thermal_safety_and_report(tmp_path: Path) -> None:
    """Verify MAPIE 90% conformal intervals, 0.0% RL thermal violation rate, and error report."""
    raw_df = generate_uk_grid_telemetry(days=5, corrupt_fraction=0.03)
    ing = ingest_grid_telemetry(
        raw_df,
        db_path=tmp_path / "test.duckdb",
        quarantine_dir=tmp_path / "quarantine",
        batch_id="forecast_test",
    )
    assert ing.quarantined_rows > 0
    enriched = detect_grid_regimes(ing.valid_df)

    mlflow_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    suite = train_conformal_forecasters(
        enriched,
        horizons=(1, 2),
        confidence_level=0.90,
        mlflow_tracking_uri=mlflow_uri,
        log_to_mlflow=True,
    )
    assert suite.mlflow_run_id is not None
    for h in (1, 2):
        c_res = suite.carbon_models[h]
        p_res = suite.price_models[h]
        assert 0.65 <= c_res.conformal_coverage_90 <= 1.0
        assert 0.65 <= p_res.conformal_coverage_90 <= 1.0
        assert (c_res.test_predictions["y_lower_90"] <= c_res.test_predictions["y_upper_90"]).all()
        assert (p_res.test_predictions["y_lower_90"] <= p_res.test_predictions["y_upper_90"]).all()

    constraints = DispatchConstraints(g99_approved=False, thermal_current_limit_amps=60.0)
    _, sim = train_and_evaluate_rl_dispatcher(enriched, constraints=constraints, epochs=12)
    assert sim.thermal_violation_rate == 0.0
    assert (sim.schedule_df["export_kw"] <= constraints.g98_export_limit_kw + 1e-3).all()
    assert (sim.schedule_df["site_current_amps"] <= constraints.thermal_current_limit_amps + 1e-3).all()

    # Direct shield stress check: exceeding G98 and critical substation headroom
    ev_kw, batt_kw, pre_violated = enforce_thermal_and_soc_limits(
        proposed_ev_kw=7.36,
        proposed_battery_kw=-5.0,
        ev_soc_pct=50.0,
        battery_soc_pct=80.0,
        base_load_kw=1.0,
        substation_headroom_pct=5.0,
        constraints=constraints,
    )
    assert pre_violated is True
    assert ev_kw <= 1.38 + 1e-4
    assert -(1.0 + ev_kw + batt_kw) <= constraints.g98_export_limit_kw + 1e-4

    report_path = tmp_path / "ERROR_ANALYSIS_REPORT.md"
    audit = generate_error_analysis_report(
        forecast_suite=suite,
        dispatch_result=sim,
        report_path=report_path,
        quarantine_count=ing.quarantined_rows,
        total_ingested_count=ing.total_rows,
    )
    assert audit.report_path.exists()
    content = report_path.read_text(encoding="utf-8")
    assert "Dunkelflaute Peak" in content
    assert "Wind Curtailment Plunge" in content
    assert "0.0000%" in content


def test_langgraph_copilot_citation_verifier_and_g98_interlock() -> None:
    """Verify BM25 hybrid retrieval, G98/G99 interlock, and citation judge verifier."""
    copilot = FlexibilityCopilot()

    # Case 1: Export > 3.68 kW without G99 approval must trigger safety interlock
    res_interlock = copilot.ask(
        query="Can I export 6.5 kW during a NESO DFS window on single-phase without G99?",
        requested_export_kw=6.5,
        g99_approved=False,
    )
    assert res_interlock["interlock_triggered"] is True
    assert res_interlock["citation_verified"] is True
    assert "ENA-G98-01" in res_interlock["citations"]
    assert "3.68" in res_interlock["answer"]

    # Case 2: Standard SEG / DFS compliance question
    res_seg = copilot.ask(
        query="What are the Ofgem Smart Export Guarantee metering rules?",
        g99_approved=False,
    )
    assert res_seg["interlock_triggered"] is False
    assert res_seg["citation_verified"] is True
    assert "OFGEM-SEG-01" in res_seg["citations"]

    # Case 3: Citation verifier rejects hallucinated or missing citations
    ok_missing, _, _ = verify_citations("You can export freely.", [{"doc_id": "ENA-G98-01"}], False)
    assert ok_missing is False

    ok_hallucinated, _, _ = verify_citations("Approved under [FAKE-DOC-99].", [{"doc_id": "ENA-G98-01"}], False)
    assert ok_hallucinated is False


def test_fastapi_endpoints_and_degraded_fallback() -> None:
    """Verify /health, /optimize-dispatch (including stale telemetry fallback), and /flexibility-copilot."""
    client = TestClient(app)

    health_resp = client.get("/health")
    assert health_resp.status_code == 200
    assert health_resp.json()["dno_regions_supported"] == 14

    # Stale telemetry triggers degraded_mode conservative fallback while staying G98 compliant
    stale_resp = client.post(
        "/optimize-dispatch",
        json={
            "dno_region_id": 10,
            "ev_soc_pct": 40.0,
            "battery_soc_pct": 50.0,
            "telemetry_age_seconds": 3600.0,
            "g99_approved": False,
        },
    )
    assert stale_resp.status_code == 200
    body = stale_resp.json()
    assert body["degraded_mode"] is True
    assert body["g98_g99_compliant"] is True
    assert body["thermal_compliant"] is True

    cop_resp = client.post(
        "/flexibility-copilot",
        json={
            "query": "Explain NESO Demand Flexibility Service baselines and G98 3.68 kW limit.",
            "requested_export_kw": 5.0,
            "g99_approved": False,
        },
    )
    assert cop_resp.status_code == 200
    cop_body = cop_resp.json()
    assert cop_body["citation_verified"] is True
    assert cop_body["interlock_triggered"] is True
