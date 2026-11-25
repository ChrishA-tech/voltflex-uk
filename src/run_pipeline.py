"""End-to-end VoltFlex UK pipeline runner with synthetic UK grid/Agile/DNO telemetry + 3% corrupt/stale rows."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.agents.flexibility_copilot import FlexibilityCopilot
from src.analytics.curtailment_detector import (
    build_curtailment_summary,
    detect_grid_regimes,
    summarize_dno_headroom,
)
from src.evaluation.error_analysis import generate_error_analysis_report
from src.ingestion.grid_telemetry import (
    DEFAULT_DB_PATH,
    DEFAULT_QUARANTINE_DIR,
    UK_DNO_REGIONS,
    ingest_grid_telemetry,
)
from src.models.forecast_and_dispatch import (
    DispatchConstraints,
    train_and_evaluate_rl_dispatcher,
    train_conformal_forecasters,
)


def generate_uk_grid_telemetry(
    days: int = 8,
    seed: int = 42,
    corrupt_fraction: float = 0.03,
) -> pd.DataFrame:
    """Generate realistic UK half-hourly grid, Agile, and 14-DNO telemetry with ~3% corrupt/stale rows."""
    rng = np.random.default_rng(seed)
    periods = days * 48
    start = datetime(2026, 11, 10, 0, 0, tzinfo=timezone.utc)
    timestamps = [start + timedelta(minutes=30 * i) for i in range(periods)]

    # National wind/solar waves creating Dunkelflaute peaks and Wind Curtailment Plunge regimes
    t_arr = np.arange(periods, dtype=float)
    wind_wave = 0.50 + 0.42 * np.sin(2.0 * np.pi * t_arr / (48.0 * 1.8)) + rng.normal(0.0, 0.05, periods)
    wind_wave = np.clip(wind_wave, 0.06, 0.98)

    rows: list[dict[str, object]] = []
    region_ids = sorted(UK_DNO_REGIONS.keys())

    for p_idx, ts in enumerate(timestamps):
        sp = ts.hour * 2 + (ts.minute // 30)
        diurnal = np.sin(2.0 * np.pi * (sp - 12) / 48.0)
        evening_peak = np.exp(-0.5 * ((sp - 36) / 3.0) ** 2)
        solar_shape = max(0.0, np.sin(np.pi * max(0, min(24, sp - 12)) / 24.0))

        w_factor = float(wind_wave[p_idx])

        for rid in region_ids:
            rated = float(UK_DNO_REGIONS[rid]["rated_capacity_mw"])
            # Scotland & North regions carry heavier wind ratios
            regional_wind_bias = 1.25 if rid in (15, 17, 18, 23) else 0.92
            demand_mw = rated * (0.48 + 0.14 * diurnal + 0.22 * evening_peak) + rng.normal(0.0, 45.0)
            demand_mw = float(np.clip(demand_mw, rated * 0.25, rated * 0.95))

            wind_mw = max(10.0, demand_mw * w_factor * regional_wind_bias + rng.normal(0.0, 35.0))
            solar_mw = max(0.0, rated * 0.14 * solar_shape * (1.0 if rid in (10, 19, 20, 22) else 0.65))

            wind_ratio = wind_mw / max(demand_mw, 1.0)
            freq_hz = 50.0 + 0.14 * (wind_ratio - 0.45) + rng.normal(0.0, 0.025)
            freq_hz = float(np.clip(freq_hz, 49.65, 50.35))

            carbon = 280.0 - 210.0 * min(wind_ratio, 1.05) + 45.0 * evening_peak + rng.normal(0.0, 12.0)
            carbon = float(np.clip(carbon, 18.0, 480.0))

            agile_p = (
                22.0
                + 24.0 * evening_peak
                - 29.0 * max(0.0, wind_ratio - 0.35)
                + rng.normal(0.0, 2.2)
            )
            agile_p = float(np.clip(agile_p, -12.0, 65.0))

            substation_load = min(rated * 0.97, max(rated * 0.22, demand_mw * (0.96 + rng.normal(0.0, 0.02))))
            ev_soc = float(np.clip(45.0 + 25.0 * np.cos(2.0 * np.pi * sp / 48.0) + rng.normal(0.0, 4.0), 12.0, 95.0))

            rows.append(
                {
                    "timestamp": ts,
                    "received_at": ts + timedelta(minutes=2),
                    "dno_region_id": rid,
                    "grid_frequency_hz": round(freq_hz, 3),
                    "wind_generation_mw": round(wind_mw, 2),
                    "solar_generation_mw": round(solar_mw, 2),
                    "demand_mw": round(demand_mw, 2),
                    "substation_load_mw": round(substation_load, 2),
                    "carbon_intensity_gco2_kwh": round(carbon, 2),
                    "agile_import_pence": round(agile_p, 2),
                    "ev_battery_soc_pct": round(ev_soc, 2),
                    "mpan": f"{rid}0001234{p_idx:05d}",
                }
            )

    df = pd.DataFrame(rows)

    # Inject ~3% corrupted and stale telemetry rows for Dead-Letter Quarantine testing
    n_corrupt = max(1, int(round(len(df) * corrupt_fraction)))
    corrupt_indices = rng.choice(df.index.to_numpy(), size=n_corrupt, replace=False)
    for k, idx in enumerate(corrupt_indices):
        mode = k % 5
        if mode == 0:
            df.loc[idx, "grid_frequency_hz"] = 47.8  # Below 49.0 Hz physical bound
        elif mode == 1:
            df.loc[idx, "wind_generation_mw"] = -125.0  # Negative generation
        elif mode == 2:
            df.loc[idx, "dno_region_id"] = 99  # Invalid UK DNO region
        elif mode == 3:
            # Stale telemetry (> 48 hours old relative to received_at)
            df.loc[idx, "timestamp"] = pd.Timestamp(df.loc[idx, "received_at"]) - timedelta(hours=72)
        else:
            df.loc[idx, "mpan"] = ""  # Missing MPAN

    return df


def run_end_to_end(
    days: int = 8,
    db_path: Path | str = DEFAULT_DB_PATH,
    quarantine_dir: Path | str = DEFAULT_QUARANTINE_DIR,
    report_path: Path | str = Path("docs/ERROR_ANALYSIS_REPORT.md"),
) -> dict[str, object]:
    """Execute the complete VoltFlex UK data, ML, RL, evaluation, and copilot pipeline."""
    raw_df = generate_uk_grid_telemetry(days=days, corrupt_fraction=0.03)
    ingestion = ingest_grid_telemetry(
        raw_df,
        db_path=db_path,
        quarantine_dir=quarantine_dir,
        batch_id="pipeline_run",
    )

    enriched = detect_grid_regimes(ingestion.valid_df)
    dno_headroom = summarize_dno_headroom(enriched)
    curtailment_summary = build_curtailment_summary(enriched)

    forecast_suite = train_conformal_forecasters(
        enriched,
        horizons=(1, 2, 4),
        confidence_level=0.90,
        log_to_mlflow=True,
    )

    _, dispatch_result = train_and_evaluate_rl_dispatcher(
        enriched,
        constraints=DispatchConstraints(g99_approved=False, thermal_current_limit_amps=60.0),
        epochs=30,
    )

    audit = generate_error_analysis_report(
        forecast_suite=forecast_suite,
        dispatch_result=dispatch_result,
        report_path=report_path,
        quarantine_count=ingestion.quarantined_rows,
        total_ingested_count=ingestion.total_rows,
    )

    copilot = FlexibilityCopilot()
    copilot_probe = copilot.ask(
        query="Can I export 6.0 kW during a NESO DFS event on single-phase G98?",
        requested_export_kw=6.0,
        g99_approved=False,
    )

    return {
        "ingestion": ingestion,
        "curtailment_summary": curtailment_summary,
        "dno_regions_count": len(dno_headroom),
        "forecast_suite": forecast_suite,
        "dispatch_result": dispatch_result,
        "audit": audit,
        "copilot_probe": copilot_probe,
    }


def main() -> None:
    results = run_end_to_end()
    ing = results["ingestion"]
    curt = results["curtailment_summary"]
    disp = results["dispatch_result"]
    audit = results["audit"]
    cop = results["copilot_probe"]

    print(
        f"[Ingestion] total={ing.total_rows}, valid={ing.valid_rows}, "
        f"quarantined={ing.quarantined_rows} ({ing.quarantine_rate:.2%}), "
        f"duckdb={ing.duckdb_path}, quarantine={ing.quarantine_path}"
    )
    print(
        f"[Analytics] regions={results['dno_regions_count']}, "
        f"curtailment_windows={curt.wind_curtailment_windows}, "
        f"negative_agile={curt.negative_agile_windows}, "
        f"dunkelflaute={curt.dunkelflaute_peak_windows}"
    )
    print(
        f"[RL Dispatch] cost=£{disp.total_cost_gbp:.2f} (baseline £{disp.baseline_cost_gbp:.2f}), "
        f"carbon={disp.total_carbon_kg:.2f}kg, thermal_violation_rate={disp.thermal_violation_rate:.4%}"
    )
    print(f"[Evaluation] report={audit.report_path}")
    print(
        f"[Copilot] verified={cop['citation_verified']}, interlock={cop['interlock_triggered']}, "
        f"citations={cop['citations']}"
    )


if __name__ == "__main__":
    main()
