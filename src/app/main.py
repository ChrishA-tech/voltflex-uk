"""FastAPI serving layer for VoltFlex UK (/health, /optimize-dispatch, /flexibility-copilot)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd
import torch
from fastapi import FastAPI
from pydantic import BaseModel, Field

from src.agents.flexibility_copilot import FlexibilityCopilot
from src.analytics.curtailment_detector import detect_grid_regimes
from src.ingestion.grid_telemetry import DEFAULT_DB_PATH, UK_DNO_REGIONS
from src.models.forecast_and_dispatch import (
    DispatchConstraints,
    EVBatteryDispatchNet,
    build_rl_state_vector,
    enforce_thermal_and_soc_limits,
)

app = FastAPI(
    title="VoltFlex UK API",
    version="0.1.0",
    description="Real-Time UK Grid Balancing, EV & Battery Dispatch Optimization, and NESO/Ofgem Flexibility Copilot",
)

_COPILOT = FlexibilityCopilot()
_DISPATCH_NET = EVBatteryDispatchNet(state_dim=7, hidden_dim=64)
_DISPATCH_NET.eval()


class DispatchRequest(BaseModel):
    dno_region_id: int = Field(default=10, ge=10, le=23)
    ev_soc_pct: float = Field(default=45.0, ge=0.0, le=100.0)
    battery_soc_pct: float = Field(default=50.0, ge=0.0, le=100.0)
    base_load_kw: float = Field(default=1.2, ge=0.0, le=25.0)
    agile_import_pence: float | None = Field(default=None)
    carbon_intensity_gco2_kwh: float | None = Field(default=None)
    substation_headroom_pct: float | None = Field(default=None)
    g99_approved: bool = Field(default=False)
    thermal_current_limit_amps: float = Field(default=60.0, ge=16.0, le=100.0)
    telemetry_age_seconds: float = Field(default=0.0, ge=0.0)


class DispatchResponse(BaseModel):
    dno_region_id: int
    dno_region_name: str
    ev_dispatch_kw: float
    battery_dispatch_kw: float
    net_site_kw: float
    site_current_amps: float
    active_export_limit_kw: float
    g98_g99_compliant: bool
    thermal_compliant: bool
    degraded_mode: bool
    fallback_reason: str | None = None
    grid_regime: str
    flexibility_signal: str


class CopilotRequest(BaseModel):
    query: str = Field(..., min_length=3)
    requested_export_kw: float | None = Field(default=None, ge=0.0)
    g99_approved: bool = Field(default=False)
    substation_headroom_pct: float | None = Field(default=None, ge=0.0, le=100.0)


class CopilotResponse(BaseModel):
    query: str
    answer: str
    citations: list[str]
    citation_verified: bool
    verification_notes: str
    interlock_triggered: bool
    interlock_reason: str
    used_safety_fallback: bool


def _load_latest_region_telemetry(dno_region_id: int) -> dict[str, Any] | None:
    db_path = Path(os.getenv("VOLTFLEX_DB_PATH") or DEFAULT_DB_PATH)
    if not db_path.exists():
        return None
    try:
        with duckdb.connect(str(db_path), read_only=True) as con:
            df = con.execute(
                """
                SELECT *
                FROM grid_telemetry
                WHERE dno_region_id = ?
                ORDER BY timestamp DESC
                LIMIT 1
                """,
                [dno_region_id],
            ).fetchdf()
        if df.empty:
            return None
        enriched = detect_grid_regimes(df)
        return enriched.iloc[0].to_dict()
    except Exception:
        return None


@app.get("/health")
def health() -> dict[str, Any]:
    """Report service health, DuckDB telemetry availability, and DNO coverage."""
    db_path = Path(os.getenv("VOLTFLEX_DB_PATH") or DEFAULT_DB_PATH)
    row_count = 0
    duckdb_ready = False
    if db_path.exists():
        try:
            with duckdb.connect(str(db_path), read_only=True) as con:
                res = con.execute("SELECT COUNT(*) FROM grid_telemetry").fetchone()
                row_count = int(res[0]) if res else 0
                duckdb_ready = True
        except Exception:
            duckdb_ready = False

    return {
        "status": "ok" if duckdb_ready else "degraded",
        "duckdb_ready": duckdb_ready,
        "telemetry_rows": row_count,
        "dno_regions_supported": len(UK_DNO_REGIONS),
        "g98_limit_kw": 3.68,
    }


@app.post("/optimize-dispatch", response_model=DispatchResponse)
def optimize_dispatch(req: DispatchRequest) -> DispatchResponse:
    """Optimize half-hourly EV & battery dispatch with degraded-telemetry fallback and G98/G99 interlock."""
    constraints = DispatchConstraints(
        g99_approved=req.g99_approved,
        thermal_current_limit_amps=req.thermal_current_limit_amps,
    )
    region_meta = UK_DNO_REGIONS.get(req.dno_region_id, {"name": f"Region {req.dno_region_id}"})

    degraded_mode = False
    fallback_reason: str | None = None

    latest = _load_latest_region_telemetry(req.dno_region_id)
    agile_p = req.agile_import_pence
    carbon_g = req.carbon_intensity_gco2_kwh
    headroom_pct = req.substation_headroom_pct

    if req.telemetry_age_seconds > 1800.0:
        degraded_mode = True
        fallback_reason = (
            f"Telemetry stale ({req.telemetry_age_seconds:.0f}s > 1800s); using conservative G98-safe fallback."
        )
    elif agile_p is None or carbon_g is None or headroom_pct is None:
        if latest is not None:
            agile_p = agile_p if agile_p is not None else float(latest["agile_import_pence"])
            carbon_g = carbon_g if carbon_g is not None else float(latest["carbon_intensity_gco2_kwh"])
            headroom_pct = headroom_pct if headroom_pct is not None else float(latest["substation_headroom_pct"])
            regime = str(latest.get("grid_regime", "Normal"))
            signal = str(latest.get("flexibility_signal", "HOLD_OR_OPTIMIZE"))
        else:
            degraded_mode = True
            fallback_reason = "Live DuckDB telemetry unavailable; falling back to conservative G98-safe schedule."
            regime = "Degraded Fallback"
            signal = "CONSERVATIVE_G98_SAFE"
    else:
        regime = "Normal"
        signal = "HOLD_OR_OPTIMIZE"

    if degraded_mode:
        agile_p = agile_p if agile_p is not None else 22.0
        carbon_g = carbon_g if carbon_g is not None else 180.0
        headroom_pct = headroom_pct if headroom_pct is not None else 20.0
        # Conservative trickle charge only, zero export risk
        proposed_ev_kw = 1.38 if req.ev_soc_pct < 80.0 else 0.0
        proposed_batt_kw = 0.0
        regime = "Degraded Fallback"
        signal = "CONSERVATIVE_G98_SAFE"
    else:
        ts = pd.Timestamp.utcnow()
        sp = ts.hour * 2 + (ts.minute // 30)
        state_np = build_rl_state_vector(
            agile_pence=float(agile_p),
            carbon_gco2_kwh=float(carbon_g),
            ev_soc_pct=req.ev_soc_pct,
            battery_soc_pct=req.battery_soc_pct,
            substation_headroom_pct=float(headroom_pct),
            settlement_period=sp,
        )
        with torch.no_grad():
            mean_act, _ = _DISPATCH_NET(torch.from_numpy(state_np).unsqueeze(0))
        proposed_ev_kw = float(mean_act[0, 0].item()) * constraints.ev_max_power_kw
        proposed_batt_kw = float(mean_act[0, 1].item()) * constraints.battery_max_power_kw
        if float(agile_p) < 0.0:
            proposed_ev_kw = max(proposed_ev_kw, 7.0)
            proposed_batt_kw = max(proposed_batt_kw, 4.5)
            regime = "Wind Curtailment Plunge"
            signal = "CHARGE_MAX_PLUNGE"
        elif float(agile_p) >= 28.0 and float(carbon_g) >= 220.0:
            proposed_ev_kw = 0.0
            proposed_batt_kw = -constraints.active_export_limit_kw
            regime = "Dunkelflaute Peak"
            signal = "DISCHARGE_DFS_PEAK"

    safe_ev_kw, safe_batt_kw, _ = enforce_thermal_and_soc_limits(
        proposed_ev_kw=proposed_ev_kw,
        proposed_battery_kw=proposed_batt_kw,
        ev_soc_pct=req.ev_soc_pct,
        battery_soc_pct=req.battery_soc_pct,
        base_load_kw=req.base_load_kw,
        substation_headroom_pct=float(headroom_pct),
        constraints=constraints,
    )

    net_site_kw = round(req.base_load_kw + safe_ev_kw + safe_batt_kw, 4)
    site_amps = round(abs(net_site_kw * 1000.0) / constraints.nominal_voltage_v, 2)
    export_kw = max(0.0, -net_site_kw)

    return DispatchResponse(
        dno_region_id=req.dno_region_id,
        dno_region_name=str(region_meta["name"]),
        ev_dispatch_kw=safe_ev_kw,
        battery_dispatch_kw=safe_batt_kw,
        net_site_kw=net_site_kw,
        site_current_amps=site_amps,
        active_export_limit_kw=constraints.active_export_limit_kw,
        g98_g99_compliant=bool(export_kw <= constraints.active_export_limit_kw + 1e-3),
        thermal_compliant=bool(site_amps <= constraints.thermal_current_limit_amps + 1e-3),
        degraded_mode=degraded_mode,
        fallback_reason=fallback_reason,
        grid_regime=regime,
        flexibility_signal=signal,
    )


@app.post("/flexibility-copilot", response_model=CopilotResponse)
def flexibility_copilot(req: CopilotRequest) -> CopilotResponse:
    """Query the guardrailed LangGraph NESO/Ofgem Flexibility Copilot."""
    result = _COPILOT.ask(
        query=req.query,
        requested_export_kw=req.requested_export_kw,
        g99_approved=req.g99_approved,
        substation_headroom_pct=req.substation_headroom_pct,
    )
    return CopilotResponse(
        query=result["query"],
        answer=result["answer"],
        citations=result["citations"],
        citation_verified=result["citation_verified"],
        verification_notes=result["verification_notes"],
        interlock_triggered=result["interlock_triggered"],
        interlock_reason=result["interlock_reason"],
        used_safety_fallback=result["used_safety_fallback"],
    )
