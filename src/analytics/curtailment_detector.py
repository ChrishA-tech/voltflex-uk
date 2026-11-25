"""Real-Time Wind Curtailment, Negative Agile Plunge-Pricing, and 14-Region DNO Headroom Detector."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.ingestion.grid_telemetry import UK_DNO_REGIONS


@dataclass
class CurtailmentSummary:
    """High-level summary of detected grid flexibility opportunities and constraints."""

    total_intervals: int
    wind_curtailment_windows: int
    negative_agile_windows: int
    dunkelflaute_peak_windows: int
    constrained_dno_regions: list[int]
    total_estimated_curtailed_mwh: float


def detect_grid_regimes(
    df: pd.DataFrame,
    plunge_price_threshold_pence: float = 0.0,
    near_zero_price_threshold_pence: float = 3.0,
    dunkelflaute_price_threshold_pence: float = 28.0,
    dunkelflaute_wind_ratio_threshold: float = 0.22,
    curtailment_wind_ratio_threshold: float = 0.62,
    curtailment_freq_hz: float = 50.08,
) -> pd.DataFrame:
    """Enrich half-hourly telemetry with wind curtailment, Agile plunge-pricing, and DNO headroom metrics."""
    out = df.copy()

    rated_cap = out["dno_region_id"].map(
        lambda rid: UK_DNO_REGIONS.get(int(rid), {}).get("rated_capacity_mw", 3200.0)
    ).astype(float)
    dno_name = out["dno_region_id"].map(
        lambda rid: UK_DNO_REGIONS.get(int(rid), {}).get("name", f"Region {rid}")
    )

    out["dno_region_name"] = dno_name
    out["substation_rated_mw"] = rated_cap
    out["substation_headroom_mw"] = (rated_cap - out["substation_load_mw"]).round(2)
    out["substation_headroom_pct"] = (
        (out["substation_headroom_mw"] / rated_cap.replace(0.0, np.nan)) * 100.0
    ).round(2)
    out["thermal_utilization_pct"] = (
        (out["substation_load_mw"] / rated_cap.replace(0.0, np.nan)) * 100.0
    ).round(2)

    out["headroom_status"] = np.select(
        [
            out["substation_headroom_pct"] < 10.0,
            out["substation_headroom_pct"] < 25.0,
            out["substation_headroom_pct"] >= 50.0,
        ],
        ["CRITICAL", "CONSTRAINED", "SURPLUS"],
        default="ADEQUATE",
    )

    wind_ratio = out["wind_generation_mw"] / out["demand_mw"].replace(0.0, np.nan)
    out["wind_penetration_ratio"] = wind_ratio.round(4)

    is_negative_plunge = out["agile_import_pence"] < plunge_price_threshold_pence
    is_near_zero_plunge = (out["agile_import_pence"] >= plunge_price_threshold_pence) & (
        out["agile_import_pence"] <= near_zero_price_threshold_pence
    )

    # Curtailment occurs when regional wind surges past local demand + export headroom or frequency rises
    is_wind_curtailment = (
        (wind_ratio >= curtailment_wind_ratio_threshold)
        & ((out["grid_frequency_hz"] >= curtailment_freq_hz) | (out["agile_import_pence"] <= near_zero_price_threshold_pence))
    ) | is_negative_plunge

    is_dunkelflaute_peak = (
        (wind_ratio <= dunkelflaute_wind_ratio_threshold)
        & (out["agile_import_pence"] >= dunkelflaute_price_threshold_pence)
        & (out["carbon_intensity_gco2_kwh"] >= 220.0)
    )

    out["is_negative_agile_plunge"] = is_negative_plunge.astype(bool)
    out["is_near_zero_plunge"] = is_near_zero_plunge.astype(bool)
    out["is_wind_curtailment_window"] = is_wind_curtailment.astype(bool)
    out["is_dunkelflaute_peak"] = is_dunkelflaute_peak.astype(bool)

    # Estimate curtailed wind MW during curtailment windows (excess above local absorption threshold)
    excess_mw = np.maximum(
        0.0,
        out["wind_generation_mw"] - curtailment_wind_ratio_threshold * out["demand_mw"],
    )
    out["estimated_curtailed_mw"] = np.where(is_wind_curtailment, np.round(excess_mw, 2), 0.0)

    out["grid_regime"] = np.select(
        [
            is_dunkelflaute_peak,
            is_wind_curtailment,
        ],
        [
            "Dunkelflaute Peak",
            "Wind Curtailment Plunge",
        ],
        default="Normal",
    )

    # Actionable flexibility signal for EV/battery orchestrators
    out["flexibility_signal"] = np.select(
        [
            is_negative_plunge,
            is_wind_curtailment,
            is_dunkelflaute_peak,
        ],
        [
            "CHARGE_MAX_PLUNGE",
            "ABSORB_CURTAILED_WIND",
            "DISCHARGE_DFS_PEAK",
        ],
        default="HOLD_OR_OPTIMIZE",
    )
    return out


def summarize_dno_headroom(enriched_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate substation transformer headroom and curtailment metrics across all 14 UK DNO regions."""
    df = enriched_df if "substation_headroom_mw" in enriched_df.columns else detect_grid_regimes(enriched_df)

    grouped = (
        df.groupby("dno_region_id", as_index=False)
        .agg(
            intervals=("timestamp", "count"),
            avg_load_mw=("substation_load_mw", "mean"),
            peak_load_mw=("substation_load_mw", "max"),
            min_headroom_mw=("substation_headroom_mw", "min"),
            avg_headroom_pct=("substation_headroom_pct", "mean"),
            curtailment_windows=("is_wind_curtailment_window", "sum"),
            negative_plunge_windows=("is_negative_agile_plunge", "sum"),
            curtailed_mwh=("estimated_curtailed_mw", lambda s: float(s.sum() * 0.5)),
        )
    )

    all_regions = pd.DataFrame(
        [
            {
                "dno_region_id": rid,
                "dno_region_name": meta["name"],
                "gsp_group": meta["gsp_group"],
                "operator": meta["operator"],
                "rated_capacity_mw": meta["rated_capacity_mw"],
            }
            for rid, meta in sorted(UK_DNO_REGIONS.items())
        ]
    )

    merged = all_regions.merge(grouped, on="dno_region_id", how="left")
    fill_zero = [
        "intervals",
        "avg_load_mw",
        "peak_load_mw",
        "curtailment_windows",
        "negative_plunge_windows",
        "curtailed_mwh",
    ]
    for col in fill_zero:
        merged[col] = merged[col].fillna(0)
    merged["min_headroom_mw"] = merged["min_headroom_mw"].fillna(merged["rated_capacity_mw"])
    merged["avg_headroom_pct"] = merged["avg_headroom_pct"].fillna(100.0)
    merged["min_headroom_pct"] = (
        (merged["min_headroom_mw"] / merged["rated_capacity_mw"]) * 100.0
    ).round(2)
    merged["headroom_status"] = np.select(
        [
            merged["min_headroom_pct"] < 10.0,
            merged["min_headroom_pct"] < 25.0,
            merged["min_headroom_pct"] >= 50.0,
        ],
        ["CRITICAL", "CONSTRAINED", "SURPLUS"],
        default="ADEQUATE",
    )
    return merged.round(2)


def build_curtailment_summary(enriched_df: pd.DataFrame) -> CurtailmentSummary:
    """Build a structured CurtailmentSummary from enriched telemetry."""
    df = enriched_df if "grid_regime" in enriched_df.columns else detect_grid_regimes(enriched_df)
    dno_summary = summarize_dno_headroom(df)
    constrained = (
        dno_summary.loc[
            dno_summary["headroom_status"].isin(["CRITICAL", "CONSTRAINED"]),
            "dno_region_id",
        ]
        .astype(int)
        .tolist()
    )
    return CurtailmentSummary(
        total_intervals=int(len(df)),
        wind_curtailment_windows=int(df["is_wind_curtailment_window"].sum()),
        negative_agile_windows=int(df["is_negative_agile_plunge"].sum()),
        dunkelflaute_peak_windows=int(df["is_dunkelflaute_peak"].sum()),
        constrained_dno_regions=constrained,
        total_estimated_curtailed_mwh=round(float(df["estimated_curtailed_mw"].sum() * 0.5), 2),
    )
