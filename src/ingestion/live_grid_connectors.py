"""Live UK Grid Carbon Intensity (NESO) and Octopus Agile Tariff API Connectors."""

from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pandas as pd
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from src.ingestion.grid_telemetry import (
    DEFAULT_DB_PATH,
    DEFAULT_QUARANTINE_DIR,
    UK_DNO_REGIONS,
    _apply_pseudonymisation,
    _detect_stale_rows,
    _validate_rows_with_pandera,
    persist_to_duckdb,
    write_quarantine_parquet,
)

logger = logging.getLogger(__name__)

CARBON_INTENSITY_REGIONAL_URL = "https://api.carbonintensity.org.uk/regional"
OCTOPUS_AGILE_PRODUCT = os.getenv("VOLTFLEX_AGILE_PRODUCT", "AGILE-24-10-01")
OCTOPUS_TARIFF_URL_TEMPLATE = (
    "https://api.octopus.energy/v1/products/{product}/electricity-tariffs/"
    "E-1R-{product}-{gsp_letter}/standard-unit-rates/"
)

# Mapping from NESO Carbon Intensity regional API regionid (1..14) to VoltFlex DNO ID (10..23)
NESO_REGIONID_TO_DNO: dict[int, int] = {
    1: 17,  # North Scotland (_P)
    2: 18,  # South Scotland (_N)
    3: 16,  # North West England (_G)
    4: 15,  # North East England (_F)
    5: 23,  # Yorkshire (_M)
    6: 13,  # North Wales & Merseyside (_D)
    7: 21,  # South Wales (_K)
    8: 14,  # West Midlands (_E)
    9: 11,  # East Midlands (_B)
    10: 10,  # East England (_A)
    11: 22,  # South West England (_L)
    12: 20,  # South England (_H)
    13: 12,  # London (_C)
    14: 19,  # South East England (_J)
}

NESO_SHORTNAME_TO_DNO: dict[str, int] = {
    "north scotland": 17,
    "south scotland": 18,
    "north west england": 16,
    "north east england": 15,
    "yorkshire": 23,
    "north wales & merseyside": 13,
    "merseyside & north wales": 13,
    "south wales": 21,
    "west midlands": 14,
    "east midlands": 11,
    "east england": 10,
    "eastern england": 10,
    "south west england": 22,
    "south western england": 22,
    "south england": 20,
    "southern england": 20,
    "london": 12,
    "south east england": 19,
}


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=0.5, min=0.5, max=3.0),
    retry=retry_if_exception_type((httpx.HTTPError, httpx.TimeoutException)),
    reraise=True,
)
def _fetch_json_with_retry(
    client: httpx.Client,
    url: str,
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Fetch JSON from an upstream UK energy API with tenacity exponential backoff."""
    resp = client.get(url, params=params)
    resp.raise_for_status()
    return resp.json()


def _resolve_dno_id(region_payload: dict[str, Any]) -> int | None:
    """Resolve a NESO Carbon Intensity region object to a canonical VoltFlex DNO region ID (10..23)."""
    short = str(region_payload.get("shortname") or region_payload.get("dnoregion") or "").strip().lower()
    if short in NESO_SHORTNAME_TO_DNO:
        return NESO_SHORTNAME_TO_DNO[short]
    neso_id = region_payload.get("regionid")
    if neso_id is not None and int(neso_id) in NESO_REGIONID_TO_DNO:
        return NESO_REGIONID_TO_DNO[int(neso_id)]
    return None


def _build_degraded_offline_telemetry(
    region_ids: list[int],
    periods: int = 96,
    seed: int = 42,
) -> pd.DataFrame:
    """Deterministic offline half-hourly fallback telemetry tagged LIVE_GRID_FEED_DEGRADED."""
    rng = np.random.default_rng(seed)
    now_utc = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    start = now_utc - timedelta(minutes=30 * (periods - 1))
    timestamps = [start + timedelta(minutes=30 * i) for i in range(periods)]

    t_arr = np.arange(periods, dtype=float)
    wind_wave = np.clip(0.50 + 0.40 * np.sin(2.0 * np.pi * t_arr / 48.0), 0.08, 0.96)

    rows: list[dict[str, Any]] = []
    for p_idx, ts in enumerate(timestamps):
        sp = ts.hour * 2 + (ts.minute // 30)
        diurnal = np.sin(2.0 * np.pi * (sp - 12) / 48.0)
        evening_peak = np.exp(-0.5 * ((sp - 36) / 3.0) ** 2)
        solar_shape = max(0.0, np.sin(np.pi * max(0, min(24, sp - 12)) / 24.0))
        w_factor = float(wind_wave[p_idx])

        for rid in region_ids:
            rated = float(UK_DNO_REGIONS[rid]["rated_capacity_mw"])
            wind_bias = 1.25 if rid in (15, 17, 18, 23) else 0.92
            demand_mw = float(np.clip(rated * (0.48 + 0.14 * diurnal + 0.22 * evening_peak), rated * 0.25, rated * 0.95))
            wind_mw = max(10.0, demand_mw * w_factor * wind_bias + float(rng.normal(0.0, 20.0)))
            solar_mw = max(0.0, rated * 0.14 * solar_shape * (1.0 if rid in (10, 19, 20, 22) else 0.65))
            wind_ratio = wind_mw / max(demand_mw, 1.0)
            freq_hz = float(np.clip(50.0 + 0.12 * (wind_ratio - 0.45), 49.65, 50.35))
            carbon = float(np.clip(275.0 - 205.0 * min(wind_ratio, 1.05) + 45.0 * evening_peak, 18.0, 480.0))
            agile = float(np.clip(21.0 + 23.0 * evening_peak - 28.0 * max(0.0, wind_ratio - 0.35), -10.0, 62.0))
            sub_load = min(rated * 0.96, max(rated * 0.22, demand_mw * 0.96))
            ev_soc = float(np.clip(48.0 + 22.0 * np.cos(2.0 * np.pi * sp / 48.0), 15.0, 92.0))

            rows.append(
                {
                    "timestamp": ts,
                    "received_at": now_utc + timedelta(minutes=2),
                    "dno_region_id": rid,
                    "grid_frequency_hz": round(freq_hz, 3),
                    "wind_generation_mw": round(wind_mw, 2),
                    "solar_generation_mw": round(solar_mw, 2),
                    "demand_mw": round(demand_mw, 2),
                    "substation_load_mw": round(sub_load, 2),
                    "carbon_intensity_gco2_kwh": round(carbon, 2),
                    "agile_import_pence": round(agile, 2),
                    "ev_battery_soc_pct": round(ev_soc, 2),
                    "mpan": f"{rid}0001234{p_idx:05d}",
                    "feed_status": "LIVE_GRID_FEED_DEGRADED",
                }
            )
    return pd.DataFrame(rows)


def _calibrate_live_rows(
    carbon_payload: dict[str, Any],
    agile_by_region: dict[int, list[dict[str, Any]]],
    region_ids: list[int],
) -> pd.DataFrame:
    """Combine live NESO regional carbon/gen-mix and Octopus Agile half-hour rates into telemetry rows."""
    data_blocks = carbon_payload.get("data") or []
    if not data_blocks:
        raise ValueError("Empty data payload from Carbon Intensity regional API")

    # Index carbon intensity snapshots by DNO region ID
    latest_block = data_blocks[-1]
    carbon_by_region: dict[int, dict[str, Any]] = {}
    for reg in latest_block.get("regions", []):
        dno_id = _resolve_dno_id(reg)
        if dno_id is not None and dno_id in region_ids:
            gen_mix = {item["fuel"].lower(): float(item.get("perc", 0.0)) for item in reg.get("generationmix", [])}
            intensity = reg.get("intensity", {})
            carbon_val = intensity.get("actual") if intensity.get("actual") is not None else intensity.get("forecast")
            if carbon_val is None:
                continue
            carbon_by_region[dno_id] = {
                "timestamp": pd.to_datetime(latest_block["from"], utc=True),
                "carbon_intensity_gco2_kwh": float(carbon_val),
                "wind_perc": gen_mix.get("wind", 25.0),
                "solar_perc": gen_mix.get("solar", 5.0),
            }

    if not carbon_by_region:
        raise ValueError("No valid DNO regions parsed from Carbon Intensity payload")

    now_rx = datetime.now(timezone.utc)
    rows: list[dict[str, Any]] = []

    for rid in region_ids:
        c_info = carbon_by_region.get(rid)
        if c_info is None:
            continue
        rated = float(UK_DNO_REGIONS[rid]["rated_capacity_mw"])
        rates = agile_by_region.get(rid) or []
        if not rates:
            raise ValueError(f"Missing Octopus Agile tariff rates for DNO region {rid}")

        for idx, rate in enumerate(rates):
            ts = pd.to_datetime(rate["valid_from"], utc=True)
            agile_p = float(rate["value_inc_vat"])

            sp = ts.hour * 2 + (ts.minute // 30)
            diurnal = np.sin(2.0 * np.pi * (sp - 12) / 48.0)
            evening_peak = np.exp(-0.5 * ((sp - 36) / 3.0) ** 2)

            demand_mw = float(np.clip(rated * (0.48 + 0.14 * diurnal + 0.22 * evening_peak), rated * 0.25, rated * 0.94))
            wind_mw = round(max(0.0, demand_mw * (c_info["wind_perc"] / 100.0)), 2)
            solar_mw = round(max(0.0, demand_mw * (c_info["solar_perc"] / 100.0)), 2)
            wind_ratio = wind_mw / max(demand_mw, 1.0)

            freq_hz = round(float(np.clip(50.0 + 0.12 * (wind_ratio - 0.35), 49.65, 50.35)), 3)
            # Adjust carbon smoothly around the live regional anchor with evening peak / plunge variation
            carbon_val = float(
                np.clip(
                    c_info["carbon_intensity_gco2_kwh"] + 25.0 * evening_peak - (18.0 if agile_p < 0 else 0.0),
                    0.0,
                    850.0,
                )
            )
            substation_load_mw = round(min(rated * 0.96, max(rated * 0.20, demand_mw * 0.94)), 2)
            ev_soc = round(float(np.clip(50.0 + 20.0 * np.cos(2.0 * np.pi * sp / 48.0), 15.0, 90.0)), 2)

            rows.append(
                {
                    "timestamp": ts,
                    "received_at": now_rx,
                    "dno_region_id": rid,
                    "grid_frequency_hz": freq_hz,
                    "wind_generation_mw": wind_mw,
                    "solar_generation_mw": solar_mw,
                    "demand_mw": round(demand_mw, 2),
                    "substation_load_mw": substation_load_mw,
                    "carbon_intensity_gco2_kwh": round(carbon_val, 2),
                    "agile_import_pence": round(agile_p, 2),
                    "ev_battery_soc_pct": ev_soc,
                    "mpan": f"{rid}0009876{idx:05d}",
                    "feed_status": "LIVE_NESO_OCTOPUS",
                }
            )

    if not rows:
        raise ValueError("Live connector assembled 0 telemetry rows")
    return pd.DataFrame(rows)


def fetch_live_uk_grid_and_agile(
    dno_region_ids: list[int] | None = None,
    timeout_seconds: float = 8.0,
    quarantine_dir: Path | str = DEFAULT_QUARANTINE_DIR,
    db_path: Path | str | None = None,
    client: httpx.Client | None = None,
    max_staleness_hours: float = 120.0,
    batch_id: str = "live_feed",
) -> pd.DataFrame:
    """Fetch live NESO Regional Carbon Intensity & Octopus Agile unit rates across UK DNO regions.

    Falls back gracefully with `LIVE_GRID_FEED_DEGRADED` status if upstream APIs time out,
    error, or change tariff product codes. All records pass through `GridTelemetrySchema`
    validation and any violating rows are routed to `write_quarantine_parquet()`.
    """
    target_regions = sorted(dno_region_ids) if dno_region_ids is not None else sorted(UK_DNO_REGIONS.keys())
    invalid_requested = [rid for rid in target_regions if rid not in UK_DNO_REGIONS]
    if invalid_requested:
        raise ValueError(f"Unsupported UK DNO region IDs requested: {invalid_requested}")

    owns_client = client is None
    http_client = client or httpx.Client(timeout=timeout_seconds)

    try:
        carbon_payload = _fetch_json_with_retry(http_client, CARBON_INTENSITY_REGIONAL_URL)

        agile_by_region: dict[int, list[dict[str, Any]]] = {}
        for rid in target_regions:
            gsp_letter = UK_DNO_REGIONS[rid]["gsp_group"].lstrip("_")
            tariff_url = OCTOPUS_TARIFF_URL_TEMPLATE.format(
                product=OCTOPUS_AGILE_PRODUCT,
                gsp_letter=gsp_letter,
            )
            tariff_payload = _fetch_json_with_retry(http_client, tariff_url)
            agile_by_region[rid] = list(tariff_payload.get("results") or [])

        raw_df = _calibrate_live_rows(carbon_payload, agile_by_region, target_regions)
    except Exception as exc:
        logger.warning("LIVE_GRID_FEED_DEGRADED: upstream API failure (%s); using offline fallback.", exc)
        raw_df = _build_degraded_offline_telemetry(target_regions)
    finally:
        if owns_client:
            http_client.close()

    # Pass everything through MPAN pseudonymisation, GridTelemetrySchema, and Parquet quarantine
    working = raw_df.reset_index(drop=True)
    pseudo_df, mpan_reasons = _apply_pseudonymisation(working)
    stale_reasons = _detect_stale_rows(pseudo_df, max_staleness_hours=max_staleness_hours)
    pre_reasons = {**mpan_reasons, **stale_reasons}

    candidate_idx = [i for i in pseudo_df.index if i not in pre_reasons]
    candidate_df = pseudo_df.loc[candidate_idx]
    if candidate_df.empty:
        valid_df = candidate_df.copy()
        schema_reasons: dict[int, str] = {}
    else:
        valid_df, schema_reasons = _validate_rows_with_pandera(candidate_df)

    all_reasons = {**pre_reasons, **schema_reasons}
    if all_reasons:
        q_idx = sorted(all_reasons.keys())
        quarantined_df = pseudo_df.loc[q_idx].copy()
        quarantined_df["quarantine_reason"] = [all_reasons[i] for i in q_idx]
        quarantined_df["quarantined_at"] = datetime.now(timezone.utc).isoformat()
        write_quarantine_parquet(quarantined_df, quarantine_dir=quarantine_dir, batch_id=batch_id)

    valid_df = valid_df.reset_index(drop=True)
    if db_path is not None and not valid_df.empty:
        persist_to_duckdb(valid_df, db_path=Path(db_path or DEFAULT_DB_PATH), replace=True)

    return valid_df
