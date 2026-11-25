"""UK Grid Telemetry Ingestion, Pandera Physical-Bound Contracts, GDPR Pseudonymisation, and Quarantine."""

from __future__ import annotations

import hashlib
import hmac
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd
import pandera.pandas as pa

DEFAULT_HMAC_SECRET = "voltflex-uk-gdpr-mpan-secret-2026"
DEFAULT_DB_PATH = Path("data/processed/voltflex.duckdb")
DEFAULT_QUARANTINE_DIR = Path("data/quarantine")

UK_DNO_REGIONS: dict[int, dict[str, Any]] = {
    10: {
        "name": "Eastern England",
        "gsp_group": "_A",
        "operator": "UK Power Networks",
        "rated_capacity_mw": 4800.0,
    },
    11: {
        "name": "East Midlands",
        "gsp_group": "_B",
        "operator": "National Grid Electricity Distribution",
        "rated_capacity_mw": 3900.0,
    },
    12: {
        "name": "London",
        "gsp_group": "_C",
        "operator": "UK Power Networks",
        "rated_capacity_mw": 5200.0,
    },
    13: {
        "name": "Merseyside & North Wales",
        "gsp_group": "_D",
        "operator": "SP Energy Networks",
        "rated_capacity_mw": 2700.0,
    },
    14: {
        "name": "West Midlands",
        "gsp_group": "_E",
        "operator": "National Grid Electricity Distribution",
        "rated_capacity_mw": 3600.0,
    },
    15: {
        "name": "North East England",
        "gsp_group": "_F",
        "operator": "Northern Powergrid",
        "rated_capacity_mw": 2500.0,
    },
    16: {
        "name": "North West England",
        "gsp_group": "_G",
        "operator": "Electricity North West",
        "rated_capacity_mw": 3500.0,
    },
    17: {
        "name": "North Scotland",
        "gsp_group": "_P",
        "operator": "Scottish & Southern Electricity Networks",
        "rated_capacity_mw": 2900.0,
    },
    18: {
        "name": "South Scotland",
        "gsp_group": "_N",
        "operator": "SP Energy Networks",
        "rated_capacity_mw": 3100.0,
    },
    19: {
        "name": "South East England",
        "gsp_group": "_J",
        "operator": "UK Power Networks",
        "rated_capacity_mw": 3700.0,
    },
    20: {
        "name": "Southern England",
        "gsp_group": "_H",
        "operator": "Scottish & Southern Electricity Networks",
        "rated_capacity_mw": 4300.0,
    },
    21: {
        "name": "South Wales",
        "gsp_group": "_K",
        "operator": "National Grid Electricity Distribution",
        "rated_capacity_mw": 2100.0,
    },
    22: {
        "name": "South Western England",
        "gsp_group": "_L",
        "operator": "National Grid Electricity Distribution",
        "rated_capacity_mw": 2800.0,
    },
    23: {
        "name": "Yorkshire",
        "gsp_group": "_M",
        "operator": "Northern Powergrid",
        "rated_capacity_mw": 3400.0,
    },
}

VALID_DNO_REGION_IDS: list[int] = sorted(UK_DNO_REGIONS.keys())

GridTelemetrySchema = pa.DataFrameSchema(
    columns={
        "timestamp": pa.Column(pa.DateTime, nullable=False),
        "dno_region_id": pa.Column(
            int,
            checks=pa.Check.isin(VALID_DNO_REGION_IDS),
            nullable=False,
            coerce=True,
        ),
        "grid_frequency_hz": pa.Column(
            float,
            checks=pa.Check.in_range(49.0, 51.0),
            nullable=False,
            coerce=True,
        ),
        "wind_generation_mw": pa.Column(
            float,
            checks=pa.Check.ge(0.0),
            nullable=False,
            coerce=True,
        ),
        "solar_generation_mw": pa.Column(
            float,
            checks=pa.Check.ge(0.0),
            nullable=False,
            coerce=True,
        ),
        "demand_mw": pa.Column(
            float,
            checks=pa.Check.gt(0.0),
            nullable=False,
            coerce=True,
        ),
        "substation_load_mw": pa.Column(
            float,
            checks=pa.Check.ge(0.0),
            nullable=False,
            coerce=True,
        ),
        "carbon_intensity_gco2_kwh": pa.Column(
            float,
            checks=pa.Check.in_range(0.0, 900.0),
            nullable=False,
            coerce=True,
        ),
        "agile_import_pence": pa.Column(
            float,
            checks=pa.Check.in_range(-50.0, 200.0),
            nullable=False,
            coerce=True,
        ),
        "ev_battery_soc_pct": pa.Column(
            float,
            checks=pa.Check.in_range(0.0, 100.0),
            nullable=False,
            coerce=True,
        ),
        "mpan_hash": pa.Column(
            str,
            checks=pa.Check.str_matches(r"^[a-f0-9]{64}$"),
            nullable=False,
        ),
    },
    strict=False,
    coerce=True,
)


@dataclass
class IngestionResult:
    """Summary of a telemetry ingestion and quarantine run."""

    valid_df: pd.DataFrame
    quarantined_df: pd.DataFrame
    duckdb_path: Path
    quarantine_path: Path | None
    total_rows: int
    valid_rows: int
    quarantined_rows: int

    @property
    def quarantine_rate(self) -> float:
        if self.total_rows == 0:
            return 0.0
        return self.quarantined_rows / self.total_rows


def pseudonymise_mpan(mpan: str, secret: str | None = None) -> str:
    """Apply UK GDPR HMAC-SHA256 pseudonymisation to a smart-meter MPAN."""
    if mpan is None or not str(mpan).strip():
        raise ValueError("MPAN cannot be empty")
    key = (secret or os.getenv("VOLTFLEX_MPAN_HMAC_SECRET") or DEFAULT_HMAC_SECRET).encode("utf-8")
    normalized = "".join(ch for ch in str(mpan).strip() if ch.isalnum())
    if len(normalized) < 6:
        raise ValueError("MPAN is too short to be valid")
    return hmac.new(key, normalized.encode("utf-8"), hashlib.sha256).hexdigest()


def _apply_pseudonymisation(df: pd.DataFrame, secret: str | None = None) -> tuple[pd.DataFrame, dict[int, str]]:
    """Replace raw MPAN values with HMAC-SHA256 digests and track invalid MPAN rows."""
    out = df.copy()
    reasons: dict[int, str] = {}
    hashes: list[str | None] = []

    has_raw_mpan = "mpan" in out.columns
    has_hashed = "mpan_hash" in out.columns

    for idx, row in out.iterrows():
        raw_val = row["mpan"] if has_raw_mpan else None
        existing_hash = row["mpan_hash"] if has_hashed else None

        if raw_val is not None and pd.notna(raw_val) and str(raw_val).strip():
            try:
                hashes.append(pseudonymise_mpan(str(raw_val), secret=secret))
            except ValueError as exc:
                hashes.append(None)
                reasons[idx] = f"invalid_mpan: {exc}"
        elif existing_hash is not None and pd.notna(existing_hash) and str(existing_hash).strip():
            hashes.append(str(existing_hash).strip())
        else:
            hashes.append(None)
            reasons[idx] = "missing_mpan"

    out["mpan_hash"] = hashes
    if has_raw_mpan:
        out = out.drop(columns=["mpan"])
    return out, reasons


def _detect_stale_rows(
    df: pd.DataFrame,
    max_staleness_hours: float = 24.0,
) -> dict[int, str]:
    """Identify stale or clock-skewed telemetry relative to `received_at` when present."""
    reasons: dict[int, str] = {}
    ts = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)

    for idx in df.index:
        if pd.isna(ts.loc[idx]):
            reasons[idx] = "invalid_timestamp"

    if "received_at" in df.columns:
        rx = pd.to_datetime(df["received_at"], errors="coerce", utc=True)
        for idx in df.index:
            if idx in reasons:
                continue
            if pd.isna(rx.loc[idx]):
                reasons[idx] = "invalid_received_at"
                continue
            age_hours = (rx.loc[idx] - ts.loc[idx]).total_seconds() / 3600.0
            if age_hours > max_staleness_hours:
                reasons[idx] = f"stale_telemetry (age={age_hours:.1f}h > {max_staleness_hours}h)"
            elif age_hours < -0.25:
                reasons[idx] = f"future_timestamp_skew (age={age_hours:.2f}h)"
    return reasons


def _validate_rows_with_pandera(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[int, str]]:
    """Validate records against GridTelemetrySchema, isolating failing row indices."""
    reasons: dict[int, str] = {}
    working = df.copy()
    working["timestamp"] = pd.to_datetime(working["timestamp"], errors="coerce", utc=True).dt.tz_localize(None)

    try:
        validated = GridTelemetrySchema.validate(working, lazy=True)
        return validated, reasons
    except pa.errors.SchemaErrors as exc:
        failure_df = exc.failure_cases
        if "index" in failure_df.columns:
            for _, case in failure_df.iterrows():
                row_idx = case["index"]
                if pd.notna(row_idx) and row_idx in working.index:
                    col = case.get("column", "schema")
                    check = case.get("check", "validation_failed")
                    prev = reasons.get(row_idx)
                    msg = f"{col}:{check}"
                    reasons[row_idx] = f"{prev}; {msg}" if prev else msg

        # Row-by-row fallback for any rows missed by vectorized coercion errors
        remaining_idx = [i for i in working.index if i not in reasons]
        if remaining_idx:
            valid_subset = working.loc[remaining_idx]
            try:
                validated = GridTelemetrySchema.validate(valid_subset, lazy=True)
                return validated, reasons
            except pa.errors.SchemaErrors as sub_exc:
                sub_failures = sub_exc.failure_cases
                if "index" in sub_failures.columns:
                    for _, case in sub_failures.iterrows():
                        row_idx = case["index"]
                        if pd.notna(row_idx) and row_idx in working.index:
                            reasons[row_idx] = f"{case.get('column', 'schema')}:{case.get('check', 'failed')}"

        clean_idx = [i for i in working.index if i not in reasons]
        if not clean_idx:
            return working.iloc[0:0].copy(), reasons
        validated = GridTelemetrySchema.validate(working.loc[clean_idx], lazy=False)
        return validated, reasons
    except pa.errors.SchemaError:
        # Handle global/type-coercion errors row-by-row
        clean_indices: list[Any] = []
        for idx in working.index:
            single = working.loc[[idx]]
            try:
                GridTelemetrySchema.validate(single, lazy=False)
                clean_indices.append(idx)
            except (pa.errors.SchemaError, pa.errors.SchemaErrors, ValueError, TypeError) as row_exc:
                reasons[idx] = f"schema_error: {str(row_exc).splitlines()[0]}"
        if not clean_indices:
            return working.iloc[0:0].copy(), reasons
        validated = GridTelemetrySchema.validate(working.loc[clean_indices], lazy=False)
        return validated, reasons


def persist_to_duckdb(
    df: pd.DataFrame,
    db_path: Path | str = DEFAULT_DB_PATH,
    table_name: str = "grid_telemetry",
    replace: bool = True,
) -> Path:
    """Persist validated telemetry DataFrame to DuckDB."""
    target = Path(db_path)
    target.parent.mkdir(parents=True, exist_ok=True)

    persist_df = df.copy()
    if "received_at" in persist_df.columns:
        persist_df["received_at"] = pd.to_datetime(persist_df["received_at"], errors="coerce", utc=True).dt.tz_localize(None)

    with duckdb.connect(str(target)) as con:
        con.register("incoming_telemetry", persist_df)
        if replace:
            con.execute(f"CREATE OR REPLACE TABLE {table_name} AS SELECT * FROM incoming_telemetry")
        else:
            con.execute(f"CREATE TABLE IF NOT EXISTS {table_name} AS SELECT * FROM incoming_telemetry WHERE 1=0")
            con.execute(f"INSERT INTO {table_name} SELECT * FROM incoming_telemetry")
    return target


def write_quarantine_parquet(
    quarantined_df: pd.DataFrame,
    quarantine_dir: Path | str = DEFAULT_QUARANTINE_DIR,
    batch_id: str | None = None,
) -> Path | None:
    """Write quarantined dead-letter telemetry rows to Parquet."""
    if quarantined_df.empty:
        return None
    target_dir = Path(quarantine_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    stamp = batch_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = target_dir / f"quarantine_{stamp}.parquet"

    safe_df = quarantined_df.copy()
    for col in safe_df.columns:
        if safe_df[col].dtype == object:
            safe_df[col] = safe_df[col].astype(str)
    safe_df.to_parquet(out_path, index=False)
    return out_path


def ingest_grid_telemetry(
    raw_df: pd.DataFrame,
    db_path: Path | str = DEFAULT_DB_PATH,
    quarantine_dir: Path | str = DEFAULT_QUARANTINE_DIR,
    mpan_secret: str | None = None,
    max_staleness_hours: float = 24.0,
    batch_id: str | None = None,
    replace_table: bool = True,
) -> IngestionResult:
    """Validate, pseudonymise, quarantine corrupted/stale rows, and persist clean telemetry."""
    if raw_df.empty:
        empty = raw_df.copy()
        return IngestionResult(
            valid_df=empty,
            quarantined_df=empty,
            duckdb_path=Path(db_path),
            quarantine_path=None,
            total_rows=0,
            valid_rows=0,
            quarantined_rows=0,
        )

    working = raw_df.copy().reset_index(drop=True)
    pseudo_df, mpan_reasons = _apply_pseudonymisation(working, secret=mpan_secret)
    stale_reasons = _detect_stale_rows(pseudo_df, max_staleness_hours=max_staleness_hours)

    pre_quarantine_reasons: dict[int, str] = {}
    for idx in pseudo_df.index:
        parts = []
        if idx in mpan_reasons:
            parts.append(mpan_reasons[idx])
        if idx in stale_reasons:
            parts.append(stale_reasons[idx])
        if parts:
            pre_quarantine_reasons[idx] = "; ".join(parts)

    candidate_idx = [i for i in pseudo_df.index if i not in pre_quarantine_reasons]
    candidate_df = pseudo_df.loc[candidate_idx]

    if candidate_df.empty:
        valid_df = candidate_df.copy()
        schema_reasons: dict[int, str] = {}
    else:
        valid_df, schema_reasons = _validate_rows_with_pandera(candidate_df)

    all_quarantine_reasons = {**pre_quarantine_reasons, **schema_reasons}
    quarantined_indices = sorted(all_quarantine_reasons.keys())

    if quarantined_indices:
        quarantined_df = pseudo_df.loc[quarantined_indices].copy()
        quarantined_df["quarantine_reason"] = [all_quarantine_reasons[i] for i in quarantined_indices]
        quarantined_df["quarantined_at"] = datetime.now(timezone.utc).isoformat()
    else:
        quarantined_df = pseudo_df.iloc[0:0].copy()
        quarantined_df["quarantine_reason"] = pd.Series(dtype="str")
        quarantined_df["quarantined_at"] = pd.Series(dtype="str")

    valid_df = valid_df.reset_index(drop=True)
    quarantined_df = quarantined_df.reset_index(drop=True)

    duck_path = persist_to_duckdb(valid_df, db_path=db_path, replace=replace_table)
    quarantine_path = write_quarantine_parquet(
        quarantined_df,
        quarantine_dir=quarantine_dir,
        batch_id=batch_id,
    )

    return IngestionResult(
        valid_df=valid_df,
        quarantined_df=quarantined_df,
        duckdb_path=duck_path,
        quarantine_path=quarantine_path,
        total_rows=len(working),
        valid_rows=len(valid_df),
        quarantined_rows=len(quarantined_df),
    )
