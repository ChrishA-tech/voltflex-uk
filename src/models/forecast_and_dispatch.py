"""Multi-Horizon LightGBM + MAPIE Conformal Forecaster and PyTorch RL EV/Battery Dispatch Optimizer."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import lightgbm as lgb
import mlflow
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from mapie.regression import SplitConformalRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.model_selection import TimeSeriesSplit

DEFAULT_MLFLOW_URI = "sqlite:///mlruns/mlflow.db"

FORECAST_FEATURE_COLUMNS = [
    "dno_region_id",
    "grid_frequency_hz",
    "wind_generation_mw",
    "solar_generation_mw",
    "demand_mw",
    "substation_load_mw",
    "wind_penetration_ratio",
    "sp_sin",
    "sp_cos",
    "dow_sin",
    "dow_cos",
    "carbon_lag_1",
    "carbon_lag_2",
    "carbon_lag_48",
    "price_lag_1",
    "price_lag_2",
    "price_lag_48",
]


@dataclass
class HorizonModelResult:
    """Artifacts and metrics for a single target + horizon conformal model."""

    target: str
    horizon_steps: int
    cv_rmse: float
    cv_mae: float
    test_rmse: float
    test_mae: float
    conformal_coverage_90: float
    mean_interval_width: float
    model: SplitConformalRegressor
    test_predictions: pd.DataFrame


@dataclass
class ForecastSuiteResult:
    """Collection of multi-horizon conformal models for carbon intensity and Agile price."""

    carbon_models: dict[int, HorizonModelResult]
    price_models: dict[int, HorizonModelResult]
    feature_columns: list[str]
    mlflow_run_id: str | None = None


def prepare_forecast_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer time-cyclic and lagged regional grid features in chronological order."""
    work = df.copy()
    work["timestamp"] = pd.to_datetime(work["timestamp"])
    work = work.sort_values(["timestamp", "dno_region_id"]).reset_index(drop=True)

    if "wind_penetration_ratio" not in work.columns:
        work["wind_penetration_ratio"] = work["wind_generation_mw"] / work["demand_mw"].replace(0.0, np.nan)
        work["wind_penetration_ratio"] = work["wind_penetration_ratio"].fillna(0.0)

    sp = work["timestamp"].dt.hour * 2 + (work["timestamp"].dt.minute // 30)
    dow = work["timestamp"].dt.day_of_week

    work["sp_sin"] = np.sin(2.0 * np.pi * sp / 48.0)
    work["sp_cos"] = np.cos(2.0 * np.pi * sp / 48.0)
    work["dow_sin"] = np.sin(2.0 * np.pi * dow / 7.0)
    work["dow_cos"] = np.cos(2.0 * np.pi * dow / 7.0)

    grp = work.groupby("dno_region_id", sort=False)
    work["carbon_lag_1"] = grp["carbon_intensity_gco2_kwh"].shift(1)
    work["carbon_lag_2"] = grp["carbon_intensity_gco2_kwh"].shift(2)
    work["carbon_lag_48"] = grp["carbon_intensity_gco2_kwh"].shift(48)

    work["price_lag_1"] = grp["agile_import_pence"].shift(1)
    work["price_lag_2"] = grp["agile_import_pence"].shift(2)
    work["price_lag_48"] = grp["agile_import_pence"].shift(48)

    for col in [
        "carbon_lag_1",
        "carbon_lag_2",
        "carbon_lag_48",
    ]:
        work[col] = work[col].fillna(work["carbon_intensity_gco2_kwh"])

    for col in [
        "price_lag_1",
        "price_lag_2",
        "price_lag_48",
    ]:
        work[col] = work[col].fillna(work["agile_import_pence"])

    return work


def _make_lgbm(random_state: int = 42) -> lgb.LGBMRegressor:
    return lgb.LGBMRegressor(
        n_estimators=120,
        learning_rate=0.06,
        num_leaves=31,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=random_state,
        verbosity=-1,
        n_jobs=1,
    )


def _train_single_horizon(
    featured_df: pd.DataFrame,
    target_col: str,
    horizon_steps: int,
    confidence_level: float = 0.90,
    random_state: int = 42,
) -> HorizonModelResult:
    """Train LightGBM + MAPIE SplitConformalRegressor with blocked time-series CV."""
    work = featured_df.copy()
    work["target_future"] = work.groupby("dno_region_id", sort=False)[target_col].shift(-horizon_steps)
    work = work.dropna(subset=["target_future"]).sort_values(["timestamp", "dno_region_id"]).reset_index(drop=True)

    X = work[FORECAST_FEATURE_COLUMNS].astype(float).to_numpy()
    y = work["target_future"].astype(float).to_numpy()

    n = len(work)
    if n < 60:
        raise ValueError(f"Insufficient samples ({n}) to train horizon={horizon_steps} forecaster")

    # Chronological blocked split: 60% train, 20% conformal calibration, 20% holdout test
    train_end = int(n * 0.60)
    cal_start = min(n - 20, train_end + horizon_steps)
    cal_end = int(n * 0.80)
    test_start = min(n - 10, cal_end + horizon_steps)

    X_train, y_train = X[:train_end], y[:train_end]
    X_cal, y_cal = X[cal_start:cal_end], y[cal_start:cal_end]
    X_test, y_test = X[test_start:], y[test_start:]

    # Blocked time-series cross-validation inside the training window
    n_splits = min(4, max(2, len(X_train) // 40))
    tscv = TimeSeriesSplit(n_splits=n_splits, gap=min(horizon_steps, max(1, len(X_train) // 20)))
    cv_rmses: list[float] = []
    cv_maes: list[float] = []
    for tr_idx, val_idx in tscv.split(X_train):
        cv_model = _make_lgbm(random_state=random_state)
        cv_model.fit(X_train[tr_idx], y_train[tr_idx])
        preds = cv_model.predict(X_train[val_idx])
        cv_rmses.append(float(root_mean_squared_error(y_train[val_idx], preds)))
        cv_maes.append(float(mean_absolute_error(y_train[val_idx], preds)))

    base_model = _make_lgbm(random_state=random_state)
    base_model.fit(X_train, y_train)

    conformal = SplitConformalRegressor(
        estimator=base_model,
        confidence_level=confidence_level,
        conformity_score="absolute",
        prefit=True,
    )
    conformal.conformalize(X_cal, y_cal)

    y_pred, y_pis = conformal.predict_interval(X_test)
    lower = y_pis[:, 0, 0]
    upper = y_pis[:, 1, 0]

    # Guard against numerical inversion
    lower_bound = np.minimum(lower, upper)
    upper_bound = np.maximum(lower, upper)
    covered = (y_test >= lower_bound) & (y_test <= upper_bound)

    test_df = work.iloc[test_start:].copy().reset_index(drop=True)
    test_df["y_true"] = y_test
    test_df["y_pred"] = y_pred
    test_df["y_lower_90"] = lower_bound
    test_df["y_upper_90"] = upper_bound
    test_df["is_covered_90"] = covered

    return HorizonModelResult(
        target=target_col,
        horizon_steps=horizon_steps,
        cv_rmse=round(float(np.mean(cv_rmses)), 4),
        cv_mae=round(float(np.mean(cv_maes)), 4),
        test_rmse=round(float(root_mean_squared_error(y_test, y_pred)), 4),
        test_mae=round(float(mean_absolute_error(y_test, y_pred)), 4),
        conformal_coverage_90=round(float(np.mean(covered)), 4),
        mean_interval_width=round(float(np.mean(upper_bound - lower_bound)), 4),
        model=conformal,
        test_predictions=test_df,
    )


def train_conformal_forecasters(
    df: pd.DataFrame,
    horizons: tuple[int, ...] = (1, 2, 4),
    confidence_level: float = 0.90,
    mlflow_tracking_uri: str | None = None,
    experiment_name: str = "voltflex-uk-conformal-forecasting",
    log_to_mlflow: bool = True,
) -> ForecastSuiteResult:
    """Train multi-horizon LightGBM + MAPIE 90% conformal regressors for carbon & Agile prices."""
    featured = prepare_forecast_features(df)
    carbon_models: dict[int, HorizonModelResult] = {}
    price_models: dict[int, HorizonModelResult] = {}

    for h in horizons:
        carbon_models[h] = _train_single_horizon(
            featured,
            target_col="carbon_intensity_gco2_kwh",
            horizon_steps=h,
            confidence_level=confidence_level,
        )
        price_models[h] = _train_single_horizon(
            featured,
            target_col="agile_import_pence",
            horizon_steps=h,
            confidence_level=confidence_level,
        )

    run_id: str | None = None
    if log_to_mlflow:
        uri = mlflow_tracking_uri or os.getenv("MLFLOW_TRACKING_URI") or DEFAULT_MLFLOW_URI
        if uri.startswith("sqlite:///"):
            db_file = Path(uri.replace("sqlite:///", "", 1))
            db_file.parent.mkdir(parents=True, exist_ok=True)
        mlflow.set_tracking_uri(uri)
        mlflow.set_experiment(experiment_name)
        with mlflow.start_run(run_name="lgbm-mapie-multihorizon") as run:
            run_id = run.info.run_id
            mlflow.log_param("horizons", list(horizons))
            mlflow.log_param("confidence_level", confidence_level)
            mlflow.log_param("n_rows", len(featured))
            for h, res in carbon_models.items():
                mlflow.log_metrics(
                    {
                        f"carbon_h{h}_cv_rmse": res.cv_rmse,
                        f"carbon_h{h}_cv_mae": res.cv_mae,
                        f"carbon_h{h}_test_rmse": res.test_rmse,
                        f"carbon_h{h}_test_mae": res.test_mae,
                        f"carbon_h{h}_coverage_90": res.conformal_coverage_90,
                        f"carbon_h{h}_width_90": res.mean_interval_width,
                    }
                )
            for h, res in price_models.items():
                mlflow.log_metrics(
                    {
                        f"price_h{h}_cv_rmse": res.cv_rmse,
                        f"price_h{h}_cv_mae": res.cv_mae,
                        f"price_h{h}_test_rmse": res.test_rmse,
                        f"price_h{h}_test_mae": res.test_mae,
                        f"price_h{h}_coverage_90": res.conformal_coverage_90,
                        f"price_h{h}_width_90": res.mean_interval_width,
                    }
                )

    return ForecastSuiteResult(
        carbon_models=carbon_models,
        price_models=price_models,
        feature_columns=list(FORECAST_FEATURE_COLUMNS),
        mlflow_run_id=run_id,
    )


# ---------------------------------------------------------------------------
# PyTorch RL / Policy-Gradient EV & Home Battery Dispatch Optimizer
# ---------------------------------------------------------------------------


@dataclass
class DispatchConstraints:
    """Physical and UK DNO G98/G99 electrical constraints for site dispatch."""

    g98_export_limit_kw: float = 3.68  # 16A single-phase @ 230V
    g99_export_limit_kw: float = 7.36  # Approved G99 export limit
    g99_approved: bool = False
    thermal_current_limit_amps: float = 60.0  # Site / feeder thermal current limit
    nominal_voltage_v: float = 230.0
    ev_capacity_kwh: float = 64.0
    ev_max_power_kw: float = 7.36  # 32A single-phase charger
    battery_capacity_kwh: float = 13.5
    battery_max_power_kw: float = 5.0
    soc_min_pct: float = 10.0
    soc_max_pct: float = 95.0
    carbon_weight_pence_per_kg: float = 8.5

    @property
    def active_export_limit_kw(self) -> float:
        return self.g99_export_limit_kw if self.g99_approved else self.g98_export_limit_kw

    @property
    def site_thermal_limit_kw(self) -> float:
        return round((self.thermal_current_limit_amps * self.nominal_voltage_v) / 1000.0, 3)


class EVBatteryDispatchNet(nn.Module):
    """Policy-Gradient (Actor-Critic) PyTorch network for half-hourly EV & home battery dispatch."""

    def __init__(self, state_dim: int = 7, hidden_dim: int = 64) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        # Actions: [ev_charge_norm in [0, 1], battery_power_norm in [-1, 1]]
        self.ev_head = nn.Linear(hidden_dim, 1)
        self.batt_head = nn.Linear(hidden_dim, 1)
        self.value_head = nn.Linear(hidden_dim, 1)
        self.log_std = nn.Parameter(torch.tensor([-1.1, -1.1], dtype=torch.float32))

    def forward(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.backbone(state)
        ev_mean = torch.sigmoid(self.ev_head(h))  # Charge-only [0, 1]
        batt_mean = torch.tanh(self.batt_head(h))  # Charge (+1) / Discharge (-1)
        mean_action = torch.cat([ev_mean, batt_mean], dim=-1)
        value = self.value_head(h).squeeze(-1)
        return mean_action, value


def build_rl_state_vector(
    agile_pence: float,
    carbon_gco2_kwh: float,
    ev_soc_pct: float,
    battery_soc_pct: float,
    substation_headroom_pct: float,
    settlement_period: int,
) -> np.ndarray:
    """Construct normalized 7-D state vector for EVBatteryDispatchNet."""
    return np.array(
        [
            np.clip(agile_pence / 40.0, -1.5, 3.0),
            np.clip(carbon_gco2_kwh / 400.0, 0.0, 2.5),
            np.clip(ev_soc_pct / 100.0, 0.0, 1.0),
            np.clip(battery_soc_pct / 100.0, 0.0, 1.0),
            np.clip(substation_headroom_pct / 100.0, 0.0, 1.0),
            np.sin(2.0 * np.pi * settlement_period / 48.0),
            np.cos(2.0 * np.pi * settlement_period / 48.0),
        ],
        dtype=np.float32,
    )


def enforce_thermal_and_soc_limits(
    proposed_ev_kw: float,
    proposed_battery_kw: float,
    ev_soc_pct: float,
    battery_soc_pct: float,
    base_load_kw: float,
    substation_headroom_pct: float,
    constraints: DispatchConstraints,
    dt_hours: float = 0.5,
) -> tuple[float, float, bool]:
    """Deterministic safety shield enforcing G98/G99 export, site thermal current, and SOC bounds.

    Returns (safe_ev_kw, safe_battery_kw, pre_shield_violated).
    """
    pre_violated = False

    # 1. Clip to asset inverter ratings
    ev_kw = float(np.clip(proposed_ev_kw, 0.0, constraints.ev_max_power_kw))
    batt_kw = float(
        np.clip(proposed_battery_kw, -constraints.battery_max_power_kw, constraints.battery_max_power_kw)
    )
    if abs(ev_kw - proposed_ev_kw) > 1e-4 or abs(batt_kw - proposed_battery_kw) > 1e-4:
        pre_violated = True

    # 2. Enforce SOC physical headroom over dt_hours
    max_ev_charge_kw = max(
        0.0, ((constraints.soc_max_pct - ev_soc_pct) / 100.0) * constraints.ev_capacity_kwh / dt_hours
    )
    if ev_kw > max_ev_charge_kw + 1e-4:
        pre_violated = True
    ev_kw = min(ev_kw, max_ev_charge_kw)

    max_batt_charge_kw = max(
        0.0, ((constraints.soc_max_pct - battery_soc_pct) / 100.0) * constraints.battery_capacity_kwh / dt_hours
    )
    max_batt_discharge_kw = max(
        0.0, ((battery_soc_pct - constraints.soc_min_pct) / 100.0) * constraints.battery_capacity_kwh / dt_hours
    )
    if batt_kw > max_batt_charge_kw + 1e-4 or batt_kw < -max_batt_discharge_kw - 1e-4:
        pre_violated = True
    batt_kw = float(np.clip(batt_kw, -max_batt_discharge_kw, max_batt_charge_kw))

    # 3. Enforce DNO substation critical headroom curtailment & site thermal current limit
    site_max_kw = constraints.site_thermal_limit_kw
    available_import_kw = max(0.0, site_max_kw - max(0.0, base_load_kw))

    if substation_headroom_pct < 10.0:
        # Critical substation thermal stress: cap controllable charging to 6A trickle regardless of battery discharge
        available_import_kw = min(available_import_kw, 1.38)
        if ev_kw + max(0.0, batt_kw) > 1.38 + 1e-4:
            pre_violated = True
            gross_charge = ev_kw + max(0.0, batt_kw)
            scale = 1.38 / gross_charge
            ev_kw *= scale
            if batt_kw > 0:
                batt_kw *= scale
    elif substation_headroom_pct < 25.0:
        available_import_kw = min(available_import_kw, site_max_kw * 0.65 - max(0.0, base_load_kw))
        available_import_kw = max(0.0, available_import_kw)

    total_device_kw = ev_kw + batt_kw
    if total_device_kw > available_import_kw + 1e-4:
        pre_violated = True
        if batt_kw > 0 and total_device_kw > 0:
            scale = available_import_kw / total_device_kw
            ev_kw *= scale
            batt_kw *= scale
        else:
            ev_kw = min(ev_kw, max(0.0, available_import_kw - batt_kw))

    # 4. Enforce G98 (3.68 kW) or G99 export limit: net_export = -(base_load + ev_kw + batt_kw)
    net_site_kw = base_load_kw + ev_kw + batt_kw
    export_limit_kw = constraints.active_export_limit_kw
    if net_site_kw < -export_limit_kw - 1e-4:
        pre_violated = True
        # Reduce battery discharge so net export stays strictly <= export_limit_kw
        batt_kw = -(export_limit_kw + base_load_kw + ev_kw)
        batt_kw = float(np.clip(batt_kw, -max_batt_discharge_kw, max_batt_charge_kw))

    # Final hard assertion of thermal current limit on net site power
    net_site_kw = base_load_kw + ev_kw + batt_kw
    site_amps = abs(net_site_kw * 1000.0) / constraints.nominal_voltage_v
    if site_amps > constraints.thermal_current_limit_amps + 1e-3:
        pre_violated = True
        if net_site_kw > 0:
            cap_dev = max(0.0, site_max_kw - base_load_kw)
            ratio = cap_dev / max(ev_kw + max(0.0, batt_kw), 1e-6)
            ev_kw *= min(1.0, ratio)
            if batt_kw > 0:
                batt_kw *= min(1.0, ratio)

    return round(float(ev_kw), 4), round(float(batt_kw), 4), pre_violated


@dataclass
class DispatchSimulationResult:
    """Outputs and safety metrics of an RL EV/battery dispatch rollout."""

    schedule_df: pd.DataFrame
    total_cost_gbp: float
    baseline_cost_gbp: float
    total_carbon_kg: float
    baseline_carbon_kg: float
    thermal_violation_rate: float
    pre_shield_violation_rate: float
    final_ev_soc_pct: float
    final_battery_soc_pct: float
    training_rewards: list[float] = field(default_factory=list)


def train_and_evaluate_rl_dispatcher(
    telemetry_df: pd.DataFrame,
    constraints: DispatchConstraints | None = None,
    epochs: int = 35,
    lr: float = 3e-3,
    seed: int = 42,
) -> tuple[EVBatteryDispatchNet, DispatchSimulationResult]:
    """Train EVBatteryDispatchNet with policy gradient (REINFORCE + baseline) and simulate dispatch."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    cfg = constraints or DispatchConstraints()
    work = telemetry_df.sort_values("timestamp").reset_index(drop=True)
    # Use a single representative DNO feeder series if multiple regions are present
    if work["dno_region_id"].nunique() > 1:
        top_region = int(work["dno_region_id"].mode().iloc[0])
        work = work[work["dno_region_id"] == top_region].sort_values("timestamp").reset_index(drop=True)

    # Cap training trajectory length for fast convergence while keeping full validation window
    episode_df = work.tail(min(len(work), 96)).reset_index(drop=True)

    net = EVBatteryDispatchNet(state_dim=7, hidden_dim=64)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    training_rewards: list[float] = []
    dt = 0.5

    for _ in range(epochs):
        ev_soc = 45.0
        batt_soc = 50.0
        log_probs: list[torch.Tensor] = []
        values: list[torch.Tensor] = []
        rewards: list[float] = []

        for _, row in episode_df.iterrows():
            ts = pd.Timestamp(row["timestamp"])
            sp = ts.hour * 2 + (ts.minute // 30)
            headroom_pct = float(row.get("substation_headroom_pct", 35.0))
            agile_p = float(row["agile_import_pence"])
            carbon_g = float(row["carbon_intensity_gco2_kwh"])

            state_np = build_rl_state_vector(
                agile_pence=agile_p,
                carbon_gco2_kwh=carbon_g,
                ev_soc_pct=ev_soc,
                battery_soc_pct=batt_soc,
                substation_headroom_pct=headroom_pct,
                settlement_period=sp,
            )
            state_t = torch.from_numpy(state_np).unsqueeze(0)
            mean_act, val = net(state_t)
            std = torch.exp(net.log_std).clamp(0.05, 0.5)
            dist = torch.distributions.Normal(mean_act.squeeze(0), std)
            sampled = dist.sample()
            log_prob = dist.log_prob(sampled).sum()

            raw_ev_kw = float(torch.clamp(sampled[0], 0.0, 1.0).item()) * cfg.ev_max_power_kw
            raw_batt_kw = float(torch.clamp(sampled[1], -1.0, 1.0).item()) * cfg.battery_max_power_kw

            safe_ev_kw, safe_batt_kw, pre_violated = enforce_thermal_and_soc_limits(
                proposed_ev_kw=raw_ev_kw,
                proposed_battery_kw=raw_batt_kw,
                ev_soc_pct=ev_soc,
                battery_soc_pct=batt_soc,
                base_load_kw=1.2,
                substation_headroom_pct=headroom_pct,
                constraints=cfg,
                dt_hours=dt,
            )

            ev_soc = float(
                np.clip(ev_soc + (safe_ev_kw * dt / cfg.ev_capacity_kwh) * 100.0, cfg.soc_min_pct, cfg.soc_max_pct)
            )
            batt_soc = float(
                np.clip(
                    batt_soc + (safe_batt_kw * dt / cfg.battery_capacity_kwh) * 100.0,
                    cfg.soc_min_pct,
                    cfg.soc_max_pct,
                )
            )

            net_kw = safe_ev_kw + safe_batt_kw
            energy_kwh = net_kw * dt
            # Export earns 15p/kWh SEG tariff when net_kw < 0, import pays agile_p
            if energy_kwh >= 0:
                cost_pence = energy_kwh * agile_p
            else:
                cost_pence = energy_kwh * 15.0  # revenue (negative cost)

            carbon_kg = energy_kwh * (carbon_g / 1000.0)
            # Encourage charging during plunge/low-carbon and keeping EV healthy
            soc_bonus = 0.08 * (ev_soc - 45.0) if sp in (12, 13, 14) else 0.0
            safety_penalty = -5.0 if pre_violated else 0.0
            reward = -(cost_pence + cfg.carbon_weight_pence_per_kg * carbon_kg) / 10.0 + soc_bonus + safety_penalty

            log_probs.append(log_prob)
            values.append(val.squeeze(0))
            rewards.append(float(reward))

        # Discounted returns
        returns: list[float] = []
        g = 0.0
        for r in reversed(rewards):
            g = r + 0.96 * g
            returns.insert(0, g)

        returns_t = torch.tensor(returns, dtype=torch.float32)
        values_t = torch.stack(values)
        advantages = returns_t - values_t.detach()
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-6)

        policy_loss = -(torch.stack(log_probs) * advantages).mean()
        value_loss = nn.functional.mse_loss(values_t, returns_t)
        loss = policy_loss + 0.5 * value_loss

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        optimizer.step()
        training_rewards.append(round(float(sum(rewards)), 3))

    sim_result = simulate_rl_dispatch(net, episode_df, constraints=cfg, training_rewards=training_rewards)
    return net, sim_result


def simulate_rl_dispatch(
    net: EVBatteryDispatchNet,
    telemetry_df: pd.DataFrame,
    constraints: DispatchConstraints | None = None,
    initial_ev_soc_pct: float = 45.0,
    initial_battery_soc_pct: float = 50.0,
    base_load_kw: float = 1.2,
    training_rewards: list[float] | None = None,
) -> DispatchSimulationResult:
    """Run deterministic policy inference with hard G98/G99 and thermal current shielding."""
    cfg = constraints or DispatchConstraints()
    work = telemetry_df.sort_values("timestamp").reset_index(drop=True)
    dt = 0.5

    ev_soc = float(initial_ev_soc_pct)
    batt_soc = float(initial_battery_soc_pct)
    rows: list[dict[str, Any]] = []
    net.eval()

    with torch.no_grad():
        for _, row in work.iterrows():
            ts = pd.Timestamp(row["timestamp"])
            sp = ts.hour * 2 + (ts.minute // 30)
            headroom_pct = float(row.get("substation_headroom_pct", 35.0))
            agile_p = float(row["agile_import_pence"])
            carbon_g = float(row["carbon_intensity_gco2_kwh"])

            state_np = build_rl_state_vector(
                agile_pence=agile_p,
                carbon_gco2_kwh=carbon_g,
                ev_soc_pct=ev_soc,
                battery_soc_pct=batt_soc,
                substation_headroom_pct=headroom_pct,
                settlement_period=sp,
            )
            mean_act, _ = net(torch.from_numpy(state_np).unsqueeze(0))
            raw_ev_kw = float(mean_act[0, 0].item()) * cfg.ev_max_power_kw
            raw_batt_kw = float(mean_act[0, 1].item()) * cfg.battery_max_power_kw

            safe_ev_kw, safe_batt_kw, pre_violated = enforce_thermal_and_soc_limits(
                proposed_ev_kw=raw_ev_kw,
                proposed_battery_kw=raw_batt_kw,
                ev_soc_pct=ev_soc,
                battery_soc_pct=batt_soc,
                base_load_kw=base_load_kw,
                substation_headroom_pct=headroom_pct,
                constraints=cfg,
                dt_hours=dt,
            )

            ev_soc = float(
                np.clip(ev_soc + (safe_ev_kw * dt / cfg.ev_capacity_kwh) * 100.0, cfg.soc_min_pct, cfg.soc_max_pct)
            )
            batt_soc = float(
                np.clip(
                    batt_soc + (safe_batt_kw * dt / cfg.battery_capacity_kwh) * 100.0,
                    cfg.soc_min_pct,
                    cfg.soc_max_pct,
                )
            )

            net_site_kw = base_load_kw + safe_ev_kw + safe_batt_kw
            site_current_amps = abs(net_site_kw * 1000.0) / cfg.nominal_voltage_v
            export_kw = max(0.0, -net_site_kw)

            thermal_violated = (
                site_current_amps > cfg.thermal_current_limit_amps + 1e-3
                or export_kw > cfg.active_export_limit_kw + 1e-3
            )

            flex_energy_kwh = (safe_ev_kw + safe_batt_kw) * dt
            cost_gbp = (
                (flex_energy_kwh * agile_p / 100.0)
                if flex_energy_kwh >= 0
                else (flex_energy_kwh * 15.0 / 100.0)
            )
            carbon_kg = flex_energy_kwh * (carbon_g / 1000.0)

            # Unoptimized dumb charging baseline (flat 3.68 kW EV import during peak/all hours)
            dumb_kwh = 1.84
            baseline_cost_gbp = dumb_kwh * agile_p / 100.0
            baseline_carbon_kg = dumb_kwh * (carbon_g / 1000.0)

            rows.append(
                {
                    "timestamp": ts,
                    "agile_import_pence": agile_p,
                    "carbon_intensity_gco2_kwh": carbon_g,
                    "substation_headroom_pct": headroom_pct,
                    "ev_dispatch_kw": safe_ev_kw,
                    "battery_dispatch_kw": safe_batt_kw,
                    "net_site_kw": round(net_site_kw, 4),
                    "site_current_amps": round(site_current_amps, 2),
                    "export_kw": round(export_kw, 4),
                    "ev_soc_pct": round(ev_soc, 2),
                    "battery_soc_pct": round(batt_soc, 2),
                    "cost_gbp": round(cost_gbp, 4),
                    "carbon_kg": round(carbon_kg, 4),
                    "baseline_cost_gbp": round(baseline_cost_gbp, 4),
                    "baseline_carbon_kg": round(baseline_carbon_kg, 4),
                    "pre_shield_violated": pre_violated,
                    "thermal_violated": thermal_violated,
                }
            )

    sched = pd.DataFrame(rows)
    return DispatchSimulationResult(
        schedule_df=sched,
        total_cost_gbp=round(float(sched["cost_gbp"].sum()), 2),
        baseline_cost_gbp=round(float(sched["baseline_cost_gbp"].sum()), 2),
        total_carbon_kg=round(float(sched["carbon_kg"].sum()), 2),
        baseline_carbon_kg=round(float(sched["baseline_carbon_kg"].sum()), 2),
        thermal_violation_rate=round(float(sched["thermal_violated"].mean()), 6),
        pre_shield_violation_rate=round(float(sched["pre_shield_violated"].mean()), 4),
        final_ev_soc_pct=round(ev_soc, 2),
        final_battery_soc_pct=round(batt_soc, 2),
        training_rewards=training_rewards or [],
    )
