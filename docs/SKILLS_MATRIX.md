# VoltFlex UK — 300-Skill Engineering & Production Capability Matrix

Comprehensive mapping of **300** concrete technical capabilities across **Real-Time Grid Streaming**, **DNO Spatial Analytics**, **Conformal Time-Series Forecasting**, **PyTorch Actor-Critic RL Dispatch**, **Guardrailed LangGraph RAG**, and **MLOps** directly to `voltflex-uk` source files and symbols.

---

## 1. Real-Time Grid Streaming (Skills 1–50)

| # | Capability | `voltflex-uk` File | Symbol / Function |
|---:|---|---|---|
| 1 | Live NESO Regional Carbon Intensity REST ingestion | `src/ingestion/live_grid_connectors.py` | `fetch_live_uk_grid_and_agile` |
| 2 | Live Octopus Energy Open Tariff API polling | `src/ingestion/live_grid_connectors.py` | `OCTOPUS_TARIFF_URL_TEMPLATE` |
| 3 | Exponential-backoff HTTP retry on transient faults | `src/ingestion/live_grid_connectors.py` | `_fetch_json_with_retry` |
| 4 | Configurable Agile product code override via env | `src/ingestion/live_grid_connectors.py` | `OCTOPUS_AGILE_PRODUCT` |
| 5 | Graceful upstream degradation flagging (`LIVE_GRID_FEED_DEGRADED`) | `src/ingestion/live_grid_connectors.py` | `_build_degraded_offline_telemetry` |
| 6 | NESO region ID (1–14) to UK DNO ID (10–23) mapping | `src/ingestion/live_grid_connectors.py` | `NESO_REGIONID_TO_DNO` |
| 7 | NESO region shortname normalization & lookup | `src/ingestion/live_grid_connectors.py` | `_resolve_dno_id` |
| 8 | Live wind/solar generation mix percentage parsing | `src/ingestion/live_grid_connectors.py` | `_calibrate_live_rows` |
| 9 | VAT-inclusive Octopus Agile half-hour rate ingestion | `src/ingestion/live_grid_connectors.py` | `_calibrate_live_rows` |
| 10 | Negative plunge tariff rate ingestion & preservation | `src/ingestion/live_grid_connectors.py` | `_calibrate_live_rows` |
| 11 | Injected `httpx.Client` dependency support for testing | `src/ingestion/live_grid_connectors.py` | `fetch_live_uk_grid_and_agile` |
| 12 | Strict DNO region allowlist validation at connector entry | `src/ingestion/live_grid_connectors.py` | `fetch_live_uk_grid_and_agile` |
| 13 | Pandera DataFrame physical-bound schema enforcement | `src/ingestion/grid_telemetry.py` | `GridTelemetrySchema` |
| 14 | UK grid frequency physical range check (49.0–51.0 Hz) | `src/ingestion/grid_telemetry.py` | `GridTelemetrySchema` |
| 15 | Non-negative wind generation contract check | `src/ingestion/grid_telemetry.py` | `GridTelemetrySchema` |
| 16 | Non-negative solar generation contract check | `src/ingestion/grid_telemetry.py` | `GridTelemetrySchema` |
| 17 | Strictly positive regional demand validation | `src/ingestion/grid_telemetry.py` | `GridTelemetrySchema` |
| 18 | Non-negative substation transformer load validation | `src/ingestion/grid_telemetry.py` | `GridTelemetrySchema` |
| 19 | Bounded carbon intensity range contract (0–900 gCO2/kWh) | `src/ingestion/grid_telemetry.py` | `GridTelemetrySchema` |
| 20 | Bounded Agile tariff range contract (-50 to +200 p/kWh) | `src/ingestion/grid_telemetry.py` | `GridTelemetrySchema` |
| 21 | Bounded EV/battery SOC contract (0–100%) | `src/ingestion/grid_telemetry.py` | `GridTelemetrySchema` |
| 22 | Canonical 14 UK DNO region ID membership check | `src/ingestion/grid_telemetry.py` | `VALID_DNO_REGION_IDS` |
| 23 | Lazy Pandera multi-error row isolation | `src/ingestion/grid_telemetry.py` | `_validate_rows_with_pandera` |
| 24 | Row-by-row coercion failure isolation fallback | `src/ingestion/grid_telemetry.py` | `_validate_rows_with_pandera` |
| 25 | UK GDPR smart-meter MPAN HMAC-SHA256 pseudonymisation | `src/ingestion/grid_telemetry.py` | `pseudonymise_mpan` |
| 26 | Secret-keyed MPAN digest environment configuration | `src/ingestion/grid_telemetry.py` | `pseudonymise_mpan` |
| 27 | MPAN normalization stripping non-alphanumeric whitespace | `src/ingestion/grid_telemetry.py` | `pseudonymise_mpan` |
| 28 | Minimum MPAN length validation prior to hashing | `src/ingestion/grid_telemetry.py` | `pseudonymise_mpan` |
| 29 | Raw MPAN column stripping prior to persistence | `src/ingestion/grid_telemetry.py` | `_apply_pseudonymisation` |
| 30 | Regex verification of 64-char SHA-256 hex digest | `src/ingestion/grid_telemetry.py` | `GridTelemetrySchema` |
| 31 | Missing/malformed MPAN dead-letter flagging | `src/ingestion/grid_telemetry.py` | `_apply_pseudonymisation` |
| 32 | Telemetry staleness detection against `received_at` | `src/ingestion/grid_telemetry.py` | `_detect_stale_rows` |
| 33 | Future clock-skew telemetry rejection (> 15 min) | `src/ingestion/grid_telemetry.py` | `_detect_stale_rows` |
| 34 | Invalid/NaT timestamp detection & quarantine | `src/ingestion/grid_telemetry.py` | `_detect_stale_rows` |
| 35 | Parquet Dead-Letter Quarantine serialization | `src/ingestion/grid_telemetry.py` | `write_quarantine_parquet` |
| 36 | Per-row quarantine reason attribution metadata | `src/ingestion/grid_telemetry.py` | `ingest_grid_telemetry` |
| 37 | UTC ISO-8601 `quarantined_at` audit stamping | `src/ingestion/grid_telemetry.py` | `ingest_grid_telemetry` |
| 38 | DuckDB analytical telemetry persistence | `src/ingestion/grid_telemetry.py` | `persist_to_duckdb` |
| 39 | Idempotent DuckDB table replacement & append modes | `src/ingestion/grid_telemetry.py` | `persist_to_duckdb` |
| 40 | Timezone-normalized UTC timestamp ingestion | `src/ingestion/grid_telemetry.py` | `_validate_rows_with_pandera` |
| 41 | Structured batch ingestion metrics (`IngestionResult`) | `src/ingestion/grid_telemetry.py` | `IngestionResult` |
| 42 | Automated quarantine rate computation | `src/ingestion/grid_telemetry.py` | `IngestionResult.quarantine_rate` |
| 43 | Empty DataFrame ingestion short-circuit safety | `src/ingestion/grid_telemetry.py` | `ingest_grid_telemetry` |
| 44 | Synthetic UK half-hourly stream generator with 3% corruption | `src/run_pipeline.py` | `generate_uk_grid_telemetry` |
| 45 | Multi-mode fault injection (freq, wind, DNO, stale, MPAN) | `src/run_pipeline.py` | `generate_uk_grid_telemetry` |
| 46 | Live-to-historical backfill blending for short windows | `src/run_pipeline.py` | `run_end_to_end` |
| 47 | CLI `--live / --no-live` feed mode selection | `src/run_pipeline.py` | `parse_args` |
| 48 | FastAPI telemetry staleness threshold guard (> 1800s) | `src/app/main.py` | `optimize_dispatch` |
| 49 | FastAPI latest regional DuckDB telemetry reader | `src/app/main.py` | `_load_latest_region_telemetry` |
| 50 | Conservative G98-safe degraded-telemetry dispatch fallback | `src/app/main.py` | `optimize_dispatch` |

---

## 2. DNO Spatial Analytics (Skills 51–100)

| # | Capability | `voltflex-uk` File | Symbol / Function |
|---:|---|---|---|
| 51 | Canonical registry of all 14 GB DNO licence areas | `src/ingestion/grid_telemetry.py` | `UK_DNO_REGIONS` |
| 52 | GSP Group suffix mapping (`_A` through `_P`) | `src/ingestion/grid_telemetry.py` | `UK_DNO_REGIONS` |
| 53 | DNO operator attribution (UKPN, NGED, SPEN, SSEN, NPG, ENW) | `src/ingestion/grid_telemetry.py` | `UK_DNO_REGIONS` |
| 54 | Regional substation transformer rated MVA/MW capacities | `src/ingestion/grid_telemetry.py` | `UK_DNO_REGIONS` |
| 55 | Live GSP letter resolution for Octopus regional tariffs | `src/ingestion/live_grid_connectors.py` | `fetch_live_uk_grid_and_agile` |
| 56 | Real-time regional transformer headroom MW calculation | `src/analytics/curtailment_detector.py` | `detect_grid_regimes` |
| 57 | Percentage substation headroom computation | `src/analytics/curtailment_detector.py` | `detect_grid_regimes` |
| 58 | Regional transformer thermal utilization percentage | `src/analytics/curtailment_detector.py` | `detect_grid_regimes` |
| 59 | Four-tier DNO headroom status (`CRITICAL`/`CONSTRAINED`/`ADEQUATE`/`SURPLUS`) | `src/analytics/curtailment_detector.py` | `detect_grid_regimes` |
| 60 | Critical substation threshold classification (< 10% headroom) | `src/analytics/curtailment_detector.py` | `detect_grid_regimes` |
| 61 | Constrained substation threshold classification (< 25% headroom) | `src/analytics/curtailment_detector.py` | `detect_grid_regimes` |
| 62 | Surplus substation headroom classification (>= 50% headroom) | `src/analytics/curtailment_detector.py` | `detect_grid_regimes` |
| 63 | Regional wind penetration ratio (`wind_mw / demand_mw`) | `src/analytics/curtailment_detector.py` | `detect_grid_regimes` |
| 64 | Negative Octopus Agile plunge-pricing detection (`< 0 p/kWh`) | `src/analytics/curtailment_detector.py` | `detect_grid_regimes` |
| 65 | Near-zero Agile plunge band detection (`0–3 p/kWh`) | `src/analytics/curtailment_detector.py` | `detect_grid_regimes` |
| 66 | Regional wind curtailment window detection (ratio + freq/price) | `src/analytics/curtailment_detector.py` | `detect_grid_regimes` |
| 67 | Estimated curtailed wind power (MW) quantification | `src/analytics/curtailment_detector.py` | `detect_grid_regimes` |
| 68 | Half-hourly curtailed wind energy (MWh) integration | `src/analytics/curtailment_detector.py` | `summarize_dno_headroom` |
| 69 | Regional `Dunkelflaute Peak` regime detection | `src/analytics/curtailment_detector.py` | `detect_grid_regimes` |
| 70 | Unified 3-class grid regime labeling | `src/analytics/curtailment_detector.py` | `detect_grid_regimes` |
| 71 | Actionable asset flexibility signal generation (`CHARGE_MAX_PLUNGE`) | `src/analytics/curtailment_detector.py` | `detect_grid_regimes` |
| 72 | Wind absorption flexibility signaling (`ABSORB_CURTAILED_WIND`) | `src/analytics/curtailment_detector.py` | `detect_grid_regimes` |
| 73 | Peak discharge flexibility signaling (`DISCHARGE_DFS_PEAK`) | `src/analytics/curtailment_detector.py` | `detect_grid_regimes` |
| 74 | Complete 14-region DNO spatial summary rollup | `src/analytics/curtailment_detector.py` | `summarize_dno_headroom` |
| 75 | Left-join completeness across missing DNO regions | `src/analytics/curtailment_detector.py` | `summarize_dno_headroom` |
| 76 | Per-DNO peak & average transformer load aggregation | `src/analytics/curtailment_detector.py` | `summarize_dno_headroom` |
| 77 | Per-DNO minimum headroom MW & percentage tracking | `src/analytics/curtailment_detector.py` | `summarize_dno_headroom` |
| 78 | Per-DNO curtailment & negative plunge event counts | `src/analytics/curtailment_detector.py` | `summarize_dno_headroom` |
| 79 | Constrained DNO region ID list extraction | `src/analytics/curtailment_detector.py` | `build_curtailment_summary` |
| 80 | National curtailed MWh aggregation across DNOs | `src/analytics/curtailment_detector.py` | `build_curtailment_summary` |
| 81 | Structured `CurtailmentSummary` dataclass reporting | `src/analytics/curtailment_detector.py` | `CurtailmentSummary` |
| 82 | Scotland & Northern wind-bias regional load synthesis | `src/run_pipeline.py` | `generate_uk_grid_telemetry` |
| 83 | Southern UK solar generation regional weighting | `src/run_pipeline.py` | `generate_uk_grid_telemetry` |
| 84 | Regional evening demand peak curve modeling | `src/run_pipeline.py` | `generate_uk_grid_telemetry` |
| 85 | Live DNO spatial demand & wind calibration | `src/ingestion/live_grid_connectors.py` | `_calibrate_live_rows` |
| 86 | Regional carbon adjustment for evening peak & plunge | `src/ingestion/live_grid_connectors.py` | `_calibrate_live_rows` |
| 87 | Regional grid frequency coupling to wind surplus | `src/ingestion/live_grid_connectors.py` | `_calibrate_live_rows` |
| 88 | Regional DNO lookup in FastAPI `/optimize-dispatch` | `src/app/main.py` | `optimize_dispatch` |
| 89 | FastAPI DNO region count health reporting (`14`) | `src/app/main.py` | `health` |
| 90 | FastAPI regional regime & signal classification | `src/app/main.py` | `optimize_dispatch` |
| 91 | Streamlit 14-region DNO headroom comparison table | `src/app/dashboard.py` | `main` |
| 92 | Streamlit interactive DNO region selector | `src/app/dashboard.py` | `main` |
| 93 | Streamlit regional Agile price time-series visualization | `src/app/dashboard.py` | `main` |
| 94 | Streamlit regional carbon intensity curve visualization | `src/app/dashboard.py` | `main` |
| 95 | Streamlit national curtailment & plunge KPI ribbon | `src/app/dashboard.py` | `main` |
| 96 | Per-region lagged target engineering without cross-DNO leakage | `src/models/forecast_and_dispatch.py` | `prepare_forecast_features` |
| 97 | Dominant feeder trajectory selection for RL simulation | `src/models/forecast_and_dispatch.py` | `train_and_evaluate_rl_dispatcher` |
| 98 | Regime-stratified spatial error audit across DNO test set | `src/evaluation/error_analysis.py` | `_evaluate_by_regime` |
| 99 | Substation headroom-aware safety interlock in LangGraph | `src/agents/flexibility_copilot.py` | `_check_g98_g99_interlock` |
| 100 | Regional DNO tariff suffix test coverage (`_C` London, etc.) | `tests/test_contracts_and_errors.py` | `test_curtailment_and_14_dno_headroom_analytics` |

---

## 3. Conformal Time-Series Forecasting (Skills 101–150)

| # | Capability | `voltflex-uk` File | Symbol / Function |
|---:|---|---|---|
| 101 | Chronological multi-region feature sorting | `src/models/forecast_and_dispatch.py` | `prepare_forecast_features` |
| 102 | UK 48-settlement-period cyclic sine/cosine encoding | `src/models/forecast_and_dispatch.py` | `prepare_forecast_features` |
| 103 | Day-of-week cyclic sine/cosine encoding | `src/models/forecast_and_dispatch.py` | `prepare_forecast_features` |
| 104 | Grouped DNO 30-min carbon lag (`carbon_lag_1`) | `src/models/forecast_and_dispatch.py` | `prepare_forecast_features` |
| 105 | Grouped DNO 60-min carbon lag (`carbon_lag_2`) | `src/models/forecast_and_dispatch.py` | `prepare_forecast_features` |
| 106 | Grouped DNO 24-hour carbon lag (`carbon_lag_48`) | `src/models/forecast_and_dispatch.py` | `prepare_forecast_features` |
| 107 | Grouped DNO 30-min Agile price lag (`price_lag_1`) | `src/models/forecast_and_dispatch.py` | `prepare_forecast_features` |
| 108 | Grouped DNO 60-min Agile price lag (`price_lag_2`) | `src/models/forecast_and_dispatch.py` | `prepare_forecast_features` |
| 109 | Grouped DNO 24-hour Agile price lag (`price_lag_48`) | `src/models/forecast_and_dispatch.py` | `prepare_forecast_features` |
| 110 | Cold-start lag imputation using current-step fallback | `src/models/forecast_and_dispatch.py` | `prepare_forecast_features` |
| 111 | Canonical 17-feature vector definition | `src/models/forecast_and_dispatch.py` | `FORECAST_FEATURE_COLUMNS` |
| 112 | Multi-horizon direct target shifting (`h = 1, 2, 4`) | `src/models/forecast_and_dispatch.py` | `_train_single_horizon` |
| 113 | Leakage-free grouped target shift per DNO region | `src/models/forecast_and_dispatch.py` | `_train_single_horizon` |
| 114 | Strict temporal Train/Calibration/Test block partitioning | `src/models/forecast_and_dispatch.py` | `_train_single_horizon` |
| 115 | Horizon-gap buffer between Train, Calibration, and Test blocks | `src/models/forecast_and_dispatch.py` | `_train_single_horizon` |
| 116 | Blocked `TimeSeriesSplit` cross-validation with gap | `src/models/forecast_and_dispatch.py` | `_train_single_horizon` |
| 117 | Adaptive fold count for variable sample sizes | `src/models/forecast_and_dispatch.py` | `_train_single_horizon` |
| 118 | Gradient-boosted tree regression via `LGBMRegressor` | `src/models/forecast_and_dispatch.py` | `_make_lgbm` |
| 119 | Deterministic LightGBM seed & single-thread safety | `src/models/forecast_and_dispatch.py` | `_make_lgbm` |
| 120 | Feature subsampling & row subsampling regularization | `src/models/forecast_and_dispatch.py` | `_make_lgbm` |
| 121 | MAPIE `SplitConformalRegressor` uncertainty quantification | `src/models/forecast_and_dispatch.py` | `_train_single_horizon` |
| 122 | Absolute conformity score calibration (`conformity_score="absolute"`) | `src/models/forecast_and_dispatch.py` | `_train_single_horizon` |
| 123 | Prefit conformal calibration on disjoint temporal block | `src/models/forecast_and_dispatch.py` | `_train_single_horizon` |
| 124 | Target 90% prediction interval generation (`predict_interval`) | `src/models/forecast_and_dispatch.py` | `_train_single_horizon` |
| 125 | Prediction interval monotonicity / non-inversion guard | `src/models/forecast_and_dispatch.py` | `_train_single_horizon` |
| 126 | Empirical 90% conformal coverage calculation | `src/models/forecast_and_dispatch.py` | `_train_single_horizon` |
| 127 | Mean conformal interval width diagnostic | `src/models/forecast_and_dispatch.py` | `_train_single_horizon` |
| 128 | Blocked CV RMSE & MAE fold averaging | `src/models/forecast_and_dispatch.py` | `_train_single_horizon` |
| 129 | Holdout test RMSE & MAE computation | `src/models/forecast_and_dispatch.py` | `_train_single_horizon` |
| 130 | Structured per-horizon artifact bundle (`HorizonModelResult`) | `src/models/forecast_and_dispatch.py` | `HorizonModelResult` |
| 131 | Multi-target carbon & Agile price training suite | `src/models/forecast_and_dispatch.py` | `train_conformal_forecasters` |
| 132 | Configurable horizon tuple & confidence level | `src/models/forecast_and_dispatch.py` | `train_conformal_forecasters` |
| 133 | Complete suite container (`ForecastSuiteResult`) | `src/models/forecast_and_dispatch.py` | `ForecastSuiteResult` |
| 134 | Numpy array alignment eliminating sklearn feature-name warnings | `src/models/forecast_and_dispatch.py` | `_train_single_horizon` |
| 135 | Minimum sample guard raising clear `ValueError` | `src/models/forecast_and_dispatch.py` | `_train_single_horizon` |
| 136 | Holdout prediction DataFrame preservation with bounds | `src/models/forecast_and_dispatch.py` | `_train_single_horizon` |
| 137 | Regime-stratified RMSE audit (`_evaluate_by_regime`) | `src/evaluation/error_analysis.py` | `_evaluate_by_regime` |
| 138 | Regime-stratified MAE audit (`Dunkelflaute`/`Plunge`/`Normal`) | `src/evaluation/error_analysis.py` | `_evaluate_by_regime` |
| 139 | Regime-stratified 90% conformal coverage verification | `src/evaluation/error_analysis.py` | `_evaluate_by_regime` |
| 140 | Regime-stratified mean interval width comparison | `src/evaluation/error_analysis.py` | `_evaluate_by_regime` |
| 141 | Empty-regime graceful `N/A` Markdown table formatting | `src/evaluation/error_analysis.py` | `generate_error_analysis_report` |
| 142 | Overall multi-horizon metrics Markdown summary table | `src/evaluation/error_analysis.py` | `generate_error_analysis_report` |
| 143 | Automated report persistence to `docs/ERROR_ANALYSIS_REPORT.md` | `src/evaluation/error_analysis.py` | `generate_error_analysis_report` |
| 144 | Structured `ErrorAnalysisAudit` return object | `src/evaluation/error_analysis.py` | `ErrorAnalysisAudit` |
| 145 | Wind penetration ratio derivation when omitted | `src/models/forecast_and_dispatch.py` | `prepare_forecast_features` |
| 146 | OpenMP single-thread guard for macOS LightGBM+PyTorch stability | `src/__init__.py` | `os.environ.setdefault` |
| 147 | Multi-horizon forecast execution in end-to-end runner | `src/run_pipeline.py` | `run_end_to_end` |
| 148 | Unit test verifying 90% coverage bounds & monotonicity | `tests/test_contracts_and_errors.py` | `test_conformal_forecasting_and_rl_thermal_safety_and_report` |
| 149 | Unit test verifying regime-stratified sections in report | `tests/test_contracts_and_errors.py` | `test_conformal_forecasting_and_rl_thermal_safety_and_report` |
| 150 | CI smoke execution of multi-horizon forecasting | `.github/workflows/ci.yml` | `Run pipeline smoke test` |

---

## 4. PyTorch Actor-Critic RL Dispatch (Skills 151–200)

| # | Capability | `voltflex-uk` File | Symbol / Function |
|---:|---|---|---|
| 151 | PyTorch Actor-Critic dispatch network (`EVBatteryDispatchNet`) | `src/models/forecast_and_dispatch.py` | `EVBatteryDispatchNet` |
| 152 | Shared 2-layer ReLU MLP feature backbone (7 -> 64 -> 64) | `src/models/forecast_and_dispatch.py` | `EVBatteryDispatchNet.__init__` |
| 153 | Sigmoid-bounded EV charge-only policy head (`[0, 1]`) | `src/models/forecast_and_dispatch.py` | `EVBatteryDispatchNet.forward` |
| 154 | Tanh-bounded bidirectional battery policy head (`[-1, 1]`) | `src/models/forecast_and_dispatch.py` | `EVBatteryDispatchNet.forward` |
| 155 | Critic state-value baseline head (`value_head`) | `src/models/forecast_and_dispatch.py` | `EVBatteryDispatchNet.forward` |
| 156 | Learnable Gaussian exploration `log_std` parameter | `src/models/forecast_and_dispatch.py` | `EVBatteryDispatchNet.__init__` |
| 157 | Clamped exploration standard deviation during rollout | `src/models/forecast_and_dispatch.py` | `train_and_evaluate_rl_dispatcher` |
| 158 | Normalized 7-D grid/asset/time RL state builder | `src/models/forecast_and_dispatch.py` | `build_rl_state_vector` |
| 159 | Clipped Agile price & carbon intensity state scaling | `src/models/forecast_and_dispatch.py` | `build_rl_state_vector` |
| 160 | Settlement-period cyclic time embedding in RL state | `src/models/forecast_and_dispatch.py` | `build_rl_state_vector` |
| 161 | Normal action distribution sampling & log-prob tracking | `src/models/forecast_and_dispatch.py` | `train_and_evaluate_rl_dispatcher` |
| 162 | Discounted return-to-go calculation (`gamma = 0.96`) | `src/models/forecast_and_dispatch.py` | `train_and_evaluate_rl_dispatcher` |
| 163 | Standardized advantage estimation (`returns - values`) | `src/models/forecast_and_dispatch.py` | `train_and_evaluate_rl_dispatcher` |
| 164 | Combined policy-gradient + MSE value loss optimization | `src/models/forecast_and_dispatch.py` | `train_and_evaluate_rl_dispatcher` |
| 165 | Gradient norm clipping (`max_norm=1.0`) with Adam | `src/models/forecast_and_dispatch.py` | `train_and_evaluate_rl_dispatcher` |
| 166 | Multi-objective cost + carbon reward formulation | `src/models/forecast_and_dispatch.py` | `train_and_evaluate_rl_dispatcher` |
| 167 | Smart Export Guarantee (15 p/kWh) export revenue credit | `src/models/forecast_and_dispatch.py` | `train_and_evaluate_rl_dispatcher` |
| 168 | Morning departure EV SOC target incentive bonus | `src/models/forecast_and_dispatch.py` | `train_and_evaluate_rl_dispatcher` |
| 169 | Pre-shield safety penalty shaping during RL training | `src/models/forecast_and_dispatch.py` | `train_and_evaluate_rl_dispatcher` |
| 170 | Declarative UK electrical constraints (`DispatchConstraints`) | `src/models/forecast_and_dispatch.py` | `DispatchConstraints` |
| 171 | ENA G98 single-phase export cap (`3.68 kW` / `16 A @ 230 V`) | `src/models/forecast_and_dispatch.py` | `DispatchConstraints.active_export_limit_kw` |
| 172 | ENA G99 approved export limit switching | `src/models/forecast_and_dispatch.py` | `DispatchConstraints.active_export_limit_kw` |
| 173 | Site main fuse thermal limit calculation (`I_max * 230V`) | `src/models/forecast_and_dispatch.py` | `DispatchConstraints.site_thermal_limit_kw` |
| 174 | Deterministic safety action shield (`enforce_thermal_and_soc_limits`) | `src/models/forecast_and_dispatch.py` | `enforce_thermal_and_soc_limits` |
| 175 | Asset inverter power rating clipping (EV 7.36 kW, Batt 5 kW) | `src/models/forecast_and_dispatch.py` | `enforce_thermal_and_soc_limits` |
| 176 | Half-hourly SOC upper-bound charge clamping (`95%`) | `src/models/forecast_and_dispatch.py` | `enforce_thermal_and_soc_limits` |
| 177 | Half-hourly SOC lower-bound discharge clamping (`10%`) | `src/models/forecast_and_dispatch.py` | `enforce_thermal_and_soc_limits` |
| 178 | Critical substation (< 10% headroom) 6 A trickle cap | `src/models/forecast_and_dispatch.py` | `enforce_thermal_and_soc_limits` |
| 179 | Constrained substation (< 25% headroom) 65% import derating | `src/models/forecast_and_dispatch.py` | `enforce_thermal_and_soc_limits` |
| 180 | Proportional multi-asset import scaling under thermal cap | `src/models/forecast_and_dispatch.py` | `enforce_thermal_and_soc_limits` |
| 181 | Battery export clamping to strictly enforce G98/G99 limit | `src/models/forecast_and_dispatch.py` | `enforce_thermal_and_soc_limits` |
| 182 | Final post-projection net site current check (`<= I_max`) | `src/models/forecast_and_dispatch.py` | `enforce_thermal_and_soc_limits` |
| 183 | Pre-shield vs post-shield violation telemetry flags | `src/models/forecast_and_dispatch.py` | `enforce_thermal_and_soc_limits` |
| 184 | Deterministic `eval()` inference simulation rollout | `src/models/forecast_and_dispatch.py` | `simulate_rl_dispatch` |
| 185 | Unoptimized dumb-charging baseline cost/carbon comparison | `src/models/forecast_and_dispatch.py` | `simulate_rl_dispatch` |
| 186 | Per-step site current (Amps) & net kW logging | `src/models/forecast_and_dispatch.py` | `simulate_rl_dispatch` |
| 187 | `0.0%` thermal violation rate verification metric | `src/models/forecast_and_dispatch.py` | `DispatchSimulationResult` |
| 188 | Structured simulation result (`DispatchSimulationResult`) | `src/models/forecast_and_dispatch.py` | `DispatchSimulationResult` |
| 189 | Reproducible PyTorch & NumPy RNG seeding | `src/models/forecast_and_dispatch.py` | `train_and_evaluate_rl_dispatcher` |
| 190 | FastAPI real-time RL dispatch endpoint (`/optimize-dispatch`) | `src/app/main.py` | `optimize_dispatch` |
| 191 | Agile plunge-regime EV/battery boost prior to safety shield | `src/app/main.py` | `optimize_dispatch` |
| 192 | Dunkelflaute peak discharge prioritization up to G98/G99 | `src/app/main.py` | `optimize_dispatch` |
| 193 | Degraded-telemetry safe trickle fallback (`1.38 kW` EV, `0 kW` Batt) | `src/app/main.py` | `optimize_dispatch` |
| 194 | Pydantic validation on dispatch requests & responses | `src/app/main.py` | `DispatchRequest` |
| 195 | Response compliance flags (`g98_g99_compliant`, `thermal_compliant`) | `src/app/main.py` | `DispatchResponse` |
| 196 | Streamlit interactive G99 & main fuse slider controls | `src/app/dashboard.py` | `main` |
| 197 | Streamlit EV/battery dispatch power & SOC trajectory charts | `src/app/dashboard.py` | `main` |
| 198 | RL cost/carbon savings audit in Markdown report | `src/evaluation/error_analysis.py` | `generate_error_analysis_report` |
| 199 | Direct shield stress test under critical substation headroom | `tests/test_contracts_and_errors.py` | `test_conformal_forecasting_and_rl_thermal_safety_and_report` |
| 200 | FastAPI stale-telemetry degraded dispatch integration test | `tests/test_contracts_and_errors.py` | `test_fastapi_endpoints_and_degraded_fallback` |

---

## 5. Guardrailed LangGraph RAG (Skills 201–250)

| # | Capability | `voltflex-uk` File | Symbol / Function |
|---:|---|---|---|
| 201 | Disk regulatory document loader (`load_flexibility_regulatory_corpus`) | `src/agents/flexibility_copilot.py` | `load_flexibility_regulatory_corpus` |
| 202 | PDF regulatory parsing via `pypdf.PdfReader` | `src/agents/flexibility_copilot.py` | `load_flexibility_regulatory_corpus` |
| 203 | Structured Markdown section parser (`## [DOC-ID] Title`) | `src/agents/flexibility_copilot.py` | `_parse_markdown_sections` |
| 204 | Authority metadata extraction from Markdown headers | `src/agents/flexibility_copilot.py` | `_parse_markdown_sections` |
| 205 | Plaintext `.txt` & `.pdf` sliding word-chunker | `src/agents/flexibility_copilot.py` | `_chunk_plain_text` |
| 206 | Citation-ID deduplication across parsed corpus files | `src/agents/flexibility_copilot.py` | `load_flexibility_regulatory_corpus` |
| 207 | Seeded NESO Demand Flexibility Service statutory doc | `data/regulatory_docs/neso_demand_flexibility_service.md` | `[NESO-DFS-01]` / `[NESO-DFS-02]` |
| 208 | Seeded ENA G98 & G99 statutory inverter limit doc | `data/regulatory_docs/ena_g98_g99_inverter_limits.md` | `[ENA-G98-01]` / `[ENA-G99-01]` |
| 209 | Seeded Ofgem SEG, Agile & UK GDPR statutory doc | `data/regulatory_docs/ofgem_smart_export_guarantee.md` | `[OFGEM-SEG-01]` |
| 210 | Immutable `RegulatoryDocument` chunk dataclass | `src/agents/flexibility_copilot.py` | `RegulatoryDocument` |
| 211 | In-memory statutory fallback corpus (`UK_FLEXIBILITY_CORPUS`) | `src/agents/flexibility_copilot.py` | `UK_FLEXIBILITY_CORPUS` |
| 212 | Regex regulatory tokenizer preserving decimal kW ratings | `src/agents/flexibility_copilot.py` | `_tokenize` |
| 213 | Lexical BM25 retrieval indexing (`BM25Okapi`) | `src/agents/flexibility_copilot.py` | `HybridRegulatoryRetriever` |
| 214 | Dense/subword TF-IDF unigram+bigram cosine retrieval | `src/agents/flexibility_copilot.py` | `HybridRegulatoryRetriever` |
| 215 | Min-max score normalization & 65/35 hybrid score fusion | `src/agents/flexibility_copilot.py` | `HybridRegulatoryRetriever.retrieve` |
| 216 | Top-k ranked retrieval with hybrid relevance scores | `src/agents/flexibility_copilot.py` | `HybridRegulatoryRetriever.retrieve` |
| 217 | Typed LangGraph workflow state schema (`CopilotState`) | `src/agents/flexibility_copilot.py` | `CopilotState` |
| 218 | Compiled `StateGraph` orchestration (`FlexibilityCopilot`) | `src/agents/flexibility_copilot.py` | `FlexibilityCopilot._build_graph` |
| 219 | Query augmentation for kW/export safety context (`_retrieve_node`) | `src/agents/flexibility_copilot.py` | `FlexibilityCopilot._retrieve_node` |
| 220 | Deterministic G98/G99 safety interlock node (`_safety_interlock_node`) | `src/agents/flexibility_copilot.py` | `FlexibilityCopilot._safety_interlock_node` |
| 221 | Regex kW extraction from free-text user queries | `src/agents/flexibility_copilot.py` | `_check_g98_g99_interlock` |
| 222 | Single-phase `3.68 kW` (`16 A`) G98 threshold enforcement | `src/agents/flexibility_copilot.py` | `G98_SINGLE_PHASE_LIMIT_KW` |
| 223 | Prompt-injection / limit-bypass phrase detection | `src/agents/flexibility_copilot.py` | `_check_g98_g99_interlock` |
| 224 | Critical DNO substation headroom (< 10%) interlock trigger | `src/agents/flexibility_copilot.py` | `_check_g98_g99_interlock` |
| 225 | Mandatory ENA-G98/G99 context injection on interlock | `src/agents/flexibility_copilot.py` | `FlexibilityCopilot._safety_interlock_node` |
| 226 | Grounded regulatory answer synthesis (`_synthesize_node`) | `src/agents/flexibility_copilot.py` | `FlexibilityCopilot._synthesize_node` |
| 227 | Inline `[DOC-ID]` citation formatting per claim | `src/agents/flexibility_copilot.py` | `FlexibilityCopilot._synthesize_node` |
| 228 | Citation-verifier judge function (`verify_citations`) | `src/agents/flexibility_copilot.py` | `verify_citations` |
| 229 | Rejection of responses lacking inline regulatory citations | `src/agents/flexibility_copilot.py` | `verify_citations` |
| 230 | Hallucinated / unretrieved `[DOC-ID]` detection | `src/agents/flexibility_copilot.py` | `verify_citations` |
| 231 | Verification that triggered interlocks state `3.68 kW`/`G98` | `src/agents/flexibility_copilot.py` | `verify_citations` |
| 232 | Judge node wrapping verification into graph state | `src/agents/flexibility_copilot.py` | `FlexibilityCopilot._citation_verifier_node` |
| 233 | Conditional LangGraph routing (`verified` vs `fallback`) | `src/agents/flexibility_copilot.py` | `FlexibilityCopilot._route_after_verifier` |
| 234 | Deterministic G98/G99 safety interlock fallback node | `src/agents/flexibility_copilot.py` | `FlexibilityCopilot._safety_fallback_node` |
| 235 | Structured `ask()` API returning answer, citations & verdict | `src/agents/flexibility_copilot.py` | `FlexibilityCopilot.ask` |
| 236 | FastAPI copilot endpoint (`/flexibility-copilot`) | `src/app/main.py` | `flexibility_copilot` |
| 237 | Pydantic copilot request & response schemas | `src/app/main.py` | `CopilotRequest` / `CopilotResponse` |
| 238 | Streamlit interactive copilot Q&A tab with interlock alerts | `src/app/dashboard.py` | `main` |
| 239 | End-to-end copilot compliance probe in pipeline runner | `src/run_pipeline.py` | `run_end_to_end` |
| 240 | Unit test verifying disk `.md`/`.txt`/`.pdf` corpus ingestion | `tests/test_contracts_and_errors.py` | `test_load_flexibility_regulatory_corpus_from_disk` |
| 241 | Unit test verifying G98 `> 3.68 kW` interlock activation | `tests/test_contracts_and_errors.py` | `test_langgraph_copilot_citation_verifier_and_g98_interlock` |
| 242 | Unit test verifying Smart Export Guarantee retrieval | `tests/test_contracts_and_errors.py` | `test_langgraph_copilot_citation_verifier_and_g98_interlock` |
| 243 | Unit test verifying missing-citation judge rejection | `tests/test_contracts_and_errors.py` | `test_langgraph_copilot_citation_verifier_and_g98_interlock` |
| 244 | Unit test verifying hallucinated-citation judge rejection | `tests/test_contracts_and_errors.py` | `test_langgraph_copilot_citation_verifier_and_g98_interlock` |
| 245 | FastAPI `/flexibility-copilot` integration test | `tests/test_contracts_and_errors.py` | `test_fastapi_endpoints_and_degraded_fallback` |
| 246 | NESO DFS double-counting rule grounding | `data/regulatory_docs/neso_demand_flexibility_service.md` | `[NESO-DFS-01]` |
| 247 | Ofgem SEG 150k-customer supplier & MCS rule grounding | `data/regulatory_docs/ofgem_smart_export_guarantee.md` | `[OFGEM-SEG-01]` |
| 248 | G100 export limitation & DNO curtailment rule grounding | `data/regulatory_docs/ena_g98_g99_inverter_limits.md` | `[ENA-G99-01]` |
| 249 | UK GDPR MPAN HMAC rule grounding in copilot corpus | `data/regulatory_docs/ofgem_smart_export_guarantee.md` | `[UK-GDPR-01]` |
| 250 | Fallback to default corpus when docs directory is empty | `src/agents/flexibility_copilot.py` | `load_flexibility_regulatory_corpus` |

---

## 6. MLOps (Skills 251–300)

| # | Capability | `voltflex-uk` File | Symbol / Function |
|---:|---|---|---|
| 251 | PEP 621 modern Python packaging & editable install | `pyproject.toml` | `[project]` |
| 252 | Reproducible Python 3.12 `uv` virtual environment setup | `.github/workflows/ci.yml` | `Set up uv` |
| 253 | Automated dependency declaration (`pypdf`, `tenacity`, `mapie`) | `pyproject.toml` | `dependencies` |
| 254 | Dev-only test & lint dependency group (`pytest`, `ruff`) | `pyproject.toml` | `[project.optional-dependencies]` |
| 255 | Ruff lint rule configuration (E, F, I, W) targeting Python 3.12 | `pyproject.toml` | `[tool.ruff]` |
| 256 | Pytest path and test-discovery configuration | `pyproject.toml` | `[tool.pytest.ini_options]` |
| 257 | MLflow SQLite tracking URI (`sqlite:///mlruns/mlflow.db`) | `src/models/forecast_and_dispatch.py` | `DEFAULT_MLFLOW_URI` |
| 258 | Automatic MLflow SQLite directory creation | `src/models/forecast_and_dispatch.py` | `train_conformal_forecasters` |
| 259 | MLflow experiment naming & run lifecycle management | `src/models/forecast_and_dispatch.py` | `train_conformal_forecasters` |
| 260 | MLflow parameter logging (`horizons`, `confidence_level`, `n_rows`) | `src/models/forecast_and_dispatch.py` | `train_conformal_forecasters` |
| 261 | MLflow per-horizon CV & test metric logging | `src/models/forecast_and_dispatch.py` | `train_conformal_forecasters` |
| 262 | MLflow conformal coverage & interval width tracking | `src/models/forecast_and_dispatch.py` | `train_conformal_forecasters` |
| 263 | MLflow run ID propagation into audit reports | `src/evaluation/error_analysis.py` | `generate_error_analysis_report` |
| 264 | Multi-stage slim Python 3.12 Docker image build | `Dockerfile` | `FROM python:3.12-slim` |
| 265 | Native `libgomp1` runtime provisioning for LightGBM | `Dockerfile` | `RUN apt-get update` |
| 266 | Container `HEALTHCHECK` probing FastAPI `/health` | `Dockerfile` | `HEALTHCHECK` |
| 267 | Non-buffered Python & bytecode-disabled container env | `Dockerfile` | `ENV PYTHONDONTWRITEBYTECODE=1` |
| 268 | Multi-service Compose stack (`pgvector` + `mlflow` + `api`) | `docker-compose.yml` | `services` |
| 269 | PostgreSQL 16 + `pgvector` health-checked service | `docker-compose.yml` | `services.pgvector` |
| 270 | Persistent Docker volume for `pgvector` database | `docker-compose.yml` | `volumes.pgvector_data` |
| 271 | Dedicated MLflow tracking server container on port 5000 | `docker-compose.yml` | `services.mlflow` |
| 272 | Compose service startup ordering via `depends_on` health checks | `docker-compose.yml` | `services.api.depends_on` |
| 273 | Environment-driven HMAC secret & DB URI injection | `docker-compose.yml` | `services.api.environment` |
| 274 | Shared bind mounts for `data/`, `docs/`, and `mlruns/` | `docker-compose.yml` | `services.api.volumes` |
| 275 | GitHub Actions CI triggered on `main` push & PRs | `.github/workflows/ci.yml` | `on` |
| 276 | Automated CI `ruff check src tests` lint gate | `.github/workflows/ci.yml` | `Lint with ruff` |
| 277 | Automated CI end-to-end pipeline smoke test | `.github/workflows/ci.yml` | `Run pipeline smoke test` |
| 278 | Automated CI `pytest -v` verification gate | `.github/workflows/ci.yml` | `Run pytest suite` |
| 279 | Git ignore hygiene for `.venv`, `mlruns/`, raw/quarantine data | `.gitignore` | `.gitignore` |
| 280 | Automated Markdown error analysis artifact generation | `src/evaluation/error_analysis.py` | `generate_error_analysis_report` |
| 281 | DuckDB read-only concurrency in FastAPI & Streamlit | `src/app/main.py` | `health` |
| 282 | Streamlit 60-second TTL data caching (`@st.cache_data`) | `src/app/dashboard.py` | `load_telemetry` |
| 283 | FastAPI `/health` readiness & degraded status probe | `src/app/main.py` | `health` |
| 284 | Uvicorn ASGI production server entrypoint | `Dockerfile` | `CMD` |
| 285 | End-to-end pipeline CLI runner (`python -m src.run_pipeline`) | `src/run_pipeline.py` | `main` |
| 286 | Isolated `tmp_path` DuckDB/Parquet/MLflow unit testing | `tests/test_contracts_and_errors.py` | `test_mpan_hmac_pseudonymisation_and_quarantine` |
| 287 | FastAPI `TestClient` contract & endpoint testing | `tests/test_contracts_and_errors.py` | `test_fastapi_endpoints_and_degraded_fallback` |
| 288 | Mocked HTTP client live-API integration testing | `tests/test_contracts_and_errors.py` | `test_live_grid_connectors_and_degraded_fallback` |
| 289 | Pipeline `--live` degraded fallback integration test | `tests/test_contracts_and_errors.py` | `test_run_pipeline_live_degraded_fallback` |
| 290 | OpenMP/BLAS thread-pinning safety guard at package import | `src/__init__.py` | `OMP_NUM_THREADS` |
| 291 | MLflow agent hint suppression for clean production logs | `src/__init__.py` | `MLFLOW_DISABLE_AGENT_HINT` |
| 292 | Structured logging of upstream feed degradation events | `src/ingestion/live_grid_connectors.py` | `logger.warning` |
| 293 | Zero-warning pandas 3.x dtype-safe Markdown export | `src/evaluation/error_analysis.py` | `generate_error_analysis_report` |
| 294 | Configurable `VOLTFLEX_DB_PATH` & `MLFLOW_TRACKING_URI` overrides | `src/app/main.py` | `_load_latest_region_telemetry` |
| 295 | Reproducible deterministic offline seed management | `src/run_pipeline.py` | `generate_uk_grid_telemetry` |
| 296 | Modular package hierarchy (`ingestion`, `analytics`, `models`, `evaluation`, `agents`, `app`) | `src/` | `src/*/__init__.py` |
| 297 | Automated 3% quarantine audit logging in pipeline summary | `src/run_pipeline.py` | `main` |
| 298 | Automated RL savings & safety telemetry printout | `src/run_pipeline.py` | `main` |
| 299 | Comprehensive quickstart & operational documentation | `README.md` | `Quickstart` |
| 300 | Full 300-capability engineering traceability matrix | `docs/SKILLS_MATRIX.md` | `docs/SKILLS_MATRIX.md` |
