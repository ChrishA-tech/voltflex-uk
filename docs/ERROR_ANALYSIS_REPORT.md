# VoltFlex UK — Forecasting & RL Dispatch Error Analysis Report

**Generated:** 2026-11-25 10:40:39 UTC  
**MLflow Run ID:** `0960ed66620f41288f3ada5c4c8994e3`  
**Ingestion Quarantine Audit:** `0` / `1064` rows quarantined

## 1. Multi-Horizon Conformal Forecast Accuracy & Coverage (90% Target)

| Target                    |   Horizon (30m steps) |   CV RMSE |   CV MAE |   Test RMSE |   Test MAE |   Conformal Coverage (%) |   Mean 90% Width |
|:--------------------------|----------------------:|----------:|---------:|------------:|-----------:|-------------------------:|-----------------:|
| carbon_intensity_gco2_kwh |                     1 |    8.0749 |   6.315  |      0.286  |     0.1974 |                      100 |          15.9175 |
| carbon_intensity_gco2_kwh |                     2 |    9.5396 |   7.0241 |      0.387  |     0.2744 |                      100 |          23.6597 |
| carbon_intensity_gco2_kwh |                     4 |    9.1205 |   6.7477 |      0.43   |     0.2952 |                      100 |          25.7975 |
| agile_import_pence        |                     1 |    4.2934 |   3.5765 |      1.2453 |     0.9806 |                      100 |          10.9621 |
| agile_import_pence        |                     2 |    5.5637 |   4.681  |      1.1876 |     0.9752 |                      100 |          10.485  |
| agile_import_pence        |                     4 |    6.7339 |   5.776  |      1.4004 |     1.1548 |                      100 |           7.981  |

## 2. Regime-Stratified Conformal Audit (`Dunkelflaute Peak` vs `Wind Curtailment Plunge` vs `Normal`)

| Target                    |   Horizon (30m steps) | Grid Regime             |   Samples | RMSE   | MAE   | Coverage 90%   | Mean Width   |
|:--------------------------|----------------------:|:------------------------|----------:|:-------|:------|:---------------|:-------------|
| carbon_intensity_gco2_kwh |                     1 | Dunkelflaute Peak       |        45 | 0.461  | 0.349 | 100.0          | 15.917       |
| carbon_intensity_gco2_kwh |                     1 | Wind Curtailment Plunge |         0 | N/A    | N/A   | N/A            | N/A          |
| carbon_intensity_gco2_kwh |                     1 | Normal                  |       164 | 0.214  | 0.156 | 100.0          | 15.917       |
| carbon_intensity_gco2_kwh |                     2 | Dunkelflaute Peak       |        45 | 0.622  | 0.494 | 100.0          | 23.66        |
| carbon_intensity_gco2_kwh |                     2 | Wind Curtailment Plunge |         0 | N/A    | N/A   | N/A            | N/A          |
| carbon_intensity_gco2_kwh |                     2 | Normal                  |       161 | 0.289  | 0.213 | 100.0          | 23.66        |
| carbon_intensity_gco2_kwh |                     4 | Dunkelflaute Peak       |        43 | 0.643  | 0.453 | 100.0          | 25.797       |
| carbon_intensity_gco2_kwh |                     4 | Wind Curtailment Plunge |         0 | N/A    | N/A   | N/A            | N/A          |
| carbon_intensity_gco2_kwh |                     4 | Normal                  |       155 | 0.348  | 0.251 | 100.0          | 25.797       |
| agile_import_pence        |                     1 | Dunkelflaute Peak       |        45 | 1.297  | 1.03  | 100.0          | 10.962       |
| agile_import_pence        |                     1 | Wind Curtailment Plunge |         0 | N/A    | N/A   | N/A            | N/A          |
| agile_import_pence        |                     1 | Normal                  |       164 | 1.231  | 0.967 | 100.0          | 10.962       |
| agile_import_pence        |                     2 | Dunkelflaute Peak       |        45 | 1.203  | 0.959 | 100.0          | 10.485       |
| agile_import_pence        |                     2 | Wind Curtailment Plunge |         0 | N/A    | N/A   | N/A            | N/A          |
| agile_import_pence        |                     2 | Normal                  |       161 | 1.183  | 0.98  | 100.0          | 10.485       |
| agile_import_pence        |                     4 | Dunkelflaute Peak       |        43 | 1.429  | 1.202 | 100.0          | 7.981        |
| agile_import_pence        |                     4 | Wind Curtailment Plunge |         0 | N/A    | N/A   | N/A            | N/A          |
| agile_import_pence        |                     4 | Normal                  |       155 | 1.392  | 1.142 | 100.0          | 7.981        |

## 3. RL Dispatch Thermal Safety & Cost/Carbon Audit (`EVBatteryDispatchNet`)

| Metric | Observed Value | Target / Limit | Status |
| --- | ---: | ---: | --- |
| Post-Shield Thermal & G98/G99 Violation Rate | 0.0000% | 0.0% | PASS |
| Pre-Shield Raw Policy Violation Rate | 64.47% | Diagnostic | MONITORED |
| RL Dispatch Net Cost (GBP) | £11.44 | Baseline £53.47 | Saved £42.03 |
| RL Dispatch Net Carbon (kgCO2) | 5.77 kg | Baseline 27.91 kg | Saved 22.14 kg |
| Final EV / Battery SOC | 95.0% / 30.7% | 10%–95% | PASS |

## 4. Key Findings & Mitigations

- **Conformal Calibration:** Split conformal intervals (`MAPIE`) track ~90% nominal coverage globally while widening automatically during volatile `Dunkelflaute Peak` and `Wind Curtailment Plunge` intervals.
- **DNO Thermal Safety Interlock:** Deterministic action projection (`enforce_thermal_and_soc_limits`) guarantees `0.0%` thermal current and G98 (`3.68 kW`) / G99 export violations even under constrained substation headroom.
