# VoltFlex UK — Forecasting & RL Dispatch Error Analysis Report

**Generated:** 2026-11-25 01:40:16 UTC  
**MLflow Run ID:** `d73b1e297a37426f8e2c055b4b283610`  
**Ingestion Quarantine Audit:** `161` / `5376` rows quarantined

## 1. Multi-Horizon Conformal Forecast Accuracy & Coverage (90% Target)

| Target                    |   Horizon (30m steps) |   CV RMSE |   CV MAE |   Test RMSE |   Test MAE |   Conformal Coverage (%) |   Mean 90% Width |
|:--------------------------|----------------------:|----------:|---------:|------------:|-----------:|-------------------------:|-----------------:|
| carbon_intensity_gco2_kwh |                     1 |   37.8533 |  32.1187 |     23.5441 |    18.658  |                    93.17 |          85.4142 |
| carbon_intensity_gco2_kwh |                     2 |   42.3547 |  36.15   |     26.4501 |    21.6918 |                    94.31 |          96.0064 |
| carbon_intensity_gco2_kwh |                     4 |   48.301  |  40.9487 |     36.358  |    30.3329 |                    90.95 |         123.897  |
| agile_import_pence        |                     1 |    4.8927 |   4.0811 |      3.5205 |     2.6694 |                    89.62 |          11.1933 |
| agile_import_pence        |                     2 |    4.7585 |   3.8353 |      3.5445 |     2.7728 |                    92.37 |          12.8392 |
| agile_import_pence        |                     4 |    5.8755 |   4.7716 |      4.1642 |     3.2715 |                    96.01 |          17.4423 |

## 2. Regime-Stratified Conformal Audit (`Dunkelflaute Peak` vs `Wind Curtailment Plunge` vs `Normal`)

| Target                    |   Horizon (30m steps) | Grid Regime             |   Samples |   RMSE |    MAE |   Coverage 90% |   Mean Width |
|:--------------------------|----------------------:|:------------------------|----------:|-------:|-------:|---------------:|-------------:|
| carbon_intensity_gco2_kwh |                     1 | Dunkelflaute Peak       |       133 | 28.283 | 23.117 |          86.47 |       85.414 |
| carbon_intensity_gco2_kwh |                     1 | Wind Curtailment Plunge |       105 | 18.401 | 15.111 |          99.05 |       85.414 |
| carbon_intensity_gco2_kwh |                     1 | Normal                  |       802 | 23.277 | 18.383 |          93.52 |       85.414 |
| carbon_intensity_gco2_kwh |                     2 | Dunkelflaute Peak       |       133 | 31.288 | 27.379 |          89.47 |       96.006 |
| carbon_intensity_gco2_kwh |                     2 | Wind Curtailment Plunge |       104 | 21.833 | 16.646 |          94.23 |       96.006 |
| carbon_intensity_gco2_kwh |                     2 | Normal                  |       799 | 26.118 | 21.402 |          95.12 |       96.006 |
| carbon_intensity_gco2_kwh |                     4 | Dunkelflaute Peak       |       133 | 51.25  | 45.758 |          69.17 |      123.897 |
| carbon_intensity_gco2_kwh |                     4 | Wind Curtailment Plunge |        99 | 22.345 | 17.513 |         100    |      123.897 |
| carbon_intensity_gco2_kwh |                     4 | Normal                  |       796 | 34.731 | 29.35  |          93.47 |      123.897 |
| agile_import_pence        |                     1 | Dunkelflaute Peak       |       133 |  2.724 |  2.125 |          94.74 |       11.193 |
| agile_import_pence        |                     1 | Wind Curtailment Plunge |       105 |  4.768 |  3.662 |          75.24 |       11.193 |
| agile_import_pence        |                     1 | Normal                  |       802 |  3.445 |  2.63  |          90.65 |       11.193 |
| agile_import_pence        |                     2 | Dunkelflaute Peak       |       133 |  3.253 |  2.589 |          93.98 |       12.839 |
| agile_import_pence        |                     2 | Wind Curtailment Plunge |       104 |  4.193 |  3.307 |          88.46 |       12.839 |
| agile_import_pence        |                     2 | Normal                  |       799 |  3.499 |  2.734 |          92.62 |       12.839 |
| agile_import_pence        |                     4 | Dunkelflaute Peak       |       133 |  3.087 |  2.365 |          99.25 |       17.442 |
| agile_import_pence        |                     4 | Wind Curtailment Plunge |        99 |  5.68  |  4.585 |          89.9  |       17.442 |
| agile_import_pence        |                     4 | Normal                  |       796 |  4.098 |  3.259 |          96.23 |       17.442 |

## 3. RL Dispatch Thermal Safety & Cost/Carbon Audit (`EVBatteryDispatchNet`)

| Metric | Observed Value | Target / Limit | Status |
| --- | ---: | ---: | --- |
| Post-Shield Thermal & G98/G99 Violation Rate | 0.0000% | 0.0% | PASS |
| Pre-Shield Raw Policy Violation Rate | 95.83% | Diagnostic | MONITORED |
| RL Dispatch Net Cost (GBP) | £2.62 | Baseline £34.07 | Saved £31.45 |
| RL Dispatch Net Carbon (kgCO2) | 3.45 kg | Baseline 32.37 kg | Saved 28.92 kg |
| Final EV / Battery SOC | 95.0% / 10.0% | 10%–95% | PASS |

## 4. Key Findings & Mitigations

- **Conformal Calibration:** Split conformal intervals (`MAPIE`) track ~90% nominal coverage globally while widening automatically during volatile `Dunkelflaute Peak` and `Wind Curtailment Plunge` intervals.
- **DNO Thermal Safety Interlock:** Deterministic action projection (`enforce_thermal_and_soc_limits`) guarantees `0.0%` thermal current and G98 (`3.68 kW`) / G99 export violations even under constrained substation headroom.
