# VoltFlex UK

Real-Time UK Grid Balancing, EV & Battery Dispatch Optimization, and NESO/Ofgem Flexibility Copilot.

## Architecture

- **Ingestion (`src/ingestion/grid_telemetry.py`)**: Pandera physical-bound contracts (`GridTelemetrySchema`), UK GDPR smart-meter MPAN HMAC-SHA256 pseudonymisation, DuckDB persistence, and Parquet Dead-Letter Quarantine.
- **Curtailment & Plunge Analytics (`src/analytics/curtailment_detector.py`)**: Wind curtailment windows, negative Octopus Agile plunge-pricing regimes, and DNO substation transformer headroom across the 14 UK DNO regions.
- **Forecasting & RL Dispatch (`src/models/forecast_and_dispatch.py`)**: Multi-horizon LightGBM + MAPIE 90% conformal regressors for half-hourly regional carbon intensity (`gCO2/kWh`) and Agile import prices (`p/kWh`) with blocked time-series CV and MLflow SQLite tracking (`sqlite:///mlruns/mlflow.db`), paired with `EVBatteryDispatchNet` (PyTorch Policy-Gradient dispatch optimizer enforcing G98/G99 thermal current limits).
- **Error Analysis (`src/evaluation/error_analysis.py`)**: Automated audit generating `docs/ERROR_ANALYSIS_REPORT.md` across `Dunkelflaute Peak`, `Wind Curtailment Plunge`, and `Normal` grid regimes.
- **Flexibility Copilot (`src/agents/flexibility_copilot.py`)**: Guardrailed LangGraph workflow with BM25 hybrid retrieval over NESO DFS, Ofgem G98/G99 (`3.68 kW` single-phase), and Smart Export Guarantee rules, citation-verifier judge node, and deterministic G98/G99 safety interlock fallback.
- **Serving (`src/app/main.py` & `src/app/dashboard.py`)**: FastAPI service (`/health`, `/optimize-dispatch`, `/flexibility-copilot`) and Streamlit live grid & EV dispatch command center.

## Quickstart

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install -e ".[dev]" streamlit tabulate
python -m src.run_pipeline
pytest -v
ruff check src tests
```
