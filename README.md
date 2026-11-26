# VoltFlex UK

Real-Time UK Grid Balancing, EV & Battery Dispatch Optimization, and NESO/Ofgem Flexibility Copilot.

## Architecture

- **Live UK Grid & Agile Connectors (`src/ingestion/live_grid_connectors.py`)**: Real-time ingestion from the National Grid ESO Regional Carbon Intensity API (`https://api.carbonintensity.org.uk/regional`) and Octopus Energy Open Tariff API (`AGILE-24-10-01`) across all 14 UK DNO regions with `tenacity` retries and `LIVE_GRID_FEED_DEGRADED` offline fallback.
- **Ingestion Contracts & GDPR (`src/ingestion/grid_telemetry.py`)**: Pandera physical-bound contracts (`GridTelemetrySchema`), UK GDPR smart-meter MPAN HMAC-SHA256 pseudonymisation, DuckDB persistence, and Parquet Dead-Letter Quarantine.
- **Curtailment & Plunge Analytics (`src/analytics/curtailment_detector.py`)**: Wind curtailment windows, negative Octopus Agile plunge-pricing regimes, and DNO substation transformer headroom across the 14 UK DNO regions.
- **Forecasting & RL Dispatch (`src/models/forecast_and_dispatch.py`)**: Multi-horizon LightGBM + MAPIE 90% conformal regressors for half-hourly regional carbon intensity (`gCO2/kWh`) and Agile import prices (`p/kWh`) with blocked time-series CV and MLflow SQLite tracking (`sqlite:///mlruns/mlflow.db`), paired with `EVBatteryDispatchNet` (PyTorch Policy-Gradient dispatch optimizer enforcing G98/G99 thermal current limits).
- **Error Analysis (`src/evaluation/error_analysis.py`)**: Automated audit generating `docs/ERROR_ANALYSIS_REPORT.md` across `Dunkelflaute Peak`, `Wind Curtailment Plunge`, and `Normal` grid regimes.
- **Flexibility Copilot (`src/agents/flexibility_copilot.py`)**: Guardrailed LangGraph workflow loading statutory `.pdf`, `.md`, and `.txt` documents from `data/regulatory_docs/` into BM25 + TF-IDF hybrid retrieval over NESO DFS, Ofgem G98/G99 (`3.68 kW` single-phase), and Smart Export Guarantee rules, with citation-verifier judge node and deterministic G98/G99 safety interlock fallback.
- **Serving & Dashboard (`src/app/main.py` & `src/app/dashboard.py`)**: FastAPI service (`/health`, `/optimize-dispatch`, `/flexibility-copilot`) and Streamlit live grid & EV dispatch command center.
- **300-Skill Matrix (`docs/SKILLS_MATRIX.md`)**: Comprehensive mapping of 300 capabilities across streaming, spatial DNO analytics, conformal forecasting, RL dispatch, LangGraph RAG, and MLOps.

## Quickstart

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install -e ".[dev]" streamlit tabulate

# Run live UK grid & Octopus Agile end-to-end pipeline (falls back gracefully if offline)
python -m src.run_pipeline --live

# Or run deterministic offline synthetic mode with 3% injected quarantine rows
python -m src.run_pipeline --no-live

# Launch FastAPI service or Streamlit live command center
uvicorn src.app.main:app --host 0.0.0.0 --port 8000
streamlit run src/app/dashboard.py

# Run verification suite
pytest -v
ruff check src tests
```


## Interactive decision replay

Explore [the dashboard](https://chrisha-tech.github.io/replay/?project=voltflex-uk) or reproduce the [scenario benchmark and error analysis](replay/README.md). The bundled replay uses synthetic scenarios.
