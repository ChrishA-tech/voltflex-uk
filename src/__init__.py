"""VoltFlex UK package."""

import os

# Guard against macOS OpenMP dual-runtime collisions between LightGBM and PyTorch.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
