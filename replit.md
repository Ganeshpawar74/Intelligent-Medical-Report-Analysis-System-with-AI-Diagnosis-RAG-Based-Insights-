# Medical AI Notebooks

Data science project containing Jupyter notebooks and trained Keras models for:
- Brain tumor classification (`Notebooks/brainTumor_v3.ipynb`)
- Pneumonia classification (`Notebooks/pneumonia_classification_v2.ipynb`)
- Disease symptom datasets (under `Data/`)

## Project Layout
- `Data/` — datasets (brain tumor images, pneumonia X-rays, disease symptoms CSVs, mtsamples.csv)
- `Notebooks/` — Jupyter notebooks
- `Notebooks/Models/` — trained `.keras` model artifacts
- `Notebooks/reports/` — training/eval report images

## Replit Setup
- Runtime: Python 3.11
- Workflow `Start application` runs JupyterLab on port 5000 (host `0.0.0.0`, auth disabled, frame embedding allowed) so the Replit preview iframe can display the Jupyter UI.
- Config file: `~/.jupyter/jupyter_server_config.py`
- Heavy ML libs (tensorflow, opencv, etc.) are not pre-installed; install on demand with the package manager when running specific notebooks.
