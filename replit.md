# MediScope AI

AI-driven medical platform with explainable diagnosis, RAG-based insights, and a Gemini-powered chatbot.

## Features
- **Image Analysis** (`/image-analysis`) — Upload a brain MRI or chest X-ray; runs the user's trained Keras CNNs (`brain_tumor_final_v3.keras`, `pneumonia_final.keras`) and uses Gemini to explain the finding in plain language.
- **Report Analyzer** (`/report-analyzer`) — Paste / upload a clinical report; Gemini summarizes it with key findings, recommendations, and precautions, grounded in the medical-transcription corpus.
- **Symptom Checker** (`/symptom-checker`) — Multinomial Naïve-Bayes classifier over the disease–symptoms dataset; ranks top conditions with descriptions and precautions.
- **RAG Chatbot** (`/chat`) — TF-IDF retrieval over the disease KB + `mtsamples.csv` medical transcriptions, generation by Gemini with cited sources.

## Architecture
- **Backend**: FastAPI + uvicorn, port 5000 (host `0.0.0.0`).
- **Templates**: Jinja2 + Tailwind CDN + vanilla JS (with `marked` for Markdown rendering).
- **ML**: TensorFlow CPU; models loaded lazily on first request.
- **RAG**: scikit-learn `TfidfVectorizer` + cosine similarity over ~1500 documents.
- **LLM**: Google Gemini via `google-genai` SDK; tries `gemini-2.5-flash`, falls back to `gemini-2.5-flash-lite`, then `gemini-2.0-flash`.

## File layout
- `app/main.py` — FastAPI routes (pages + JSON APIs).
- `app/ml.py` — image preprocessing & Keras model loaders.
- `app/symptoms.py` — symptom→disease classifier.
- `app/rag.py` — knowledge-base build, retrieval, and Gemini generation.
- `app/templates/` — Jinja2 templates (base, index, image, report, symptoms, chat).
- `app/static/` — CSS + JS.
- `Notebooks/Models/` — pretrained `.keras` models.
- `Data/` — datasets (brain tumor, pneumonia, disease symptoms CSVs, mtsamples.csv).

## Setup
- Runtime: Python 3.11.
- Required secret: `GEMINI_API_KEY`.
- Workflow `Start application` runs:
  `uvicorn app.main:app --host 0.0.0.0 --port 5000 --proxy-headers --forwarded-allow-ips='*'`
- Deployment target: VM (long-running web server) with the same uvicorn command.

## Endpoints (JSON)
- `POST /api/image/predict` — multipart form with `task` (`brain_tumor` | `pneumonia`) and `file`.
- `POST /api/symptoms/predict` — `{symptoms: [...], top_k}`.
- `POST /api/report/analyze` — multipart form with `text` and/or `.txt` `file`.
- `POST /api/chat` — `{message, history: [{role, content}]}`.
- `GET /api/health` — diagnostics.
