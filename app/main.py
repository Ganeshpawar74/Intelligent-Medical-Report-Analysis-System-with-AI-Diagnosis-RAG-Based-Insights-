"""FastAPI entrypoint for the Medical AI platform."""
from __future__ import annotations
import os
from typing import List, Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from . import ml, symptoms as sym_mod, rag

app = FastAPI(title="MediScope AI", version="1.0")

BASE_DIR = os.path.dirname(__file__)
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")


# ----------- Pages ---------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
def page_home(request: Request):
    return templates.TemplateResponse(request, "index.html", {"page": "home"})


@app.get("/image-analysis", response_class=HTMLResponse)
def page_image(request: Request):
    return templates.TemplateResponse(request, "image_analysis.html", {"page": "image"})


@app.get("/report-analyzer", response_class=HTMLResponse)
def page_report(request: Request):
    return templates.TemplateResponse(request, "report_analyzer.html", {"page": "report"})


@app.get("/symptom-checker", response_class=HTMLResponse)
def page_symptoms(request: Request):
    return templates.TemplateResponse(
        request,
        "symptom_checker.html",
        {"page": "symptoms", "symptoms": sym_mod.all_symptoms()},
    )


@app.get("/chat", response_class=HTMLResponse)
def page_chat(request: Request):
    return templates.TemplateResponse(request, "chat.html", {"page": "chat"})


# ----------- API: Image classification ------------------------------------

@app.post("/api/image/predict")
async def api_image_predict(
    task: str = Form(...),
    file: UploadFile = File(...),
):
    if task not in ("brain_tumor", "pneumonia"):
        raise HTTPException(400, "task must be 'brain_tumor' or 'pneumonia'")
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(400, "Please upload an image file (PNG, JPG, JPEG).")
    data = await file.read()
    if not data:
        raise HTTPException(400, "Empty file.")
    try:
        if task == "brain_tumor":
            result = ml.predict_brain(data)
        else:
            result = ml.predict_pneumonia(data)
    except Exception as e:
        raise HTTPException(500, f"Model inference failed: {e}")
    result["preview"] = ml.encode_image_b64(data)
    try:
        result["explanation"] = rag.explain_image_finding(result)
    except Exception as e:
        result["explanation"] = f"(AI explanation unavailable: {e})"
    return result


# ----------- API: Symptom checker -----------------------------------------

class SymptomRequest(BaseModel):
    symptoms: List[str]
    top_k: int = 3


@app.post("/api/symptoms/predict")
def api_symptoms_predict(req: SymptomRequest):
    if not req.symptoms:
        raise HTTPException(400, "Provide at least one symptom.")
    return {"results": sym_mod.predict(req.symptoms, top_k=req.top_k)}


@app.get("/api/symptoms/list")
def api_symptoms_list():
    return {"symptoms": sym_mod.all_symptoms()}


# ----------- API: Report analyzer -----------------------------------------

class ReportRequest(BaseModel):
    text: str


@app.post("/api/report/analyze")
async def api_report_analyze(
    text: str = Form(default=""), file: Optional[UploadFile] = File(default=None)
):
    body = (text or "").strip()
    if file and file.filename:
        raw = await file.read()
        try:
            body = (body + "\n" + raw.decode("utf-8", errors="ignore")).strip()
        except Exception:
            raise HTTPException(400, "Could not read uploaded file as text.")
    if len(body) < 20:
        raise HTTPException(400, "Please provide at least a short medical report (20+ characters).")
    return rag.analyze_report(body[:8000])


# ----------- API: Chat -----------------------------------------------------

class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    message: str
    history: List[ChatMessage] = []


@app.post("/api/chat")
def api_chat(req: ChatRequest):
    if not req.message.strip():
        raise HTTPException(400, "Empty message.")
    history = [m.model_dump() for m in req.history]
    return rag.chat(req.message.strip(), history=history)


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "gemini_configured": bool(os.environ.get("GEMINI_API_KEY")),
        "kb_docs": len(rag.KB.docs),
        "diseases": len(sym_mod.all_diseases()),
        "symptoms": len(sym_mod.all_symptoms()),
    }
