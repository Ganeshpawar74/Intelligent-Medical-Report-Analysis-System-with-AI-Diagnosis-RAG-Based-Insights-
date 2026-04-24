"""RAG knowledge base + Gemini-powered chat / report analysis.

Uses TF-IDF over the medical-transcription samples, disease descriptions,
and precaution sheets. Generation is delegated to Google Gemini.
"""
from __future__ import annotations
import os, csv, re, json
from pathlib import Path
from typing import List, Dict, Optional
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from . import symptoms as sym_mod

DATA_DIR = Path("Data")
MT_PATH = DATA_DIR / "mtsamples.csv"
DIS_DIR = DATA_DIR / "Disease Symtoms"

GEMINI_MODELS = [
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite",
]


# ---------------------------------------------------------------------------
# Knowledge base
# ---------------------------------------------------------------------------

def _load_mt_chunks(max_rows: int = 1500) -> List[Dict]:
    out = []
    with open(MT_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i >= max_rows:
                break
            text = (row.get("transcription") or "").strip()
            if len(text) < 50:
                continue
            out.append({
                "source": "mtsamples",
                "title": (row.get("sample_name") or "").strip() or "Sample",
                "specialty": (row.get("medical_specialty") or "").strip(),
                "keywords": (row.get("keywords") or "").strip(),
                "text": text[:1800],
            })
    return out


def _load_disease_chunks() -> List[Dict]:
    out = []
    desc, prec = {}, {}
    with open(DIS_DIR / "symptom_Description.csv", newline="", encoding="utf-8") as f:
        for row in csv.reader(f):
            if len(row) >= 2 and row[0] != "Disease":
                desc[row[0].strip()] = row[1].strip()
    with open(DIS_DIR / "symptom_precaution.csv", newline="", encoding="utf-8") as f:
        for row in csv.reader(f):
            if len(row) >= 2 and row[0] != "Disease":
                prec[row[0].strip()] = [p.strip() for p in row[1:] if p and p.strip()]
    for d in set(list(desc) + list(prec)):
        text = f"Disease: {d}.\n{desc.get(d, '')}\nPrecautions: {', '.join(prec.get(d, []))}"
        out.append({
            "source": "disease_kb",
            "title": d,
            "specialty": "General",
            "keywords": d,
            "text": text,
        })
    return out


class KnowledgeBase:
    def __init__(self):
        self.docs: List[Dict] = []
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.matrix = None

    def build(self):
        self.docs = _load_disease_chunks() + _load_mt_chunks()
        corpus = [
            f"{d['title']} {d['specialty']} {d['keywords']} {d['text']}"
            for d in self.docs
        ]
        self.vectorizer = TfidfVectorizer(
            stop_words="english", max_features=40000, ngram_range=(1, 2)
        )
        self.matrix = self.vectorizer.fit_transform(corpus)

    def search(self, query: str, k: int = 5) -> List[Dict]:
        if self.vectorizer is None or self.matrix is None or not query.strip():
            return []
        q = self.vectorizer.transform([query])
        sims = cosine_similarity(q, self.matrix)[0]
        idx = np.argsort(sims)[::-1][:k]
        results = []
        for i in idx:
            if sims[i] <= 0:
                continue
            d = self.docs[i].copy()
            d["score"] = float(sims[i])
            results.append(d)
        return results


KB = KnowledgeBase()
KB.build()


# ---------------------------------------------------------------------------
# Gemini client
# ---------------------------------------------------------------------------

def _gemini_client():
    direct_key = os.environ.get("GEMINI_API_KEY")
    if direct_key:
        try:
            from google import genai
            return genai.Client(api_key=direct_key)
        except Exception:
            return None
    api_key = os.environ.get("AI_INTEGRATIONS_GEMINI_API_KEY")
    base_url = os.environ.get("AI_INTEGRATIONS_GEMINI_BASE_URL")
    if not api_key or not base_url:
        return None
    try:
        from google import genai
        return genai.Client(
            api_key=api_key,
            http_options={"api_version": "", "base_url": base_url},
        )
    except Exception:
        return None


def _gemini_generate(prompt: str, system: Optional[str] = None) -> str:
    client = _gemini_client()
    if client is None:
        return (
            "AI service is not configured. The retrieved knowledge-base passages "
            "are still shown above."
        )
    from google.genai import types
    config_kwargs = {"max_output_tokens": 8192}
    if system:
        config_kwargs["system_instruction"] = system
    config = types.GenerateContentConfig(**config_kwargs)
    last_err = None
    for model in GEMINI_MODELS:
        try:
            resp = client.models.generate_content(
                model=model, contents=prompt, config=config
            )
            text = (resp.text or "").strip()
            if text:
                return text
        except Exception as e:
            last_err = e
            continue
    return f"AI service is temporarily unavailable. Please try again in a moment. ({last_err})"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

CHAT_SYSTEM = (
    "You are MedAssist, a helpful medical knowledge assistant. "
    "Answer the user's question in clear, simple, friendly language. "
    "Always cite information from the provided context when relevant, and clearly label "
    "anything that is general medical knowledge. Add a short 'Important' note that you "
    "are not a substitute for a licensed clinician and the user should consult a doctor "
    "for diagnosis and treatment. Never invent medications or dosages."
)

REPORT_SYSTEM = (
    "You are a medical-report explainer. The user will give you a medical report "
    "(possibly with jargon, lab values, or radiology language). Your job is to: "
    "1) Summarize the report in plain everyday language a non-medical person can "
    "understand. 2) List the key findings as bullet points. 3) Suggest reasonable "
    "next-step recommendations and precautions, drawing on the provided context "
    "passages where relevant. 4) Add a clear disclaimer that this is informational only "
    "and the user should follow up with a qualified clinician.\n\n"
    "STRICT OUTPUT FORMAT — you MUST follow this exactly:\n"
    "- Respond in Markdown.\n"
    "- Do NOT write any preamble or intro sentence before the first heading.\n"
    "- Use these EXACT five section headings, each on its own line, prefixed with '## ' (two hash signs and a space):\n"
    "  ## Summary\n"
    "  ## Key Findings\n"
    "  ## Possible Conditions Mentioned\n"
    "  ## Recommendations & Precautions\n"
    "  ## Disclaimer\n"
    "- Under each heading, write the relevant content (paragraphs or bullet points starting with '- ').\n"
    "- Never use bare headings without the '## ' prefix. Never bold a heading instead of using '## '."
)


def chat(question: str, history: Optional[List[Dict]] = None) -> Dict:
    ctx = KB.search(question, k=5)
    ctx_text = "\n\n".join(
        f"[{i+1}] {c['title']} ({c['source']}, {c['specialty']})\n{c['text'][:900]}"
        for i, c in enumerate(ctx)
    ) or "(no relevant passages found)"
    history_text = ""
    if history:
        history_text = "\n".join(
            f"{m['role'].upper()}: {m['content']}" for m in history[-6:]
        )
    prompt = (
        f"Conversation so far:\n{history_text}\n\n"
        f"Knowledge base context:\n{ctx_text}\n\n"
        f"User question: {question}\n\n"
        "Write your answer in Markdown."
    )
    answer = _gemini_generate(prompt, system=CHAT_SYSTEM)
    return {"answer": answer, "sources": ctx}


def analyze_report(report_text: str) -> Dict:
    # try to find diseases mentioned in the symptom-disease KB by simple matching
    diseases_mentioned = []
    low = report_text.lower()
    for d in sym_mod.all_diseases():
        if d.lower() in low:
            diseases_mentioned.append(d)
    # retrieve similar transcriptions / disease entries
    ctx = KB.search(report_text[:1500], k=6)
    ctx_text = "\n\n".join(
        f"[{i+1}] {c['title']} ({c['source']})\n{c['text'][:900]}"
        for i, c in enumerate(ctx)
    ) or "(no relevant passages found)"
    prompt = (
        f"Medical report to explain:\n\"\"\"\n{report_text.strip()}\n\"\"\"\n\n"
        f"Possibly relevant reference passages from a medical-transcription corpus "
        f"and disease knowledge base:\n{ctx_text}\n\n"
        "Now produce the structured plain-language explanation."
    )
    explanation = _gemini_generate(prompt, system=REPORT_SYSTEM)
    return {
        "explanation": explanation,
        "diseases_mentioned": diseases_mentioned,
        "sources": ctx,
    }


def explain_image_finding(finding: Dict) -> str:
    """Friendly explanation of an image-classification result, with KB context."""
    label = finding.get("label", "")
    task = finding.get("task", "")
    confidence = finding.get("confidence", 0.0)
    query = f"{label} {task} diagnosis treatment precautions"
    ctx = KB.search(query, k=3)
    ctx_text = "\n\n".join(
        f"[{i+1}] {c['title']}\n{c['text'][:700]}" for i, c in enumerate(ctx)
    ) or "(no relevant passages found)"
    prompt = (
        f"An AI image classifier examined a medical image (task: {task}). "
        f"It predicted: {label} with confidence {confidence:.1%}. "
        f"All class probabilities: {finding.get('all_probs')}.\n\n"
        f"Reference passages:\n{ctx_text}\n\n"
        "In simple plain language, explain what this finding means, what the user "
        "should do next, what precautions are reasonable, and remind them this is "
        "an automated screening tool that does not replace a radiologist or doctor. "
        "Use Markdown with sections: What this means, Recommended next steps, "
        "Precautions, Disclaimer."
    )
    return _gemini_generate(prompt, system=CHAT_SYSTEM)
