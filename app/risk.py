"""Risk / severity scoring for image findings and report text."""
from __future__ import annotations
import re
from typing import Dict, List
from . import symptoms as sym_mod

# High-risk keywords that should escalate severity if found in a report
ALARM_KEYWORDS = [
    "myocardial infarction", "heart attack", "stroke", "hemorrhage", "haemorrhage",
    "sepsis", "septic", "anaphylaxis", "respiratory failure", "cardiac arrest",
    "stemi", "nstemi", "pulmonary embolism", "embolism", "tumor", "tumour",
    "malignant", "malignancy", "cancer", "carcinoma", "metastasis", "metastatic",
    "severe", "acute", "critical", "emergency", "urgent",
    "shortness of breath", "chest pain", "loss of consciousness", "seizure",
    "unresponsive", "hypotension", "shock",
]

LEVELS = {
    "low":      {"label": "Low risk", "color": "emerald",
                 "advice": "No urgent action indicated. Continue with routine self-care and follow general health guidance."},
    "moderate": {"label": "Moderate risk", "color": "amber",
                 "advice": "Consider scheduling a visit with a general physician soon to discuss these findings and confirm with proper clinical evaluation."},
    "high":     {"label": "High risk", "color": "rose",
                 "advice": "Please consult a qualified doctor or visit a nearby clinic / hospital as soon as possible for proper evaluation. If you experience worsening symptoms, seek emergency care immediately."},
}


def _pack(level: str, score: int, reasons: List[str]) -> Dict:
    info = LEVELS[level]
    return {
        "level": level,
        "label": info["label"],
        "color": info["color"],
        "score": int(score),
        "advice": info["advice"],
        "reasons": reasons,
    }


# ---------------------------------------------------------------------------
# Image findings
# ---------------------------------------------------------------------------

def assess_image_finding(finding: Dict) -> Dict:
    task = finding.get("task")
    label = finding.get("label", "")
    conf = float(finding.get("confidence", 0.0))
    reasons: List[str] = []

    if task == "brain_tumor":
        if label == "notumor":
            level = "low"
            score = 1
            reasons.append("No tumor detected by the classifier.")
        else:
            # any detected tumor type
            level = "high" if conf >= 0.6 else "moderate"
            score = 8 if level == "high" else 5
            reasons.append(f"Possible {label} detected (confidence {conf:.0%}).")
            reasons.append("Brain tumors require specialist evaluation regardless of subtype.")
    elif task == "pneumonia":
        if label == "NORMAL":
            level = "low"
            score = 1
            reasons.append("Chest X-ray classified as normal.")
        else:
            level = "high" if conf >= 0.7 else "moderate"
            score = 7 if level == "high" else 4
            reasons.append(f"Pneumonia features detected (confidence {conf:.0%}).")
            reasons.append("Lung infections can worsen quickly; clinical confirmation is important.")
    else:
        level = "moderate"
        score = 3
        reasons.append("Unknown task — defaulting to a cautious moderate-risk assessment.")

    if conf < 0.55 and level != "low":
        reasons.append("Model confidence is borderline — a clinician should confirm.")

    return _pack(level, score, reasons)


# ---------------------------------------------------------------------------
# Report text
# ---------------------------------------------------------------------------

_SYMPTOM_PATTERNS = None

def _build_patterns():
    global _SYMPTOM_PATTERNS
    pats = []
    for s in sym_mod.all_symptoms():
        # convert "skin_rash" -> regex matching "skin rash" or "skin_rash"
        words = s.split("_")
        pat = r"\b" + r"[\s_-]+".join(re.escape(w) for w in words) + r"\b"
        pats.append((s, re.compile(pat, re.IGNORECASE), sym_mod._SEV.get(s, 1)))
    _SYMPTOM_PATTERNS = pats


def assess_report(text: str) -> Dict:
    if _SYMPTOM_PATTERNS is None:
        _build_patterns()

    found_symptoms: List[Dict] = []
    seen = set()
    severity_sum = 0
    for name, pat, weight in _SYMPTOM_PATTERNS:
        if pat.search(text) and name not in seen:
            seen.add(name)
            found_symptoms.append({
                "name": name.replace("_", " "),
                "weight": int(weight),
            })
            severity_sum += int(weight)

    found_alarms: List[str] = []
    low = text.lower()
    for kw in ALARM_KEYWORDS:
        if kw in low and kw not in found_alarms:
            found_alarms.append(kw)

    # scoring
    score = severity_sum + 4 * len(found_alarms)
    if found_alarms or score >= 12:
        level = "high"
    elif score >= 5 or len(found_symptoms) >= 3:
        level = "moderate"
    else:
        level = "low"

    reasons = []
    if found_alarms:
        reasons.append(
            f"Detected {len(found_alarms)} high-risk indicator(s) in the report: "
            + ", ".join(found_alarms[:6])
            + ("…" if len(found_alarms) > 6 else "")
        )
    if found_symptoms:
        top = sorted(found_symptoms, key=lambda x: -x["weight"])[:6]
        reasons.append(
            "Symptoms recognised: "
            + ", ".join(f"{s['name']} (severity {s['weight']})" for s in top)
            + (f" and {len(found_symptoms)-6} more" if len(found_symptoms) > 6 else "")
        )
    if not reasons:
        reasons.append("No specific high-risk indicators detected in the text.")

    out = _pack(level, score, reasons)
    out["matched_symptoms"] = found_symptoms
    out["alarm_keywords"] = found_alarms
    out["severity_sum"] = int(severity_sum)
    return out
