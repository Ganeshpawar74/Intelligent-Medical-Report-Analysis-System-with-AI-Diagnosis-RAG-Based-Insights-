"""Organ / body-part detection from medical reports.

Looks at the user's report text and the AI explanation (if available) and
returns the most likely affected organ together with positional info that
the front-end uses to highlight a 3D body model.

Pure-Python keyword matching, no external services.
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple


# Each organ entry has:
#   keys      : list of regex / keyword patterns (case-insensitive, word-bounded)
#   label     : display label
#   region    : "head" | "chest" | "abdomen" | "pelvis" | "limbs" | "skin"
#   color     : hex color used by the 3D scene to highlight the part
#   description : one-liner shown beside the model
ORGANS: Dict[str, Dict] = {
    "brain": {
        "keys": [
            "brain", "cerebr", "cerebellum", "glioma", "meningioma", "pituitary",
            "tumor of the brain", "intracranial", "stroke", "ischemic stroke",
            "hemorrhag", "encephal", "neurolog", "migraine", "headache",
            "seizure", "epilep", "alzheimer", "parkinson",
        ],
        "label": "Brain",
        "region": "head",
        "color": "#ef4444",
        "description": "Central nervous system organ — controls thought, movement, senses and vital functions.",
    },
    "eye": {
        "keys": ["eye", "ocular", "retina", "vision", "conjunctiv", "cataract", "glaucoma"],
        "label": "Eyes",
        "region": "head",
        "color": "#f59e0b",
        "description": "Sensory organ for sight.",
    },
    "ear": {
        "keys": ["ear", "auditory", "tinnitus", "otitis", "hearing loss", "cochlea"],
        "label": "Ears",
        "region": "head",
        "color": "#f59e0b",
        "description": "Sensory organ for hearing and balance.",
    },
    "nose_sinus": {
        "keys": ["nose", "nasal", "sinus", "sinusitis", "rhinitis", "rhinor"],
        "label": "Nose / Sinuses",
        "region": "head",
        "color": "#f59e0b",
        "description": "Upper airway involved in breathing and smell.",
    },
    "throat": {
        "keys": [
            "throat", "pharyn", "larynx", "tonsil", "vocal cord",
            "sore throat", "strep",
        ],
        "label": "Throat",
        "region": "head",
        "color": "#f59e0b",
        "description": "Passageway for air and food, contains the voice box.",
    },
    "thyroid": {
        "keys": ["thyroid", "hypothyroid", "hyperthyroid", "goiter", "tsh"],
        "label": "Thyroid",
        "region": "head",
        "color": "#10b981",
        "description": "Endocrine gland in the neck that regulates metabolism.",
    },
    "lungs": {
        "keys": [
            "lung", "pulmonary", "pneumon", "bronch", "asthma",
            "copd", "tuberculosis", "tb ", "respiratory", "chest x-?ray",
            "pleural", "pleura", "alveol", "spiculated nodule",
            "pulmonary nodule", "shortness of breath", "dyspnea",
        ],
        "label": "Lungs",
        "region": "chest",
        "color": "#3b82f6",
        "description": "Pair of organs responsible for breathing and gas exchange.",
    },
    "heart": {
        "keys": [
            "heart", "cardiac", "cardio", "myocard", "infarct", "angina",
            "ecg", "ekg", "ischem", "atrial", "ventricul", "coronary",
            "tachycard", "bradycard", "arrhythm", "hypertension",
            "high blood pressure", "chest pain", "st elevation",
            "troponin", "valve",
        ],
        "label": "Heart",
        "region": "chest",
        "color": "#ef4444",
        "description": "Muscular organ that pumps blood throughout the body.",
    },
    "esophagus": {
        "keys": ["esophag", "oesophag", "reflux", "gerd", "heartburn"],
        "label": "Esophagus",
        "region": "chest",
        "color": "#a855f7",
        "description": "Tube carrying food from the throat to the stomach.",
    },
    "stomach": {
        "keys": [
            "stomach", "gastric", "gastritis", "ulcer", "peptic",
            "indigestion", "dyspepsia", "nausea", "vomit",
        ],
        "label": "Stomach",
        "region": "abdomen",
        "color": "#a855f7",
        "description": "Digestive organ that breaks down food using acids and enzymes.",
    },
    "liver": {
        "keys": [
            "liver", "hepat", "jaundice", "cirrhosis", "hepatitis",
            "ast ", "alt ", "alkaline phosphatase",
        ],
        "label": "Liver",
        "region": "abdomen",
        "color": "#b45309",
        "description": "Largest internal organ — filters blood and produces bile.",
    },
    "gallbladder": {
        "keys": ["gallbladder", "cholecyst", "gallstone", "biliary"],
        "label": "Gallbladder",
        "region": "abdomen",
        "color": "#84cc16",
        "description": "Small pouch under the liver that stores bile.",
    },
    "pancreas": {
        "keys": ["pancrea", "diabet", "insulin", "glucose", "hyperglyc", "hypoglyc"],
        "label": "Pancreas",
        "region": "abdomen",
        "color": "#eab308",
        "description": "Gland that produces insulin and digestive enzymes.",
    },
    "spleen": {
        "keys": ["spleen", "splenomeg"],
        "label": "Spleen",
        "region": "abdomen",
        "color": "#9333ea",
        "description": "Organ that filters blood and supports the immune system.",
    },
    "kidney": {
        "keys": [
            "kidney", "renal", "nephr", "ureter", "creatinine",
            "uti ", "urinary tract infection",
        ],
        "label": "Kidneys",
        "region": "abdomen",
        "color": "#06b6d4",
        "description": "Pair of organs that filter blood and produce urine.",
    },
    "bladder": {
        "keys": ["bladder", "cystitis", "incontinence", "urinary"],
        "label": "Bladder",
        "region": "pelvis",
        "color": "#06b6d4",
        "description": "Hollow organ that stores urine before it leaves the body.",
    },
    "intestines": {
        "keys": [
            "intestin", "bowel", "colon", "colitis", "crohn",
            "diarrhea", "constipation", "ibs", "ileum", "jejun", "duoden",
            "appendix", "appendicitis",
        ],
        "label": "Intestines",
        "region": "abdomen",
        "color": "#f97316",
        "description": "Long tube where most digestion and absorption happen.",
    },
    "reproductive_f": {
        "keys": ["ovar", "uterus", "uterine", "cervix", "menstr", "pregnan"],
        "label": "Reproductive system",
        "region": "pelvis",
        "color": "#ec4899",
        "description": "Organs involved in reproduction and hormonal regulation.",
    },
    "reproductive_m": {
        "keys": ["prostate", "testic", "scrotum"],
        "label": "Reproductive system",
        "region": "pelvis",
        "color": "#ec4899",
        "description": "Organs involved in reproduction and hormonal regulation.",
    },
    "skin": {
        "keys": [
            "skin", "rash", "dermat", "eczema", "psoriasis", "acne",
            "itch", "hive", "urticaria", "lesion on the skin",
        ],
        "label": "Skin",
        "region": "skin",
        "color": "#fb7185",
        "description": "Largest organ — barrier protecting against infection and injury.",
    },
    "joints": {
        "keys": [
            "joint", "arthrit", "osteoarthr", "rheumatoid",
            "knee pain", "elbow pain", "shoulder pain",
        ],
        "label": "Joints",
        "region": "limbs",
        "color": "#22d3ee",
        "description": "Where two bones meet — allow movement of the body.",
    },
    "bones": {
        "keys": [
            "bone", "fracture", "osteoporos", "osteomyel", "spine",
            "vertebr", "back pain",
        ],
        "label": "Bones / Spine",
        "region": "limbs",
        "color": "#94a3b8",
        "description": "Skeletal structures providing form, support and protection.",
    },
    "blood": {
        "keys": [
            "anemia", "anaemia", "leuk", "lymphom", "platelet",
            "hemoglobin", "haemoglobin", "wbc", "rbc",
            "blood count",
        ],
        "label": "Blood / Circulation",
        "region": "chest",
        "color": "#dc2626",
        "description": "Body-wide circulation of cells, oxygen and nutrients.",
    },
}


# Penalty tokens that often appear without actually meaning the organ is the primary site.
NEGATIVE_CONTEXT = re.compile(
    r"\b(no|denies|without|unremark|negative for)\s+\w+",
    re.IGNORECASE,
)


def _count_hits(text: str, patterns: List[str]) -> int:
    """Count case-insensitive whole-word/substring hits."""
    score = 0
    for p in patterns:
        if " " in p or "-" in p or "?" in p:
            # treat as a regex/phrase
            score += len(re.findall(p, text, re.IGNORECASE))
        else:
            score += len(re.findall(rf"\b{re.escape(p)}\w*", text, re.IGNORECASE))
    return score


def detect_organ(report_text: str, explanation: str = "") -> Optional[Dict]:
    """Return the most likely affected organ (or None) and runner-ups.

    Output shape:
        {
          "key": "lungs",
          "label": "Lungs",
          "region": "chest",
          "color": "#3b82f6",
          "description": "...",
          "score": 7,
          "candidates": [{"key": "...", "label": "...", "score": 3}, ...]
        }
    """
    if not report_text and not explanation:
        return None

    # The explanation is generally cleaner medical language; weight it slightly higher.
    text_report = report_text or ""
    text_expl = explanation or ""

    # Drop obvious "no X" phrases so 'no acute pulmonary process' doesn't light up the lungs.
    text_report_clean = NEGATIVE_CONTEXT.sub("", text_report)
    text_expl_clean = NEGATIVE_CONTEXT.sub("", text_expl)

    scored: List[Tuple[str, int]] = []
    for key, meta in ORGANS.items():
        s = _count_hits(text_report_clean, meta["keys"]) * 2
        s += _count_hits(text_expl_clean, meta["keys"]) * 3
        if s > 0:
            scored.append((key, s))

    if not scored:
        return None

    scored.sort(key=lambda kv: kv[1], reverse=True)
    top_key, top_score = scored[0]
    meta = ORGANS[top_key]

    candidates = [
        {"key": k, "label": ORGANS[k]["label"], "region": ORGANS[k]["region"], "score": s}
        for k, s in scored[:5]
    ]

    return {
        "key": top_key,
        "label": meta["label"],
        "region": meta["region"],
        "color": meta["color"],
        "description": meta["description"],
        "score": top_score,
        "candidates": candidates,
    }
