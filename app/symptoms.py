"""Symptom-based disease prediction.

Uses the disease-symptoms dataset to build a multinomial-NB classifier
plus per-disease descriptions and precautions.
"""
from __future__ import annotations
import csv, re
from pathlib import Path
from typing import List, Dict
from collections import defaultdict
import numpy as np

DATA = Path("Data/Disease Symtoms")


def _norm(s: str) -> str:
    return re.sub(r"\s+", "_", s.strip().lower()).strip("_")


def _load():
    rows = []
    with open(DATA / "dataset.csv", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            disease = row[0].strip()
            syms = [_norm(s) for s in row[1:] if s and s.strip()]
            if disease and syms:
                rows.append((disease, syms))

    severity: Dict[str, int] = {}
    with open(DATA / "Symptom-severity.csv", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            if len(row) >= 2:
                severity[_norm(row[0])] = int(row[1])

    desc: Dict[str, str] = {}
    with open(DATA / "symptom_Description.csv", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            if len(row) >= 2:
                desc[row[0].strip()] = row[1].strip()

    prec: Dict[str, List[str]] = {}
    with open(DATA / "symptom_precaution.csv", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            if not row:
                continue
            d = row[0].strip()
            ps = [p.strip() for p in row[1:] if p and p.strip()]
            if d:
                prec[d] = ps
    return rows, severity, desc, prec


_ROWS, _SEV, _DESC, _PREC = _load()
_ALL_SYMPTOMS = sorted({s for _, syms in _ROWS for s in syms})
_SYM_INDEX = {s: i for i, s in enumerate(_ALL_SYMPTOMS)}
_DISEASES = sorted({d for d, _ in _ROWS})
_DIS_INDEX = {d: i for i, d in enumerate(_DISEASES)}


def _build_priors_likelihoods():
    # Multinomial NB-ish with Laplace smoothing
    n_d = len(_DISEASES)
    n_s = len(_ALL_SYMPTOMS)
    counts = np.ones((n_d, n_s), dtype=np.float64)  # +1 smoothing
    totals = np.full(n_d, n_s, dtype=np.float64)
    priors = np.ones(n_d, dtype=np.float64)
    for d, syms in _ROWS:
        di = _DIS_INDEX[d]
        priors[di] += 1
        for s in syms:
            counts[di, _SYM_INDEX[s]] += 1
            totals[di] += 1
    log_likelihood = np.log(counts / totals[:, None])
    log_prior = np.log(priors / priors.sum())
    return log_prior, log_likelihood


_LOG_PRIOR, _LOG_LIK = _build_priors_likelihoods()


def all_symptoms() -> List[str]:
    return list(_ALL_SYMPTOMS)


def predict(symptoms: List[str], top_k: int = 3) -> List[dict]:
    selected = [_norm(s) for s in symptoms if _norm(s) in _SYM_INDEX]
    if not selected:
        return []
    vec = np.zeros(len(_ALL_SYMPTOMS))
    for s in selected:
        vec[_SYM_INDEX[s]] = 1.0
    scores = _LOG_PRIOR + _LOG_LIK @ vec
    # softmax
    e = np.exp(scores - scores.max())
    probs = e / e.sum()
    order = np.argsort(probs)[::-1][:top_k]
    out = []
    for i in order:
        d = _DISEASES[i]
        out.append({
            "disease": d,
            "probability": float(probs[i]),
            "description": _DESC.get(d, ""),
            "precautions": _PREC.get(d, []),
            "matched_symptoms": [s for s in selected if any(s == sym for sym in [_norm(x) for x in _SYM_INDEX])],
        })
    # severity score for selected symptoms
    severity_total = sum(_SEV.get(s, 0) for s in selected)
    for o in out:
        o["severity_score"] = severity_total
    return out


def disease_info(disease: str) -> dict:
    return {
        "disease": disease,
        "description": _DESC.get(disease, ""),
        "precautions": _PREC.get(disease, []),
    }


def all_diseases() -> List[str]:
    return list(_DISEASES)
