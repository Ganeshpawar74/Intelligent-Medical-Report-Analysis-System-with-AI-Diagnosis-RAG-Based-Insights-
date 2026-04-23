"""Medical glossary: curated definitions + LRU cache + Gemini fallback."""
from __future__ import annotations
import functools
import os
import re
from typing import Dict, List, Optional

# Curated dictionary of common medical terms (plain-language explanations).
GLOSSARY: Dict[str, Dict[str, str]] = {
    # --- Brain / Neuro ---
    "glioma":         {"plain": "A tumour that grows from glial cells, the support cells in the brain or spinal cord.", "category": "Brain tumour"},
    "meningioma":     {"plain": "A usually slow-growing tumour that forms in the meninges, the membranes covering the brain and spinal cord. Most are benign.", "category": "Brain tumour"},
    "pituitary":      {"plain": "Relating to the pituitary gland, a pea-sized gland at the base of the brain that controls hormones.", "category": "Anatomy"},
    "pituitary tumor": {"plain": "An abnormal growth in the pituitary gland. Most are non-cancerous but can affect hormone levels.", "category": "Brain tumour"},
    "tumor":          {"plain": "An abnormal lump or growth of cells. Can be benign (non-cancerous) or malignant (cancerous).", "category": "General"},
    "tumour":         {"plain": "An abnormal lump or growth of cells. Can be benign (non-cancerous) or malignant (cancerous).", "category": "General"},
    "benign":         {"plain": "Not cancerous. Benign growths usually do not spread to other parts of the body.", "category": "General"},
    "malignant":      {"plain": "Cancerous. Malignant cells can invade nearby tissue and spread to other parts of the body.", "category": "General"},
    "metastasis":     {"plain": "The spread of cancer cells from where they first formed to another part of the body.", "category": "Oncology"},
    "lesion":         {"plain": "An area of abnormal tissue — could be an injury, infection, or tumour.", "category": "General"},
    "edema":          {"plain": "Swelling caused by extra fluid trapped in body tissues.", "category": "General"},
    "mri":            {"plain": "Magnetic Resonance Imaging — a scan that uses magnets and radio waves to make detailed pictures of inside the body.", "category": "Imaging"},
    "ct":             {"plain": "Computed Tomography — an X-ray scan that produces cross-sectional pictures of the body.", "category": "Imaging"},
    "ct scan":        {"plain": "Computed Tomography scan — uses X-rays to build cross-sectional pictures of the body.", "category": "Imaging"},
    "biopsy":         {"plain": "Removing a small piece of tissue to examine it under a microscope, usually to check for cancer.", "category": "Procedure"},

    # --- Lungs / Respiratory ---
    "pneumonia":      {"plain": "An infection that inflames the air sacs in the lungs, which may fill with fluid.", "category": "Lung disease"},
    "pneumonitis":    {"plain": "Inflammation of lung tissue, usually not caused by infection.", "category": "Lung disease"},
    "consolidation":  {"plain": "An area of the lung that is filled with fluid or pus instead of air, often seen in pneumonia.", "category": "Imaging finding"},
    "infiltrate":     {"plain": "An area on a chest X-ray that looks denser than normal lung — often suggests fluid, pus, or cells where they shouldn't be.", "category": "Imaging finding"},
    "opacity":        {"plain": "A whitish area on a scan that blocks X-rays — often indicating fluid, mass, or scarring.", "category": "Imaging finding"},
    "nodule":         {"plain": "A small, rounded lump in the body. In the lungs, a nodule may need follow-up to rule out cancer.", "category": "Imaging finding"},
    "spiculated":     {"plain": "Having spike-like edges. A spiculated nodule is more concerning for cancer.", "category": "Imaging finding"},
    "pleural effusion": {"plain": "Fluid build-up between the layers of tissue lining the lungs and the chest cavity.", "category": "Lung disease"},
    "bronchitis":     {"plain": "Inflammation of the airways (bronchi) that carry air to the lungs.", "category": "Lung disease"},
    "asthma":         {"plain": "A long-term condition where the airways become narrow and inflamed, causing wheezing and shortness of breath.", "category": "Lung disease"},
    "copd":           {"plain": "Chronic Obstructive Pulmonary Disease — a group of lung diseases that block airflow and make breathing difficult.", "category": "Lung disease"},
    "dyspnea":        {"plain": "Shortness of breath or difficulty breathing.", "category": "Symptom"},
    "shortness of breath": {"plain": "Feeling unable to get enough air; medically called dyspnea.", "category": "Symptom"},
    "tachypnea":      {"plain": "Breathing faster than normal.", "category": "Symptom"},
    "hypoxia":        {"plain": "When tissues do not get enough oxygen.", "category": "Symptom"},

    # --- Cardiac ---
    "myocardial infarction": {"plain": "A heart attack — happens when blood flow to part of the heart is blocked, damaging heart muscle.", "category": "Cardiac"},
    "ischemia":       {"plain": "Reduced blood supply to an organ or part of the body, often the heart.", "category": "Cardiac"},
    "angina":         {"plain": "Chest pain caused by reduced blood flow to the heart muscle.", "category": "Cardiac"},
    "arrhythmia":     {"plain": "An irregular heartbeat — too fast, too slow, or uneven.", "category": "Cardiac"},
    "hypertension":   {"plain": "High blood pressure — a long-term condition that strains blood vessels and the heart.", "category": "Cardiac"},
    "hypotension":    {"plain": "Low blood pressure.", "category": "Cardiac"},
    "tachycardia":    {"plain": "A heart rate that is faster than normal (over 100 bpm at rest).", "category": "Cardiac"},
    "bradycardia":    {"plain": "A heart rate that is slower than normal (under 60 bpm).", "category": "Cardiac"},
    "ecg":            {"plain": "Electrocardiogram — a quick test that records the electrical activity of the heart.", "category": "Diagnostic test"},
    "ekg":            {"plain": "Electrocardiogram — a quick test that records the electrical activity of the heart.", "category": "Diagnostic test"},
    "st elevation":   {"plain": "A pattern on an ECG that often signals a serious heart attack involving full-thickness damage to the heart muscle.", "category": "Cardiac"},
    "troponin":       {"plain": "A protein released into the blood when heart muscle is damaged. High levels suggest a heart attack.", "category": "Lab test"},
    "diaphoresis":    {"plain": "Heavy sweating, often associated with serious illness like a heart attack.", "category": "Symptom"},

    # --- General / lab / vitals ---
    "hemoglobin":     {"plain": "The protein in red blood cells that carries oxygen. Low levels mean anemia.", "category": "Lab test"},
    "anemia":         {"plain": "Having fewer healthy red blood cells than normal, leading to tiredness and weakness.", "category": "Blood disorder"},
    "leukocytosis":   {"plain": "A higher than normal white blood cell count, often a sign of infection or inflammation.", "category": "Lab finding"},
    "leukopenia":     {"plain": "A lower than normal white blood cell count, which can increase infection risk.", "category": "Lab finding"},
    "thrombocytopenia": {"plain": "Low platelet count, which can cause easy bruising and bleeding.", "category": "Lab finding"},
    "glucose":        {"plain": "Blood sugar — the main source of energy for the body's cells.", "category": "Lab test"},
    "hyperglycemia":  {"plain": "High blood sugar.", "category": "Lab finding"},
    "hypoglycemia":   {"plain": "Low blood sugar.", "category": "Lab finding"},
    "diabetes":       {"plain": "A condition where blood sugar levels are too high because the body cannot use insulin properly.", "category": "Endocrine"},
    "hyperthyroidism":{"plain": "An overactive thyroid gland, producing too much thyroid hormone.", "category": "Endocrine"},
    "hypothyroidism": {"plain": "An underactive thyroid gland, producing too little thyroid hormone.", "category": "Endocrine"},
    "fever":          {"plain": "A body temperature higher than normal (above 38°C / 100.4°F), usually a sign of infection.", "category": "Symptom"},
    "sepsis":         {"plain": "A life-threatening reaction to infection that can damage organs.", "category": "Emergency"},
    "inflammation":   {"plain": "The body's response to injury or infection — causing redness, warmth, swelling, or pain.", "category": "General"},
    "edematous":      {"plain": "Swollen due to fluid build-up.", "category": "General"},
    "erythema":       {"plain": "Redness of the skin.", "category": "Skin"},
    "pruritus":       {"plain": "Itching of the skin.", "category": "Skin"},
    "urticaria":      {"plain": "Hives — itchy, raised welts on the skin, usually due to an allergic reaction.", "category": "Skin"},
    "rhinitis":       {"plain": "Inflammation of the inside of the nose, causing sneezing, congestion, and a runny nose.", "category": "ENT"},
    "allergic rhinitis":{"plain": "Hay fever — sneezing and runny nose triggered by allergens like pollen.", "category": "ENT"},
    "gastritis":      {"plain": "Inflammation of the lining of the stomach, often causing pain and nausea.", "category": "GI"},
    "hepatitis":      {"plain": "Inflammation of the liver, usually caused by a virus.", "category": "GI"},
    "nephritis":      {"plain": "Inflammation of the kidneys.", "category": "Renal"},
    "renal":          {"plain": "Relating to the kidneys.", "category": "Anatomy"},
    "hepatic":        {"plain": "Relating to the liver.", "category": "Anatomy"},
    "cardiac":        {"plain": "Relating to the heart.", "category": "Anatomy"},
    "pulmonary":      {"plain": "Relating to the lungs.", "category": "Anatomy"},
    "cerebral":       {"plain": "Relating to the brain.", "category": "Anatomy"},
    "acute":          {"plain": "Sudden in onset and usually short-term, often severe.", "category": "Descriptor"},
    "chronic":        {"plain": "Long-lasting, often persisting for months or years.", "category": "Descriptor"},
    "bilateral":      {"plain": "Affecting both sides of the body.", "category": "Descriptor"},
    "unilateral":     {"plain": "Affecting only one side of the body.", "category": "Descriptor"},
    "prognosis":      {"plain": "The likely course and outcome of a disease.", "category": "General"},
    "etiology":       {"plain": "The cause of a disease.", "category": "General"},
    "differential diagnosis": {"plain": "A list of possible conditions that could be causing the patient's symptoms.", "category": "General"},
    "asymptomatic":   {"plain": "Showing no symptoms.", "category": "Descriptor"},
    "symptomatic":    {"plain": "Showing symptoms of a condition.", "category": "Descriptor"},
    "remission":      {"plain": "A period when symptoms of a chronic disease lessen or disappear.", "category": "General"},
    "relapse":        {"plain": "The return of a disease or symptoms after a period of improvement.", "category": "General"},

    # --- Common symptoms ---
    "headache":       {"plain": "Pain or discomfort in the head, scalp, or neck.", "category": "Symptom"},
    "nausea":         {"plain": "A feeling of sickness with an urge to vomit.", "category": "Symptom"},
    "vomiting":       {"plain": "Forcefully ejecting stomach contents through the mouth.", "category": "Symptom"},
    "fatigue":        {"plain": "Extreme tiredness or lack of energy.", "category": "Symptom"},
    "dizziness":      {"plain": "Feeling lightheaded, unsteady, or like the room is spinning.", "category": "Symptom"},
    "vertigo":        {"plain": "A spinning sensation, as if you or your surroundings are moving.", "category": "Symptom"},
    "syncope":        {"plain": "Fainting — a brief loss of consciousness due to a temporary drop in blood flow to the brain.", "category": "Symptom"},
    "palpitations":   {"plain": "A feeling that your heart is pounding, fluttering, or beating irregularly.", "category": "Symptom"},
    "chest pain":     {"plain": "Pain or discomfort in the chest. Can have many causes — some serious like heart attack, others minor.", "category": "Symptom"},
    "abdominal pain": {"plain": "Pain felt anywhere between the chest and the groin.", "category": "Symptom"},
    "back pain":      {"plain": "Pain in any part of the back, often in the lower back.", "category": "Symptom"},
    "joint pain":     {"plain": "Discomfort, aches, or soreness in any of the body's joints.", "category": "Symptom"},
    "skin rash":      {"plain": "A noticeable change in the texture or color of the skin — may be itchy, red, or bumpy.", "category": "Symptom"},
    "itching":        {"plain": "An irritating sensation that makes you want to scratch.", "category": "Symptom"},
    "sore throat":    {"plain": "Pain, scratchiness, or irritation of the throat.", "category": "Symptom"},
    "runny nose":     {"plain": "Mucus discharge from the nose, often due to a cold or allergies.", "category": "Symptom"},
    "congestion":     {"plain": "A blocked or stuffy nose, usually from inflammation of the nasal passages.", "category": "Symptom"},
    "high fever":     {"plain": "A body temperature above 39°C / 102°F — should be taken seriously.", "category": "Symptom"},
    "breathlessness": {"plain": "Feeling like you can't get enough air; same as shortness of breath.", "category": "Symptom"},
}

# Compile a single regex matching any glossary term as a whole word/phrase, longest first.
_TERMS_SORTED = sorted(GLOSSARY.keys(), key=len, reverse=True)
_TERM_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(t) for t in _TERMS_SORTED) + r")\b",
    flags=re.IGNORECASE,
)


def all_terms() -> List[str]:
    return list(GLOSSARY.keys())


def lookup_local(term: str) -> Optional[Dict[str, str]]:
    if not term:
        return None
    t = term.strip().lower()
    if t in GLOSSARY:
        d = GLOSSARY[t]
        return {"term": t, "plain": d["plain"], "category": d.get("category", "Medical"), "source": "curated"}
    return None


@functools.lru_cache(maxsize=512)
def _gemini_define(term: str) -> Optional[str]:
    """Ask Gemini for a one-line plain-language definition. Cached."""
    try:
        from . import rag  # reuse the configured Gemini client
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            return None
        prompt = (
            f"Define the medical term \"{term}\" in ONE short sentence (max 25 words) "
            f"using plain everyday language a patient with no medical background would understand. "
            f"Return only the definition, no preamble."
        )
        text = rag._gemini_generate(prompt)
        if not text:
            return None
        text = text.strip().strip('"').strip()
        # Trim to first sentence.
        if "." in text:
            text = text.split(".")[0] + "."
        return text[:280]
    except Exception:
        return None


def define(term: str) -> Dict[str, str]:
    local = lookup_local(term)
    if local:
        return local
    plain = _gemini_define(term.lower())
    if plain:
        return {"term": term.lower(), "plain": plain, "category": "Medical", "source": "ai"}
    return {
        "term": term.lower(),
        "plain": "No definition is available for this term yet. Please consult a clinician for clarification.",
        "category": "Unknown",
        "source": "none",
    }
