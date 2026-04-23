"""Image classification: brain tumor (4-class) and pneumonia (binary)."""
from __future__ import annotations
import os, io, base64, threading
from typing import Optional
import numpy as np
from PIL import Image

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

BRAIN_MODEL_PATH = "Notebooks/Models/brain_tumor_final_v3.keras"
PNEU_MODEL_PATH = "Notebooks/Models/pneumonia_final.keras"

BRAIN_CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]
PNEU_CLASSES = ["NORMAL", "PNEUMONIA"]
PNEU_THRESHOLD = 0.5
IMG_SIZE = (224, 224)

_lock = threading.Lock()
_brain_model = None
_pneu_model = None
_preprocess = None


def _ensure_tf():
    global _preprocess
    import tensorflow as tf  # noqa: F401
    from tensorflow.keras.applications.efficientnet import preprocess_input
    _preprocess = preprocess_input
    return tf


def get_brain_model():
    global _brain_model
    with _lock:
        if _brain_model is None:
            tf = _ensure_tf()
            _brain_model = tf.keras.models.load_model(BRAIN_MODEL_PATH, compile=False)
        return _brain_model


def get_pneu_model():
    global _pneu_model
    with _lock:
        if _pneu_model is None:
            tf = _ensure_tf()
            _pneu_model = tf.keras.models.load_model(PNEU_MODEL_PATH, compile=False)
        return _pneu_model


def _load_image(file_bytes: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(file_bytes)).convert("RGB").resize(IMG_SIZE)
    arr = np.asarray(img, dtype=np.float32)
    arr = _preprocess(arr.copy())
    return np.expand_dims(arr, 0)


def predict_brain(file_bytes: bytes) -> dict:
    model = get_brain_model()
    x = _load_image(file_bytes)
    probs = model.predict(x, verbose=0)[0]
    idx = int(np.argmax(probs))
    return {
        "task": "brain_tumor",
        "label": BRAIN_CLASSES[idx],
        "confidence": float(probs[idx]),
        "all_probs": {c: float(p) for c, p in zip(BRAIN_CLASSES, probs)},
        "is_normal": BRAIN_CLASSES[idx] == "notumor",
    }


def predict_pneumonia(file_bytes: bytes) -> dict:
    model = get_pneu_model()
    x = _load_image(file_bytes)
    out = model.predict(x, verbose=0)[0]
    # binary sigmoid -> shape (1,) prob of PNEUMONIA (class index 1)
    if out.shape == (1,):
        p_pneu = float(out[0])
    else:
        p_pneu = float(out[1])
    label = "PNEUMONIA" if p_pneu >= PNEU_THRESHOLD else "NORMAL"
    return {
        "task": "pneumonia",
        "label": label,
        "confidence": p_pneu if label == "PNEUMONIA" else 1.0 - p_pneu,
        "all_probs": {"NORMAL": 1.0 - p_pneu, "PNEUMONIA": p_pneu},
        "is_normal": label == "NORMAL",
    }


def encode_image_b64(file_bytes: bytes) -> str:
    img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    img.thumbnail((512, 512))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
