"""
FastAPI service for Indian grain classification model.

Endpoints:
    GET  /health   -> basic health check + model info
    GET  /classes  -> list of supported class names
    POST /predict  -> upload an image, get probabilities for all classes + top prediction
"""

import io
import os
import secrets
import logging

import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException, Security, Depends
from fastapi.security import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import tflite_runtime.interpreter as tflite

from class_names import CLASS_NAMES, num_classes

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("grain-api")

MODEL_PATH = os.environ.get("MODEL_PATH", "models/Grain.tflite")
API_KEY = os.environ.get("API_KEY")

app = FastAPI(
    title="Grain Classifier API",
    description="Upload an image of a grain/pulse and get class probabilities.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

api_key_header = APIKeyHeader(name="x-api-key", auto_error=False)


def verify_api_key(provided_key: str = Security(api_key_header)):
    if not API_KEY:
        return
    if not provided_key or not secrets.compare_digest(provided_key, API_KEY):
        raise HTTPException(status_code=401, detail="Invalid or missing API key.")


interpreter = None
input_details = None
output_details = None
input_size = (224, 224)


@app.on_event("startup")
def load_model():
    global interpreter, input_details, output_details, input_size

    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(
            f"Model file not found at '{MODEL_PATH}'. "
            f"Place your trained model at that path or set the MODEL_PATH env var."
        )

    logger.info(f"Loading TFLite model from {MODEL_PATH} ...")
    interpreter = tflite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    shape = input_details[0]['shape']
    if len(shape) == 4 and shape[1] and shape[2]:
        input_size = (int(shape[1]), int(shape[2]))

    out_units = output_details[0]['shape'][-1]
    if out_units != num_classes():
        logger.warning(
            f"Model output units ({out_units}) does not match number of "
            f"CLASS_NAMES ({num_classes()}). Predictions may be mislabeled."
        )

    logger.info(f"Model loaded. Input size: {input_size}, output classes: {out_units}")


def preprocess_image(file_bytes: bytes) -> np.ndarray:
    try:
        img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid image.")

    img = img.resize(input_size)
    # No /255 division, no preprocess_input — EfficientNetB0 (original) has
    # rescaling built into the base model; training fed raw 0-255 pixels.
    arr = np.array(img, dtype=np.float32)
    arr = np.expand_dims(arr, axis=0)
    return arr


@app.get("/")
def root():
    return {"status": "ok", "message": "Grain Classifier API is running. See /health, /classes, /predict, /docs."}


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": interpreter is not None,
        "input_size": input_size,
        "num_classes": num_classes(),
    }


@app.get("/classes")
def classes():
    return {"num_classes": num_classes(), "classes": CLASS_NAMES}


@app.post("/predict")
async def predict(file: UploadFile = File(...), _auth: None = Depends(verify_api_key)):
    if interpreter is None:
        raise HTTPException(status_code=503, detail="Model is not loaded yet.")

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload an image file.")

    file_bytes = await file.read()
    input_tensor = preprocess_image(file_bytes)

    interpreter.set_tensor(input_details[0]['index'], input_tensor)
    interpreter.invoke()
    preds = interpreter.get_tensor(output_details[0]['index'])[0]

    if not np.isclose(preds.sum(), 1.0, atol=1e-2):
        exp = np.exp(preds - np.max(preds))
        preds = exp / exp.sum()

    probabilities = {
        CLASS_NAMES[i]: float(preds[i]) for i in range(min(len(CLASS_NAMES), len(preds)))
    }

    top_idx = int(np.argmax(preds))
    predicted_label = CLASS_NAMES[top_idx] if top_idx < len(CLASS_NAMES) else str(top_idx)
    confidence = float(preds[top_idx])

    return {
        "predicted_label": predicted_label,
        "confidence": confidence,
        "probabilities": probabilities,
    }