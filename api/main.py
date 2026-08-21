#!/usr/bin/env python3
"""
FastAPI Production API for AI Customer Support Ticket Classifier.
Endpoints: /predict, /predict/batch, /explain, /health
Auto-docs at: http://localhost:8000/docs
"""

import sys
import pickle
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# ═══════════════════════════════════════════════════════════
# Load Model Artifacts (once at startup)
# ═══════════════════════════════════════════════════════════
MODEL_DIR = Path("models/production_v2")

with open(MODEL_DIR / "model.pkl", "rb") as f:
    MODEL = pickle.load(f)
with open(MODEL_DIR / "vectorizer.pkl", "rb") as f:
    VECTORIZER = pickle.load(f)
with open(MODEL_DIR / "encoder.pkl", "rb") as f:
    ENCODER = pickle.load(f)

# Load metrics for /health
with open(MODEL_DIR / "metrics.json", "r") as f:
    METRICS = json.load(f)

CLASSES = list(ENCODER.classes_)


# ═══════════════════════════════════════════════════════════
# Pydantic Models (Request/Response Validation)
# ═══════════════════════════════════════════════════════════
class PredictRequest(BaseModel):
    """Single ticket prediction request."""
    text: str = Field(
        ...,
        min_length=1,
        max_length=5000,
        description="Raw customer support ticket text",
        example="I was charged twice for my subscription this month"
    )


class PredictResponse(BaseModel):
    """Single ticket prediction response."""
    category: str = Field(..., description="Predicted category")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    all_probabilities: Dict[str, float] = Field(
        ..., description="Probability for each class"
    )
    model_version: str = "v1.0"
    timestamp: str


class BatchPredictRequest(BaseModel):
    """Batch prediction request."""
    texts: List[str] = Field(
        ...,
        min_items=1,
        max_items=100,
        description="List of ticket texts",
        example=["I was charged twice", "My login is not working"]
    )


class BatchPredictResponse(BaseModel):
    """Batch prediction response."""
    predictions: List[PredictResponse]
    total: int
    model_version: str = "v1.0"


class ExplainResponse(BaseModel):
    """Explanation response with top contributing words."""
    text: str
    category: str
    confidence: float
    top_words: List[Dict[str, Any]] = Field(
        ..., description="Words that contributed most to prediction"
    )


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model: str
    test_f1_macro: float
    test_accuracy: float
    classes: List[str]
    timestamp: str


# ═══════════════════════════════════════════════════════════
# FastAPI App
# ═══════════════════════════════════════════════════════════
app = FastAPI(
    title="AI Customer Support Ticket Classifier API",
    description="""
    Production-ready API for classifying customer support tickets
    into 4 categories: Account, Billing, Technical Support, Refund.
    
    Built with: SVM + TF-IDF on 13K real banking tickets.
    """,
    version="2.0.0",
    contact={
        "name": "Mohammad Hussein",
        "email": "king.mohamd.09876@gmail.com",
        "url": "https://mohammad-hussein-dev.github.io"
    },
)


def _predict_single(text: str) -> Dict[str, Any]:
    """Core prediction logic."""
    # Vectorize
    X = VECTORIZER.transform([text])
    
    # Predict
    pred_idx = MODEL.predict(X)[0]
    category = ENCODER.inverse_transform([pred_idx])[0]
    
    # Probabilities
    if hasattr(MODEL, "predict_proba"):
        proba = MODEL.predict_proba(X)[0]
        confidence = float(np.max(proba))
        all_proba = {
            cls: float(p) for cls, p in zip(CLASSES, proba)
        }
    else:
        confidence = 1.0
        all_proba = {category: 1.0}
    
    return {
        "category": category,
        "confidence": confidence,
        "all_probabilities": all_proba,
    }


def _explain_prediction(text: str) -> List[Dict[str, Any]]:
    """
    Simple explanation: show TF-IDF weighted words.
    Returns top contributing words.
    """
    X = VECTORIZER.transform([text])
    feature_names = VECTORIZER.get_feature_names_out()
    
    # Get TF-IDF scores for this text
    scores = X.toarray()[0]
    top_idx = np.argsort(scores)[-10:][::-1]
    
    top_words = []
    for idx in top_idx:
        if scores[idx] > 0:
            top_words.append({
                "word": feature_names[idx],
                "tfidf_score": float(scores[idx]),
            })
    
    return top_words


# ═══════════════════════════════════════════════════════════
# Endpoints
# ═══════════════════════════════════════════════════════════
@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    """
    Classify a single customer support ticket.
    
    Returns predicted category, confidence, and all class probabilities.
    """
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    
    result = _predict_single(request.text)
    
    return PredictResponse(
        category=result["category"],
        confidence=result["confidence"],
        all_probabilities=result["all_probabilities"],
        model_version="v1.0",
        timestamp=datetime.utcnow().isoformat()
    )


@app.post("/predict/batch", response_model=BatchPredictResponse)
async def predict_batch(request: BatchPredictRequest):
    """
    Classify multiple tickets in one request (batch processing).
    
    Max 100 texts per request for performance.
    """
    predictions = []
    for text in request.texts:
        if not text.strip():
            raise HTTPException(
                status_code=400,
                detail=f"Empty text at index {len(predictions)}"
            )
        
        result = _predict_single(text)
        predictions.append(PredictResponse(
            category=result["category"],
            confidence=result["confidence"],
            all_probabilities=result["all_probabilities"],
            model_version="v1.0",
            timestamp=datetime.utcnow().isoformat()
        ))
    
    return BatchPredictResponse(
        predictions=predictions,
        total=len(predictions),
        model_version="v1.0"
    )


@app.post("/explain", response_model=ExplainResponse)
async def explain(request: PredictRequest):
    """
    Explain why the model made this prediction.
    
    Returns top TF-IDF weighted words that influenced the decision.
    """
    result = _predict_single(request.text)
    top_words = _explain_prediction(request.text)
    
    return ExplainResponse(
        text=request.text,
        category=result["category"],
        confidence=result["confidence"],
        top_words=top_words
    )


@app.get("/health", response_model=HealthResponse)
async def health():
    """
    Health check endpoint.
    
    Returns model status, performance metrics, and available classes.
    """
    return HealthResponse(
        status="healthy",
        model=METRICS["best_model"],
        test_f1_macro=METRICS["models"][METRICS["best_model"]]["test_f1_macro"],
        test_accuracy=METRICS["models"][METRICS["best_model"]]["test_accuracy"],
        classes=CLASSES,
        timestamp=datetime.utcnow().isoformat()
    )


@app.get("/")
async def root():
    """API root with links."""
    return {
        "message": "AI Customer Support Ticket Classifier API",
        "version": "2.0.0",
        "docs": "/docs",
        "health": "/health",
        "endpoints": {
            "predict": "POST /predict",
            "batch_predict": "POST /predict/batch",
            "explain": "POST /explain",
            "health": "GET /health"
        }
    }
