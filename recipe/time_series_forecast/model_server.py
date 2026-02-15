#!/usr/bin/env python3
"""
Unified Time Series Prediction Service

A FastAPI service that provides time series prediction using multiple models:
- Chronos2: Foundation model for time series
- PatchTST: Patch-based Transformer
- iTransformer: Inverted Transformer

This service can run on a dedicated GPU, separate from the training framework.

Usage:
    # Start the server on GPU 3 (loads all available models)
    CUDA_VISIBLE_DEVICES=3 python model_server.py --port 8994
    
    # Or with uvicorn directly
    CUDA_VISIBLE_DEVICES=3 uvicorn model_server:app --host 0.0.0.0 --port 8994
"""

import argparse
import os
import json
from typing import List, Optional, Dict, Any
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

from recipe.time_series_forecast.config_utils import get_default_lengths


# =============================================================================
# Configuration
# =============================================================================

# Base path for models
MODELS_BASE_PATH = Path(__file__).parent / "models"

# Global model caches
_models: Dict[str, Any] = {
    "chronos2": None,
    "patchtst": None,
    "itransformer": None,
}

_configs: Dict[str, dict] = {}

# Model directories
_MODEL_DIRS = {
    "chronos2": MODELS_BASE_PATH / "chronos-2",
    "patchtst": MODELS_BASE_PATH / "patchtst",
    "itransformer": MODELS_BASE_PATH / "itransformer",
}

# Default lengths resolved from env/base.yaml
DEFAULT_LOOKBACK_WINDOW, DEFAULT_FORECAST_HORIZON = get_default_lengths()


# =============================================================================
# Request/Response Models
# =============================================================================

class PredictRequest(BaseModel):
    """Request model for prediction endpoint"""
    timestamps: List[str]  # List of timestamp strings
    values: List[float]    # List of time series values
    series_id: str = "series_0"
    prediction_length: int = DEFAULT_FORECAST_HORIZON
    model_name: str = "chronos2"  # Model to use


class PredictResponse(BaseModel):
    """Response model for prediction endpoint"""
    timestamps: List[str]   # Predicted timestamps
    values: List[float]     # Predicted values
    model_used: str
    status: str = "success"


class HealthResponse(BaseModel):
    """Response model for health check"""
    status: str
    models_loaded: Dict[str, bool]
    device: str


class ModelsInfoResponse(BaseModel):
    """Response model for models info"""
    available_models: List[str]
    models_status: Dict[str, Dict[str, Any]]


# =============================================================================
# Model Loading Functions
# =============================================================================

def load_config(model_name: str) -> dict:
    """Load model configuration from config.json."""
    config_path = _MODEL_DIRS.get(model_name, MODELS_BASE_PATH / model_name) / "config.json"
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = json.load(f)
    else:
        config = {}

    # Override lengths to align with runtime configuration
    config["seq_len"] = DEFAULT_LOOKBACK_WINDOW
    config["pred_len"] = DEFAULT_FORECAST_HORIZON

    return config


def load_chronos2(device: str = "cuda"):
    """Load the Chronos2 model using BaseChronosPipeline"""
    global _models
    
    if _models["chronos2"] is None:
        model_dir = _MODEL_DIRS["chronos2"]
        if not model_dir.exists():
            print(f"[WARNING] Chronos2 model directory not found: {model_dir}")
            return None
        
        try:
            from chronos import BaseChronosPipeline
            print(f"Loading Chronos2 model from {model_dir} on device {device}...")
            _models["chronos2"] = BaseChronosPipeline.from_pretrained(str(model_dir), device_map=device)
            print("Chronos2 model loaded successfully!")
        except ImportError:
            print("[WARNING] chronos package is not installed. Skipping Chronos2 model.")
            return None
        except Exception as e:
            print(f"[WARNING] Failed to load Chronos2 model: {e}")
            return None
    
    return _models["chronos2"]


def load_patchtst(device: str = "cuda"):
    """Load the PatchTST model"""
    global _models, _configs
    
    if _models["patchtst"] is None:
        model_dir = _MODEL_DIRS["patchtst"]
        checkpoint_path = model_dir / "checkpoint.pth"
        
        if not checkpoint_path.exists():
            print(f"[WARNING] PatchTST checkpoint not found: {checkpoint_path}")
            return None
        
        try:
            # Import model definition
            from recipe.time_series_forecast.models.patchtst.model import create_patchtst_model
            
            # Load config
            config = load_config("patchtst")
            _configs["patchtst"] = config
            
            # Create model
            model = create_patchtst_model(config)
            
            # Load checkpoint
            checkpoint = torch.load(checkpoint_path, map_location=device)
            if isinstance(checkpoint, dict):
                if 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                elif 'state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['state_dict'])
                else:
                    model.load_state_dict(checkpoint)
            else:
                model.load_state_dict(checkpoint)
            
            model.to(device)
            model.eval()
            _models["patchtst"] = model
            print(f"PatchTST model loaded successfully from {checkpoint_path}!")
            
        except Exception as e:
            print(f"[WARNING] Failed to load PatchTST model: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    return _models["patchtst"]


def load_itransformer(device: str = "cuda"):
    """Load the iTransformer model"""
    global _models, _configs
    
    if _models["itransformer"] is None:
        model_dir = _MODEL_DIRS["itransformer"]
        checkpoint_path = model_dir / "checkpoint.pth"
        
        if not checkpoint_path.exists():
            print(f"[WARNING] iTransformer checkpoint not found: {checkpoint_path}")
            return None
        
        try:
            # Import model definition
            from recipe.time_series_forecast.models.itransformer.model import create_itransformer_model
            
            # Load config
            config = load_config("itransformer")
            _configs["itransformer"] = config
            
            # Create model
            model = create_itransformer_model(config)
            
            # Load checkpoint
            checkpoint = torch.load(checkpoint_path, map_location=device)
            if isinstance(checkpoint, dict):
                if 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                elif 'state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['state_dict'])
                else:
                    model.load_state_dict(checkpoint)
            else:
                model.load_state_dict(checkpoint)
            
            model.to(device)
            model.eval()
            _models["itransformer"] = model
            print(f"iTransformer model loaded successfully from {checkpoint_path}!")
            
        except Exception as e:
            print(f"[WARNING] Failed to load iTransformer model: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    return _models["itransformer"]


def load_all_models(device: str = "cuda"):
    """Load all available models"""
    print("=" * 60)
    print("Loading all available models...")
    print("=" * 60)
    
    load_chronos2(device)
    load_patchtst(device)
    load_itransformer(device)
    
    loaded = [name for name, model in _models.items() if model is not None]
    print("=" * 60)
    print(f"Models loaded: {loaded}")
    print("=" * 60)


# =============================================================================
# Prediction Functions
# =============================================================================

def predict_with_chronos2(request: PredictRequest) -> PredictResponse:
    """
    Generate predictions using Chronos2 with Non-stationary Transformer normalization.
    Uses BaseChronosPipeline.predict_quantiles (same as Time-Series-Library).
    """
    pipeline = _models["chronos2"]
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Chronos2 model not loaded")
    
    datetime_list = [pd.to_datetime(ts) for ts in request.timestamps]
    values = np.array(request.values, dtype=np.float32)
    
    # Convert to tensor: [batch=1, seq_len, n_vars=1]
    x_enc = torch.FloatTensor(values).unsqueeze(0).unsqueeze(-1)
    
    # Non-stationary Transformer Normalization
    means = x_enc.mean(1, keepdim=True).detach()
    x_enc = x_enc - means
    stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
    x_enc = x_enc / stdev
    
    # Reshape for Chronos: [batch, n_vars, seq_len]
    x_enc = x_enc.permute(0, 2, 1)
    
    # Predict using predict_quantiles
    quantiles, _ = pipeline.predict_quantiles(
        x_enc.cpu().numpy(),
        prediction_length=request.prediction_length,
        quantile_levels=[0.1, 0.5, 0.9]
    )
    
    # quantiles[0] shape: [batch, pred_len, num_quantiles]
    # Take median (index 1 in last dimension)
    dec_out = quantiles[0][:, :, 1]  # [batch, pred_len]
    dec_out = dec_out.unsqueeze(-1)  # [batch, pred_len, 1]
    
    # De-Normalization
    dec_out = dec_out * stdev[:, 0, :].unsqueeze(1).repeat(1, request.prediction_length, 1)
    dec_out = dec_out + means[:, 0, :].unsqueeze(1).repeat(1, request.prediction_length, 1)
    
    pred_values = dec_out.numpy().squeeze().tolist()
    
    # Generate timestamps
    freq = datetime_list[-1] - datetime_list[-2] if len(datetime_list) >= 2 else pd.Timedelta(hours=1)
    last_ts = datetime_list[-1]
    pred_timestamps = [(last_ts + freq * (i + 1)).strftime('%Y-%m-%d %H:%M:%S') 
                       for i in range(request.prediction_length)]
    
    return PredictResponse(
        timestamps=pred_timestamps,
        values=pred_values,
        model_used="chronos2",
        status="success"
    )


def predict_with_pytorch_model(request: PredictRequest, model_name: str) -> PredictResponse:
    """
    Generate predictions using PatchTST or iTransformer.
    
    Note: Models have Non-stationary Transformer normalization built-in (same as Time-Series-Library).
    Input raw data, output is in original scale.
    """
    model = _models[model_name]
    if model is None:
        raise HTTPException(status_code=503, detail=f"{model_name} model not loaded")
    
    device = next(model.parameters()).device
    
    # Prepare data
    values = np.array(request.values, dtype=np.float32)
    datetime_list = [pd.to_datetime(ts) for ts in request.timestamps]
    freq = datetime_list[-1] - datetime_list[-2] if len(datetime_list) >= 2 else pd.Timedelta(hours=1)
    
    # Prepare input tensor: [batch, seq_len, n_vars]
    input_tensor = torch.FloatTensor(values).unsqueeze(0).unsqueeze(-1).to(device)
    
    # Predict (model handles normalization/denormalization internally)
    with torch.no_grad():
        predictions = model(input_tensor)
    
    # Convert to numpy
    pred_values = predictions.cpu().numpy().squeeze()
    
    # Ensure correct length
    if len(pred_values) > request.prediction_length:
        pred_values = pred_values[:request.prediction_length]
    elif len(pred_values) < request.prediction_length:
        pad_len = request.prediction_length - len(pred_values)
        pred_values = np.concatenate([pred_values, [pred_values[-1]] * pad_len])
    
    # Generate timestamps
    last_ts = datetime_list[-1]
    pred_timestamps = [(last_ts + freq * (i + 1)).strftime('%Y-%m-%d %H:%M:%S') 
                       for i in range(request.prediction_length)]
    
    return PredictResponse(
        timestamps=pred_timestamps,
        values=pred_values.tolist(),
        model_used=model_name,
        status="success"
    )


# =============================================================================
# FastAPI App
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for model loading"""
    device = os.environ.get("MODEL_DEVICE", "cuda")
    load_all_models(device)
    yield


app = FastAPI(
    title="Time Series Prediction Service",
    description="Unified time series prediction service supporting Chronos2, PatchTST, and iTransformer",
    version="2.0.0",
    lifespan=lifespan
)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        device = f"cuda:{torch.cuda.current_device()}"
    
    return HealthResponse(
        status="healthy",
        models_loaded={name: (model is not None) for name, model in _models.items()},
        device=device
    )


@app.get("/models", response_model=ModelsInfoResponse)
async def models_info():
    """Get information about available models"""
    models_status = {}
    for name, model in _models.items():
        models_status[name] = {
            "loaded": model is not None,
            "config": _configs.get(name, {}),
            "checkpoint_exists": (_MODEL_DIRS.get(name, MODELS_BASE_PATH / name) / "checkpoint.pth").exists() 
                                 if name != "chronos2" else (_MODEL_DIRS["chronos2"] / "model.safetensors").exists()
        }
    
    return ModelsInfoResponse(
        available_models=["chronos2", "patchtst", "itransformer"],
        models_status=models_status
    )


@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    """
    Generate time series predictions using the specified model.
    
    Args:
        request: PredictRequest containing timestamps, values, model_name, and prediction parameters
        
    Returns:
        PredictResponse with predicted timestamps and values
    """
    # Validate input
    if len(request.timestamps) != len(request.values):
        raise HTTPException(
            status_code=400, 
            detail="timestamps and values must have the same length"
        )
    
    if len(request.values) < 2:
        raise HTTPException(
            status_code=400,
            detail="At least 2 data points are required"
        )
    
    model_name = request.model_name.lower().strip()
    
    try:
        if model_name == "chronos2":
            return predict_with_chronos2(request)
        elif model_name in ["patchtst", "itransformer"]:
            return predict_with_pytorch_model(request, model_name)
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown model: {model_name}. Available: chronos2, patchtst, itransformer"
            )
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        error_msg = f"{type(e).__name__}: {str(e)}"
        print(f"[ERROR] Prediction failed: {error_msg}")
        print(f"[ERROR] Traceback:\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=error_msg)


def main():
    parser = argparse.ArgumentParser(description="Time Series Prediction Service")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind")
    parser.add_argument("--port", type=int, default=8994, help="Port to bind")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use (cuda, cpu, cuda:0, etc.)")
    args = parser.parse_args()
    
    # Set device via environment variable
    os.environ["MODEL_DEVICE"] = args.device
    
    print(f"Starting Time Series Prediction Service on {args.host}:{args.port} with device {args.device}")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
