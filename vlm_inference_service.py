#!/usr/bin/env python3
"""
Real-Time VLM Inference Service

This module provides a real-time inference service for VLM predictions,
supporting concurrent requests, model management, and performance monitoring.

Features:
- Async inference with FastAPI
- Model caching and hot-swapping
- Request batching and optimization
- Performance monitoring and metrics
- Health checks and graceful shutdown
- Concurrent request handling
"""

import os
import json
import asyncio
import threading
import time
from typing import Dict, List, Any, Optional, Tuple, Union
from pathlib import Path
import logging
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import numpy as np
from PIL import Image
import io
import base64

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PredictionRequest(BaseModel):
    """Request model for VLM predictions"""
    chart_image: str = Field(..., description="Base64 encoded chart image")
    text_descriptions: List[str] = Field(..., description="List of text descriptions for matching")
    technical_features: Optional[Dict[str, Any]] = Field(None, description="Technical indicators for multimodal fusion")
    model_version: Optional[str] = Field("latest", description="Model version to use")
    confidence_threshold: Optional[float] = Field(0.5, description="Minimum confidence threshold")

class PredictionResponse(BaseModel):
    """Response model for VLM predictions"""
    prediction: int
    confidence: float
    probabilities: List[float]
    processing_time: float
    model_version: str
    timestamp: str
    multimodal: bool = False
    clip_result: Optional[Dict[str, Any]] = None
    xgb_result: Optional[Dict[str, Any]] = None

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    uptime: float
    model_loaded: bool
    active_requests: int
    total_predictions: int
    average_latency: float

class MetricsResponse(BaseModel):
    """Metrics response"""
    total_requests: int
    successful_predictions: int
    failed_predictions: int
    average_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    throughput_rps: float
    model_versions: List[str]

class VLMInferenceService:
    """
    Real-time VLM inference service with async processing
    """

    def __init__(self,
                 model_dir: str = "vlm/models",
                 host: str = "0.0.0.0",
                 port: int = 8000,
                 max_workers: int = 4,
                 enable_multimodal: bool = True):
        """
        Initialize the inference service

        Args:
            model_dir: Directory containing trained models
            host: Host address for the service
            port: Port for the service
            max_workers: Maximum number of worker threads
            enable_multimodal: Whether to enable multimodal fusion
        """
        self.model_dir = Path(model_dir)
        self.host = host
        self.port = port
        self.max_workers = max_workers
        self.enable_multimodal = enable_multimodal

        # Service state
        self.start_time = datetime.now()
        self.is_running = False
        self.active_requests = 0
        self.total_predictions = 0
        self.failed_predictions = 0

        # Performance tracking
        self.latencies = []
        self.request_times = []

        # Model management
        self.models = {}  # Cache for loaded models
        self.current_model_version = "latest"

        # Thread pool for CPU-intensive tasks
        self.executor = ThreadPoolExecutor(max_workers=max_workers)

        # Locks for thread safety
        self.request_lock = threading.Lock()

        # Initialize FastAPI app
        self.app = FastAPI(title="VLM Trading Inference Service", version="1.0.0")
        self._setup_routes()

        logger.info("[INIT] VLM Inference Service initialized")

    def _setup_routes(self):
        """Set up FastAPI routes"""

        @self.app.get("/health", response_model=HealthResponse)
        async def health_check():
            """Health check endpoint"""
            with self.request_lock:
                uptime = (datetime.now() - self.start_time).total_seconds()
                model_loaded = len(self.models) > 0
                avg_latency = np.mean(self.latencies) if self.latencies else 0.0

            return HealthResponse(
                status="healthy" if model_loaded else "degraded",
                uptime=uptime,
                model_loaded=model_loaded,
                active_requests=self.active_requests,
                total_predictions=self.total_predictions,
                average_latency=avg_latency
            )

        @self.app.get("/metrics", response_model=MetricsResponse)
        async def get_metrics():
            """Get service metrics"""
            latencies_ms = [lat * 1000 for lat in self.latencies] if self.latencies else []

            return MetricsResponse(
                total_requests=self.total_predictions + self.failed_predictions,
                successful_predictions=self.total_predictions,
                failed_predictions=self.failed_predictions,
                average_latency_ms=np.mean(latencies_ms) if latencies_ms else 0.0,
                p95_latency_ms=np.percentile(latencies_ms, 95) if latencies_ms else 0.0,
                p99_latency_ms=np.percentile(latencies_ms, 99) if latencies_ms else 0.0,
                throughput_rps=self._calculate_throughput(),
                model_versions=list(self.models.keys())
            )

        @self.app.post("/predict", response_model=PredictionResponse)
        async def predict(request: PredictionRequest, background_tasks: BackgroundTasks):
            """Main prediction endpoint"""
            start_time = time.time()

            try:
                with self.request_lock:
                    self.active_requests += 1

                # Process prediction in thread pool
                result = await asyncio.get_event_loop().run_in_executor(
                    self.executor, self._process_prediction, request
                )

                processing_time = time.time() - start_time

                # Update metrics
                with self.request_lock:
                    self.active_requests -= 1
                    self.total_predictions += 1
                    self.latencies.append(processing_time)
                    self.request_times.append(datetime.now())

                # Keep only recent latencies (last 1000)
                if len(self.latencies) > 1000:
                    self.latencies = self.latencies[-1000:]
                    self.request_times = self.request_times[-1000:]

                # Add processing time to result
                result.processing_time = processing_time
                result.timestamp = datetime.now().isoformat()

                return result

            except Exception as e:
                processing_time = time.time() - start_time

                with self.request_lock:
                    self.active_requests -= 1
                    self.failed_predictions += 1
                    self.latencies.append(processing_time)

                logger.error(f"[ERROR] Prediction failed: {e}")
                raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

        @self.app.post("/predict/batch")
        async def predict_batch(requests: List[PredictionRequest]):
            """Batch prediction endpoint"""
            if len(requests) > 10:  # Limit batch size
                raise HTTPException(status_code=400, detail="Batch size limited to 10 requests")

            start_time = time.time()

            try:
                # Process batch in parallel
                tasks = []
                for req in requests:
                    task = asyncio.get_event_loop().run_in_executor(
                        self.executor, self._process_prediction, req
                    )
                    tasks.append(task)

                results = await asyncio.gather(*tasks)

                processing_time = time.time() - start_time

                # Update metrics
                with self.request_lock:
                    self.total_predictions += len(results)
                    for _ in results:
                        self.latencies.append(processing_time / len(results))

                return {
                    "results": results,
                    "batch_size": len(results),
                    "total_processing_time": processing_time,
                    "average_processing_time": processing_time / len(results)
                }

            except Exception as e:
                logger.error(f"[ERROR] Batch prediction failed: {e}")
                raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")

        @self.app.post("/models/load/{version}")
        async def load_model(version: str):
            """Load a specific model version"""
            try:
                await asyncio.get_event_loop().run_in_executor(
                    self.executor, self._load_model_version, version
                )
                return {"status": "success", "message": f"Model {version} loaded"}
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")

        @self.app.get("/models/list")
        async def list_models():
            """List available model versions"""
            model_files = list(self.model_dir.glob("*.pt"))
            versions = [f.stem for f in model_files]
            return {"models": versions, "current": self.current_model_version}

    def _load_model_version(self, version: str):
        """Load a specific model version"""
        model_path = self.model_dir / f"{version}.pt"

        if not model_path.exists():
            raise FileNotFoundError(f"Model {version} not found")

        if version in self.models:
            logger.info(f"[MODEL] Model {version} already loaded")
            return

        try:
            if self.enable_multimodal:
                from multimodal_fusion import MultimodalFusion
                model = MultimodalFusion(
                    clip_model_path=str(model_path),
                    fusion_method="weighted_average"
                )
            else:
                from vlm.clip_calibration import CLIPCalibrator
                model = CLIPCalibrator(str(model_path))

            self.models[version] = model
            self.current_model_version = version

            logger.info(f"[MODEL] Loaded model version: {version}")

        except Exception as e:
            logger.error(f"[ERROR] Failed to load model {version}: {e}")
            raise

    def _process_prediction(self, request: PredictionRequest) -> PredictionResponse:
        """Process a single prediction request"""
        try:
            # Get model
            model = self.models.get(request.model_version) or self.models.get("latest")
            if not model:
                raise RuntimeError("No model loaded")

            # Decode image
            image_data = base64.b64decode(request.chart_image)
            image = Image.open(io.BytesIO(image_data)).convert('RGB')

            # Prepare predictions
            if self.enable_multimodal and hasattr(model, 'predict_multimodal'):
                # Multimodal prediction
                if not request.technical_features:
                    raise ValueError("Technical features required for multimodal prediction")

                result = model.predict_multimodal(
                    image,
                    request.text_descriptions,
                    request.technical_features
                )

                return PredictionResponse(
                    prediction=result['prediction'],
                    confidence=result['confidence'],
                    probabilities=result['probabilities'].tolist(),
                    processing_time=0.0,  # Will be set by caller
                    model_version=request.model_version or "latest",
                    timestamp="",  # Will be set by caller
                    multimodal=True,
                    clip_result=result.get('clip_result'),
                    xgb_result=result.get('xgb_result')
                )

            else:
                # CLIP-only prediction
                result = model.predict_calibrated(
                    image,
                    request.text_descriptions,
                    calibration_method="temperature"
                )

                return PredictionResponse(
                    prediction=result['predictions'][0],
                    confidence=result['confidences'][0],
                    probabilities=result['probabilities'][0].tolist(),
                    processing_time=0.0,  # Will be set by caller
                    model_version=request.model_version or "latest",
                    timestamp="",  # Will be set by caller
                    multimodal=False
                )

        except Exception as e:
            logger.error(f"[ERROR] Prediction processing failed: {e}")
            raise

    def _calculate_throughput(self) -> float:
        """Calculate requests per second"""
        if not self.request_times:
            return 0.0

        # Calculate throughput over last minute
        one_minute_ago = datetime.now() - timedelta(minutes=1)
        recent_requests = [t for t in self.request_times if t > one_minute_ago]

        return len(recent_requests) / 60.0

    def preload_models(self):
        """Preload available models"""
        model_files = list(self.model_dir.glob("*.pt"))

        for model_file in model_files:
            version = model_file.stem
            try:
                self._load_model_version(version)
                logger.info(f"[PRELOAD] Preloaded model: {version}")
            except Exception as e:
                logger.warning(f"[PRELOAD] Failed to preload {version}: {e}")

    def start_service(self):
        """Start the inference service"""
        logger.info(f"[START] Starting VLM Inference Service on {self.host}:{self.port}")

        # Preload models
        self.preload_models()

        self.is_running = True

        # Start server
        uvicorn.run(
            self.app,
            host=self.host,
            port=self.port,
            log_level="info"
        )

    def stop_service(self):
        """Stop the inference service"""
        logger.info("[STOP] Stopping VLM Inference Service")
        self.is_running = False
        self.executor.shutdown(wait=True)

    async def __aenter__(self):
        """Async context manager entry"""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        self.stop_service()

# Utility functions
def create_inference_service(model_dir: str = "vlm/models",
                           enable_multimodal: bool = True) -> VLMInferenceService:
    """
    Create and configure VLM inference service

    Args:
        model_dir: Directory containing models
        enable_multimodal: Whether to enable multimodal fusion

    Returns:
        Configured VLMInferenceService instance
    """
    service = VLMInferenceService(
        model_dir=model_dir,
        enable_multimodal=enable_multimodal
    )

    return service

def run_inference_service(host: str = "0.0.0.0",
                         port: int = 8000,
                         model_dir: str = "vlm/models"):
    """
    Run the VLM inference service

    Args:
        host: Host address
        port: Port number
        model_dir: Model directory
    """
    service = create_inference_service(model_dir)
    service.start_service()

# Monitoring and management utilities
class ServiceMonitor:
    """Monitor for the VLM inference service"""

    def __init__(self, service_url: str = "http://localhost:8000"):
        self.service_url = service_url

    def check_health(self) -> Dict[str, Any]:
        """Check service health"""
        try:
            import requests
            response = requests.get(f"{self.service_url}/health")
            return response.json()
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}

    def get_metrics(self) -> Dict[str, Any]:
        """Get service metrics"""
        try:
            import requests
            response = requests.get(f"{self.service_url}/metrics")
            return response.json()
        except Exception as e:
            return {"error": str(e)}

    def list_models(self) -> Dict[str, Any]:
        """List available models"""
        try:
            import requests
            response = requests.get(f"{self.service_url}/models/list")
            return response.json()
        except Exception as e:
            return {"error": str(e)}

if __name__ == "__main__":
    # Run the service
    run_inference_service()