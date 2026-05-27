from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field
import uuid
from pathlib import Path
from typing import Dict, Any, Optional, List

from src.app.infrastructure.config import get_config
from src.app.infrastructure.logging import setup_logging, get_logger
from src.app.messaging.connection import AsynchronousZMQClient
from src.app.domain.exceptions import (
    MessagingError, DownloadError, InvalidImageError, SlabDetectionError, InferenceError
)
from src.app.preprocessing.pipeline import OpenCVPipeline
from src.app.dataset.services.downloader import DownloaderService
from src.app.ml.inference.service import StoneColorInferenceService
from src.app.ml.application.use_cases import PredictStoneColorUseCase
from src.app.ml.domain.models import PredictionRequest as DomainPredictionRequest

# Setup logging
setup_logging()
logger = get_logger(__name__)
config = get_config()

# Global ZMQ Client instance
zmq_client = AsynchronousZMQClient(
    broker_url=config.zmq_broker_url,
    timeout_ms=config.zmq_response_timeout_ms
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Start ZeroMQ Client
    logger.info("FastAPI service starting up. Initializing ZMQ Client connection...")
    try:
        await zmq_client.start()
        logger.info("ZeroMQ Client connection established.")
    except Exception as e:
        logger.error("Failed to connect ZMQ Client during startup", extra={"error": str(e)})
        # We don't crash the server immediately, but log it so developers can troubleshoot ZMQ broker issues.
    
    yield
    
    # Shutdown: Stop ZeroMQ Client
    logger.info("FastAPI service shutting down. Closing ZMQ Client connection...")
    await zmq_client.stop()
    logger.info("ZeroMQ Client connection closed.")


app = FastAPI(
    title="Stone Slab Preprocessing API",
    description="DDD & Event-Driven Microservice for Stone Slab Extraction",
    version="1.0.0",
    lifespan=lifespan
)


class SlabExtractionRequest(BaseModel):
    image_url: str
    request_id: Optional[str] = None


class SlabExtractionResponse(BaseModel):
    success: bool
    cropped_image_path: str
    request_id: str
    metadata: Dict[str, Any]


class ColorPredictionRequest(BaseModel):
    image_url: str


class TopPredictionCandidate(BaseModel):
    class_: str = Field(..., alias="class")
    confidence: float

    class Config:
        allow_population_by_field_name = True
        populate_by_name = True


class ColorPredictionResponse(BaseModel):
    predicted_color: str
    confidence: float
    top_predictions: List[TopPredictionCandidate]
    processing_time_ms: int
    model_version: Optional[str] = None


@app.get("/health", status_code=status.HTTP_200_OK)
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "zmq_connected": zmq_client.socket is not None}


@app.post(
    "/extract-slab",
    response_model=SlabExtractionResponse,
    status_code=status.HTTP_200_OK,
    summary="Extract and Clean Stone Slab",
    description="Accepts an image URL, processes it asynchronously via internal ZeroMQ workers to detect/crop the slab, and returns the path to the cropped image."
)
async def extract_slab(request: SlabExtractionRequest):
    request_id = request.request_id or str(uuid.uuid4())
    logger.info("Received slab extraction REST request", extra={"url": request.image_url, "request_id": request_id})
    
    # Send request over ZeroMQ
    payload = {
        "image_url": request.image_url,
        "request_id": request_id
    }
    
    try:
        response = await zmq_client.send_request(payload)
        
        if not response.get("success", False):
            # Error returned from the worker
            error_msg = response.get("error", "Unknown processing error occurred in worker.")
            logger.error("Worker processing failed", extra={"url": request.image_url, "request_id": request_id, "error": error_msg})
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=error_msg
            )
            
        return SlabExtractionResponse(
            success=True,
            cropped_image_path=response["cropped_image_path"],
            request_id=response.get("request_id", request_id),
            metadata=response["metadata"]
        )

    except MessagingError as e:
        logger.error("ZeroMQ Messaging error", extra={"url": request.image_url, "error": str(e)})
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Internal messaging service unavailable: {e}"
        )
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error("Unexpected error in API handler", extra={"url": request.image_url, "error": str(e)})
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected internal error occurred: {e}"
        )


predictor_use_case: Optional[PredictStoneColorUseCase] = None


def get_predictor_use_case() -> PredictStoneColorUseCase:
    """Lazily loads and returns the PredictStoneColorUseCase instance."""
    global predictor_use_case
    if predictor_use_case is None:
        model_path = str(config.ml_model_dir / "stone_color_model.pt")
        if not Path(model_path).exists():
            logger.warning(
                "ML model file not found during service initialization!",
                extra={"path": model_path}
            )

        downloader = DownloaderService(
            timeout_seconds=config.download_timeout_seconds,
            retry_count=config.download_retry_count
        )
        slab_detector = OpenCVPipeline()
        inference_service = StoneColorInferenceService(
            model_path=model_path,
            device="cpu"
        )
        predictor_use_case = PredictStoneColorUseCase(
            downloader=downloader,
            slab_detector=slab_detector,
            inference_service=inference_service,
            model_version="1.0.0"
        )
    return predictor_use_case


@app.post(
    "/predict-color",
    response_model=ColorPredictionResponse,
    status_code=status.HTTP_200_OK,
    summary="Predict Stone Color",
    description="Downloads a stone image, localizes the slab contour, crops the region, and runs inference for color classification."
)
async def predict_color(request: ColorPredictionRequest):
    logger.info("Received predict-color request", extra={"url": request.image_url})
    
    use_case = get_predictor_use_case()
    domain_request = DomainPredictionRequest(image_url=request.image_url)

    try:
        result = await use_case.execute(domain_request)
        
        return ColorPredictionResponse(
            predicted_color=result.predicted_color,
            confidence=result.confidence,
            top_predictions=[
                TopPredictionCandidate(**{"class": c.class_name, "confidence": c.confidence})
                for c in result.top_predictions
            ],
            processing_time_ms=int(round(result.processing_time_ms)),
            model_version=result.model_version
        )
        
    except DownloadError as e:
        logger.error("Download failure in predict color endpoint", extra={"url": request.image_url, "error": str(e)})
        # Differentiate gateway timeouts from invalid hosts
        if "timeout" in str(e).lower() or "timed out" in str(e).lower():
            raise HTTPException(
                status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                detail=f"Image download timed out: {e}"
            )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to download image from the provided URL: {e}"
        )
        
    except InvalidImageError as e:
        logger.error("Invalid or corrupt image in predict color endpoint", extra={"url": request.image_url, "error": str(e)})
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported or corrupt image file: {e}"
        )
        
    except SlabDetectionError as e:
        logger.error("Slab detection failure in predict color endpoint", extra={"url": request.image_url, "error": str(e)})
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Slab preprocessing/cropping failed: {e}"
        )
        
    except InferenceError as e:
        logger.error("Inference failure in predict color endpoint", extra={"url": request.image_url, "error": str(e)})
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Model inference failed: {e}"
        )
        
    except Exception as e:
        logger.error("Unexpected failure in predict color endpoint", extra={"url": request.image_url, "error": str(e)})
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected prediction error occurred: {e}"
        )

