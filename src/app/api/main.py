from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel
import uuid
from typing import Dict, Any, Optional
from src.app.infrastructure.config import get_config
from src.app.infrastructure.logging import setup_logging, get_logger
from src.app.messaging.connection import AsynchronousZMQClient
from src.app.domain.exceptions import MessagingError

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
