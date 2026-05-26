import pytest
import asyncio
from unittest.mock import MagicMock
from src.app.messaging.connection import AsynchronousZMQClient, ZMQWorker
from src.app.domain.exceptions import MessagingError


@pytest.mark.asyncio
async def test_messaging_flow_integration():
    """
    Integration test of AsynchronousZMQClient and ZMQWorker
    using a real ephemeral port binding.
    """
    port = 55655
    broker_url = f"tcp://127.0.0.1:{port}"

    # Setup dummy request handler on the worker
    def dummy_handler(payload: dict) -> dict:
        url = payload.get("image_url")
        if url == "https://error.com/error.jpg":
            return {"success": False, "error": "Mocked processing error"}
        return {
            "success": True,
            "cropped_image_path": "/mock/storage/cropped.png",
            "metadata": {"width": 800, "height": 600}
        }

    # Start ZMQ Worker
    worker = ZMQWorker(broker_url=broker_url, request_handler=dummy_handler)
    worker_task = asyncio.create_task(worker.start())

    # Start ZMQ Client
    client = AsynchronousZMQClient(broker_url=broker_url, timeout_ms=2000)
    await client.start()

    try:
        # 1. Success case request
        payload = {"image_url": "https://example.com/slab.jpg"}
        response = await client.send_request(payload)
        
        assert response["success"] is True
        assert response["cropped_image_path"] == "/mock/storage/cropped.png"
        assert response["metadata"]["width"] == 800

        # 2. Worker logic failure case request
        error_payload = {"image_url": "https://error.com/error.jpg"}
        error_response = await client.send_request(error_payload)
        assert error_response["success"] is False
        assert "Mocked processing error" in error_response["error"]

    finally:
        # Cleanup
        await client.stop()
        worker.stop()
        try:
            await worker_task
        except asyncio.CancelledError:
            pass


@pytest.mark.asyncio
async def test_messaging_client_timeout():
    """Tests that client raises MessagingError if request times out."""
    port = 55656
    broker_url = f"tcp://127.0.0.1:{port}"

    # Start ZMQ Client but no worker
    client = AsynchronousZMQClient(broker_url=broker_url, timeout_ms=500)
    await client.start()

    try:
        payload = {"image_url": "https://example.com/slab.jpg"}
        with pytest.raises(MessagingError) as exc_info:
            await client.send_request(payload)
        assert "Request timed out" in str(exc_info.value)
    finally:
        await client.stop()
