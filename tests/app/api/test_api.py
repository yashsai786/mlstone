import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, patch

from src.app.api.main import app, zmq_client
from src.app.domain.exceptions import MessagingError

client = TestClient(app)


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


@patch("src.app.api.main.zmq_client.send_request", new_callable=AsyncMock)
def test_extract_slab_endpoint_success(mock_send_request):
    # Mock ZMQ client response
    mock_send_request.return_value = {
        "success": True,
        "cropped_image_path": "/storage/cropped/abc.png",
        "metadata": {
            "width": 1024,
            "height": 768,
            "confidence": 0.95
        }
    }

    payload = {"image_url": "https://example.com/slabs/slab1.jpg"}
    response = client.post("/extract-slab", json=payload)

    assert response.status_code == 200
    res_data = response.json()
    assert res_data["success"] is True
    assert res_data["cropped_image_path"] == "/storage/cropped/abc.png"
    assert res_data["metadata"]["width"] == 1024

    mock_send_request.assert_called_once()
    call_args = mock_send_request.call_args[0][0]
    assert call_args["image_url"] == "https://example.com/slabs/slab1.jpg"
    assert "request_id" in call_args


@patch("src.app.api.main.zmq_client.send_request", new_callable=AsyncMock)
def test_extract_slab_endpoint_worker_failure(mock_send_request):
    mock_send_request.return_value = {
        "success": False,
        "error": "No stone slabs detected in the image."
    }

    payload = {"image_url": "https://example.com/empty.jpg"}
    response = client.post("/extract-slab", json=payload)

    assert response.status_code == 422
    assert "No stone slabs detected" in response.json()["detail"]


@patch("src.app.api.main.zmq_client.send_request", new_callable=AsyncMock)
def test_extract_slab_endpoint_messaging_error(mock_send_request):
    # Simulate ZeroMQ communication timeout / crash
    mock_send_request.side_effect = MessagingError("ZMQ request timed out.")

    payload = {"image_url": "https://example.com/slab.jpg"}
    response = client.post("/extract-slab", json=payload)

    assert response.status_code == 503
    assert "messaging service unavailable" in response.json()["detail"]


@patch("src.app.api.main.zmq_client.send_request", new_callable=AsyncMock)
def test_extract_slab_endpoint_internal_server_error(mock_send_request):
    # Simulate some unhandled exception
    mock_send_request.side_effect = RuntimeError("Fatal crash")

    payload = {"image_url": "https://example.com/slab.jpg"}
    response = client.post("/extract-slab", json=payload)

    assert response.status_code == 500
    assert "unexpected internal error" in response.json()["detail"]
