import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch
from src.app.api.main import app, get_predictor_use_case
from src.app.ml.domain.models import PredictionResult, PredictionCandidate
from src.app.domain.exceptions import DownloadError, InvalidImageError, SlabDetectionError, InferenceError


@pytest.fixture
def api_client():
    return TestClient(app)


def test_predict_color_api_success(api_client):
    # Setup mock output
    mock_candidates = [
        PredictionCandidate(class_name="beige", confidence=0.98),
        PredictionCandidate(class_name="grey", confidence=0.01),
        PredictionCandidate(class_name="gold", confidence=0.01)
    ]
    mock_result = PredictionResult(
        predicted_color="beige",
        confidence=0.98,
        top_predictions=mock_candidates,
        processing_time_ms=143.0,
        model_version="1.0.0"
    )
    
    mock_use_case = MagicMock()
    # Since execute is an async method, we make it return an awaitable coroutine
    async def mock_execute(*args, **kwargs):
        return mock_result
    mock_use_case.execute = mock_execute

    # Patch dependency injector
    with patch("src.app.api.main.get_predictor_use_case", return_value=mock_use_case):
        response = api_client.post(
            "/predict-color",
            json={"image_url": "https://example.com/slabs/stone1.jpg"}
        )
        
        assert response.status_code == 200
        payload = response.json()
        
        assert payload["predicted_color"] == "beige"
        assert payload["confidence"] == 0.98
        assert payload["processing_time_ms"] == 143
        assert payload["model_version"] == "1.0.0"
        
        # Verify alias serialized "class" successfully
        top_list = payload["top_predictions"]
        assert len(top_list) == 3
        assert top_list[0]["class"] == "beige"
        assert top_list[0]["confidence"] == 0.98
        assert "class" in top_list[0]


def test_predict_color_api_invalid_url(api_client):
    mock_use_case = MagicMock()
    async def mock_execute(*args, **kwargs):
        raise DownloadError("Invalid image URL format provided.")
    mock_use_case.execute = mock_execute

    with patch("src.app.api.main.get_predictor_use_case", return_value=mock_use_case):
        response = api_client.post(
            "/predict-color",
            json={"image_url": "invalid_url_no_http"}
        )
        
        assert response.status_code == 400
        payload = response.json()
        assert "Failed to download image" in payload["detail"]


def test_predict_color_api_download_timeout(api_client):
    mock_use_case = MagicMock()
    async def mock_execute(*args, **kwargs):
        raise DownloadError("Connection timed out waiting for server.")
    mock_use_case.execute = mock_execute

    with patch("src.app.api.main.get_predictor_use_case", return_value=mock_use_case):
        response = api_client.post(
            "/predict-color",
            json={"image_url": "https://slow-server.com/image.jpg"}
        )
        
        assert response.status_code == 504
        payload = response.json()
        assert "timed out" in payload["detail"]


def test_predict_color_api_corrupt_image(api_client):
    mock_use_case = MagicMock()
    async def mock_execute(*args, **kwargs):
        raise InvalidImageError("Downloaded image could not be decoded.")
    mock_use_case.execute = mock_execute

    with patch("src.app.api.main.get_predictor_use_case", return_value=mock_use_case):
        response = api_client.post(
            "/predict-color",
            json={"image_url": "https://example.com/corrupt.png"}
        )
        
        assert response.status_code == 415
        payload = response.json()
        assert "Unsupported or corrupt image" in payload["detail"]


def test_predict_color_api_slab_detection_failure(api_client):
    mock_use_case = MagicMock()
    async def mock_execute(*args, **kwargs):
        raise SlabDetectionError("No stone slabs detected in the image.")
    mock_use_case.execute = mock_execute

    with patch("src.app.api.main.get_predictor_use_case", return_value=mock_use_case):
        response = api_client.post(
            "/predict-color",
            json={"image_url": "https://example.com/no_slabs.jpg"}
        )
        
        assert response.status_code == 422
        payload = response.json()
        assert "preprocessing/cropping failed" in payload["detail"]


def test_predict_color_api_inference_failure(api_client):
    mock_use_case = MagicMock()
    async def mock_execute(*args, **kwargs):
        raise InferenceError("Model color classification inference execution failed.")
    mock_use_case.execute = mock_execute

    with patch("src.app.api.main.get_predictor_use_case", return_value=mock_use_case):
        response = api_client.post(
            "/predict-color",
            json={"image_url": "https://example.com/infer_fail.jpg"}
        )
        
        assert response.status_code == 502
        payload = response.json()
        assert "Model inference failed" in payload["detail"]
