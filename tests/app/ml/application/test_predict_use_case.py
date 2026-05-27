import pytest
from unittest.mock import AsyncMock, MagicMock
import numpy as np
import cv2

from src.app.ml.application.use_cases import PredictStoneColorUseCase
from src.app.ml.domain.models import PredictionRequest, StoneClassification
from src.app.domain.models import SlabRegion, BoundingBox
from src.app.domain.exceptions import DownloadError, InvalidImageError, SlabDetectionError, InferenceError


@pytest.fixture
def mock_downloader():
    downloader = MagicMock()
    downloader.download = AsyncMock(return_value=b"fake_image_bytes")
    return downloader


@pytest.fixture
def mock_slab_detector():
    detector = MagicMock()
    
    # Setup mock region
    region = SlabRegion(
        bounding_box=BoundingBox(0, 0, 100, 100),
        contour=[(0, 0), (100, 0), (100, 100), (0, 100)],
        confidence=0.95
    )
    detector.detect_slabs = MagicMock(return_value=[region])
    detector.crop_slab = MagicMock(return_value=np.zeros((100, 100, 3), dtype=np.uint8))
    return detector


@pytest.fixture
def mock_inference_service():
    service = MagicMock()
    classification = StoneClassification(
        predicted_class="beige",
        confidence=0.98,
        top_k={"beige": 0.98, "grey": 0.01, "gold": 0.01},
        inference_time_ms=10.0,
        device_used="cpu"
    )
    service.predict = MagicMock(return_value=classification)
    return service


@pytest.mark.asyncio
async def test_predict_use_case_success(mock_downloader, mock_slab_detector, mock_inference_service):
    # Patch cv2.imdecode to return a valid numpy array
    dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
    
    use_case = PredictStoneColorUseCase(
        downloader=mock_downloader,
        slab_detector=mock_slab_detector,
        inference_service=mock_inference_service,
        model_version="1.0.0"
    )
    
    request = PredictionRequest(image_url="https://example.com/stone.jpg")

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(cv2, "imdecode", lambda *args: dummy_img)
        
        result = await use_case.execute(request)
        
        assert result.predicted_color == "beige"
        assert result.confidence == 0.98
        assert len(result.top_predictions) == 3
        assert result.top_predictions[0].class_name == "beige"
        assert result.top_predictions[0].confidence == 0.98
        assert result.model_version == "1.0.0"
        
        # Verify call dependencies
        mock_downloader.download.assert_called_once_with("https://example.com/stone.jpg")
        mock_slab_detector.detect_slabs.assert_called_once()
        mock_slab_detector.crop_slab.assert_called_once()
        mock_inference_service.predict.assert_called_once()


@pytest.mark.asyncio
async def test_predict_use_case_invalid_url(mock_downloader, mock_slab_detector, mock_inference_service):
    use_case = PredictStoneColorUseCase(
        downloader=mock_downloader,
        slab_detector=mock_slab_detector,
        inference_service=mock_inference_service
    )
    
    # Invalid protocol
    request = PredictionRequest(image_url="ftp://example.com/stone.jpg")
    
    with pytest.raises(DownloadError) as exc_info:
        await use_case.execute(request)
    assert "Invalid image URL format" in str(exc_info.value)


@pytest.mark.asyncio
async def test_predict_use_case_download_timeout(mock_downloader, mock_slab_detector, mock_inference_service):
    mock_downloader.download.side_effect = DownloadError("Request timed out after 10s")
    
    use_case = PredictStoneColorUseCase(
        downloader=mock_downloader,
        slab_detector=mock_slab_detector,
        inference_service=mock_inference_service
    )
    
    request = PredictionRequest(image_url="https://example.com/stone.jpg")
    
    with pytest.raises(DownloadError) as exc_info:
        await use_case.execute(request)
    assert "Request timed out" in str(exc_info.value)


@pytest.mark.asyncio
async def test_predict_use_case_decode_failure(mock_downloader, mock_slab_detector, mock_inference_service):
    use_case = PredictStoneColorUseCase(
        downloader=mock_downloader,
        slab_detector=mock_slab_detector,
        inference_service=mock_inference_service
    )
    
    request = PredictionRequest(image_url="https://example.com/stone.jpg")
    
    # Force imdecode to fail (return None)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(cv2, "imdecode", lambda *args: None)
        
        with pytest.raises(InvalidImageError) as exc_info:
            await use_case.execute(request)
        assert "could not be decoded" in str(exc_info.value)


@pytest.mark.asyncio
async def test_predict_use_case_no_slabs_detected(mock_downloader, mock_slab_detector, mock_inference_service):
    mock_slab_detector.detect_slabs = MagicMock(return_value=[])
    dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
    
    use_case = PredictStoneColorUseCase(
        downloader=mock_downloader,
        slab_detector=mock_slab_detector,
        inference_service=mock_inference_service
    )
    
    request = PredictionRequest(image_url="https://example.com/stone.jpg")
    
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(cv2, "imdecode", lambda *args: dummy_img)
        
        with pytest.raises(SlabDetectionError) as exc_info:
            await use_case.execute(request)
        assert "No stone slabs detected" in str(exc_info.value)


@pytest.mark.asyncio
async def test_predict_use_case_crop_failure(mock_downloader, mock_slab_detector, mock_inference_service):
    mock_slab_detector.crop_slab = MagicMock(return_value=None)
    dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
    
    use_case = PredictStoneColorUseCase(
        downloader=mock_downloader,
        slab_detector=mock_slab_detector,
        inference_service=mock_inference_service
    )
    
    request = PredictionRequest(image_url="https://example.com/stone.jpg")
    
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(cv2, "imdecode", lambda *args: dummy_img)
        
        with pytest.raises(SlabDetectionError) as exc_info:
            await use_case.execute(request)
        assert "Failed to crop stone slab" in str(exc_info.value)


@pytest.mark.asyncio
async def test_predict_use_case_inference_failure(mock_downloader, mock_slab_detector, mock_inference_service):
    mock_inference_service.predict.side_effect = Exception("Runtime CUDA out of memory")
    dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
    
    use_case = PredictStoneColorUseCase(
        downloader=mock_downloader,
        slab_detector=mock_slab_detector,
        inference_service=mock_inference_service
    )
    
    request = PredictionRequest(image_url="https://example.com/stone.jpg")
    
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(cv2, "imdecode", lambda *args: dummy_img)
        
        with pytest.raises(InferenceError) as exc_info:
            await use_case.execute(request)
        assert "color classification inference execution failed" in str(exc_info.value)
