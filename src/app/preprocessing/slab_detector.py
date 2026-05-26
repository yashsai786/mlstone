import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional
from src.app.application.ports import SlabDetectorPort
from src.app.domain.models import SlabRegion, RectangleCandidate
from src.app.domain.exceptions import SlabDetectionError
from src.app.domain.slab_detection import SlabDetectionResult
from src.app.domain.value_objects import BoundingBox, DetectionConfidence, CropMetadata
from src.app.preprocessing.morphology import BilateralFilterService, MorphologyGradientService
from src.app.preprocessing.thresholding import ThresholdingService
from src.app.preprocessing.contour_scorer import ContourExtractionService, RectangleScoringService
from src.app.preprocessing.cropper import SafeCropper


class SlabDetector(SlabDetectorPort):
    """
    Production-grade slab detector orchestrator implementing SlabDetectorPort.
    Uses experimentally validated morphology-gradient based slab localization.
    """

    def __init__(
        self,
        bilateral_service: Optional[BilateralFilterService] = None,
        morphology_service: Optional[MorphologyGradientService] = None,
        threshold_service: Optional[ThresholdingService] = None,
        contour_service: Optional[ContourExtractionService] = None,
        scoring_service: Optional[RectangleScoringService] = None,
        cropper: Optional[SafeCropper] = None
    ):
        self.bilateral_service = bilateral_service or BilateralFilterService()
        self.morphology_service = morphology_service or MorphologyGradientService()
        self.threshold_service = threshold_service or ThresholdingService()
        self.contour_service = contour_service or ContourExtractionService()
        self.scoring_service = scoring_service or RectangleScoringService()
        self.cropper = cropper or SafeCropper()
        self.debug_images: Dict[str, np.ndarray] = {}

    def detect(self, image: np.ndarray) -> SlabDetectionResult:
        """
        Runs the morphology-gradient slab localization pipeline and returns a SlabDetectionResult.
        """
        if image is None or image.size == 0:
            raise SlabDetectionError("Input image is empty or invalid.")

        orig_h, orig_w = image.shape[:2]

        # 1. Grayscale conversion
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # 2. Bilateral filtering
        filtered = self.bilateral_service.filter(gray, 9, 75, 75)

        # 3. Morphological gradient
        gradient_map = self.morphology_service.compute_gradient(filtered, (5, 5))

        # 4. OTSU thresholding
        thresh = self.threshold_service.apply_otsu(gradient_map)

        # 5. Large-kernel morphology closing/opening
        closed_opened = self.morphology_service.apply_close_open(thresh, (41, 41))

        # Cache debug outputs for diagnostics
        self.debug_images["original"] = image.copy()
        self.debug_images["gray"] = gray.copy()
        self.debug_images["blurred"] = filtered.copy()
        self.debug_images["edges"] = thresh.copy()
        self.debug_images["morphology"] = closed_opened.copy()

        # Draw contours for debug contours.jpg
        contours = self.contour_service.extract_contours(closed_opened)
        contours_img = image.copy()
        cv2.drawContours(contours_img, contours, -1, (0, 0, 255), 2)
        self.debug_images["contours"] = contours_img

        best_x, best_y, best_w, best_h = 0, 0, orig_w, orig_h
        max_area = 0
        is_slab_locked = False
        best_contour = np.array([])
        best_score = 0.0

        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            area = w * h
            
            score, is_locked = self.scoring_service.score_contour(c, orig_w, orig_h)
            
            if is_locked:
                if area > max_area:
                    max_area = area
                    best_x, best_y, best_w, best_h = x, y, w, h
                    is_slab_locked = True
                    best_contour = c
                    best_score = score

        if not is_slab_locked:
            # Fallback coordinate setup (7% inset of total area)
            best_x = int(orig_w * 0.07)
            best_y = int(orig_h * 0.07)
            best_w = int(orig_w * 0.86)
            best_h = int(orig_h * 0.86)
            best_score = 0.10
            best_contour = np.array([
                [[best_x, best_y]],
                [[best_x + best_w, best_y]],
                [[best_x + best_w, best_y + best_h]],
                [[best_x, best_y + best_h]]
            ], dtype=np.int32)

        # Apply safe rectangular crop
        cropped_img, (cx, cy, cw, ch), is_fallback = self.cropper.crop(
            image,
            (best_x, best_y, best_w, best_h),
            is_slab_locked
        )

        # Safe bounding box & metadata instantiation
        bbox = BoundingBox(x=cx, y=cy, width=cw, height=ch)
        confidence = DetectionConfidence(best_score)
        metadata = CropMetadata(
            original_width=orig_w,
            original_height=orig_h,
            cropped_width=cw,
            cropped_height=ch,
            is_fallback=is_fallback,
            confidence=best_score,
            warning="Fallback crop executed" if (is_fallback or not is_slab_locked) else None
        )

        contour_pts = [tuple(pt[0]) for pt in best_contour] if len(best_contour) > 0 else []

        return SlabDetectionResult(
            bounding_box=bbox,
            confidence=confidence,
            crop_metadata=metadata,
            contour=contour_pts
        )

    # --- Legacy Compatibility & Delegators for tests ---

    def preprocess_image(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        h, w = image.shape[:2]
        max_dim = 1000
        scale = 1.0
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            resized = cv2.resize(image, (int(w * scale), int(h * scale)))
        else:
            resized = image.copy()
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        return resized, gray, scale

    def apply_smoothing(self, gray: np.ndarray, std_val: float) -> np.ndarray:
        if std_val < 15.0:
            return cv2.GaussianBlur(gray, (5, 5), 0)
        return self.bilateral_service.filter(gray)

    def detect_edges(self, image: np.ndarray) -> np.ndarray:
        return self.threshold_service.apply_otsu(
            self.morphology_service.compute_gradient(image)
        )

    def apply_morphology(self, image: np.ndarray) -> np.ndarray:
        return self.morphology_service.apply_close_open(image)

    def score_contour(self, contour: np.ndarray, total_area: float, edges: np.ndarray, w: int, h: int) -> Tuple[float, RectangleCandidate]:
        score, is_locked = self.scoring_service.score_contour(contour, w, h)
        x, y, cw, ch = cv2.boundingRect(contour)
        cand = RectangleCandidate(
            center=(float(x + cw/2), float(y + ch/2)),
            size=(float(cw), float(ch)),
            angle=0.0,
            score=score,
            contour_area=float(cv2.contourArea(contour))
        )
        return score, cand

    # --- SlabDetectorPort Interface Compliance ---

    def detect_slabs(self, image: np.ndarray) -> List[SlabRegion]:
        """
        Detects stone slabs using the morphology-gradient localization algorithm.
        """
        try:
            res = self.detect(image)
        except Exception as e:
            if isinstance(e, SlabDetectionError):
                raise e
            raise SlabDetectionError(str(e))
        
        # Build backward-compatible SlabRegion
        region = SlabRegion(
            bounding_box=BboxWrapper(res.bounding_box),
            contour=res.contour,
            confidence=res.confidence.value,
            is_rotated=False,
            rotation_angle=0.0,
            polygon_points=res.contour,
            contour_area_ratio=res.crop_metadata.cropped_width * res.crop_metadata.cropped_height / (res.crop_metadata.original_width * res.crop_metadata.original_height)
        )
        return [region]

    def crop_slab(self, image: np.ndarray, region: SlabRegion) -> np.ndarray:
        """
        Returns a crop of the slab without perspective transformation or rotation warping.
        """
        if image is None or image.size == 0:
            raise SlabDetectionError("Input image is empty.")
            
        res = self.detect(image)
        cropped_img, _, _ = self.cropper.crop(
            image,
            (res.bounding_box.x, res.bounding_box.y, res.bounding_box.width, res.bounding_box.height),
            is_locked=not res.crop_metadata.is_fallback
        )
        return cropped_img

    def remove_background(self, image: np.ndarray, region: SlabRegion) -> np.ndarray:
        """
        Returns BGR crop converted to RGBA with complete opaque alpha channel (no transparent background).
        """
        crop = self.crop_slab(image, region)
        rgba = cv2.cvtColor(crop, cv2.COLOR_BGR2BGRA)
        rgba[:, :, 3] = 255
        return rgba


class BboxWrapper:
    """Wrapper class ensuring backward-compatibility with old domain.models.BoundingBox."""
    def __init__(self, bbox: BoundingBox):
        self.x = bbox.x
        self.y = bbox.y
        self.width = bbox.width
        self.height = bbox.height
        
    def to_tuple(self) -> Tuple[int, int, int, int]:
        return self.x, self.y, self.width, self.height
