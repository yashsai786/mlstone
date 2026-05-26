# 💎 Stone Slab Preprocessing & Localization API

An ultra-high-performance, production-grade ML microservice built using **Domain-Driven Design (DDD)**, **Clean Architecture**, and an **Event-Driven Architecture** communicating via **ZeroMQ (pyzmq)**. 

This service implements an experimentally validated **morphological-gradient based slab localization pipeline** in OpenCV to precisely isolate the primary stone slab rectangular region from raw warehouse photos—handling low contrast, perspective tilts, shadows, straps, and holders flawlessly without aggressive pixel-level masking or warping.

---

## 🏗️ Architectural Philosophy

The project adheres strictly to **Clean Architecture** principles and **Dependency Inversion**:

```
      ▲  [Infrastructure]  (FastAPI, ZMQ Worker, Decoders, Storage)
      │        │
      │        ▼
      │  [Application]     (Use Cases, Ports, Orchestrators)
      │        │
      │        ▼
      │  [Preprocessing]   (Morphology Services, Scoring, Safe Cropper)
      │        │
      │        ▼
      └──[Domain]          (Pure Entities, Value Objects, exceptions)
```

- **Domain Layer**: Contains immutable value objects (`BoundingBox`, `DetectionConfidence`, `CropMetadata`) and pure model states (`SlabDetectionResult`) completely free of framework or OpenCV dependencies.
- **Preprocessing Layer**: Structured into decoupled SOLID services (`BilateralFilterService`, `MorphologyGradientService`, `ThresholdingService`, `ContourExtractionService`, `RectangleScoringService`, `SafeCropper`).
- **Application Layer**: Contains framework-agnostic orchestrators (`SlabExtractionUseCase`).
- **Infrastructure Layer**: Direct adapters (FastAPI endpoints, ZeroMQ asynchronous worker, image decoders, local filesystem storage).

---

## 🔬 Morphology-Gradient CV Pipeline

Rather than relying on brittle Canny border approximations or fragile pixel-level foreground segmentation, the localization engine operates on high-efficiency structural gradient density:

```
  Original Image
         │
         ▼
     Grayscale
         │
         ▼
  Bilateral Filter       <-- Suppresses fine granite/marble texture noise
         │
         ▼
Morphological Gradient   <-- Highlights boundary transitions with MORPH_GRADIENT
         │
         ▼
   Otsu Threshold        <-- Computes optimal dynamic binarization threshold
         │
         ▼
 Morphology Close/Open   <-- Bridges gaps, smooths straps & holders (Large (41,41) kernel)
         │
         ▼
  Contour Scorer         <-- Computes multi-attribute scoring & locks primary slab
         │
         ▼
    Safe Cropper         <-- Dynamic pad insets with 30% min dimension guardrail
         │
         ▼
    Clean Crop
```

---

## 🚀 Setup & Installation

### Prerequisite
- **Python 3.11+** installed on a Linux/macOS environment.

### 1. Initialize Virtual Environment & Dependencies
```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Install production and development dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 🏃 Running the Application

The system uses asynchronous ZeroMQ internal communication. You can run it either in **REST fallback mode** or **Full ZMQ Event-Driven Mode**.

### Mode A: Full Event-Driven Stack (Recommended)

1. **Start the ZMQ Broker / Coordinator**
   ```bash
   python -m src.app.messaging.broker
   ```
2. **Start the Asynchronous Preprocessing Worker**
   ```bash
   python -m src.app.messaging.worker
   ```
3. **Start the REST API Gateway (FastAPI)**
   ```bash
   ./venv/bin/uvicorn src.app.api.main:app --host 0.0.0.0 --port 8000 --reload
   ```

### Mode B: Lightweight REST Gateway Only
If the ZeroMQ coordinator broker is not running, the FastAPI service automatically falls back safely to synchronous local in-memory execution to maintain 100% uptime.
```bash
./venv/bin/uvicorn src.app.api.main:app --host 0.0.0.0 --port 8000
```

---

## 🧪 Running the Test Suite

The project includes an exhaustive mirrored test suite covering **61 distinct test cases**, including all 20 mandatory synthetic warehouse scenarios (low contrast, rotated, noisy, straps/holders crossings, partial visibility, and border safety):

```bash
# Run the entire suite with verbose logging
./venv/bin/pytest -v
```

---

## 🔌 API Documentation

### POST `/extract-slab`
Submits a raw stone image URL for preprocessing, localization, and cropping.

#### **Request Body**
```json
{
  "image_url": "https://iblocky.work/black-eagle/black-eagle/260107/Bundle/260107_Taj_Mahal_bundle_1_20260526074523.webp",
  "request_id": "optional-custom-uuid-12345"
}
```

#### **Success Response (200 OK)**
```json
{
  "request_id": "optional-custom-uuid-12345",
  "status": "success",
  "data": {
    "bounding_box": {
      "x": 38,
      "y": 50,
      "width": 320,
      "height": 300
    },
    "confidence": 0.85,
    "crop_metadata": {
      "original_width": 400,
      "original_height": 400,
      "cropped_width": 320,
      "cropped_height": 300,
      "is_fallback": false,
      "confidence": 0.85,
      "warning": null
    }
  }
}
```

---

## 📁 Preprocessing Intermediate Outputs
For every request processed, the pipeline persists high-fidelity step-by-step debug images under the `debug_outputs/{request_id}/` folder:
- `original.jpg` - Raw downloaded image
- `gray.jpg` - Clean grayscale conversion
- `blurred.jpg` - High-preservation Bilateral filter output
- `edges.jpg` - Otsu binarized structural boundary gradient
- `morphology.jpg` - Bridged structural contour maps (Close/Open output)
- `contours.jpg` - Superimposed detected boundary contours
