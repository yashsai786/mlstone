from datetime import datetime
from src.app.ml.domain.models import StoneClassification, EvaluationReport, TrainingMetadata


def test_stone_classification_model():
    classification = StoneClassification(
        predicted_class="beige",
        confidence=0.89,
        top_k={"beige": 0.89, "grey": 0.08, "black": 0.03},
        inference_time_ms=12.5,
        device_used="cpu"
    )
    
    assert classification.predicted_class == "beige"
    assert classification.confidence == 0.89
    assert classification.top_k["grey"] == 0.08
    assert classification.inference_time_ms == 12.5
    assert classification.device_used == "cpu"


def test_evaluation_report_model():
    report = EvaluationReport(
        accuracy=0.92,
        macro_f1=0.91,
        weighted_f1=0.92,
        per_class_metrics={
            "beige": {"precision": 0.90, "recall": 0.92, "f1-score": 0.91, "support": 100}
        },
        confusion_matrix=[[92, 8], [6, 94]],
        class_labels=["beige", "grey"]
    )
    
    assert report.accuracy == 0.92
    assert report.macro_f1 == 0.91
    assert report.confusion_matrix[0][1] == 8
    assert isinstance(report.generated_at, str)


def test_training_metadata_model():
    metadata = TrainingMetadata(
        best_accuracy=0.95,
        best_loss=0.12,
        epochs_trained=8,
        class_to_idx={"beige": 0, "grey": 1},
        idx_to_class={0: "beige", 1: "grey"}
    )
    
    assert metadata.best_accuracy == 0.95
    assert metadata.epochs_trained == 8
    assert metadata.class_to_idx["grey"] == 1
    assert metadata.idx_to_class[0] == "beige"
    assert isinstance(metadata.trained_at, str)
