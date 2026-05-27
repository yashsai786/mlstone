#!/usr/bin/env python3
import sys
from pathlib import Path

# Add project root to python path to ensure imports work seamlessly
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import click
import torch
from src.app.infrastructure.config import get_config
from src.app.infrastructure.logging import setup_logging, get_logger
from src.app.ml.inference.service import StoneColorInferenceService

logger = get_logger("scripts.test_inference")


@click.command()
@click.option(
    "--image-path",
    required=True,
    type=click.Path(exists=True),
    help="Path to the target stone slab image file."
)
@click.option(
    "--device",
    default="cuda" if torch.cuda.is_available() else "cpu",
    help="Force inference device (cpu or cuda)."
)
@click.option(
    "--top-k",
    type=int,
    default=3,
    help="Number of top probability predictions to show."
)
def main(image_path, device, top_k):
    """
    Trained Model Single-Image Inference CLI Tester.
    Takes a path, runs classification, and outputs confidence statistics.
    """
    setup_logging()
    logger.info("Initializing ML Inference CLI tester...")

    config = get_config()
    model_path = str(config.ml_model_dir / "stone_color_model.pt")

    # 1. Instantiate the inference service
    try:
        service = StoneColorInferenceService(
            model_path=model_path,
            device=device
        )
        
        # 2. Run prediction
        logger.info("Executing prediction on test image...", extra={"path": image_path})
        classification = service.predict(
            image_input=image_path,
            top_k=top_k
        )
        
        # 3. Print formatted console predictions
        click.echo("\n" + "=" * 55)
        click.echo("             SINGLE-IMAGE INFERENCE REPORT")
        click.echo("=" * 55)
        click.echo(f"Target Image Path:     {image_path}")
        click.echo(f"Device Used:           {classification.device_used.upper()}")
        click.echo(f"Inference Time:        {classification.inference_time_ms:.2f} ms")
        click.echo("-" * 55)
        click.echo(f"PREDICTED COLOR CLASS: {classification.predicted_class.upper()}")
        click.echo(f"Confidence Score:      {classification.confidence * 100:.2f}%")
        click.echo("-" * 55)
        click.echo(f"Top {top_k} Predictions:")
        for name, confidence in classification.top_k.items():
            click.echo(f"  - {name:<12}: {confidence * 100:.2f}%")
        click.echo("=" * 55 + "\n")
        
    except Exception as e:
        logger.error("Inference execution failed", extra={"error": str(e)})
        click.echo(f"\n❌ Error during inference execution: {e}\n", err=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
