#!/usr/bin/env python3
import asyncio
import sys
from pathlib import Path

# Add project root to python path to ensure imports work seamlessly
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import click
import torch
from src.app.infrastructure.config import get_config
from src.app.infrastructure.logging import setup_logging, get_logger
from src.app.ml.application.use_cases import TrainModelUseCase

logger = get_logger("scripts.train_model")


@click.command()
@click.option(
    "--epochs",
    type=int,
    help="Override default training epochs count."
)
@click.option(
    "--batch-size",
    type=int,
    help="Override default training batch size."
)
@click.option(
    "--learning-rate",
    type=float,
    help="Override default training learning rate."
)
@click.option(
    "--device",
    default="cuda" if torch.cuda.is_available() else "cpu",
    help="Force training device (cpu or cuda)."
)
@click.option(
    "--seed",
    type=int,
    default=42,
    help="Define training deterministic random seed."
)
def main(epochs, batch_size, learning_rate, device, seed):
    """
    Supervised Machine Learning Training Entry Point.
    Trains an EfficientNet-B0 classifier model for stone color classification.
    """
    setup_logging()
    logger.info("Initializing ML Training CLI entry point...")

    config = get_config()
    
    # 1. Instantiate the orchestrator use-case
    use_case = TrainModelUseCase(config=config)
    
    # 2. Execute training
    try:
        metadata = use_case.execute(
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            device=device,
            seed=seed
        )
        
        # 3. Print formatted console results
        click.echo("\n" + "=" * 55)
        click.echo("             MODEL TRAINING PIPELINE COMPLETED")
        click.echo("=" * 55)
        click.echo(f"Device Used:           {device.upper()}")
        click.echo(f"Final Model Version:   {metadata.model_version}")
        click.echo(f"Epochs Completed:      {metadata.epochs_trained}")
        click.echo("-" * 55)
        click.echo(f"Best Validation Acc:   {metadata.best_accuracy * 100:.2f}%")
        click.echo(f"Best Validation Loss:  {metadata.best_loss:.4f}")
        click.echo(f"Total Unique Classes:  {len(metadata.class_to_idx)}")
        click.echo(f"Class Mappings:        {list(metadata.class_to_idx.keys())}")
        click.echo(f"Model File Path:       {config.ml_model_dir}/stone_color_model.pt")
        click.echo("=" * 55 + "\n")
        
    except Exception as e:
        logger.error("ML model training execution crashed", extra={"error": str(e)})
        click.echo(f"\n❌ Error during training execution: {e}\n", err=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
