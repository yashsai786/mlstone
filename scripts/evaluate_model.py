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
from src.app.ml.application.use_cases import EvaluateModelUseCase

logger = get_logger("scripts.evaluate_model")


@click.command()
@click.option(
    "--device",
    default="cuda" if torch.cuda.is_available() else "cpu",
    help="Force evaluation device (cpu or cuda)."
)
@click.option(
    "--seed",
    type=int,
    default=42,
    help="Define evaluation deterministic random seed."
)
def main(device, seed):
    """
    Trained Model Evaluation Entry Point.
    Generates standard classification reports, confusion matrices, and failure diagnostics.
    """
    setup_logging()
    logger.info("Initializing ML Evaluation CLI entry point...")

    config = get_config()
    
    # 1. Instantiate the evaluation use-case
    use_case = EvaluateModelUseCase(config=config)
    
    # 2. Execute evaluation
    try:
        report, save_path = use_case.execute(device=device, seed=seed)
        
        # 3. Print formatted console reports
        click.echo("\n" + "=" * 55)
        click.echo("             MODEL EVALUATION REPORT SUMMARY")
        click.echo("=" * 55)
        click.echo(f"Device Used:           {device.upper()}")
        click.echo(f"Target Save Path:      {save_path}")
        click.echo(f"Evaluation Accuracy:   {report.accuracy * 100:.2f}%")
        click.echo(f"Macro Average F1:      {report.macro_f1:.4f}")
        click.echo(f"Weighted Average F1:   {report.weighted_f1:.4f}")
        click.echo("-" * 55)
        click.echo("                     PER-CLASS PERFORMANCE")
        click.echo("-" * 55)
        click.echo(f"{'Color Class':<15} | {'Precision':<9} | {'Recall':<9} | {'F1-Score':<9}")
        click.echo("-" * 55)
        for name, metrics in report.per_class_metrics.items():
            click.echo(
                f"{name:<15} | "
                f"{metrics['precision']*100:<8.1f}% | "
                f"{metrics['recall']*100:<8.1f}% | "
                f"{metrics['f1-score']*100:<8.1f}%"
            )
        click.echo("-" * 55)
        click.echo("Confusion Matrix:")
        for row in report.confusion_matrix:
            click.echo(f"  {row}")
        click.echo("=" * 55 + "\n")
        
    except Exception as e:
        logger.error("ML model evaluation execution crashed", extra={"error": str(e)})
        click.echo(f"\n❌ Error during evaluation execution: {e}\n", err=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
