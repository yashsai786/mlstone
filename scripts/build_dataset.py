#!/usr/bin/env python3
import asyncio
import sys
from pathlib import Path

# Add project root to python path to ensure imports work seamlessly
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import click
from src.app.infrastructure.config import get_config
from src.app.infrastructure.logging import setup_logging, get_logger
from src.app.dataset.infrastructure.adapters import LocalDatasetReader, LocalDatasetWriter
from src.app.dataset.services.downloader import DownloaderService
from src.app.dataset.application.use_cases import IngestDatasetUseCase, ProgressMetrics

logger = get_logger("scripts.build_dataset")


async def run_pipeline(
    single_color: click.Parameter,
    resume: click.Parameter,
    dry_run: click.Parameter,
    concurrency: click.Parameter,
    output_size: click.Parameter,
    retry_count: click.Parameter,
    timeout: click.Parameter
):
    # Setup global configuration
    config = get_config()

    # Apply command-line overrides to config
    if concurrency:
        config.dataset_concurrency = concurrency
    if output_size:
        config.output_image_size = output_size
    if retry_count:
        config.download_retry_count = retry_count
    if timeout:
        config.download_timeout_seconds = timeout

    # Setup infrastructure and services dependencies
    reader = LocalDatasetReader()
    writer = LocalDatasetWriter(
        processed_base_dir=config.processed_dataset_dir,
        failed_base_dir=config.failed_dir,
        metadata_path=config.metadata_csv_path
    )
    downloader = DownloaderService(
        timeout_seconds=config.download_timeout_seconds,
        retry_count=config.download_retry_count
    )

    # Initialize use-case
    use_case = IngestDatasetUseCase(
        config=config,
        reader=reader,
        writer=writer,
        downloader_service=downloader
    )

    # Run ingestion pipeline
    metrics: ProgressMetrics = await use_case.execute(
        single_color=single_color,
        resume_failed=resume,
        dry_run=dry_run
    )

    # Print beautiful human-readable metrics report to stderr/stdout
    click.echo("\n" + "=" * 55)
    click.echo("             DATASET INGESTION PIPELINE REPORT")
    click.echo("=" * 55)
    click.echo(f"Execution Mode:        {'DRY-RUN' if dry_run else 'PRODUCTION'}")
    click.echo(f"Active Filter Color:   {single_color or 'None (Full Dataset)'}")
    click.echo(f"Resume Failed Jobs:    {str(resume).upper()}")
    click.echo(f"Configured Concurrency: {config.dataset_concurrency}")
    click.echo(f"Output Resized Side:   {config.output_image_size}px")
    click.echo("-" * 55)
    click.echo(f"Total Unique URLs Found: {metrics.total_found}")
    click.echo(f"Already Completed:       {metrics.skipped_completed}")
    click.echo(f"Scheduled to Process:    {metrics.to_process}")
    if not dry_run:
        click.echo(f"Successfully Cropped:    {metrics.success_count}")
        click.echo(f"Failed Downloads:        {metrics.failed_download}")
        click.echo(f"Failed Crops/Detections: {metrics.failed_crop}")
        click.echo(f"Other Processing Errors: {metrics.failed_other}")
        click.echo(f"Overall Ingest Success:  {metrics.success_rate:.2f}%")
        click.echo(f"Total Run Duration:      {metrics.duration_seconds:.2f} seconds")
        click.echo(f"Average Ingest Speed:    {metrics.items_per_second:.2f} items/sec")
    click.echo("=" * 55 + "\n")


@click.command()
@click.option(
    "--single-color",
    default=None,
    help="Filter ingestion to a single color directory class name (e.g., beige, black)."
)
@click.option(
    "--resume",
    is_flag=True,
    help="Resume pipeline by retrying failed download/crop URLs registered in metadata.csv."
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Scan files and print planned metrics without downloading or cropping images."
)
@click.option(
    "--concurrency",
    type=int,
    help="Override default dataset ingestion worker concurrency count."
)
@click.option(
    "--output-size",
    type=int,
    help="Override default processed image maximum bounding box dimensions."
)
@click.option(
    "--retry-count",
    type=int,
    help="Override downloader retry attempts before failing."
)
@click.option(
    "--timeout",
    type=int,
    help="Override download timeout in seconds."
)
def main(single_color, resume, dry_run, concurrency, output_size, retry_count, timeout):
    """
    Stone Slab Dataset Ingestion and Preprocessing Pipeline CLI entry point.
    """
    setup_logging()
    logger.info("Initializing dataset CLI entry point...")

    # Execute async pipeline flow
    asyncio.run(
        run_pipeline(
            single_color=single_color,
            resume=resume,
            dry_run=dry_run,
            concurrency=concurrency,
            output_size=output_size,
            retry_count=retry_count,
            timeout=timeout
        )
    )


if __name__ == "__main__":
    main()
