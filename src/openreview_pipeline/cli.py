import logging
import os
import sys
from pathlib import Path
from typing import Optional

import click

from openreview_pipeline.runner import (
    load_config,
    run_download_stage,
    run_filter_stage,
    run_filter_queries_stage,
    run_generate_queries_stage,
    run_hard_negative_mining_stage,
    run_selected_stages,
    run_summarize_stage,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

CONFIG_PATH = Path(__file__).parent.parent.parent / "config.yaml"


@click.group()
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
def cli(verbose: bool):
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)


@cli.command()
@click.option("--output", "-o", type=click.Path(), default="data/00_downloaded.json", help="Output path")
@click.option("--venue", default="ICLR", help="Venue to download from (e.g., ICLR, NeurIPS)")
@click.option("--year", default=None, type=int, help="Year to download (default: current year)")
@click.option("--username", default=None, help="OpenReview username/email (overrides config)")
@click.option("--password", default=None, help="OpenReview password (overrides config)")
@click.option("--token", default=None, help="OpenReview API token (overrides config)")
@click.option("--limit", default=None, type=int, help="Limit number of papers to fetch")
def download(output: str, venue: str, year: int, username: str, password: str, token: str, limit: int):
    logger.info(f"Downloading papers from {venue} {year or 'current year'}")
    run_download_stage(
        output_path=Path(output),
        venue=venue,
        year=year,
        config_path=CONFIG_PATH,
        username=username,
        password=password,
        token=token,
        limit=limit,
    )
    logger.info(f"Download complete: {output}")


@cli.command()
@click.option("--input", "-i", "input_path", type=click.Path(exists=True), required=True, help="Input dataset path")
@click.option("--output", "-o", type=click.Path(), default="data/01_filtered.json", help="Output path")
def filter(input_path: str, output: str):
    logger.info(f"Running filter stage: {input_path} -> {output}")
    run_filter_stage(
        input_path=Path(input_path),
        output_path=Path(output),
    )
    logger.info(f"Filter complete: {output}")


@cli.command()
@click.option("--input", "-i", "input_path", type=click.Path(exists=True), required=True, help="Input dataset path")
@click.option("--output", "-o", type=click.Path(), default="data/02_summarized.json", help="Output path")
@click.option("--base-url", default=None, help="LLM API base URL (overrides config)")
@click.option("--api-token", default=None, help="LLM API token (overrides config)")
@click.option("--model", default=None, help="Model name (overrides config)")
def summarize(input_path: str, output: str, base_url: str, api_token: str, model: str):
    logger.info(f"Running summarize stage: {input_path} -> {output}")
    run_summarize_stage(
        input_path=Path(input_path),
        output_path=Path(output),
        config_path=CONFIG_PATH,
        base_url=base_url,
        api_token=api_token,
        model=model,
    )
    logger.info(f"Summarize complete: {output}")


@cli.command("generate-queries")
@click.option("--input", "-i", "input_path", type=click.Path(exists=True), required=True, help="Input dataset path")
@click.option("--output", "-o", type=click.Path(), default="data/03_queries.json", help="Output path")
@click.option("--base-url", default=None, help="LLM API base URL (overrides config)")
@click.option("--api-token", default=None, help="LLM API token (overrides config)")
@click.option("--model", default=None, help="Model name (overrides config)")
def generate_queries(input_path: str, output: str, base_url: str, api_token: str, model: str):
    logger.info(f"Running generate-queries stage: {input_path} -> {output}")
    run_generate_queries_stage(
        input_path=Path(input_path),
        output_path=Path(output),
        config_path=CONFIG_PATH,
        base_url=base_url,
        api_token=api_token,
        model=model,
    )
    logger.info(f"Generate queries complete: {output}")


@cli.command("filter-queries")
@click.option("--input", "-i", "input_path", type=click.Path(exists=True), required=True, help="Input dataset path")
@click.option("--output", "-o", type=click.Path(), default="data/06_filtered_queries.json", help="Output path")
@click.option("--base-url", default=None, help="LLM API base URL (overrides config)")
@click.option("--api-token", default=None, help="LLM API token (overrides config)")
@click.option("--model", default=None, help="Model name (overrides config)")
@click.option("--threshold", type=float, default=0.5, help="Quality threshold")
def filter_queries(input_path: str, output: str, base_url: str, api_token: str, model: str, threshold: float):
    logger.info(f"Running filter-queries stage: {input_path} -> {output}")
    run_filter_queries_stage(
        input_path=Path(input_path),
        output_path=Path(output),
        config_path=CONFIG_PATH,
        base_url=base_url,
        api_token=api_token,
        model=model,
        threshold=threshold,
    )
    logger.info(f"Filter queries complete: {output}")


@cli.command("hard-negative-mining")
@click.option("--input", "-i", "input_path", type=click.Path(exists=True), required=True, help="Input dataset path (filtered queries)")
@click.option("--output", "-o", type=click.Path(), default="data/05_hard_negatives.json", help="Output path")
@click.option("--base-url", default=None, help="LLM API base URL (overrides config)")
@click.option("--api-token", default=None, help="LLM API token (overrides config)")
@click.option("--model", default=None, help="Model name (overrides config)")
def hard_negative_mining(input_path: str, output: str, base_url: str, api_token: str, model: str):
    logger.info(f"Running hard-negative-mining stage: {input_path} -> {output}")
    run_hard_negative_mining_stage(
        input_path=Path(input_path),
        output_path=Path(output),
        config_path=CONFIG_PATH,
        base_url=base_url,
        api_token=api_token,
        model=model,
    )
    logger.info(f"Hard negative mining complete: {output}")


@cli.command("run-all")
@click.option("--output-dir", "-o", type=click.Path(), default="data", help="Output directory")
@click.option("--venue", default="ICLR", help="Venue to download from (ICLR, NeurIPS, ICML)")
@click.option("--year", default=None, type=int, help="Year to download (default: current year)")
@click.option("--max-papers", default=30, type=int, help="Maximum papers to download")
@click.option("--llm-limit", default=5, type=int, help="Maximum papers to process with LLM (for cost saving)")
@click.option("--base-url", default=None, help="LLM API base URL (overrides config)")
@click.option("--api-token", default=None, help="LLM API token (overrides config)")
@click.option("--model", default=None, help="Model name (overrides config)")
def run_all(output_dir: str, venue: str, year: int, max_papers: int, llm_limit: int, base_url: str, api_token: str, model: str):
    run_selected_stages(
        "0-4",
        output_dir=Path(output_dir),
        venue=venue,
        year=year,
        download_limit=max_papers,
        llm_limit=llm_limit,
        config_path=CONFIG_PATH,
        base_url=base_url,
        api_token=api_token,
        model=model,
    )
    logger.info("Pipeline complete!")


def main():
    cli()


if __name__ == "__main__":
    main()
