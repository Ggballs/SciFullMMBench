import logging
import os
import sys
from pathlib import Path
from typing import Optional

import click
import yaml

from openreview_pipeline.llm import MockLLMBackend, OpenAICompatibleBackend
from openreview_pipeline.stages import (
    DatasetDownloader,
    RuleBasedFilter,
    Summarizer,
    QueryGenerator,
    HardNegativeMiner,
    QueryFilter,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

CONFIG_PATH = Path(__file__).parent.parent.parent / "config.yaml"


def load_config() -> dict:
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH) as f:
            return yaml.safe_load(f) or {}
    return {}


def get_llm_backend(
    name: str = "mock",
    base_url: Optional[str] = None,
    api_token: Optional[str] = None,
    model: str = "gpt-4o-mini",
):
    if name == "mock":
        return MockLLMBackend()
    elif name == "openai-compatible" or (base_url and api_token):
        if not base_url or not api_token:
            logger.warning("base_url and api_token required for openai-compatible, using mock")
            return MockLLMBackend()
        return OpenAICompatibleBackend(
            base_url=base_url,
            api_token=api_token,
            model=model,
        )
    else:
        logger.warning(f"Unknown LLM backend '{name}', using mock")
        return MockLLMBackend()


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
    from datetime import datetime

    config = load_config()
    target_year = year or datetime.now().year
    logger.info(f"Downloading papers from {venue} {target_year}")

    downloader = DatasetDownloader(venue=venue, year_threshold=target_year)

    openreview_creds = config.get("openreview", {})
    username = username or openreview_creds.get("username", "")
    password = password or openreview_creds.get("password", "")
    token = token or openreview_creds.get("token", "")

    if username or password or token:
        downloader.set_openreview_credentials(
            username=username,
            password=password,
            token=token,
        )
    else:
        logger.warning("No OpenReview credentials provided in config.yaml or arguments. Using stub data.")

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    downloader.run(output_path)
    logger.info(f"Download complete: {output}")


@cli.command()
@click.option("--input", "-i", "input_path", type=click.Path(exists=True), required=True, help="Input dataset path")
@click.option("--output", "-o", type=click.Path(), default="data/01_filtered.json", help="Output path")
def filter(input_path: str, output: str):
    logger.info(f"Running filter stage: {input_path} -> {output}")
    filter_stage = RuleBasedFilter()
    filter_stage.run(Path(input_path), Path(output))
    logger.info(f"Filter complete: {output}")


@cli.command()
@click.option("--input", "-i", "input_path", type=click.Path(exists=True), required=True, help="Input dataset path")
@click.option("--output", "-o", type=click.Path(), default="data/02_summarized.json", help="Output path")
@click.option("--base-url", default=None, help="LLM API base URL (overrides config)")
@click.option("--api-token", default=None, help="LLM API token (overrides config)")
@click.option("--model", default=None, help="Model name (overrides config)")
def summarize(input_path: str, output: str, base_url: str, api_token: str, model: str):
    config = load_config()
    llm_config = config.get("llm", {})

    base_url = base_url or llm_config.get("base_url", "")
    api_token = api_token or llm_config.get("api_token", "")
    model = model or llm_config.get("model", "gpt-4o-mini")

    logger.info(f"Running summarize stage: {input_path} -> {output}")
    llm_backend = get_llm_backend(
        name="openai-compatible" if base_url and api_token else "mock",
        base_url=base_url,
        api_token=api_token,
        model=model,
    )
    summarizer = Summarizer(llm=llm_backend)
    summarizer.run(Path(input_path), Path(output))
    logger.info(f"Summarize complete: {output}")


@cli.command("generate-queries")
@click.option("--input", "-i", "input_path", type=click.Path(exists=True), required=True, help="Input dataset path")
@click.option("--output", "-o", type=click.Path(), default="data/03_queries.json", help="Output path")
@click.option("--base-url", default=None, help="LLM API base URL (overrides config)")
@click.option("--api-token", default=None, help="LLM API token (overrides config)")
@click.option("--model", default=None, help="Model name (overrides config)")
def generate_queries(input_path: str, output: str, base_url: str, api_token: str, model: str):
    config = load_config()
    llm_config = config.get("llm", {})

    base_url = base_url or llm_config.get("base_url", "")
    api_token = api_token or llm_config.get("api_token", "")
    model = model or llm_config.get("model", "gpt-4o-mini")

    logger.info(f"Running generate-queries stage: {input_path} -> {output}")
    llm_backend = get_llm_backend(
        name="openai-compatible" if base_url and api_token else "mock",
        base_url=base_url,
        api_token=api_token,
        model=model,
    )
    generator = QueryGenerator(llm=llm_backend)
    generator.run(Path(input_path), Path(output))
    logger.info(f"Generate queries complete: {output}")


@cli.command("filter-queries")
@click.option("--input", "-i", "input_path", type=click.Path(exists=True), required=True, help="Input dataset path")
@click.option("--output", "-o", type=click.Path(), default="data/06_filtered_queries.json", help="Output path")
@click.option("--base-url", default=None, help="LLM API base URL (overrides config)")
@click.option("--api-token", default=None, help="LLM API token (overrides config)")
@click.option("--model", default=None, help="Model name (overrides config)")
@click.option("--threshold", type=float, default=0.5, help="Quality threshold")
def filter_queries(input_path: str, output: str, base_url: str, api_token: str, model: str, threshold: float):
    config = load_config()
    llm_config = config.get("llm", {})

    base_url = base_url or llm_config.get("base_url", "")
    api_token = api_token or llm_config.get("api_token", "")
    model = model or llm_config.get("model", "gpt-4o-mini")

    logger.info(f"Running filter-queries stage: {input_path} -> {output}")
    llm_backend = get_llm_backend(
        name="openai-compatible" if base_url and api_token else "mock",
        base_url=base_url,
        api_token=api_token,
        model=model,
    )
    query_filter = QueryFilter(llm=llm_backend, threshold=threshold)
    query_filter.run(Path(input_path), Path(output))
    logger.info(f"Filter queries complete: {output}")


@cli.command("hard-negative-mining")
@click.option("--input", "-i", "input_path", type=click.Path(exists=True), required=True, help="Input dataset path (filtered queries)")
@click.option("--output", "-o", type=click.Path(), default="data/05_hard_negatives.json", help="Output path")
@click.option("--base-url", default=None, help="LLM API base URL (overrides config)")
@click.option("--api-token", default=None, help="LLM API token (overrides config)")
@click.option("--model", default=None, help="Model name (overrides config)")
def hard_negative_mining(input_path: str, output: str, base_url: str, api_token: str, model: str):
    config = load_config()
    llm_config = config.get("llm", {})

    base_url = base_url or llm_config.get("base_url", "")
    api_token = api_token or llm_config.get("api_token", "")
    model = model or llm_config.get("model", "gpt-4o-mini")

    logger.info(f"Running hard-negative-mining stage: {input_path} -> {output}")
    llm_backend = get_llm_backend(
        name="openai-compatible" if base_url and api_token else "mock",
        base_url=base_url,
        api_token=api_token,
        model=model,
    )
    miner = HardNegativeMiner(llm=llm_backend)
    miner.run(Path(input_path), Path(output))
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
    from datetime import datetime

    config = load_config()
    llm_config = config.get("llm", {})
    openreview_creds = config.get("openreview", {})

    base_url = base_url or llm_config.get("base_url", "")
    api_token = api_token or llm_config.get("api_token", "")
    model = model or llm_config.get("model", "gpt-4o-mini")
    target_year = year or datetime.now().year

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    llm_backend = get_llm_backend(
        name="openai-compatible" if base_url and api_token else "mock",
        base_url=base_url,
        api_token=api_token,
        model=model,
    )

    download_path = output_path / "00_downloaded.json"
    filtered_path = output_path / "01_filtered.json"
    summarized_path = output_path / "02_summarized.json"
    queries_path = output_path / "03_queries.json"
    final_path = output_path / "04_filtered_queries.json"

    logger.info(f"Stage 0: Downloading up to {max_papers} papers from {venue} {target_year}")
    downloader = DatasetDownloader(venue=venue, year_threshold=target_year)
    downloader.set_openreview_credentials(
        username=openreview_creds.get("username"),
        password=openreview_creds.get("password"),
        token=openreview_creds.get("token"),
    )
    downloader.fetch_recent_papers(limit=max_papers)
    downloader.run(download_path)

    logger.info("Stage 1: Filtering papers")
    filter_stage = RuleBasedFilter()
    filter_stage.run(download_path, filtered_path)

    logger.info(f"Stage 2: Summarizing papers (LLM limit: {llm_limit})")
    summarizer = Summarizer(llm=llm_backend, llm_limit=llm_limit)
    summarizer.run(filtered_path, summarized_path)

    logger.info("Stage 3: Generating queries")
    generator = QueryGenerator(llm=llm_backend)
    generator.run(summarized_path, queries_path)

    logger.info("Stage 4: Filtering queries")
    query_filter = QueryFilter(llm=llm_backend)
    query_filter.run(queries_path, final_path)

    logger.info("Pipeline complete!")


def main():
    cli()
