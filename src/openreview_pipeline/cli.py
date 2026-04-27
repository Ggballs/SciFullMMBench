import logging
from pathlib import Path

import click

from openreview_pipeline.pipeline_output_builder import build_pipeline_output, write_pipeline_output
from openreview_pipeline.runner import (
    run_download_stage,
    run_filter_stage,
    run_generate_queries_stage,
    run_hard_negative_mining_stage,
    run_query_analysis_stage,
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
@click.option("--forum-id", default=None, help="Download exactly one paper by OpenReview forum id")
@click.option("--username", default=None, help="OpenReview username/email (overrides config)")
@click.option("--password", default=None, help="OpenReview password (overrides config)")
@click.option("--token", default=None, help="OpenReview API token (overrides config)")
@click.option("--limit", default=None, type=int, help="Limit number of papers to fetch")
def download(
    output: str,
    venue: str,
    year: int,
    forum_id: str,
    username: str,
    password: str,
    token: str,
    limit: int,
):
    run_download_stage(
        output_path=Path(output),
        venue=venue,
        year=year,
        forum_id=forum_id,
        config_path=CONFIG_PATH,
        username=username,
        password=password,
        token=token,
        limit=limit,
    )


@cli.command()
@click.option("--input", "-i", "input_path", type=click.Path(exists=True), required=True, help="Input dataset path")
@click.option("--output", "-o", type=click.Path(), default="data/01_filtered.json", help="Output path")
def filter(input_path: str, output: str):
    run_filter_stage(
        input_path=Path(input_path),
        output_path=Path(output),
    )


@cli.command()
@click.option("--input", "-i", "input_path", type=click.Path(exists=True), required=True, help="Input dataset path")
@click.option("--output", "-o", type=click.Path(), default="data/02_summarized.json", help="Output path")
@click.option("--base-url", default=None, help="LLM API base URL (overrides config)")
@click.option("--api-token", default=None, help="LLM API token (overrides config)")
@click.option("--model", default=None, help="Model name (overrides config)")
def summarize(input_path: str, output: str, base_url: str, api_token: str, model: str):
    run_summarize_stage(
        input_path=Path(input_path),
        output_path=Path(output),
        config_path=CONFIG_PATH,
        base_url=base_url,
        api_token=api_token,
        model=model,
    )


@cli.command("generate-queries")
@click.option("--input", "-i", "input_path", type=click.Path(exists=True), required=True, help="Input summarized dataset path")
@click.option("--output", "-o", type=click.Path(), default="data/03_queries.json", help="Output path")
@click.option("--base-url", default=None, help="LLM API base URL (overrides config)")
@click.option("--api-token", default=None, help="LLM API token (overrides config)")
@click.option("--model", default=None, help="Model name (overrides config)")
def generate_queries(input_path: str, output: str, base_url: str, api_token: str, model: str):
    run_generate_queries_stage(
        input_path=Path(input_path),
        output_path=Path(output),
        config_path=CONFIG_PATH,
        base_url=base_url,
        api_token=api_token,
        model=model,
    )


@cli.command("hard-negative-mining")
@click.option("--input", "-i", "input_path", type=click.Path(exists=True), required=True, help="Input generated queries dataset path")
@click.option("--query-analysis-input", type=click.Path(exists=True), default=None, help="Optional stage-4 query analysis directory used to keep only surviving queries")
@click.option("--output", "-o", type=click.Path(), default="data/05_hard_negatives.json", help="Output path")
@click.option("--base-url", default=None, help="LLM API base URL (overrides config)")
@click.option("--api-token", default=None, help="LLM API token (overrides config)")
@click.option("--model", default=None, help="Model name (overrides config)")
@click.option("--scholar-provider", default=None, help="Google Scholar backend: 'serpapi' or 'scholarly' (overrides config)")
@click.option("--serpapi-api-key", default=None, help="SerpAPI key for real Google Scholar search (overrides config)")
@click.option("--scholar-max-results", default=None, type=int, help="Maximum Google Scholar candidates to retrieve per query")
@click.option("--scholar-language", default=None, help="Google Scholar language code, e.g. 'en'")
def hard_negative_mining(
    input_path: str,
    query_analysis_input: str,
    output: str,
    base_url: str,
    api_token: str,
    model: str,
    scholar_provider: str,
    serpapi_api_key: str,
    scholar_max_results: int,
    scholar_language: str,
):
    run_hard_negative_mining_stage(
        input_path=Path(input_path),
        output_path=Path(output),
        query_analysis_output_dir=Path(query_analysis_input) if query_analysis_input else None,
        config_path=CONFIG_PATH,
        base_url=base_url,
        api_token=api_token,
        model=model,
        scholar_provider=scholar_provider,
        serpapi_api_key=serpapi_api_key,
        scholar_max_results=scholar_max_results,
        scholar_language=scholar_language,
    )


@cli.command("query-analysis")
@click.option("--summarized-input", type=click.Path(exists=True), required=True, help="Stage-2 summarized dataset path")
@click.option("--queries-input", type=click.Path(exists=True), required=True, help="Stage-3 generated queries dataset path")
@click.option("--downloaded-input", type=click.Path(exists=True), default=None, help="Optional stage-0 downloaded dataset path")
@click.option("--output-dir", "-o", type=click.Path(), default="data/04_query_analysis", help="Output directory")
@click.option("--base-url", default=None, help="LLM API base URL (overrides config)")
@click.option("--api-token", default=None, help="LLM API token (overrides config)")
@click.option("--model", default=None, help="Model name (overrides config)")
@click.option("--llm-batch-size", default=25, type=int, help="Batch size for LLM-as-a-judge style scoring")
@click.option("--llm-judge-mode", default="batch", help="LLM judge mode: batch or single_query")
@click.option("--llm-max-concurrency", default=1, type=int, help="Max concurrent style-judge batches")
def query_analysis(
    summarized_input: str,
    queries_input: str,
    downloaded_input: str,
    output_dir: str,
    base_url: str,
    api_token: str,
    model: str,
    llm_batch_size: int,
    llm_judge_mode: str,
    llm_max_concurrency: int,
):
    run_query_analysis_stage(
        summarized_path=Path(summarized_input),
        queries_path=Path(queries_input),
        downloaded_path=Path(downloaded_input) if downloaded_input else None,
        output_dir=Path(output_dir),
        config_path=CONFIG_PATH,
        base_url=base_url,
        api_token=api_token,
        model=model,
        llm_batch_size=llm_batch_size,
        llm_judge_mode=llm_judge_mode,
        llm_max_concurrency=llm_max_concurrency,
    )


@cli.command("run-all")
@click.option("--output-dir", "-o", type=click.Path(), default="data", help="Output directory")
@click.option("--venue", default="ICLR", help="Venue to download from (ICLR, NeurIPS, ICML)")
@click.option("--year", default=None, type=int, help="Year to download (default: current year)")
@click.option("--forum-id", default=None, help="Download exactly one paper by OpenReview forum id")
@click.option("--max-papers", default=30, type=int, help="Maximum papers to download")
@click.option("--llm-limit", default=5, type=int, help="Maximum papers to process with LLM")
@click.option("--base-url", default=None, help="LLM API base URL (overrides config)")
@click.option("--api-token", default=None, help="LLM API token (overrides config)")
@click.option("--model", default=None, help="Model name (overrides config)")
@click.option("--llm-batch-size", default=25, type=int, help="Batch size for LLM-as-a-judge style scoring")
@click.option("--llm-judge-mode", default="batch", help="LLM judge mode: batch or single_query")
@click.option("--llm-max-concurrency", default=1, type=int, help="Max concurrent style-judge batches")
@click.option("--final-output", type=click.Path(), default=None, help="Final combined JSON path")
def run_all(
    output_dir: str,
    venue: str,
    year: int,
    forum_id: str,
    max_papers: int,
    llm_limit: int,
    base_url: str,
    api_token: str,
    model: str,
    llm_batch_size: int,
    llm_judge_mode: str,
    llm_max_concurrency: int,
    final_output: str,
):
    paths = run_selected_stages(
        "0-5",
        output_dir=Path(output_dir),
        venue=venue,
        year=year,
        forum_id=forum_id,
        download_limit=max_papers,
        llm_limit=llm_limit,
        config_path=CONFIG_PATH,
        base_url=base_url,
        api_token=api_token,
        model=model,
        llm_batch_size=llm_batch_size,
        llm_judge_mode=llm_judge_mode,
        llm_max_concurrency=llm_max_concurrency,
    )
    final_output_path = (
        Path(final_output).expanduser().resolve()
        if final_output
        else paths.output_dir / "final_pipeline_output.json"
    )
    artifact = build_pipeline_output(
        downloaded_path=paths.downloaded_path,
        filtered_path=paths.filtered_path,
        summarized_path=paths.summarized_path,
        queries_path=paths.queries_path,
        hard_negatives_path=paths.hard_negatives_path,
        query_analysis_output_dir=paths.query_analysis_output_dir,
    )
    write_pipeline_output(final_output_path, artifact)
    click.echo(str(final_output_path))


@cli.command("update-final-json")
@click.option("--base-dir", type=click.Path(exists=True), required=True, help="Directory containing stage output files")
@click.option("--output", "-o", type=click.Path(), default=None, help="Output path for final combined JSON")
@click.option("--downloaded-path", type=click.Path(exists=True), default=None, help="Optional override for 00_downloaded.json")
@click.option("--filtered-path", type=click.Path(exists=True), default=None, help="Optional override for 01_filtered.json")
@click.option("--summarized-path", type=click.Path(exists=True), default=None, help="Optional override for 02_summarized.json")
@click.option("--queries-path", type=click.Path(exists=True), default=None, help="Optional override for 03_queries.json")
@click.option("--hard-negatives-path", type=click.Path(exists=True), default=None, help="Optional override for 05_hard_negatives.json")
@click.option("--query-analysis-dir", type=click.Path(exists=True), default=None, help="Optional override for 04_query_analysis directory")
def update_final_json(
    base_dir: str,
    output: str,
    downloaded_path: str,
    filtered_path: str,
    summarized_path: str,
    queries_path: str,
    hard_negatives_path: str,
    query_analysis_dir: str,
):
    base = Path(base_dir).expanduser().resolve()
    final_output_path = Path(output).expanduser().resolve() if output else (base / "final_pipeline_output.json")
    final_output_path.parent.mkdir(parents=True, exist_ok=True)

    artifact = build_pipeline_output(
        downloaded_path=Path(downloaded_path).expanduser().resolve() if downloaded_path else (base / "00_downloaded.json"),
        filtered_path=Path(filtered_path).expanduser().resolve() if filtered_path else (base / "01_filtered.json"),
        summarized_path=Path(summarized_path).expanduser().resolve() if summarized_path else (base / "02_summarized.json"),
        queries_path=Path(queries_path).expanduser().resolve() if queries_path else (base / "03_queries.json"),
        hard_negatives_path=Path(hard_negatives_path).expanduser().resolve() if hard_negatives_path else (base / "05_hard_negatives.json"),
        query_analysis_output_dir=Path(query_analysis_dir).expanduser().resolve() if query_analysis_dir else (base / "04_query_analysis"),
    )
    write_pipeline_output(final_output_path, artifact)
    click.echo(str(final_output_path))


def main():
    cli()


if __name__ == "__main__":
    main()
