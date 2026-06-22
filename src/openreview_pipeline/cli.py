import logging
from pathlib import Path
from typing import Optional

import click

from utils.app_logging import configure_project_logging
from utils.project_paths import DEFAULT_CONFIG_PATH
from openreview_pipeline.pipeline_output_builder import build_pipeline_output, write_pipeline_output
from openreview_pipeline.stage2_prompt_dump import write_stage2_text_prompt
from openreview_pipeline.runner import (
    build_llm_backend,
    resolve_generate_query_settings,
    resolve_pipeline_paths,
    run_download_stage,
    run_filter_stage,
    run_generate_queries_stage,
    run_hard_negative_mining_stage,
    run_query_analysis_stage,
    run_selected_stages,
    run_summarize_stage,
)
from utils.db.golden_query_embeddings import (
    ensure_schema,
    get_engine,
)
from utils.golden_retrieval_icl import (
    DEFAULT_OUTPUT_PATH,
    IR_CONSENSUS_PATH,
    QA_CONSENSUS_PATH,
    build_examples_from_csv_paths,
    write_examples_json,
)
configure_project_logging()
logger = logging.getLogger(__name__)

CONFIG_PATH = DEFAULT_CONFIG_PATH


@click.group()
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
def cli(verbose: bool):
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)


@cli.command()
@click.option("--output", "-o", type=click.Path(), default="data/00_downloaded.json", help="Output path")
@click.option("--venue", default="ICLR", help="Venue to download from (e.g., ICLR, NeurIPS)")
@click.option("--year", default=None, type=int, help="Year to download (default: current year)")
@click.option("--forum-id", default=None, help="OpenReview forum id or comma-separated ids")
@click.option("--username", default=None, help="OpenReview username/email (overrides config)")
@click.option("--password", default=None, help="OpenReview password (overrides config)")
@click.option("--token", default=None, help="OpenReview API token (overrides config)")
@click.option("--max-papers", "limit", default=None, type=int, help="Maximum papers to download")
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
@click.option("--model", default=None, help="Model name (overrides config)")
def summarize(input_path: str, output: str, base_url: str, model: str):
    run_summarize_stage(
        input_path=Path(input_path),
        output_path=Path(output),
        config_path=CONFIG_PATH,
        base_url=base_url,
        model=model,
    )


@cli.command("dump-stage2-text-prompt")
@click.option("--input", "-i", "input_path", type=click.Path(exists=True), required=True, help="Input filtered dataset path")
@click.option("--output", "-o", type=click.Path(), required=True, help="Output text file path")
@click.option("--paper-id", default=None, help="Paper id to render; defaults to the first passed paper")
def dump_stage2_text_prompt(input_path: str, output: str, paper_id: str):
    output_path = write_stage2_text_prompt(
        filtered_input_path=Path(input_path),
        output_path=Path(output),
        paper_id=paper_id,
    )
    click.echo(str(output_path))


@cli.command("generate-queries")
@click.option("--input", "-i", "input_path", type=click.Path(exists=True), required=True, help="Input summarized dataset path")
@click.option("--output", "-o", type=click.Path(), default="data/03_queries.json", help="Output path")
@click.option("--base-url", default=None, help="LLM API base URL (overrides config)")
@click.option("--model", default=None, help="Model name (overrides config)")
def generate_queries(input_path: str, output: str, base_url: str, model: str):
    run_generate_queries_stage(
        input_path=Path(input_path),
        output_path=Path(output),
        config_path=CONFIG_PATH,
        base_url=base_url,
        model=model,
    )


@cli.command("init-golden-query-embeddings")
@click.option("--db-url", default=None, help="PostgreSQL SQLAlchemy URL (overrides config)")
@click.option("--embedding-dimension", default=1024, type=int, help="pgvector dimension")
def init_golden_query_embeddings(db_url: str, embedding_dimension: int):
    settings = resolve_generate_query_settings(CONFIG_PATH)
    engine = get_engine(db_url or str(settings["golden_embedding_db_url"]))
    ensure_schema(engine, embedding_dimension=embedding_dimension)
    click.echo("Initialized golden_query_embeddings schema.")


@cli.command("prepare-golden-retrieval-icl-examples")
@click.option("--ir-csv", type=click.Path(exists=True), default=str(IR_CONSENSUS_PATH), help="Final human consensus IR CSV")
@click.option("--qa-csv", type=click.Path(exists=True), default=str(QA_CONSENSUS_PATH), help="Final human consensus QA CSV")
@click.option("--output", type=click.Path(), default=str(DEFAULT_OUTPUT_PATH), help="Output normalized examples JSON")
@click.option("--resolve-web-titles", is_flag=True, help="Resolve missing QA titles from DOI/arXiv URLs")
@click.option("--generate-answer-tldr", is_flag=True, help="Use configured LLM to generate QA answer_tldr values")
@click.option("--base-url", default=None, help="LLM API base URL override for answer TLDR generation")
@click.option("--model", default=None, help="LLM model override for answer TLDR generation")
def prepare_golden_retrieval_icl_examples(
    ir_csv: str,
    qa_csv: str,
    output: str,
    resolve_web_titles: bool,
    generate_answer_tldr: bool,
    base_url: str,
    model: str,
):
    answer_tldr_generator = None
    if generate_answer_tldr:
        llm = build_llm_backend(CONFIG_PATH, base_url=base_url, model=model)

        def answer_tldr_generator(answer_text: str) -> str:
            return llm.generate(
                "Summarize this CrossValidated/StackExchange answer for retrieval-ICL.\n\n"
                "Return one concise paragraph, 2-4 sentences. Preserve the main claim, "
                "cited-paper use, and important experimental/method detail. Do not add facts.\n\n"
                f"Answer:\n{answer_text}"
            ).strip()

    examples, report = build_examples_from_csv_paths(
        [Path(ir_csv), Path(qa_csv)],
        resolve_web_titles=resolve_web_titles,
        answer_tldr_generator=answer_tldr_generator,
    )
    output_path = Path(output)
    write_examples_json(examples, output_path, report=report)
    click.echo(f"Wrote {len(examples)} retrieval-ICL examples to {output_path}.")
    click.echo(f"Wrote preparation report to {output_path.with_name(output_path.stem + '_report.json')}.")


@cli.command("hard-negative-mining")
@click.option("--input", "-i", "input_path", type=click.Path(exists=True), required=True, help="Input generated queries dataset path")
@click.option("--query-analysis-input", type=click.Path(exists=True), default=None, help="Optional stage-4 query analysis directory used to keep only surviving queries")
@click.option("--output", "-o", type=click.Path(), default="data/05_hard_negatives.json", help="Output path")
@click.option("--base-url", default=None, help="LLM API base URL (overrides config)")
@click.option("--model", default=None, help="Model name (overrides config)")
@click.option("--scholar-provider", default=None, help="Search backend: 'serpapi', 'scholarly', 'arxiv', or 'semantic_scholar' (overrides config)")
@click.option("--serpapi-api-key", default=None, help="SerpAPI key when using the 'serpapi' backend")
@click.option("--scholar-max-results", default=None, type=int, help="Maximum candidates to retrieve per query")
@click.option("--scholar-language", default=None, help="Google Scholar language code, e.g. 'en'")
@click.option("--mode", type=click.Choice(["direct", "queue"]), default="direct", show_default=True, help="Stage5 execution mode")
@click.option("--queue-table-name", default="", help="Optional PostgreSQL queue table override for queue mode")
@click.option("--scheduler-mode", type=click.Choice(["independent_worker", "batch_size"]), default="independent_worker", show_default=True, help="Queue review scheduling mode")
@click.option("--batch-size", default=20, type=int, show_default=True, help="Batch size when queue scheduler mode is batch_size")
@click.option("--review-max-workers", default=None, type=int, help="Override review worker count")
@click.option("--docling-pool-size", default=4, type=int, show_default=True, help="Docling parse concurrency for queue mode")
@click.option("--http-proxy", default="http://127.0.0.1:7890", show_default=True, help="HTTP proxy used by queue-mode PDF download")
@click.option("--https-proxy", default="http://127.0.0.1:7890", show_default=True, help="HTTPS proxy used by queue-mode PDF download")
@click.option("--pdf-output-dir", default=None, type=click.Path(), help="Optional PDF cache directory for queue mode")
@click.option("--monitor-file", default=None, type=click.Path(), help="Optional queue monitor JSON path")
@click.option("--monitor-interval", default=300, type=int, show_default=True, help="Queue monitor update interval in seconds")
@click.option("--retrieve-done-flag", default=None, type=click.Path(), help="Optional queue retrieve-done flag path")
@click.option("--retrieval-search-max-workers", default=1, type=int, show_default=True, help="Queue retrieval only: Semantic Scholar search concurrency across keys")
@click.option("--retrieval-rerank-max-workers", default=1, type=int, show_default=True, help="Queue retrieval only: DeepSeek rerank concurrency across queries")
@click.option("--parse-only", is_flag=True, help="Queue mode only: skip retrieval/review and only download+Docling-parse existing queue candidates")
@click.option("--review-only", is_flag=True, help="Queue mode only: skip retrieval and only review existing queue candidates")
@click.option("--download-only", is_flag=True, help="Queue mode only: only download PDFs for existing queue candidates")
@click.option("--parse-min-query-candidates", default=25, type=int, show_default=True, help="Queue parse-only: only parse queries with at least this many retrieved candidates")
@click.option("--paper-id", "paper_ids", multiple=True, help="Restrict queue/direct run to these paper ids; can be repeated")
@click.option("--skip-query-analysis-filter", is_flag=True, help="Do not apply stage-4 Keep filtering before Stage5")
@click.option("--retrieval-only", is_flag=True, help="Queue mode only: stop after Semantic Scholar retrieval + DeepSeek rerank and persist pending review rows")
@click.option("--task", default=None, help="Queue mode: only process rows with this task tag (triggers download→parse→review pipeline)")
@click.option("--download-workers", default=1, type=int, show_default=True, help="Task pipeline: download workers (Phase 1)")
@click.option("--parse-workers", default=72, type=int, show_default=True, help="Task pipeline: parse workers (Phase 2)")
@click.option("--review-workers", default=200, type=int, show_default=True, help="Task pipeline: review workers (Phase 3)")
def hard_negative_mining(
    input_path: str,
    query_analysis_input: str,
    output: str,
    base_url: str,
    model: str,
    scholar_provider: str,
    serpapi_api_key: str,
    scholar_max_results: int,
    scholar_language: str,
    mode: str,
    queue_table_name: str,
    scheduler_mode: str,
    batch_size: int,
    review_max_workers: int,
    docling_pool_size: int,
    http_proxy: str,
    https_proxy: str,
    pdf_output_dir: str,
    monitor_file: str,
    monitor_interval: int,
    retrieve_done_flag: str,
    retrieval_search_max_workers: int,
    retrieval_rerank_max_workers: int,
    parse_only: bool,
    review_only: bool,
    download_only: bool,
    parse_min_query_candidates: int,
    paper_ids: tuple[str, ...],
    skip_query_analysis_filter: bool,
    retrieval_only: bool,
    task: str,
    download_workers: int,
    parse_workers: int,
    review_workers: int,
):
    run_hard_negative_mining_stage(
        input_path=Path(input_path),
        output_path=Path(output),
        query_analysis_output_dir=Path(query_analysis_input) if query_analysis_input else None,
        config_path=CONFIG_PATH,
        base_url=base_url,
        model=model,
        scholar_provider=scholar_provider,
        serpapi_api_key=serpapi_api_key,
        scholar_max_results=scholar_max_results,
        scholar_language=scholar_language,
        mode=mode,
        queue_table_name=queue_table_name,
        scheduler_mode=scheduler_mode,
        batch_size=batch_size,
        review_max_workers=review_max_workers,
        docling_pool_size=docling_pool_size,
        http_proxy=http_proxy,
        https_proxy=https_proxy,
        pdf_output_dir=Path(pdf_output_dir) if pdf_output_dir else None,
        monitor_file=Path(monitor_file) if monitor_file else None,
        monitor_interval=monitor_interval,
        retrieve_done_flag=Path(retrieve_done_flag) if retrieve_done_flag else None,
        retrieval_search_max_workers=retrieval_search_max_workers,
        retrieval_rerank_max_workers=retrieval_rerank_max_workers,
        parse_only=parse_only,
        review_only=review_only,
        download_only=download_only,
        parse_min_query_candidates=parse_min_query_candidates,
        query_subset_paper_ids=list(paper_ids) if paper_ids else None,
        apply_query_analysis_filter=not skip_query_analysis_filter,
        retrieval_only=retrieval_only,
        task_filter=task if task else None,
        download_workers=download_workers,
        parse_workers=parse_workers,
        review_workers=review_workers,
    )


@cli.command("query-analysis")
@click.option("--summarized-input", type=click.Path(exists=True), required=True, help="Stage-2 summarized dataset path")
@click.option("--queries-input", type=click.Path(exists=True), required=True, help="Stage-3 generated queries dataset path")
@click.option("--downloaded-input", type=click.Path(exists=True), default=None, help="Optional stage-0 downloaded dataset path")
@click.option("--output-dir", "-o", type=click.Path(), default="data/04_query_analysis", help="Output directory")
@click.option("--base-url", default=None, help="LLM API base URL (overrides config)")
@click.option("--model", default=None, help="Model name (overrides config)")
@click.option(
    "--analysis",
    "analysis_modes",
    multiple=True,
    type=click.Choice(["retrieval", "style", "embedding"]),
    help="Stage-4 analysis blocks to run. Repeatable. Default: run all.",
)
def query_analysis(
    summarized_input: str,
    queries_input: str,
    downloaded_input: str,
    output_dir: str,
    base_url: str,
    model: str,
    analysis_modes: tuple[str, ...],
):
    run_query_analysis_stage(
        summarized_path=Path(summarized_input),
        queries_path=Path(queries_input),
        downloaded_path=Path(downloaded_input) if downloaded_input else None,
        output_dir=Path(output_dir),
        config_path=CONFIG_PATH,
        base_url=base_url,
        model=model,
        analysis_modes=list(analysis_modes) if analysis_modes else None,
    )


@cli.command("run-all")
@click.option("--output-dir", "-o", type=click.Path(), default="data", help="Output directory")
@click.option("--venue", default="ICLR", help="Venue to download from (ICLR, NeurIPS, ICML)")
@click.option("--year", default=None, type=int, help="Year to download (default: current year)")
@click.option("--forum-id", default=None, help="OpenReview forum id or comma-separated ids")
@click.option("--max-papers", default=30, type=int, help="Maximum papers to download")
@click.option("--summarize-limit", default=None, type=int, help="Maximum papers to summarize with the LLM")
@click.option("--base-url", default=None, help="LLM API base URL (overrides config)")
@click.option("--model", default=None, help="Model name (overrides config)")
@click.option("--skip-filter", is_flag=True, help="Mark all downloaded papers as passed instead of applying stage-1 filtering")
@click.option("--final-output", type=click.Path(), default=None, help="Final combined JSON path")
def run_all(
    output_dir: str,
    venue: str,
    year: int,
    forum_id: str,
    max_papers: int,
    summarize_limit: int,
    base_url: str,
    model: str,
    skip_filter: bool,
    final_output: str,
):
    paths = run_selected_stages(
        "0-5",
        output_dir=Path(output_dir),
        venue=venue,
        year=year,
        forum_id=forum_id,
        download_limit=max_papers,
        llm_limit=summarize_limit,
        config_path=CONFIG_PATH,
        base_url=base_url,
        model=model,
        skip_filter=skip_filter,
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
        queries_path=_resolve_queries_dataset_path(paths.output_dir, None),
        hard_negatives_path=paths.hard_negatives_path,
        query_analysis_output_dir=paths.query_analysis_output_dir,
    )
    write_pipeline_output(final_output_path, artifact)
    click.echo(str(final_output_path))


@cli.command("run-pipeline")
@click.option("--stages", default="0-4", help="Stage range/spec to run. Default: 0-4")
@click.option("--output-dir", "-o", type=click.Path(), default="data", help="Output directory")
@click.option("--venue", default="ICLR", help="Venue to download from (ICLR, NeurIPS, ICML)")
@click.option("--year", default=None, type=int, help="Year to download (default: current year)")
@click.option("--forum-id", default=None, help="OpenReview forum id or comma-separated ids")
@click.option("--max-papers", default=30, type=int, help="Maximum papers to download")
@click.option("--summarize-limit", default=None, type=int, help="Maximum papers to summarize with the LLM")
@click.option("--base-url", default=None, help="LLM API base URL (overrides config)")
@click.option("--model", default=None, help="Model name (overrides config)")
@click.option("--skip-filter", is_flag=True, help="Mark all downloaded papers as passed instead of applying stage-1 filtering")
@click.option("--final-output", type=click.Path(), default=None, help="Final combined JSON path")
@click.option("--downloaded-input", type=click.Path(exists=True), default=None, help="Reuse existing 00_downloaded.json (skip stage-0 download)")
@click.option("--input-path", type=click.Path(exists=True), default=None, help="Input path for the first selected stage (overrides --downloaded-input as input)")
@click.option("--sync-gradio", is_flag=True, help="Sync final output to Gradio display")
@click.option(
    "--query-analysis",
    "query_analysis_modes",
    multiple=True,
    type=click.Choice(["retrieval", "style", "embedding"]),
    help="Stage-4 analysis blocks to run when query analysis is selected. Repeatable. Default: run all.",
)
def run_pipeline(
    stages: str,
    output_dir: str,
    venue: str,
    year: int,
    forum_id: str,
    max_papers: int,
    summarize_limit: int,
    base_url: str,
    model: str,
    skip_filter: bool,
    final_output: str,
    downloaded_input: str,
    input_path: str,
    sync_gradio: bool,
    query_analysis_modes: tuple[str, ...],
):
    stage_input = Path(input_path) if input_path else (Path(downloaded_input) if downloaded_input else None)
    paths = run_selected_stages(
        stages,
        input_path=stage_input,
        output_dir=Path(output_dir),
        venue=venue,
        year=year,
        forum_id=forum_id,
        download_limit=max_papers,
        llm_limit=summarize_limit,
        config_path=CONFIG_PATH,
        base_url=base_url,
        model=model,
        query_analysis_modes=list(query_analysis_modes) if query_analysis_modes else None,
        skip_filter=skip_filter,
        downloaded_path=Path(downloaded_input) if downloaded_input else None,
    )

    # Generate simplified queries JSON right after stage 3
    if paths.queries_path and paths.queries_path.exists():
        _simplify_queries_json(paths.output_dir / "final_pipeline_output.json")

    final_output_path = (
        Path(final_output).expanduser().resolve()
        if final_output
        else paths.output_dir / "final_pipeline_output.json"
    )
    final_output_path.parent.mkdir(parents=True, exist_ok=True)
    artifact = build_pipeline_output(
        downloaded_path=paths.downloaded_path,
        filtered_path=paths.filtered_path,
        summarized_path=paths.summarized_path,
        queries_path=_resolve_queries_dataset_path(paths.output_dir, None),
        hard_negatives_path=paths.hard_negatives_path,
        query_analysis_output_dir=paths.query_analysis_output_dir,
    )
    write_pipeline_output(final_output_path, artifact)
    click.echo(str(final_output_path))

    _generate_markdown_report(final_output_path)
    click.echo(str(final_output_path.parent / "query_bullet_original_comments.md"))

    _simplify_queries_json(final_output_path)
    click.echo(str(final_output_path.parent / "03_queries_simplified.json"))

    if sync_gradio:
        _sync_to_gradio(final_output_path.parent, final_output_path)


def _generate_markdown_report(final_json_path: Path) -> None:
    try:
        from scripts.render_query_bullet_comments_md import render_report, load_pipeline
        output = final_json_path.parent / "query_bullet_original_comments.md"
        data = load_pipeline(final_json_path)
        output.write_text(render_report(data, max_comment_chars=1800), encoding="utf-8")
    except Exception as exc:
        click.echo(f"Markdown report generation skipped: {exc}", err=True)


def _simplify_queries_json(final_json_path: Path) -> None:
    try:
        from scripts.simplify_queries import simplify
        queries_path = final_json_path.parent / "03_queries.json"
        output = final_json_path.parent / "03_queries_simplified.json"
        if queries_path.exists():
            simplify(str(queries_path), str(output))
    except Exception as exc:
        click.echo(f"Simplified queries generation skipped: {exc}", err=True)


GRADIO_OUTPUT_DIR = Path("outputs")


def _sync_to_gradio(output_dir: Path, final_json_path: Path) -> None:
    try:
        import shutil
        target_json = GRADIO_OUTPUT_DIR / "final_pipeline_output.json"
        target_analysis = GRADIO_OUTPUT_DIR / "04_query_analysis"
        target_md = GRADIO_OUTPUT_DIR / "query_bullet_original_comments.md"
        shutil.copy2(final_json_path, target_json)
        if target_analysis.exists():
            shutil.rmtree(target_analysis)
        analysis_src = output_dir / "04_query_analysis"
        if analysis_src.exists():
            shutil.copytree(analysis_src, target_analysis)
        md_src = output_dir / "query_bullet_original_comments.md"
        if md_src.exists():
            shutil.copy2(md_src, target_md)
        click.echo(f"Synced to Gradio: {target_json}")
    except Exception as exc:
        click.echo(f"Gradio sync skipped: {exc}", err=True)


def _resolve_queries_dataset_path(base: Path, override: Optional[str]) -> Path:
    if override:
        return Path(override).expanduser().resolve()
    paths = resolve_pipeline_paths(output_dir=base)
    if (
        paths.decontextualized_queries_path.is_file()
        and (
            not paths.queries_path.is_file()
            or paths.decontextualized_queries_path.stat().st_mtime >= paths.queries_path.stat().st_mtime
        )
    ):
        return paths.decontextualized_queries_path
    return paths.queries_path


@cli.command("update-final-json")
@click.option("--base-dir", type=click.Path(exists=True), required=True, help="Directory containing stage output files")
@click.option("--output", "-o", type=click.Path(), default=None, help="Output path for final combined JSON")
@click.option("--downloaded-path", type=click.Path(exists=True), default=None, help="Optional override for 00_downloaded.json")
@click.option("--filtered-path", type=click.Path(exists=True), default=None, help="Optional override for 01_filtered.json")
@click.option("--summarized-path", type=click.Path(exists=True), default=None, help="Optional override for 02_summarized.json")
@click.option("--queries-path", type=click.Path(exists=True), default=None, help="Optional override for 03_queries.json")
@click.option("--hard-negatives-path", type=click.Path(exists=True), default=None, help="Optional override for 05_hard_negatives.json")
@click.option("--query-analysis-dir", type=click.Path(exists=True), default=None, help="Optional override for 04_query_analysis directory")
@click.option("--sync-gradio", is_flag=True, help="Sync final output to Gradio display")
def update_final_json(
    base_dir: str,
    output: str,
    downloaded_path: str,
    filtered_path: str,
    summarized_path: str,
    queries_path: str,
    hard_negatives_path: str,
    query_analysis_dir: str,
    sync_gradio: bool,
):
    base = Path(base_dir).expanduser().resolve()
    final_output_path = Path(output).expanduser().resolve() if output else (base / "final_pipeline_output.json")
    final_output_path.parent.mkdir(parents=True, exist_ok=True)

    artifact = build_pipeline_output(
        downloaded_path=Path(downloaded_path).expanduser().resolve() if downloaded_path else (base / "00_downloaded.json"),
        filtered_path=Path(filtered_path).expanduser().resolve() if filtered_path else (base / "01_filtered.json"),
        summarized_path=Path(summarized_path).expanduser().resolve() if summarized_path else (base / "02_summarized.json"),
        queries_path=_resolve_queries_dataset_path(base, queries_path),
        hard_negatives_path=Path(hard_negatives_path).expanduser().resolve() if hard_negatives_path else (base / "05_hard_negatives.json"),
        query_analysis_output_dir=Path(query_analysis_dir).expanduser().resolve() if query_analysis_dir else (base / "04_query_analysis"),
    )
    write_pipeline_output(final_output_path, artifact)
    click.echo(str(final_output_path))

    _generate_markdown_report(final_output_path)
    click.echo(str(final_output_path.parent / "query_bullet_original_comments.md"))

    _simplify_queries_json(final_output_path)
    click.echo(str(final_output_path.parent / "03_queries_simplified.json"))

    if sync_gradio:
        _sync_to_gradio(final_output_path.parent, final_output_path)


def main():
    cli()


if __name__ == "__main__":
    main()
