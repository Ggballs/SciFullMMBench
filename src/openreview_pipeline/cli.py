import logging
import json
from pathlib import Path

import click

from openreview_pipeline.app_logging import configure_project_logging
from openreview_pipeline.pipeline_output_builder import build_pipeline_output, write_pipeline_output
from openreview_pipeline.runner import (
    build_llm_backend,
    resolve_generate_query_settings,
    run_download_stage,
    run_filter_stage,
    run_generate_queries_stage,
    run_hard_negative_mining_stage,
    run_query_analysis_stage,
    run_selected_stages,
    run_summarize_stage,
)
from openreview_pipeline.utils.db.golden_query_embeddings import (
    GoldenQueryEmbeddingRow,
    ensure_schema,
    get_engine,
    upsert_golden_query_embeddings,
)
from openreview_pipeline.utils.golden_retrieval_icl import (
    DEFAULT_OUTPUT_PATH,
    IR_CONSENSUS_PATH,
    QA_CONSENSUS_PATH,
    build_examples_from_csv_paths,
    write_examples_json,
)
from openreview_pipeline.utils.embeddings import BGEM3Embedder

configure_project_logging()
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


@cli.command("import-golden-query-embeddings")
@click.option("--db-url", default=None, help="PostgreSQL SQLAlchemy URL (overrides config)")
@click.option("--golden-classifications-path", type=click.Path(exists=True), default=None, help="Normalized retrieval-ICL examples JSON")
@click.option("--bge-model-path", default=None, help="BGE-M3 model path")
@click.option("--bge-device", default=None, help="BGE device")
@click.option("--embedding-dimension", default=1024, type=int, help="pgvector dimension")
def import_golden_query_embeddings(
    db_url: str,
    golden_classifications_path: str,
    bge_model_path: str,
    bge_device: str,
    embedding_dimension: int,
):
    settings = resolve_generate_query_settings(CONFIG_PATH)
    path = Path(golden_classifications_path or str(settings["golden_classifications_path"]))
    raw_rows = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    eligible = [row for row in raw_rows if str(row.get("indexing_content") or "").strip()]

    embedder = BGEM3Embedder(
        model_path=str(bge_model_path or settings["bge_model_path"]),
        device=str(bge_device or settings["bge_device"]),
    )
    vectors = embedder.embed_texts([str(row.get("indexing_content") or "") for row in eligible])
    rows = [
        GoldenQueryEmbeddingRow(
            example_id=str(row.get("example_id") or ""),
            query_id=str(row.get("query_id") or ""),
            query_type=str(row.get("query_type") or "").strip().upper(),
            view_label=str(row.get("view_label") or "").strip(),
            query_text=str(row.get("query") or "").strip(),
            target_papers=[str(title) for title in row.get("target_papers", [])],
            answer_original_content=str(row.get("answer_original_content") or ""),
            answer_tldr=str(row.get("answer_tldr") or ""),
            human_view_note=str(row.get("human_view_note") or ""),
            indexing_content=str(row.get("indexing_content") or "").strip(),
            retrieval_content=str(row.get("retrieval_content") or "").strip(),
            embedding=embedding,
        )
        for row, embedding in zip(eligible, vectors)
    ]
    engine = get_engine(db_url or str(settings["golden_embedding_db_url"]))
    ensure_schema(engine, embedding_dimension=embedding_dimension)
    count = upsert_golden_query_embeddings(engine, rows)
    click.echo(f"Imported {count} golden query embedding rows.")


@cli.command("hard-negative-mining")
@click.option("--input", "-i", "input_path", type=click.Path(exists=True), required=True, help="Input generated queries dataset path")
@click.option("--query-analysis-input", type=click.Path(exists=True), default=None, help="Optional stage-4 query analysis directory used to keep only surviving queries")
@click.option("--output", "-o", type=click.Path(), default="data/05_hard_negatives.json", help="Output path")
@click.option("--base-url", default=None, help="LLM API base URL (overrides config)")
@click.option("--model", default=None, help="Model name (overrides config)")
@click.option("--scholar-provider", default=None, help="Google Scholar backend: 'serpapi' or 'scholarly' (overrides config)")
@click.option("--serpapi-api-key", default=None, help="SerpAPI key for real Google Scholar search (overrides config)")
@click.option("--scholar-max-results", default=None, type=int, help="Maximum Google Scholar candidates to retrieve per query")
@click.option("--scholar-language", default=None, help="Google Scholar language code, e.g. 'en'")
@click.option("--download-selected-pdfs", is_flag=True, help="Download selected hard-negative/positive PDFs instead of storing URL-only metadata")
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
    download_selected_pdfs: bool,
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
        download_selected_pdfs=download_selected_pdfs,
    )


@cli.command("query-analysis")
@click.option("--summarized-input", type=click.Path(exists=True), required=True, help="Stage-2 summarized dataset path")
@click.option("--queries-input", type=click.Path(exists=True), required=True, help="Stage-3 generated queries dataset path")
@click.option("--downloaded-input", type=click.Path(exists=True), default=None, help="Optional stage-0 downloaded dataset path")
@click.option("--output-dir", "-o", type=click.Path(), default="data/04_query_analysis", help="Output directory")
@click.option("--base-url", default=None, help="LLM API base URL (overrides config)")
@click.option("--model", default=None, help="Model name (overrides config)")
def query_analysis(
    summarized_input: str,
    queries_input: str,
    downloaded_input: str,
    output_dir: str,
    base_url: str,
    model: str,
):
    run_query_analysis_stage(
        summarized_path=Path(summarized_input),
        queries_path=Path(queries_input),
        downloaded_path=Path(downloaded_input) if downloaded_input else None,
        output_dir=Path(output_dir),
        config_path=CONFIG_PATH,
        base_url=base_url,
        model=model,
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
@click.option("--download-selected-pdfs", is_flag=True, help="Download selected hard-negative/positive PDFs instead of storing URL-only metadata")
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
    download_selected_pdfs: bool,
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
        download_selected_pdfs=download_selected_pdfs,
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
