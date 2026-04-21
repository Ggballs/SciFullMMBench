import sys
sys.path.insert(0, 'src')

import yaml
import json
import logging
from pathlib import Path

from openreview_pipeline.llm import OpenAICompatibleBackend
from openreview_pipeline.stages.stage3_generate_queries import QueryGenerator

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

with open("config.yaml") as f:
    config = yaml.safe_load(f)

llm_config = config.get("llm", {})
backend = OpenAICompatibleBackend(
    base_url=llm_config["base_url"],
    api_token=llm_config["api_token"],
    model=llm_config["model"],
)

summarized_path = Path("data/iclr_2026/02_summarized_single.json")
queries_path = Path("data/iclr_2026/03_queries_single_new.json")

logger.info("=== Stage 3: Generating queries with new prompt ===")
generator = QueryGenerator(llm=backend)
generator.run(summarized_path, queries_path)

with open(queries_path) as f:
    queries = json.load(f)
logger.info(f"Generated {queries['total_queries']} queries")
logger.info(f"Saved to {queries_path}")