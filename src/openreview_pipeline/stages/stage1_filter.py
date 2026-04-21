import logging
import re
from pathlib import Path
from typing import Dict, Optional, List

import yaml

from openreview_pipeline.schemas import DownloadedPapersDataset
from openreview_pipeline.schemas.schemas_filter import FilteredPapersDataset, FilterResult, FilterRuleResult
from openreview_pipeline.utils import load_json, save_json

logger = logging.getLogger(__name__)

DEFAULT_RULES_PATH = Path(__file__).parent.parent.parent.parent / "configs" / "rules.yaml"


class RuleConfig:
    def __init__(self, config_path: Optional[Path] = None):
        self.config_path = config_path or DEFAULT_RULES_PATH
        self._config = self._load_config()

    def _load_config(self) -> Dict:
        if not self.config_path.exists():
            logger.warning(f"Config file not found: {self.config_path}, using defaults")
            return self._default_config()
        with open(self.config_path) as f:
            return yaml.safe_load(f)

    def _default_config(self) -> Dict:
        return {
            "accepted": {
                "accept_keywords": ["accept", "accepted", "oral", "poster", "spotlight", "strong accept"],
                "reject_keywords": ["reject", "rejected", "strong reject", "not accept"],
            },
            "similar_paper": {
                "citation_threshold": 0.05,
                "max_citations_per_word": 0.05,
                "citation_patterns": ["et al", "arxiv", "paper~", "proceedings"],
            },
            "multimodal_info": {
                "min_multimodal_mentions": 2,
                "multimodal_keywords": ["figure", "fig", "table", "diagram", "chart", "equation", "formula", "plot"],
                "content_sources": ["abstract", "review", "rebuttal", "comments"],
            },
        }

    @property
    def accept_keywords(self) -> List[str]:
        return self._config.get("accepted", {}).get("accept_keywords", [])

    @property
    def reject_keywords(self) -> List[str]:
        return self._config.get("accepted", {}).get("reject_keywords", [])

    @property
    def citation_threshold(self) -> float:
        return self._config.get("similar_paper", {}).get("citation_threshold", 0.05)

    @property
    def citation_patterns(self) -> List[str]:
        return self._config.get("similar_paper", {}).get("citation_patterns", [])

    @property
    def min_multimodal_mentions(self) -> int:
        return self._config.get("multimodal_info", {}).get("min_multimodal_mentions", 2)

    @property
    def multimodal_keywords(self) -> List[str]:
        return self._config.get("multimodal_info", {}).get("multimodal_keywords", [])

    @property
    def content_sources(self) -> List[str]:
        return self._config.get("multimodal_info", {}).get("content_sources", [])


class RuleBasedFilter:
    def __init__(self, config_path: Optional[Path] = None, rules_config: Optional[RuleConfig] = None, limit: Optional[int] = None):
        self.config = rules_config or RuleConfig(config_path)
        self.limit = limit

    def check_accepted(self, paper: Dict) -> bool:
        decision = paper.get("decision")
        if decision:
            decision_content = decision.get("content", {})
            decision_value = decision_content.get("decision", "")
            if isinstance(decision_value, dict):
                decision_value = decision_value.get("value", "")
            decision_value = str(decision_value).lower()

            has_accept = any(kw in decision_value for kw in self.config.accept_keywords)
            has_reject = any(kw in decision_value for kw in self.config.reject_keywords)

            if has_reject and not has_accept:
                return False
            return has_accept

        title = paper.get("paper", {}).get("title", "").lower()
        abstract = paper.get("paper", {}).get("abstract", "").lower()
        venue = paper.get("paper", {}).get("venue", "").lower()
        text = f"{title} {abstract} {venue}"

        has_accept = any(kw in text for kw in self.config.accept_keywords)
        has_reject = any(kw in text for kw in self.config.reject_keywords)

        if has_reject and not has_accept:
            return False
        return has_accept

    def check_similar_paper(self, paper: Dict) -> bool:
        abstract = paper.get("paper", {}).get("abstract", "").lower()
        title = paper.get("paper", {}).get("title", "").lower()
        text = f"{title} {abstract}"

        word_count = len(text.split())
        if word_count == 0:
            return False

        citation_count = 0
        for pattern in self.config.citation_patterns:
            citation_count += len(re.findall(re.escape(pattern), text))

        citation_ratio = citation_count / word_count
        return citation_ratio >= self.config.citation_threshold

    def check_multimodal_info(self, paper: Dict) -> bool:
        content_sources = self.config.content_sources
        multimodal_keywords = self.config.multimodal_keywords

        content_parts = []
        for source in content_sources:
            if source in ["abstract", "title"]:
                content_parts.append(paper.get("paper", {}).get(source, "").lower())
            elif source in paper:
                content_parts.append(str(paper.get(source, "")).lower())
            elif source in paper.get("paper", {}):
                content_parts.append(paper.get("paper", {}).get(source, "").lower())

        combined_content = " ".join(content_parts)

        mention_count = 0
        for keyword in multimodal_keywords:
            mention_count += len(re.findall(r'\b' + re.escape(keyword) + r'\b', combined_content))

        return mention_count >= self.config.min_multimodal_mentions

    def apply(self, dataset: DownloadedPapersDataset) -> FilteredPapersDataset:
        total = dataset.total_count
        limit = self.limit if self.limit else total
        logger.info(f"Applying filter to {min(limit, total)} of {total} papers (limit={self.limit})")

        results = []
        passed_count = 0

        for paper_meta in dataset.papers[:limit]:
            try:
                paper_dict = paper_meta.model_dump()

                accepted = self.check_accepted(paper_dict)
                similar_paper = self.check_similar_paper(paper_dict)
                multimodal_info = self.check_multimodal_info(paper_dict)

                passed = accepted and not similar_paper and multimodal_info

                if passed:
                    passed_count += 1

                results.append(
                    FilterResult(
                        paper=paper_meta,
                        passed=passed,
                        details=FilterRuleResult(
                            accepted=accepted,
                            similar_paper=similar_paper,
                            multimodal_info=multimodal_info,
                        ),
                    )
                )
            except Exception as e:
                logger.error(f"Error filtering paper {paper_meta.paper.id}: {e}")
                import traceback
                logger.error(f"Stack trace: {traceback.format_exc()}")
                results.append(
                    FilterResult(
                        paper=paper_meta,
                        passed=False,
                        details=FilterRuleResult(
                            accepted=False,
                            similar_paper=False,
                            multimodal_info=False,
                        ),
                    )
                )

        logger.info(f"Filter complete: {passed_count}/{len(results)} passed")

        return FilteredPapersDataset(
            results=results,
            total_input=len(results),
            total_passed=passed_count,
            total_filtered=len(results) - passed_count,
        )

    def run(self, input_path: Path, output_path: Path) -> None:
        logger.info(f"Running filter stage: {input_path} -> {output_path}")
        dataset = load_json(input_path, DownloadedPapersDataset)
        result = self.apply(dataset)
        save_json(output_path, result)
