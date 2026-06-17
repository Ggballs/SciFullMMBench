from __future__ import annotations

import argparse
import gc
import logging
from typing import Optional

import torch
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmbedRequest(BaseModel):
    texts: list[str] = Field(default_factory=list)


class EmbedResponse(BaseModel):
    embeddings: list[list[float]]


# ---- Model registry ---------------------------------------------------------

MODEL_REGISTRY = {
    "qwen3-embed-8b": {
        "path": "/data3/weiyiyang/model_cache/Qwen3-Embedding-8B",
        "display_name": "Qwen3-Embedding-8B",
    },
    "instructor-xl": {
        "path": "/data3/weiyiyang/model_cache/instructor-xl",
        "display_name": "Instructor-XL",
    },
    "gritlm-7b": {
        "path": "/data3/weiyiyang/model_cache/GritLM-7B",
        "display_name": "GritLM-7B",
    },
    "specter2": {
        "path": "/data3/weiyiyang/model_cache/specter2_base",
        "display_name": "SPECTER2",
    },
    "scincl": {
        "path": "/data3/weiyiyang/model_cache/scincl",
        "display_name": "SCiNCL",
    },
}


# ---- Embedders --------------------------------------------------------------

class Specter2Embedder:
    """SPECTER2 uses vanilla BERT — load with AutoModel + mean pooling."""

    def __init__(self, model_path: str, device: str):
        from transformers import AutoModel, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(model_path)
        self._model = AutoModel.from_pretrained(model_path).to(device)
        self._model.eval()
        self._device = device

    @torch.no_grad()
    def encode(self, texts: list[str], batch_size: int = 64) -> list[list[float]]:
        all_embeddings: list[list[float]] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            inputs = self._tokenizer(
                batch, padding=True, truncation=True,
                max_length=self._tokenizer.model_max_length,
                return_tensors="pt"
            ).to(self._device)
            outputs = self._model(**inputs)
            # mean pooling over token dimension (excluding padding)
            attention_mask = inputs["attention_mask"]
            hidden = outputs.last_hidden_state
            mask_expanded = attention_mask.unsqueeze(-1).expand(hidden.size()).float()
            summed = (hidden * mask_expanded).sum(dim=1)
            counts = mask_expanded.sum(dim=1).clamp(min=1e-9)
            pooled = summed / counts
            all_embeddings.extend(
                [list(map(float, vec)) for vec in pooled.cpu().numpy()]
            )
        return all_embeddings


class SentenceTransformerEmbedder:
    def __init__(self, model_path: str, device: str):
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(
            model_path, device=device, trust_remote_code=True
        )

    def encode(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = self._model.encode(texts, show_progress_bar=False)
        return [list(map(float, vec)) for vec in vectors]


def _patch_for_gritlm():
    """Monkey-patch for GritLM-7B compatibility with transformers >= 5.x."""
    from transformers import MistralConfig
    from transformers.cache_utils import DynamicCache

    # 1. MistralConfig: accept rope_theta (removed in transformers 5.x)
    if not getattr(MistralConfig, "_rope_theta_patched", False):
        _orig_init = MistralConfig.__init__

        def _patched_init(self, **kwargs):
            rope_theta = kwargs.pop("rope_theta", 10000.0)
            _orig_init(self, **kwargs)
            self.rope_theta = rope_theta

        MistralConfig.__init__ = _patched_init
        MistralConfig._rope_theta_patched = True

    # 2. DynamicCache: add from_legacy_cache (removed in transformers 5.x)
    if not hasattr(DynamicCache, "from_legacy_cache"):

        @staticmethod
        def _from_legacy_cache(past_key_values):
            cache = DynamicCache()
            if past_key_values is not None:
                for layer_idx, layer_states in enumerate(past_key_values):
                    cache.update(layer_states[0], layer_states[1], layer_idx)
            return cache

        DynamicCache.from_legacy_cache = _from_legacy_cache

    # 3. DynamicCache: add get_usable_length (renamed to get_seq_length in transformers 5.x)
    if not hasattr(DynamicCache, "get_usable_length"):

        def _get_usable_length(self, seq_length, layer_idx=0):
            return min(seq_length, self.get_seq_length(layer_idx))

        DynamicCache.get_usable_length = _get_usable_length

    # 4. DynamicCache: add to_legacy_cache (removed in transformers 5.x)
    if not hasattr(DynamicCache, "to_legacy_cache"):

        def _to_legacy_cache(self):
            return tuple(
                (self.key_cache[i], self.value_cache[i])
                for i in range(len(self.key_cache))
            )

        DynamicCache.to_legacy_cache = _to_legacy_cache


def _build_embedder(model_name: str, device: str):
    entry = MODEL_REGISTRY[model_name]
    path = entry["path"]
    if model_name == "specter2":
        return Specter2Embedder(path, device)
    if model_name == "gritlm-7b":
        _patch_for_gritlm()
    return SentenceTransformerEmbedder(path, device)


# ---- FastAPI app ------------------------------------------------------------

def build_app(device: str):
    from fastapi import FastAPI

    app = FastAPI(title="SciFullMMBench Multi-Model Embedding Service")
    embedders: dict[str, object] = {}
    current_model_name: Optional[str] = None

    def _release_other_models(keep_model: str) -> None:
        nonlocal current_model_name
        stale_names = [name for name in embedders.keys() if name != keep_model]
        if not stale_names:
            return
        for name in stale_names:
            logger.info("Unloading model '%s' to free memory before '%s'.", name, keep_model)
            embedders.pop(name, None)
        gc.collect()
        if device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()
        current_model_name = keep_model if keep_model in embedders else None

    @app.get("/health")
    def health() -> dict:
        return {
            "status": "ok",
            "device": device,
            "loaded_models": list(embedders.keys()),
            "current_model": current_model_name,
            "available_models": list(MODEL_REGISTRY.keys()),
        }

    @app.post("/embed/{model_name}", response_model=EmbedResponse)
    def embed(model_name: str, payload: EmbedRequest) -> EmbedResponse:
        nonlocal current_model_name
        if model_name not in MODEL_REGISTRY:
            return EmbedResponse(embeddings=[])  # let caller handle 404-like case
        if not payload.texts:
            return EmbedResponse(embeddings=[])

        if model_name not in embedders:
            _release_other_models(keep_model=model_name)
            logger.info("Loading model '%s' on %s ...", model_name, device)
            embedders[model_name] = _build_embedder(model_name, device)
            current_model_name = model_name
            logger.info("Model '%s' loaded.", model_name)

        encoder = embedders[model_name]
        vectors = encoder.encode(payload.texts)
        return EmbedResponse(embeddings=vectors)

    return app


# ---- CLI --------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Serve multiple embedding models over HTTP."
    )
    parser.add_argument("--device", default="cuda:3")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=18081)
    return parser


def main() -> int:
    import uvicorn

    args = build_parser().parse_args()
    app = build_app(device=str(args.device))
    uvicorn.run(app, host=str(args.host), port=int(args.port))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
