"""Serve a single multimodal embedding model over HTTP.

Usage:
  python serve_multimodal_embedding.py --model qwen3-vl-embed-8b --device cuda:0 --port 18083
  python serve_multimodal_embedding.py --model openclip-vit-g-14   --device cuda:0 --port 18084
  python serve_multimodal_embedding.py --model vlm2vec-qwen2vl-7b  --device cuda:0 --port 18085
  python serve_multimodal_embedding.py --model ops-mm-embed-7b     --device cuda:0 --port 18086

Each instance loads ONE model. Run 4 instances on different GPUs/ports.

POST /embed  body: {"texts": [...], "images": [...]}
  texts[i] and images[i] are paired (either or both can be provided).
  images entries can be local file paths or base64 data URIs.
"""
from __future__ import annotations

import argparse
import base64
import logging
import os
import tempfile
from pathlib import Path
from typing import Optional

import torch
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_CHOICES = [
    "qwen3-vl-embed-8b",
    "openclip-vit-g-14",
    "vlm2vec-qwen2vl-7b",
    "ops-mm-embed-7b",
]


class EmbedRequest(BaseModel):
    texts: list[str] = Field(default_factory=list)
    images: list[str] = Field(default_factory=list,
                              description="File paths or base64 data URIs per item")


class EmbedResponse(BaseModel):
    embeddings: list[list[float]]


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

def _resolve_images(image_specs: list[str]) -> list[str]:
    """Convert base64 data URIs to temp file paths; pass local paths through."""
    resolved: list[str] = []
    for spec in image_specs:
        if spec.startswith("data:") and ";base64," in spec:
            _, b64 = spec.split(";base64,", 1)
            fd, path = tempfile.mkstemp(suffix=".png", prefix="mmembed_")
            os.close(fd)
            Path(path).write_bytes(base64.b64decode(b64))
            resolved.append(path)
        else:
            resolved.append(spec)
    return resolved


# ---------------------------------------------------------------------------
# Embedders
# ---------------------------------------------------------------------------

class Qwen3VLEmbedder:
    """Qwen3-VL-Embedding via SentenceTransformer — needs transformers >= 5.x for qwen3_vl support.

    Uses .conda-bge transformers 5.8.1 (NOT the /tmp/txfm452 override).
    """

    def __init__(self, model_path: str, device: str):
        # Ensure .conda-bge transformers (5.8.1) is used, not the 4.52.3 override
        import sys
        _stale = "/tmp/txfm452"
        if _stale in sys.path:
            sys.path.remove(_stale)
        from sentence_transformers import SentenceTransformer

        logger.info("Loading Qwen3-VL-Embedding from %s on %s ...", model_path, device)
        self._model = SentenceTransformer(model_path, device=device, trust_remote_code=True)
        self._dim = 4096
        logger.info("Loaded. Embedding dim: %d", self._dim)

    @property
    def dim(self) -> int:
        return self._dim

    def encode(self, texts: list[str], images: list[str]) -> list[list[float]]:
        n = max(len(texts), len(images))
        if n == 0:
            return []
        inputs = []
        for i in range(n):
            item: dict[str, object] = {}
            if i < len(texts) and texts[i]:
                item["text"] = texts[i]
            if i < len(images) and images[i]:
                item["image"] = images[i]
            inputs.append(item)
        vectors = self._model.encode(inputs, show_progress_bar=False)
        return [list(map(float, v)) for v in vectors]


class OpenCLIPEmbedder:
    """OpenCLIP ViT-G/14 — separate text/image encoders, L2 normalized."""

    def __init__(self, model_path: str, device: str):
        import os
        os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
        import open_clip

        self._device = device
        model_key = "ViT-g-14"
        pretrained = "laion2b_s34b_b88k"
        logger.info("Loading OpenCLIP %s (%s) on %s ...", model_key, pretrained, device)
        self._model, _, self._preprocess = open_clip.create_model_and_transforms(
            model_key, pretrained=pretrained, cache_dir="/tmp/weiyiyang/model_cache",
        )
        self._model = self._model.to(device)
        self._model.eval()
        self._tokenizer = open_clip.get_tokenizer(model_key)
        self._dim = 1024
        logger.info("Loaded. Embedding dim: %d", self._dim)

    @property
    def dim(self) -> int:
        return self._dim

    def encode(self, texts: list[str], images: list[str]) -> list[list[float]]:
        n = max(len(texts), len(images))
        if n == 0:
            return []

        emb_texts: list[torch.Tensor] = []
        emb_images: list[torch.Tensor] = []

        # Batch text encoding
        valid_texts = [(i, t) for i, t in enumerate(texts) if t]
        if valid_texts:
            indices, batch = zip(*valid_texts)
            tokens = self._tokenizer(list(batch)).to(self._device)
            with torch.no_grad():
                feats = self._model.encode_text(tokens)
                feats = feats / feats.norm(dim=-1, keepdim=True)
            emb_texts = list(zip(indices, feats))

        # Batch image encoding
        valid_images = [(i, p) for i, p in enumerate(images) if p]
        if valid_images:
            from PIL import Image
            indices, paths = zip(*valid_images)
            imgs = torch.stack([
                self._preprocess(Image.open(p).convert("RGB")).to(self._device)
                for p in paths
            ])
            with torch.no_grad():
                ifeats = self._model.encode_image(imgs)
                ifeats = ifeats / ifeats.norm(dim=-1, keepdim=True)
            emb_images = list(zip(indices, ifeats))

        # Merge: if both text+image, concatenate; else use whatever exists
        results: list[list[float]] = []
        for i in range(n):
            t = next((f for idx, f in emb_texts if idx == i), None)
            img = next((f for idx, f in emb_images if idx == i), None)
            if t is not None and img is not None:
                fused = torch.cat([t, img])  # 2048-dim
            elif t is not None:
                fused = t
            elif img is not None:
                fused = img
            else:
                fused = torch.zeros(self._dim, device=self._device)
            # Re-normalize concatenated vector
            fused = fused / fused.norm()
            results.append(list(map(float, fused.cpu().numpy())))
        return results


class VLM2VecEmbedder:
    """VLM2Vec-Qwen2VL-7B — needs VLM2Vec repo cloned to /tmp/weiyiyang/VLM2Vec."""

    def __init__(self, model_path: str, device: str, base_model_path: str = "/tmp/weiyiyang/model_cache/Qwen2-VL-7B-Instruct"):
        import sys
        # Use transformers 4.52.3 (VLM2Vec requirements.txt pin)
        _txfm_path = "/tmp/txfm452"
        if _txfm_path not in sys.path:
            sys.path.insert(0, _txfm_path)
        # Flash-attn stub
        if "/tmp" not in sys.path:
            sys.path.insert(0, "/tmp")
        _vlm2vec_root = "/tmp/weiyiyang/VLM2Vec"
        if _vlm2vec_root not in sys.path:
            sys.path.insert(0, _vlm2vec_root)

        # HybridCache stub for transformers >= 5.x compat
        from transformers.cache_utils import Cache
        if not hasattr(__import__("transformers.cache_utils"), "HybridCache"):
            class _HybridCacheStub(Cache):
                pass
            import transformers.cache_utils as _cu
            _cu.HybridCache = _HybridCacheStub

        from src.model.model import MMEBModel
        from src.arguments import ModelArguments

        logger.info("Loading VLM2Vec from %s (base: %s) on %s ...", model_path, base_model_path, device)

        # VLM2Vec load() hardcodes flash_attention_2 on the config — intercept
        # the actual model from_pretrained() and force sdpa instead.
        from src.model.vlm_backbone.qwen2_vl import Qwen2VLForConditionalGeneration
        _orig_fp = Qwen2VLForConditionalGeneration.from_pretrained
        @classmethod
        def _sdpa_from_pretrained(cls, *a, **kw):
            if "config" in kw:
                kw["config"]._attn_implementation = "sdpa"
                if hasattr(kw["config"], "vision_config") and kw["config"].vision_config:
                    kw["config"].vision_config._attn_implementation = "sdpa"
            return _orig_fp.__func__(cls, *a, **kw)
        Qwen2VLForConditionalGeneration.from_pretrained = _sdpa_from_pretrained
        try:
            model_args = ModelArguments(
                model_name=base_model_path,  # local path (HF blocked)
                checkpoint_path=model_path,
                pooling="last",
                normalize=True,
                model_backbone="qwen2_vl",
                lora=True,
            )
            self._model = MMEBModel.load(model_args)
        finally:
            Qwen2VLForConditionalGeneration.from_pretrained = _orig_fp
        self._model = self._model.to(device, dtype=torch.bfloat16)
        self._model.eval()

        from src.model.processor import load_processor
        self._processor = load_processor(model_args)
        self._device = device
        self._dim = 3584
        logger.info("Loaded. Embedding dim: %d", self._dim)

    @property
    def dim(self) -> int:
        return self._dim

    def encode(self, texts: list[str], images: list[str]) -> list[list[float]]:
        from src.model.processor import QWEN2_VL, VLM_IMAGE_TOKENS

        n = max(len(texts), len(images))
        if n == 0:
            return []

        results: list[list[float]] = []
        for i in range(n):
            text = texts[i] if i < len(texts) else ""
            image = images[i] if i < len(images) else ""

            has_image = bool(image)
            if has_image:
                text = f"{VLM_IMAGE_TOKENS[QWEN2_VL]}{text}"

            inputs = self._processor(
                text=text or None,
                images=image or None,
                return_tensors="pt",
            )
            inputs = {k: v.to(self._device) for k, v in inputs.items()}
            if has_image and "pixel_values" in inputs:
                for k in ("pixel_values", "image_grid_thw"):
                    if k in inputs and inputs[k].dim() == 1:
                        inputs[k] = inputs[k].unsqueeze(0)

            with torch.no_grad():
                if has_image:
                    output = self._model(qry=inputs)["qry_reps"]
                else:
                    output = self._model(tgt=inputs)["tgt_reps"]
            results.append(list(map(float, output[0].cpu().numpy())))
        return results


class OpsMMEmbedder:
    """Ops-MM-embedding-v1-7B — custom OpsMMEmbeddingV1 class from HF."""

    def __init__(self, model_path: str, device: str):
        logger.info("Loading Ops-MM-Embedding from %s on %s ...", model_path, device)
        self._model = None  # lazy load
        self._path = model_path
        self._device = device
        self._dim = 3584
        self._loaded = False

    @property
    def dim(self) -> int:
        return self._dim

    def _ensure_loaded(self):
        if self._loaded:
            return
        import sys
        _ops_mm_root = "/tmp/weiyiyang"
        if _ops_mm_root not in sys.path:
            sys.path.insert(0, _ops_mm_root)
        from ops_mm_embedding_v1 import OpsMMEmbeddingV1
        self._model = OpsMMEmbeddingV1(
            self._path, device=self._device, attn_implementation="sdpa",
        )
        self._loaded = True
        logger.info("Loaded. Embedding dim: %d", self._dim)

    def encode(self, texts: list[str], images: list[str]) -> list[list[float]]:
        n = max(len(texts), len(images))
        if n == 0:
            return []
        self._ensure_loaded()

        # Determine strategy based on inputs
        has_texts = any(t for t in texts)
        has_images = any(p for p in images)

        if has_texts and has_images:
            # Fused text+image embeddings
            return [
                list(map(float, v))
                for v in self._model.get_fused_embeddings(
                    texts=texts, images=images, instruction=""
                )
            ]
        elif has_images:
            return [
                list(map(float, v))
                for v in self._model.get_image_embeddings(images)
            ]
        else:
            return [
                list(map(float, v))
                for v in self._model.get_text_embeddings(texts)
            ]


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

MODEL_CONFIG: dict[str, dict] = {
    "qwen3-vl-embed-8b": {
        "default_path": "/tmp/weiyiyang/model_cache/Qwen3-VL-Embedding-8B",
        "display_name": "Qwen3-VL-Embedding-8B",
        "embedder_class": Qwen3VLEmbedder,
    },
    "openclip-vit-g-14": {
        "default_path": "ViT-g-14",
        "display_name": "OpenCLIP ViT-G/14",
        "embedder_class": OpenCLIPEmbedder,
    },
    "vlm2vec-qwen2vl-7b": {
        "default_path": "/tmp/weiyiyang/model_cache/VLM2Vec-Qwen2VL-7B",
        "display_name": "VLM2Vec-Qwen2VL-7B",
        "embedder_class": VLM2VecEmbedder,
        "base_model_path": "/tmp/weiyiyang/model_cache/Qwen2-VL-7B-Instruct",
    },
    "ops-mm-embed-7b": {
        "default_path": "/tmp/weiyiyang/model_cache/Ops-MM-embedding-v1-7B",
        "display_name": "Ops-MM-Embedding-7B",
        "embedder_class": OpsMMEmbedder,
    },
}


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

def build_app(model_name: str, model_path: str, device: str):
    from fastapi import FastAPI

    cfg = MODEL_CONFIG[model_name]
    embedder: Optional[object] = None

    app = FastAPI(title=f"SciFullMMBench Multimodal Embedding — {cfg['display_name']}")

    @app.get("/health")
    def health() -> dict:
        return {
            "status": "ok",
            "model": model_name,
            "display_name": cfg["display_name"],
            "embed_dim": embedder.dim if embedder else None,
            "device": device,
        }

    @app.post("/embed", response_model=EmbedResponse)
    def embed(payload: EmbedRequest) -> EmbedResponse:
        nonlocal embedder
        if embedder is None:
            cls = cfg["embedder_class"]
            kwargs = {}
            if model_name == "vlm2vec-qwen2vl-7b":
                kwargs["base_model_path"] = cfg.get("base_model_path", "/tmp/weiyiyang/model_cache/Qwen2-VL-7B-Instruct")
            embedder = cls(model_path, device, **kwargs)
        texts = payload.texts or []
        images = payload.images or []
        if images:
            images = _resolve_images(images)
        vectors = embedder.encode(texts, images)
        return EmbedResponse(embeddings=vectors)

    return app


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Serve a single multimodal embedding model over HTTP."
    )
    parser.add_argument("--model", required=True, choices=MODEL_CHOICES)
    parser.add_argument("--model-path", type=str, default=None,
                        help="Override default model path.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=18083)
    return parser


def main() -> int:
    import uvicorn

    args = build_parser().parse_args()
    model_path = args.model_path or MODEL_CONFIG[args.model]["default_path"]
    cfg = MODEL_CONFIG[args.model]
    logger.info("Model: %s (%s)", args.model, cfg["display_name"])
    logger.info("Path: %s", model_path)
    logger.info("Device: %s, Port: %d", args.device, args.port)

    app = build_app(args.model, model_path, str(args.device))
    uvicorn.run(app, host=str(args.host), port=int(args.port))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
