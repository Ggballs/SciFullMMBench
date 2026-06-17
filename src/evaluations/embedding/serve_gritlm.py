"""Standalone GritLM-7B embedding service (transformers 4.44 compatible)."""
from __future__ import annotations

import argparse
import logging

from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_PATH = "/data3/weiyiyang/model_cache/GritLM-7B"


class EmbedRequest(BaseModel):
    texts: list[str] = Field(default_factory=list)


class EmbedResponse(BaseModel):
    embeddings: list[list[float]]


def build_app(device: str):
    from fastapi import FastAPI

    app = FastAPI(title="GritLM-7B Embedding Service")
    model: SentenceTransformer | None = None

    def _ensure_model():
        nonlocal model
        if model is None:
            logger.info("Loading GritLM-7B on %s ...", device)
            model = SentenceTransformer(
                MODEL_PATH, device=device, trust_remote_code=True
            )
            logger.info("GritLM-7B loaded.")

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "device": device, "model_loaded": model is not None}

    @app.post("/embed", response_model=EmbedResponse)
    def embed(payload: EmbedRequest) -> EmbedResponse:
        if not payload.texts:
            return EmbedResponse(embeddings=[])
        _ensure_model()
        vectors = model.encode(payload.texts, show_progress_bar=False)
        return EmbedResponse(embeddings=[list(map(float, v)) for v in vectors])

    return app


def main() -> int:
    import uvicorn

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:3")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=18082)
    args = parser.parse_args()

    app = build_app(device=str(args.device))
    uvicorn.run(app, host=str(args.host), port=int(args.port))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
