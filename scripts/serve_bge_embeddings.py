from __future__ import annotations

import argparse
from typing import Optional

from pydantic import BaseModel, Field


class EmbedRequest(BaseModel):
    texts: list[str] = Field(default_factory=list)


class EmbedResponse(BaseModel):
    embeddings: list[list[float]]


def build_app(model_path: str, device: str):
    from fastapi import FastAPI
    from sentence_transformers import SentenceTransformer

    app = FastAPI(title="SciFullMMBench BGE Embedding Service")
    model: Optional[SentenceTransformer] = None

    def get_model() -> SentenceTransformer:
        nonlocal model
        if model is None:
            model = SentenceTransformer(model_path, device=device, trust_remote_code=True)
        return model

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "model_path": model_path, "device": device}

    @app.post("/embed", response_model=EmbedResponse)
    def embed(payload: EmbedRequest) -> EmbedResponse:
        if not payload.texts:
            return EmbedResponse(embeddings=[])
        vectors = get_model().encode(payload.texts)
        return EmbedResponse(
            embeddings=[[float(value) for value in vector] for vector in vectors]
        )

    return app


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Serve BGE-M3 embeddings over HTTP.")
    parser.add_argument("--model-path", default="/data3/yangyinghao/bge-m3")
    parser.add_argument("--device", default="cuda:3")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=18080)
    return parser


def main() -> int:
    import uvicorn

    args = build_parser().parse_args()
    app = build_app(model_path=str(args.model_path), device=str(args.device))
    uvicorn.run(app, host=str(args.host), port=int(args.port))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
