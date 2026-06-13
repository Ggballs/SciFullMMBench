from __future__ import annotations

import json
from urllib import request
from typing import Optional, Protocol


class TextEmbedder(Protocol):
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        ...


class BGEM3Embedder:
    def __init__(self, model_path: str, device: str = "cuda:2"):
        self.model_path = model_path
        self.device = device
        self._model = None

    def _load_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(
                self.model_path,
                device=self.device,
                trust_remote_code=True,
            )
        return self._model

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = self._load_model().encode(texts)
        return [list(map(float, vector)) for vector in vectors]


class HttpTextEmbedder:
    def __init__(self, service_url: str, timeout_seconds: float = 120.0):
        self.service_url = service_url
        self.timeout_seconds = float(timeout_seconds)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        payload = json.dumps({"texts": texts}).encode("utf-8")
        req = request.Request(
            self.service_url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with request.urlopen(req, timeout=self.timeout_seconds) as response:
            raw = response.read().decode("utf-8")
        data = json.loads(raw)
        embeddings = data.get("embeddings")
        if not isinstance(embeddings, list):
            raise ValueError("Embedding service response must contain an 'embeddings' list.")
        if len(embeddings) != len(texts):
            raise ValueError(
                "Embedding service returned "
                f"{len(embeddings)} vectors for {len(texts)} input texts."
            )
        return [[float(value) for value in vector] for vector in embeddings]


def build_text_embedder(
    *,
    model_path: str,
    device: str,
    service_url: Optional[str] = None,
    timeout_seconds: float = 120.0,
) -> TextEmbedder:
    if service_url:
        return HttpTextEmbedder(service_url=service_url, timeout_seconds=timeout_seconds)
    return BGEM3Embedder(model_path=model_path, device=device)
