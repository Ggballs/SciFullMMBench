from __future__ import annotations

from typing import Protocol


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
