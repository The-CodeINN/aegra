"""Thin wrapper that truncates texts before forwarding to the real embeddings model.

The langchain_aws ``BedrockEmbeddings`` class validates Cohere v3 text lengths
client-side (hard 2048-char limit) *before* sending the request to the API.
This means the ``truncate="END"`` model kwarg never gets a chance to help.

``TruncatingEmbeddings`` sits in front of the real model and clips every input
string to ``max_chars`` so the validation passes.
"""

from __future__ import annotations

from typing import Any

from langchain_core.embeddings import Embeddings


class TruncatingEmbeddings(Embeddings):
    """Proxy that truncates input texts then delegates to a wrapped model."""

    def __init__(self, wrapped: Embeddings, max_chars: int = 2000) -> None:
        self._wrapped = wrapped
        self._max_chars = max_chars

    def _truncate(self, texts: list[str]) -> list[str]:
        return [t[: self._max_chars] if len(t) > self._max_chars else t for t in texts]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._wrapped.embed_documents(self._truncate(texts))

    def embed_query(self, text: str) -> list[float]:
        if len(text) > self._max_chars:
            text = text[: self._max_chars]
        return self._wrapped.embed_query(text)

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        return await self._wrapped.aembed_documents(self._truncate(texts))

    async def aembed_query(self, text: str) -> list[float]:
        if len(text) > self._max_chars:
            text = text[: self._max_chars]
        return await self._wrapped.aembed_query(text)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._wrapped, name)
