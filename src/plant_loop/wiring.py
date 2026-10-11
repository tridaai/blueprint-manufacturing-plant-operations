"""Composition root: build a PlantService from configuration.

Every concrete collaborator is chosen here — reasoning backend
(REASONING_BACKEND), embedder, vector index, record store, object
store — and injected into the service. Nothing else in the package
reads backend-selection environment variables.
"""

from __future__ import annotations

from .backends import get_reasoner
from .embeddings import get_embedder, get_vector_index
from .kb import KBRetriever
from .objectstore import default_object_store
from .service import PlantService
from .store import default_store


def build_service() -> PlantService:
    """The service for this process, wired from the environment."""
    embedder = get_embedder()
    index = get_vector_index()
    return PlantService(
        reasoner=get_reasoner(),
        retriever=KBRetriever(embedder, index),
        store=default_store(),
        objects=default_object_store(),
    )
