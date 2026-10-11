-- Knowledge-base chunk embeddings for SOP / history retrieval (pgvector).
--
-- The embedding column is deliberately dimension-free (`vector` with
-- no length): the deployment picks the embedding model, and the model
-- name is stored per row so a model switch re-embeds instead of
-- mixing vector spaces. Similarity is cosine distance via the <=>
-- operator.
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS kb_embeddings (
    chunk_id    TEXT PRIMARY KEY,
    model       TEXT NOT NULL,
    embedding   vector NOT NULL,
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);
