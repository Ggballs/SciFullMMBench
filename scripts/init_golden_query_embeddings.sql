CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS golden_query_embeddings (
  example_id TEXT PRIMARY KEY,
  query_id TEXT NOT NULL,
  query_type TEXT NOT NULL,
  view_label TEXT NOT NULL,
  query_text TEXT NOT NULL,
  target_papers JSONB NOT NULL DEFAULT '[]'::jsonb,
  answer_original_content TEXT NOT NULL DEFAULT '',
  answer_tldr TEXT NOT NULL DEFAULT '',
  human_view_note TEXT NOT NULL DEFAULT '',
  indexing_content TEXT NOT NULL DEFAULT '',
  retrieval_content TEXT NOT NULL DEFAULT '',
  embedding vector(1024) NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,

  CONSTRAINT chk_golden_query_type CHECK (query_type IN ('IR', 'QA')),
  CONSTRAINT chk_golden_view_label CHECK (view_label IN ('motivation', 'method', 'experiment/result'))
);

CREATE INDEX IF NOT EXISTS idx_golden_query_embeddings_type_view
  ON golden_query_embeddings (query_type, view_label);

CREATE INDEX IF NOT EXISTS idx_golden_query_embeddings_embedding_cosine
  ON golden_query_embeddings
  USING ivfflat (embedding vector_cosine_ops)
  WITH (lists = 100);

CREATE OR REPLACE FUNCTION set_golden_query_embeddings_updated_at()
RETURNS TRIGGER AS $$
BEGIN
  NEW.updated_at = CURRENT_TIMESTAMP;
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_golden_query_embeddings_updated_at
  ON golden_query_embeddings;

CREATE TRIGGER trg_golden_query_embeddings_updated_at
BEFORE UPDATE ON golden_query_embeddings
FOR EACH ROW
EXECUTE FUNCTION set_golden_query_embeddings_updated_at();
