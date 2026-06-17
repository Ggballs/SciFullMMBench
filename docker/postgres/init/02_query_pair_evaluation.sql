CREATE TABLE IF NOT EXISTS query_pair_evaluation (
  id BIGSERIAL PRIMARY KEY,

  pair_id VARCHAR(128) NOT NULL,
  mode VARCHAR(32) NOT NULL,
  paper_id VARCHAR(128) NOT NULL,

  reviewer_username VARCHAR(64) NOT NULL,

  choice VARCHAR(32) NOT NULL,
  confidence VARCHAR(16),
  note TEXT,

  ordering_seed INTEGER NOT NULL,
  pair_snapshot JSONB,

  created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,

  CONSTRAINT uq_query_pair_evaluation UNIQUE (
    pair_id,
    reviewer_username
  )
);

CREATE INDEX IF NOT EXISTS idx_query_pair_eval_pair_id
  ON query_pair_evaluation (pair_id);

CREATE INDEX IF NOT EXISTS idx_query_pair_eval_reviewer_username
  ON query_pair_evaluation (reviewer_username);

CREATE INDEX IF NOT EXISTS idx_query_pair_eval_mode
  ON query_pair_evaluation (mode);

CREATE OR REPLACE FUNCTION set_query_pair_evaluation_updated_at()
RETURNS TRIGGER AS $$
BEGIN
  IF NEW.updated_at = OLD.updated_at THEN
    NEW.updated_at = CURRENT_TIMESTAMP;
  END IF;
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_query_pair_evaluation_updated_at ON query_pair_evaluation;

CREATE TRIGGER trg_query_pair_evaluation_updated_at
BEFORE UPDATE ON query_pair_evaluation
FOR EACH ROW
EXECUTE FUNCTION set_query_pair_evaluation_updated_at();
