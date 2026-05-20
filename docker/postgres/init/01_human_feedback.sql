CREATE TABLE IF NOT EXISTS human_feedback (
  id BIGSERIAL PRIMARY KEY,

  paper_forum_id VARCHAR(128) NOT NULL,
  query_id VARCHAR(64) NOT NULL,
  query_text TEXT NOT NULL,

  feedback_item_id VARCHAR(64) NOT NULL,
  reviewer_username VARCHAR(64) NOT NULL,

  judgement VARCHAR(32) NOT NULL,
  selection_type JSONB,
  reason_note TEXT,

  feedback_raw_json JSONB,

  created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,

  CONSTRAINT uq_human_feedback UNIQUE (
    paper_forum_id,
    query_id,
    feedback_item_id,
    reviewer_username
  )
);

CREATE INDEX IF NOT EXISTS idx_human_feedback_query_id
  ON human_feedback (query_id);

CREATE INDEX IF NOT EXISTS idx_human_feedback_reviewer_username
  ON human_feedback (reviewer_username);

CREATE OR REPLACE FUNCTION set_human_feedback_updated_at()
RETURNS TRIGGER AS $$
BEGIN
  IF NEW.updated_at = OLD.updated_at THEN
    NEW.updated_at = CURRENT_TIMESTAMP;
  END IF;
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_human_feedback_updated_at ON human_feedback;

CREATE TRIGGER trg_human_feedback_updated_at
BEFORE UPDATE ON human_feedback
FOR EACH ROW
EXECUTE FUNCTION set_human_feedback_updated_at();
