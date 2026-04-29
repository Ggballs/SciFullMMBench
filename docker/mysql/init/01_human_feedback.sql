CREATE TABLE IF NOT EXISTS human_feedback (
  id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT
    COMMENT 'Internal primary key for one feedback row.',

  paper_forum_id VARCHAR(128) NOT NULL
    COMMENT 'OpenReview forum id for the paper associated with this query.',
  query_id VARCHAR(64) NOT NULL
    COMMENT 'Stable query identifier.',
  query_text TEXT NOT NULL
    COMMENT 'Original generated retrieval query shown to the human reviewer.',

  feedback_item_id VARCHAR(64) NOT NULL
    COMMENT 'Stable feedback item id. Example: query_relevance, human_like, hard_negative:1, positive:1.',
  reviewer_username VARCHAR(64) NOT NULL
    COMMENT 'Username authenticated by the Gradio login layer.',

  judgement VARCHAR(32) NOT NULL
    COMMENT 'Human judgement value, usually Yes, No, or Unsure.',
  selection_type JSON DEFAULT NULL
    COMMENT 'Selected issue/type values from the UI, e.g. ["Too specific", "Other"].',
  reason_note TEXT DEFAULT NULL
    COMMENT 'Reviewer explanation, note, or custom Other text.',

  feedback_raw_json JSON DEFAULT NULL
    COMMENT 'Raw feedback payload from the UI for future compatibility/debugging.',

  created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
    COMMENT 'Row creation time.',
  updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
    COMMENT 'Last update time.',

  PRIMARY KEY (id),

  UNIQUE KEY uq_human_feedback (
    paper_forum_id,
    query_id,
    feedback_item_id,
    reviewer_username
  ),

  KEY idx_query_id (query_id),
  KEY idx_reviewer_username (reviewer_username)
) ENGINE=InnoDB
  DEFAULT CHARSET=utf8mb4
  COLLATE=utf8mb4_unicode_ci
  COMMENT='Human feedback for query relevance, human-like query quality, and candidate label checks.';
