-- =====================================================================
-- 03c_reject_or_disable.sql
-- Save as a query named "dq_reject". Fill in the boxes, then click Run.
--
--   :target_table   e.g. workspace.default.demand
--   :rule_names     comma-separated rule names (leave empty for none)
--   :mapping_column column for the mappings below (leave empty for none)
--   :raw_values     comma-separated raw values in that column
--   :note           why, e.g. "coincidence, not a real constraint"
--
-- For each name/value: if it's a pending proposal, it's rejected. If it's
-- already enforced, it's disabled (kept for the record, but the pipeline
-- ignores it from the next run). Nothing is deleted.
-- =====================================================================

-- Reject pending rule proposals
UPDATE workspace.data_quality.rule_proposals
SET status = 'rejected', reviewed_by = current_user(), reviewed_at = current_timestamp(),
    review_note = :note
WHERE status = 'proposed'
  AND target_table = :target_table
  AND array_contains(transform(split(:rule_names, ','), s -> trim(s)), rule_name);

-- Disable enforced rules
UPDATE workspace.data_quality.rules
SET enabled = false
WHERE enabled
  AND target_table = :target_table
  AND array_contains(transform(split(:rule_names, ','), s -> trim(s)), rule_name);

-- Reject pending mapping proposals
UPDATE workspace.data_quality.mapping_proposals
SET status = 'rejected', reviewed_by = current_user(), reviewed_at = current_timestamp(),
    review_note = :note
WHERE status = 'proposed'
  AND target_table = :target_table
  AND column_name = :mapping_column
  AND array_contains(transform(split(:raw_values, ','), s -> trim(s)), raw_value);

-- Disable enforced mappings
UPDATE workspace.data_quality.value_mappings
SET enabled = false
WHERE enabled
  AND target_table = :target_table
  AND column_name = :mapping_column
  AND array_contains(transform(split(:raw_values, ','), s -> trim(s)), raw_value);

-- What's still enforced for this table.
SELECT 'rule' AS kind, rule_name AS name, action, constraint_expr AS detail
FROM workspace.data_quality.rules
WHERE target_table = :target_table AND enabled
UNION ALL
SELECT 'mapping', column_name, NULL, concat(raw_value, ' -> ', clean_value)
FROM workspace.data_quality.value_mappings
WHERE target_table = :target_table AND enabled;
