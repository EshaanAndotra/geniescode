-- =====================================================================
-- 03b_approve.sql
-- Save as a query named "dq_approve". Fill in the boxes above the editor,
-- then click Run. Nothing in the SQL needs editing.
--
--   :target_table   table the proposals are for, e.g. workspace.default.demand
--   :rule_names     comma-separated rule names to approve (leave empty for none)
--   :mapping_column column whose mappings you're approving (leave empty for none)
--   :raw_values     comma-separated raw values in that column to approve
--
-- Only proposals with status 'proposed' are touched, so running it twice
-- does nothing the second time. The last statement shows the result.
-- =====================================================================

-- Rules: copy into the enforced table...
MERGE INTO workspace.data_quality.rules AS r
USING (
  SELECT * FROM workspace.data_quality.rule_proposals
  WHERE status = 'proposed'
    AND target_table = :target_table
    AND array_contains(transform(split(:rule_names, ','), s -> trim(s)), rule_name)
) AS p
  ON r.target_table = p.target_table AND r.rule_name = p.rule_name
WHEN MATCHED THEN UPDATE SET
  constraint_expr    = p.constraint_expr,
  action             = p.action,
  tag                = p.tag,
  enabled            = true,
  source_proposal_id = p.proposal_id,
  approved_by        = current_user(),
  approved_at        = current_timestamp()
WHEN NOT MATCHED THEN INSERT
  (rule_name, target_table, tag, constraint_expr, action, enabled,
   source_proposal_id, approved_by, approved_at)
VALUES
  (p.rule_name, p.target_table, p.tag, p.constraint_expr, p.action, true,
   p.proposal_id, current_user(), current_timestamp());

-- ...then mark the proposals approved.
UPDATE workspace.data_quality.rule_proposals
SET status = 'approved', reviewed_by = current_user(), reviewed_at = current_timestamp()
WHERE status = 'proposed'
  AND target_table = :target_table
  AND array_contains(transform(split(:rule_names, ','), s -> trim(s)), rule_name);

-- Mappings: same two steps.
MERGE INTO workspace.data_quality.value_mappings AS v
USING (
  SELECT * FROM workspace.data_quality.mapping_proposals
  WHERE status = 'proposed'
    AND target_table = :target_table
    AND column_name = :mapping_column
    AND array_contains(transform(split(:raw_values, ','), s -> trim(s)), raw_value)
) AS p
  ON v.target_table = p.target_table AND v.column_name = p.column_name AND v.raw_value = p.raw_value
WHEN MATCHED THEN UPDATE SET
  clean_value        = p.clean_value,
  enabled            = true,
  source_proposal_id = p.proposal_id,
  approved_by        = current_user(),
  approved_at        = current_timestamp()
WHEN NOT MATCHED THEN INSERT
  (target_table, column_name, raw_value, clean_value, enabled,
   source_proposal_id, approved_by, approved_at)
VALUES
  (p.target_table, p.column_name, p.raw_value, p.clean_value, true,
   p.proposal_id, current_user(), current_timestamp());

UPDATE workspace.data_quality.mapping_proposals
SET status = 'approved', reviewed_by = current_user(), reviewed_at = current_timestamp()
WHERE status = 'proposed'
  AND target_table = :target_table
  AND column_name = :mapping_column
  AND array_contains(transform(split(:raw_values, ','), s -> trim(s)), raw_value);

-- What's now enforced for this table.
SELECT 'rule' AS kind, rule_name AS name, action, constraint_expr AS detail, approved_at
FROM workspace.data_quality.rules
WHERE target_table = :target_table AND enabled
UNION ALL
SELECT 'mapping', column_name, NULL, concat(raw_value, ' -> ', clean_value), approved_at
FROM workspace.data_quality.value_mappings
WHERE target_table = :target_table AND enabled
ORDER BY approved_at DESC;
