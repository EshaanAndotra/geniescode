-- =====================================================================
-- 03a_review.sql  (read-only, no parameters)
-- Save as a query named "dq_review" and click Run whenever you want to see
-- what's waiting. Each statement's result shows up in its own tab.
--
-- How to read fail_pct: the share of rows in the table TODAY that would fail
-- the rule. Rows are never removed, so a warn rule only adds a flag.
--   ~0%      Rule describes the data as it is. Check it's a real constraint,
--            not a coincidence.
--   small %  Usually the useful ones: the rule catches actual bad rows.
--   large %  The data is badly off or the rule is wrong. Look before approving.
-- Every 'fail' rule shown here has fail_pct = 0. The notebook downgrades any
-- fail rule that today's data already breaks to warn.
-- =====================================================================

-- 1. Rules waiting for review
SELECT target_table, rule_name, action, round(fail_rate * 100, 2) AS fail_pct,
       constraint_expr, rationale, column_name, proposed_at
FROM workspace.data_quality.rule_proposals
WHERE status = 'proposed'
ORDER BY target_table, fail_rate DESC;

-- 2. Mappings waiting for review
SELECT target_table, column_name, raw_value, clean_value, rows_in_sample, rationale
FROM workspace.data_quality.mapping_proposals
WHERE status = 'proposed'
ORDER BY target_table, column_name, clean_value, rows_in_sample DESC;

-- 3. What the model got wrong in recent runs (useful for tuning the prompt)
SELECT target_table, rule_name, constraint_expr, validation_error, proposed_at
FROM workspace.data_quality.rule_proposals
WHERE status = 'invalid'
ORDER BY proposed_at DESC
LIMIT 50;

-- 4. What the pipeline enforces right now
SELECT target_table, action, rule_name, constraint_expr, approved_by, approved_at
FROM workspace.data_quality.rules
WHERE enabled
ORDER BY target_table, action, rule_name;

SELECT target_table, column_name, raw_value, clean_value, approved_at
FROM workspace.data_quality.value_mappings
WHERE enabled
ORDER BY target_table, column_name, clean_value;

-- 5. Current versions of the enforced tables. When you build a training set,
--    put these in the pipeline config as dq.rules_version / dq.mappings_version.
DESCRIBE HISTORY workspace.data_quality.rules LIMIT 1;
DESCRIBE HISTORY workspace.data_quality.value_mappings LIMIT 1;
