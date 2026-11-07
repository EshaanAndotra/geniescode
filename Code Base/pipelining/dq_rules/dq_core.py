"""Shared logic for proposing data quality rules with an LLM.

Imported by the 02_propose_rules notebook. Plain Python with no Databricks-only
calls, so it can be tested locally with PySpark.

The cleaned tables feed a risk model and then a prediction model, so the rules
never remove rows. A rule is either:
  warn  - the row is kept and the rule name is added to its _dq_flags column
  fail  - the pipeline update stops (for structural problems such as a missing key)
"""
import json
import re
import uuid
from typing import Literal

from langchain_core.utils.function_calling import convert_to_openai_function
from pydantic import BaseModel, ConfigDict, Field
from pyspark.sql import functions as F
from pyspark.sql.types import (
    ArrayType, BinaryType, BooleanType, DateType, MapType, NumericType,
    StringType, StructType, TimestampType,
)

LOW_CARDINALITY = 50     # columns with <= this many distinct values get full value counts
EXAMPLES_PER_COLUMN = 5  # example values for higher-cardinality text columns

NAME_RE = re.compile(r"^[a-z][a-z0-9_]{2,63}$")
# The expression will run on every row inside the pipeline, so keep it to plain
# scalar logic. A reviewer still reads every rule; this just catches the obvious stuff.
BLOCKED_RE = re.compile(r";|\bselect\b|\bai_\w+\s*\(|\bhttp_request\b|\bsecret\s*\(|\bremote_query\b", re.I)


def qcol(name):
    """Column reference that survives odd column names."""
    return F.col("`" + name.replace("`", "``") + "`")


def profile_table(df, profile_rows, send_values):
    """Return (profile dict, value_counts_by_column, sampled_row_count)."""
    total_rows = df.count()
    frac = min(1.0, profile_rows / total_rows) if total_rows else 1.0
    sample = df.sample(fraction=frac, seed=42) if frac < 1.0 else df

    fields = sample.schema.fields
    complex_types = (ArrayType, MapType, StructType, BinaryType)

    aggs = [F.count(F.lit(1)).alias("__n")]
    for i, f in enumerate(fields):
        c = qcol(f.name)
        aggs.append(F.sum(F.when(c.isNull(), 1).otherwise(0)).alias(f"nulls{i}"))
        if isinstance(f.dataType, complex_types):
            continue
        aggs.append(F.approx_count_distinct(c).alias(f"distinct{i}"))
        if isinstance(f.dataType, (NumericType, DateType, TimestampType)):
            aggs += [F.min(c).alias(f"min{i}"), F.max(c).alias(f"max{i}")]
        elif isinstance(f.dataType, StringType):
            aggs += [
                F.min(F.length(c)).alias(f"minlen{i}"),
                F.max(F.length(c)).alias(f"maxlen{i}"),
                F.sum(F.when(c != F.trim(c), 1).otherwise(0)).alias(f"edge_ws{i}"),
                F.sum(F.when(F.trim(c) == "", 1).otherwise(0)).alias(f"blank{i}"),
            ]
    stats = sample.agg(*aggs).first().asDict()
    n = stats["__n"]

    columns, value_counts = [], {}
    for i, f in enumerate(fields):
        col = {
            "name": f.name,
            "type": f.dataType.simpleString(),
            "null_fraction": round(stats[f"nulls{i}"] / n, 4) if n else None,
        }
        if f"distinct{i}" in stats:
            col["approx_distinct"] = stats[f"distinct{i}"]
        if f"min{i}" in stats:
            col["min"], col["max"] = stats[f"min{i}"], stats[f"max{i}"]
        if f"minlen{i}" in stats:
            col["min_length"], col["max_length"] = stats[f"minlen{i}"], stats[f"maxlen{i}"]
            col["rows_with_leading_or_trailing_space"] = stats[f"edge_ws{i}"]
            col["blank_rows"] = stats[f"blank{i}"]

        distinct = col.get("approx_distinct")
        if send_values and distinct is not None:
            c = qcol(f.name)
            if distinct <= LOW_CARDINALITY:
                rows = (sample.where(c.isNotNull()).groupBy(c.alias("v")).count()
                        .orderBy(F.desc("count")).limit(LOW_CARDINALITY * 2).collect())
                counts = {str(r["v"]): r["count"] for r in rows}
                col["value_counts"] = counts
                if isinstance(f.dataType, StringType):
                    value_counts[f.name] = counts
            elif isinstance(f.dataType, StringType):
                rows = (sample.where(c.isNotNull()).select(c.alias("v")).distinct()
                        .limit(EXAMPLES_PER_COLUMN).collect())
                col["examples"] = [r["v"] for r in rows]
        columns.append(col)

    profile = {"rows_in_table": total_rows, "rows_profiled": n, "columns": columns}
    return profile, value_counts, n


class ProposedRule(BaseModel):
    model_config = ConfigDict(extra="forbid")
    rule_name: str = Field(description="Unique snake_case name, e.g. order_amount_non_negative")
    column_name: str = Field(description="The main column this rule checks")
    constraint_expr: str = Field(description="Spark SQL boolean expression over one row; the row passes when it is TRUE")
    action: Literal["warn", "fail"]
    rationale: str = Field(description="One sentence naming the profile evidence behind the rule")


class ProposedMapping(BaseModel):
    model_config = ConfigDict(extra="forbid")
    column_name: str
    raw_value: str = Field(description="A value that appears in the column's value_counts")
    clean_value: str = Field(description="The canonical value it should become")
    rationale: str


class Proposals(BaseModel):
    """Proposed data quality rules and category mappings for one table."""
    model_config = ConfigDict(extra="forbid")
    rules: list[ProposedRule]
    mappings: list[ProposedMapping]


SYSTEM_PROMPT = """You help a data engineer write data quality rules for one table in Databricks.

You get a profile of the table: column names, types, null fractions, distinct counts, ranges, and for some columns value counts or example values. Propose two things.

1. Expectation rules. Each rule is a Spark SQL boolean expression evaluated on one row, used as a Lakeflow pipeline expectation. A row passes when the expression is TRUE. Rules describe data quality; they must not transform, impute or cap values.
- Reference columns exactly as named in the profile. Wrap names that aren't plain identifiers in backticks.
- Make null handling explicit. If nulls are acceptable, write `col IS NULL OR <check>`. If they aren't, include `col IS NOT NULL`.
- Use only built-in scalar SQL functions. No subqueries, no SELECT, no other tables, no ai_* functions.
- Base each rule on evidence in the profile or an obvious domain constraint (IDs present, amounts not negative, dates not in the future, values in a known set). Don't invent business rules the profile doesn't support.
- Choose the action. The cleaned table trains a risk model, so rows are never removed: a failing row is kept and flagged.
  "warn": the row is kept and flagged. Use this for almost everything, including outliers, odd values and missing data, because unusual rows are often the ones a risk model needs to learn from.
  "fail": stops the whole pipeline. Use it only for structural breakage, such as an ID or key column that should never be null. Never use "fail" for a value-range check.
- Give each rule a unique snake_case name and a one-sentence rationale citing the profile evidence.
- Don't repeat a rule listed under existing rules.

2. Category mappings, only for text columns that have value_counts. Map a raw value to a canonical value only when both clearly mean the same thing (case, whitespace, spelling, abbreviation). Prefer an existing value as the canonical form, usually the most frequent spelling. Only use raw values that appear in value_counts. Don't merge values that could be genuinely different categories. Return an empty list if nothing needs mapping.

Fewer solid rules beat many weak ones. A person reviews everything you propose."""


def build_human_message(source_table, profile, existing_rules, existing_mappings):
    return (
        f"Table: {source_table}\n\n"
        f"Existing rules (don't repeat):\n{json.dumps(existing_rules, indent=1, default=str)}\n\n"
        f"Existing mappings (don't repeat):\n{json.dumps(existing_mappings, indent=1, default=str)}\n\n"
        f"Profile:\n{json.dumps(profile, indent=1, default=str)}"
    )


def _message_text(msg):
    """Text of a chat response. Some endpoints return a list of blocks (reasoning + text)."""
    content = msg.content
    if isinstance(content, str):
        return content
    parts = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict) and block.get("type") in ("text", "output_text"):
            parts.append(block.get("text", ""))
    return "".join(parts)


def _parse_proposals(text):
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"Model didn't return JSON. First 500 chars:\n{text[:500]}")
    return Proposals.model_validate_json(text[start:end + 1])


def ask_model(llm, system, human):
    """Get Proposals from any LangChain chat model that speaks the OpenAI chat format.

    Tries structured output (response_format=json_schema) first. If the endpoint
    rejects that, retries with the schema in the prompt and validates the reply.
    """
    fn = convert_to_openai_function(Proposals, strict=True)  # inlines $refs, sets additionalProperties=false
    schema = fn["parameters"]
    response_format = {"type": "json_schema",
                       "json_schema": {"name": "proposals", "schema": schema, "strict": True}}
    messages = [("system", system), ("human", human)]
    try:
        return _parse_proposals(_message_text(llm.bind(response_format=response_format).invoke(messages)))
    except Exception as first_error:
        fallback_system = (system + "\n\nReturn only a JSON object, no prose, matching this JSON schema:\n"
                           + json.dumps(schema))
        try:
            return _parse_proposals(_message_text(llm.invoke([("system", fallback_system), ("human", human)])))
        except Exception as second_error:
            raise RuntimeError(f"Structured output failed ({first_error}); "
                               f"fallback failed ({second_error})") from second_error


def validate_rules(df, rules, existing_names):
    """Check each proposed rule. Returns list of dicts with validation_error / fail_rate / status."""
    out, seen = [], set(existing_names)
    for r in rules:
        rec = r.model_dump()
        rec["constraint_expr"] = rec["constraint_expr"].strip()
        rec["fail_rate"], rec["validation_error"] = None, None
        if rec["rule_name"] in seen:
            continue  # already exists or duplicated within this batch; skip silently
        seen.add(rec["rule_name"])
        if not NAME_RE.match(rec["rule_name"]):
            rec["validation_error"] = "rule_name must be snake_case, 3-64 chars"
        elif BLOCKED_RE.search(rec["constraint_expr"]):
            rec["validation_error"] = "expression uses a blocked construct (;, SELECT, ai_*, http_request, secret, remote_query)"
        else:
            try:
                dtype = df.select(F.expr(rec["constraint_expr"]).alias("x")).schema["x"].dataType
                if not isinstance(dtype, BooleanType):
                    rec["validation_error"] = f"expression returns {dtype.simpleString()}, not boolean"
            except Exception as e:  # parse or analysis error, e.g. unknown column
                rec["validation_error"] = str(e).strip().splitlines()[0][:500]
        out.append(rec)

    # One pass over the table for every rule that parsed: count rows where the
    # expression is FALSE (would fail) and rows where it is NULL. A rule that
    # returns NULL has ambiguous null handling, so it goes back as invalid.
    def counts(i):
        e = F.expr(out[i]["constraint_expr"])
        return [F.sum(F.when(~e, 1).otherwise(0)).alias(f"false{i}"),
                F.sum(F.when(e.isNull(), 1).otherwise(0)).alias(f"null{i}")]

    ok = [i for i, rec in enumerate(out) if rec["validation_error"] is None]
    results = {}  # i -> (false_count, null_count)
    if ok:
        try:
            row = df.agg(F.count(F.lit(1)).alias("__n"), *[a for i in ok for a in counts(i)]).first()
            n = row["__n"]
            results = {i: (row[f"false{i}"], row[f"null{i}"]) for i in ok}
        except Exception:
            # Some expression parses fine but errors at runtime (e.g. an ANSI cast
            # error on a string column). Re-check one at a time to find which.
            n = df.count()
            for i in ok:
                try:
                    r = df.agg(*counts(i)).first()
                    results[i] = (r[f"false{i}"], r[f"null{i}"])
                except Exception as e:
                    out[i]["validation_error"] = "runtime error: " + str(e).strip().splitlines()[0][:480]

    for i, (false_count, null_count) in results.items():
        out[i]["fail_rate"] = (false_count / n) if n else None
        if null_count:
            out[i]["validation_error"] = (
                f"returns NULL for {null_count:,} rows; add explicit IS NULL / IS NOT NULL handling")
        elif out[i]["action"] == "fail" and false_count:
            # A fail rule that today's data already breaks would stop the next
            # pipeline run. Keep it, but as a warn, and say why.
            out[i]["action"] = "warn"
            out[i]["rationale"] = (f"[downgraded from fail: {false_count:,} rows fail today] "
                                   + out[i]["rationale"])

    for rec in out:
        rec["status"] = "invalid" if rec["validation_error"] else "proposed"
    return out


def validate_mappings(mappings, value_counts, existing_pairs):
    out, seen = [], set(existing_pairs)
    for m in mappings:
        rec = m.model_dump()
        rec["validation_error"], rec["rows_in_sample"] = None, None
        key = (rec["column_name"], rec["raw_value"])
        if rec["raw_value"] == rec["clean_value"] or key in seen:
            continue
        seen.add(key)
        counts = value_counts.get(rec["column_name"])
        if counts is None:
            rec["validation_error"] = "column has no value counts (not a low-cardinality text column)"
        elif rec["raw_value"] not in counts:
            rec["validation_error"] = "raw_value doesn't appear in the profiled values"
        else:
            rec["rows_in_sample"] = counts[rec["raw_value"]]
        out.append(rec)

    # Chains (a -> b and b -> c) apply only one step in the pipeline and leave
    # both b and c in the output, so flag every mapping that touches the
    # middle value and let the reviewer pick the final form.
    valid = [r for r in out if r["validation_error"] is None]
    raws = {(r["column_name"], r["raw_value"]) for r in valid}
    cleans = {(r["column_name"], r["clean_value"]) for r in valid}
    middle = raws & cleans
    for rec in valid:
        if (rec["column_name"], rec["raw_value"]) in middle or (rec["column_name"], rec["clean_value"]) in middle:
            rec["validation_error"] = "part of a chain (a value is both mapped from and mapped to); pick one final form"
    for rec in out:
        rec["status"] = "invalid" if rec["validation_error"] else "proposed"
    return out


RULE_PROPOSAL_SCHEMA = """
    proposal_id STRING, run_id STRING, target_table STRING, tag STRING, rule_name STRING,
    column_name STRING, constraint_expr STRING, action STRING, rationale STRING, fail_rate DOUBLE,
    validation_error STRING, status STRING, model STRING, proposed_by STRING,
    reviewed_by STRING, reviewed_at TIMESTAMP, review_note STRING"""

MAPPING_PROPOSAL_SCHEMA = """
    proposal_id STRING, run_id STRING, target_table STRING, column_name STRING, raw_value STRING,
    clean_value STRING, rationale STRING, rows_in_sample BIGINT, validation_error STRING,
    status STRING, model STRING, proposed_by STRING,
    reviewed_by STRING, reviewed_at TIMESTAMP, review_note STRING"""


def propose_for_table(spark, llm, source_table, *, dq_schema, run_id, user, model_name,
                      tag=None, profile_rows=100000, send_values=True):
    """Profile one table, ask the model, validate, and append to the proposal tables.

    Returns (rule_records, mapping_records).
    """
    rule_proposals, mapping_proposals = f"{dq_schema}.rule_proposals", f"{dq_schema}.mapping_proposals"
    rules_table, mappings_table = f"{dq_schema}.rules", f"{dq_schema}.value_mappings"

    df = spark.read.table(source_table)
    profile, value_counts, _ = profile_table(df, profile_rows, send_values)

    is_target = F.col("target_table") == source_table
    existing_rules = [r.asDict() for r in spark.read.table(rules_table)
                      .where(is_target & F.col("enabled"))
                      .select("rule_name", "constraint_expr", "action").collect()]
    pending_rule_names = [r["rule_name"] for r in spark.read.table(rule_proposals)
                          .where(is_target & (F.col("status") == "proposed"))
                          .select("rule_name").collect()]
    existing_mappings = [r.asDict() for r in spark.read.table(mappings_table)
                         .where(is_target & F.col("enabled"))
                         .select("column_name", "raw_value", "clean_value").collect()]
    pending_mapping_pairs = [(r["column_name"], r["raw_value"]) for r in spark.read.table(mapping_proposals)
                             .where(is_target & (F.col("status") == "proposed"))
                             .select("column_name", "raw_value").collect()]

    proposals = ask_model(llm, SYSTEM_PROMPT, build_human_message(
        source_table, profile,
        existing_rules + [{"rule_name": n, "status": "pending review"} for n in pending_rule_names],
        existing_mappings,
    ))

    rule_recs = validate_rules(df, proposals.rules,
                               [r["rule_name"] for r in existing_rules] + pending_rule_names)
    mapping_recs = validate_mappings(proposals.mappings, value_counts,
                                     [(m["column_name"], m["raw_value"]) for m in existing_mappings]
                                     + pending_mapping_pairs)

    common = {"run_id": run_id, "target_table": source_table, "model": model_name, "proposed_by": user,
              "reviewed_by": None, "reviewed_at": None, "review_note": None}
    if rule_recs:
        rows = [{**common, "proposal_id": str(uuid.uuid4()), "tag": tag,
                 **{k: r[k] for k in ("rule_name", "column_name", "constraint_expr", "action", "rationale",
                                      "fail_rate", "validation_error", "status")}}
                for r in rule_recs]
        (spark.createDataFrame(rows, schema=RULE_PROPOSAL_SCHEMA)
         .withColumn("proposed_at", F.current_timestamp())
         .write.mode("append").saveAsTable(rule_proposals))
    if mapping_recs:
        rows = [{**common, "proposal_id": str(uuid.uuid4()),
                 **{k: m[k] for k in ("column_name", "raw_value", "clean_value", "rationale",
                                      "rows_in_sample", "validation_error", "status")}}
                for m in mapping_recs]
        (spark.createDataFrame(rows, schema=MAPPING_PROPOSAL_SCHEMA)
         .withColumn("proposed_at", F.current_timestamp())
         .write.mode("append").saveAsTable(mapping_proposals))
    return rule_recs, mapping_recs
