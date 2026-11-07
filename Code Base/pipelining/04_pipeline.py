# Lakeflow Spark Declarative Pipeline source file.
#
# One pipeline covers every source table. For each table in dq.source_tables it
# builds <table>_clean in the pipeline's default schema (workspace.silver):
#   - approved category mappings applied
#   - every row kept; _dq_flags lists the approved "warn" rules the row failed
#   - approved "fail" rules stop the update if any row breaks them
#
# No rows are dropped. The output trains a risk model, and the rows that fail
# checks are often the ones it most needs; the modeling step decides what to do
# with them, and can use _dq_flags as features.
#
# Settings -> Configuration:
#   dq.source_tables     workspace.default.demand,workspace.default.prices,...
#   dq.rules_table       workspace.data_quality.rules
#   dq.mappings_table    workspace.data_quality.value_mappings
#   dq.rules_version     (optional) Delta version of rules_table to use
#   dq.mappings_version  (optional) Delta version of mappings_table to use
#
# Pin the two versions to whatever was current when you trained a model, so data
# you score later is cleaned exactly like the training data. Leave them empty to
# use the latest approved rules. Nothing here writes to the source tables.

from pyspark import pipelines as dp
from pyspark.sql import functions as F

SOURCE_TABLES = [t.strip() for t in spark.conf.get("dq.source_tables").split(",") if t.strip()]
RULES_TABLE = spark.conf.get("dq.rules_table")
MAPPINGS_TABLE = spark.conf.get("dq.mappings_table")
RULES_VERSION = spark.conf.get("dq.rules_version", "").strip()
MAPPINGS_VERSION = spark.conf.get("dq.mappings_version", "").strip()


def qcol(name):
    return F.col("`" + name.replace("`", "``") + "`")


def read_at_version(table, version):
    if version:
        return spark.sql(f"SELECT * FROM {table} VERSION AS OF {int(version)}")
    return spark.read.table(table)


def load_rules():
    """Approved, enabled rules: {target_table: {'warn': {name: expr}, 'fail': {name: expr}}}."""
    out = {}
    for r in (read_at_version(RULES_TABLE, RULES_VERSION).where(F.col("enabled"))
              .select("target_table", "rule_name", "constraint_expr", "action").collect()):
        action = "fail" if r["action"] == "fail" else "warn"  # anything else is treated as warn
        out.setdefault(r["target_table"], {"warn": {}, "fail": {}})[action][r["rule_name"]] = r["constraint_expr"]
    return out


def load_mappings():
    """Approved, enabled mappings: {target_table: {column_name: {raw_value: clean_value}}}."""
    out = {}
    for r in (read_at_version(MAPPINGS_TABLE, MAPPINGS_VERSION).where(F.col("enabled"))
              .select("target_table", "column_name", "raw_value", "clean_value").collect()):
        out.setdefault(r["target_table"], {}).setdefault(r["column_name"], {})[r["raw_value"]] = r["clean_value"]
    return out


def apply_mappings(df, mappings):
    """Replace raw values with canonical ones. Values without a mapping pass through unchanged."""
    for col_name, m in mappings.items():
        lookup = F.create_map(*[F.lit(x) for pair in m.items() for x in pair])
        c = qcol(col_name)
        df = df.withColumn(col_name, F.coalesce(F.try_element_at(lookup, c), c))
    return df


def flags_col(rules):
    """Array of rule names whose expression is not TRUE for this row (empty array if none)."""
    if not rules:
        return F.array().cast("array<string>")
    return F.array_compact(F.array(*[
        F.when(~F.coalesce(F.expr(expr), F.lit(False)), F.lit(name))
        for name, expr in rules.items()
    ]))


ALL_RULES = load_rules()
ALL_MAPPINGS = load_mappings()


def define_clean_table(source_table):
    short_name = source_table.split(".")[-1]
    rules = ALL_RULES.get(source_table, {"warn": {}, "fail": {}})
    mappings = ALL_MAPPINGS.get(source_table, {})

    def clean():
        # Mappings run first, so rules and flags see the cleaned values.
        df = apply_mappings(spark.read.table(source_table), mappings)
        return df.withColumn("_dq_flags", flags_col(rules["warn"]))

    # Expectations give per-rule pass/fail counts on the Data quality tab.
    # warn = keep the row and record the metric; fail = stop the update.
    fn = clean
    if rules["warn"]:
        fn = dp.expect_all(rules["warn"])(fn)
    if rules["fail"]:
        fn = dp.expect_all_or_fail(rules["fail"])(fn)

    dp.materialized_view(
        name=f"{short_name}_clean",
        comment=(f"{source_table} with approved mappings applied. No rows removed; "
                 f"_dq_flags lists failed warn rules."),
    )(fn)


for t in SOURCE_TABLES:
    define_clean_table(t)
