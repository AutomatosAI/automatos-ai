"""F077 (B) — an answer grouped by a code names what the code stands for.

Retest of refresh 4, question #6 ("How many active subscribers are on each
plan?", expected Harvest Club 63 / Regular 232 / Taster 88). NL2SQL wrote
``SELECT s.plan_code, COUNT(...) ... GROUP BY s.plan_code``, so the answer read
REGULAR / TASTER / CLUB: every number right, no plan named. The schema had the
foreign key subscribers.plan_code -> subscription_plans.plan_code (F067), and the
plans table has a name, but nothing told the model to use it. The prompt now says
so, marks each relationship with the column its target is read by, and brings a
chosen table's lookup tables into the schema.
"""
from __future__ import annotations

from modules.nl2sql.query.nl2sql_service import NaturalLanguageToSQLService


def _col(name, kind="text", **extra):
    return {"name": name, "type": kind, **extra}


SHOP = {
    "tables": [
        {"name": "subscribers", "columns": [_col("subscriber_id", "integer", primary_key=True),
                                            _col("plan_code"), _col("status")]},
        {"name": "subscription_plans", "columns": [_col("plan_code", primary_key=True), _col("name"),
                                                   _col("price_gbp", "numeric")]},
        {"name": "wholesale_accounts", "columns": [_col("account_id", "integer", primary_key=True), _col("cafe_name")]},
        {"name": "wholesale_orders", "columns": [_col("order_id", "integer", primary_key=True),
                                                 _col("account_id", "integer"), _col("kg", "numeric")]},
    ],
    "relationships": [
        {"from_table": "subscribers", "from_column": "plan_code", "to_table": "subscription_plans",
         "to_column": "plan_code", "type": "foreign_key"},
        {"from_table": "wholesale_orders", "from_column": "account_id", "to_table": "wholesale_accounts",
         "to_column": "account_id", "type": "foreign_key"},
    ],
}


def _prompt(question):
    return NaturalLanguageToSQLService(llm_provider=None)._build_prompt(
        question=question, schema_metadata=SHOP, semantic_layer=None, dialect="postgresql", examples=None)


def test_the_prompt_says_to_name_what_a_code_stands_for():
    prompt = _prompt("How many active subscribers are on each plan?")
    assert "join that table and show\n   its name" in prompt
    assert "show the code only if the question asks for it" in prompt
    assert ("subscribers.plan_code -> subscription_plans.plan_code (foreign_key; label: subscription_plans.name)"
            in prompt)
    assert "wholesale_orders.account_id -> wholesale_accounts.account_id (foreign_key; label: wholesale_accounts.cafe_name)" in prompt


def test_a_lookup_table_comes_with_the_table_that_codes_into_it():
    prompt = _prompt("How many active subscribers are there in each group?")  # names no plan
    schema = prompt.split("DATABASE SCHEMA:")[1].split("RELATIONSHIPS:")[0]
    assert "Table: subscribers" in schema
    assert "Table: subscription_plans" in schema and "  - name (text)" in schema


def test_the_label_is_the_column_a_person_reads_a_row_by():
    from modules.nl2sql.query.nl2sql_service import label_column

    assert label_column(SHOP["tables"][1]) == "name"
    assert label_column(SHOP["tables"][2]) == "cafe_name"
    assert label_column({"columns": [_col("sku"), _col("description")]}) == "description"
    assert label_column({"columns": [_col("id", "integer")]}) is None
