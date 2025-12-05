# 🗣️ NL2SQL - Talk to Your Database

> **"SQL is hard. English is easy."**

---

## 💡 The Problem

**Data is locked away.**
- "How many users signed up last week?"
- "What's the average revenue per active subscription?"
- "Show me the top 10 error logs from yesterday."

To answer these, you usually need:
1.  A developer.
2.  Access to the database.
3.  Knowledge of the schema.
4.  Time to write a SQL query.

**This is a bottleneck.** ⏳

---

## ✨ The Solution

**NL2SQL** (Natural Language to SQL) lets you ask questions in plain English and get answers instantly.

It's not just a "text-to-SQL" wrapper. It's a **safe, schema-aware query engine**.

### 🛡️ Safety First
- **Read-Only by Default**: We strictly enforce `SELECT` only. No `DROP TABLE`, no `DELETE`, no `UPDATE`.
- **Query Validation**: Every generated query is parsed and validated before execution.
- **Role-Based Access**: Respects the permissions of the user asking the question.

---

## 🚀 Capabilities

### 1️⃣ **Complex Joins? No Problem.**
"Show me the email of users who bought a 'Pro' plan in 2024."

**Generated SQL:**
```sql
SELECT u.email 
FROM users u
JOIN subscriptions s ON u.id = s.user_id
JOIN plans p ON s.plan_id = p.id
WHERE p.name = 'Pro' 
  AND s.created_at >= '2024-01-01';
```

### 2️⃣ **Schema Introspection**
We don't hallucinate table names. We read your *actual* database schema.
- Tables & Columns
- Foreign Keys
- Data Types
- Comments/Descriptions

### 3️⃣ **Self-Correction**
If the database returns an error (e.g., "column not found"), the agent **sees the error**, corrects the query, and retries automatically.

---

## 🏗️ Architecture

```
modules/nl2sql/
├── introspection/       # Schema discovery
│   ├── scanner.py       # Reads DB structure
│   └── mapper.py        # Maps schema to vector embeddings
├── generation/          # Query creation
│   ├── prompt.py        # Context-aware prompting
│   └── validator.py     # SQL safety checks
├── execution/           # Running queries
│   ├── runner.py        # Safe execution engine
│   └── formatter.py     # Result formatting (CSV, JSON, Markdown)
└── service.py           # Main API
```

---

## ⚡ How It Works

1.  **Question**: User asks "Who are my top customers?"
2.  **Retrieval**: We search the schema for relevant tables (`users`, `orders`, `payments`).
3.  **Prompting**: We construct a prompt with the *relevant* schema subset.
4.  **Generation**: The LLM generates a SQL query.
5.  **Validation**: We check for forbidden keywords (`DROP`, `ALTER`) and syntax errors.
6.  **Execution**: We run the query against the database.
7.  **Explanation**: We explain the result in plain English.

---

## 🛠️ Usage

### **Basic Query**

```python
from modules.nl2sql import NL2SQLService

service = NL2SQLService()

result = await service.query(
    question="How many active users do we have?",
    connection_id="prod-db"
)

print(result.answer) 
# "You have 1,245 active users."

print(result.sql)
# "SELECT count(*) FROM users WHERE status = 'active'"
```

### **Data Analysis**

```python
result = await service.analyze(
    question="What is the trend of signups over the last 6 months?",
    output_format="chart_data"
)

# Returns JSON ready for Chart.js / Recharts
```

---

## 🔮 The Future

- **Multi-Database Support**: Join data across Postgres and Snowflake.
- **Semantic Layer**: Define metrics like "Revenue" once, use everywhere.
- **Visualization**: Auto-generate charts and dashboards from questions.

**Unlock your data. Just ask.** 🔓
