# rag_feedback (and sibling zero-row learning surfaces)

Captured 2026-07-04, read-only.

## Query

```sql
SELECT 'rag_feedback', count(*) FROM rag_feedback;  -- part of the census UNION query
```

**rag_feedback: 0 rows.** No sample possible.

Sibling learning/quality surfaces that are also empty (same census query):

| table | rows |
|---|---:|
| rag_feedback | 0 |
| knowledge_nodes | 0 |
| knowledge_edges | 0 |
| memory_items | 0 |
| database_query_audit | 0 |
| intent_classification_cache | 0 |
| harness_prescriptions | 0 |
| nl2sql_benchmark_runs / _results | 0 / 0 |
| approval_grants | 0 |

Context: the RAG corpus itself is populated (644 documents, 19,130 document_chunks) and chat ran through 2026-06-27, so there was retrieval activity that *could* have produced feedback.

## First look

The feedback loop around retrieval has never captured a single signal in production: `rag_feedback` is empty despite months of chat over a real 19k-chunk corpus. The persistent Knowledge Graph tables (`knowledge_nodes`/`knowledge_edges`) are also empty — whatever KG the product surfaces is being built elsewhere (per-workspace graph blobs, see workspace-graphs.md) or on the fly. For dossier teams: any code path claiming to *learn from retrieval feedback* is currently learning from nothing; treat uplift claims accordingly.
