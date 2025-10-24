# PRD 09: Context Engineering Real Data Integration

**Status:** Draft  
**Priority:** High  
**Effort:** 6-8 hours  
**Dependencies:** PRD-08 (Document System)

---

## Executive Summary

The Context Engineering page currently displays a mix of **mock data**, **hardcoded values**, and **incomplete real data**. This PRD outlines the work required to connect all Context Engineering features to real data sources, implement proper tracking, and remove all mock/test data.

---

## Current State Analysis

### API Endpoints Status

| Endpoint | Status | Issues |
|----------|--------|--------|
| `GET /api/context/stats` | ✅ Exists | Returns zeros for queries/performance, only embeddings count is real (292) |
| `GET /api/context/performance` | ✅ Exists | Returns mock time-series data (all zeros) |
| `GET /api/context/sources` | ✅ Exists | Returns hardcoded mock data ("Technical Docs", "Configuration", etc.) |
| `GET /api/context/patterns` | ✅ Exists | Returns RAG configs but includes test/XSS junk data |
| `GET /api/context/queries/recent` | ✅ Exists | Returns empty array (no tracking implemented) |

### Frontend Components Status

| Component | Status | Issues |
|-----------|--------|--------|
| **RAG Context Builder** | ✅ Working | Already implemented in PRD-08, fully functional! |
| **Performance Charts** | ⚠️ Partial | Shows charts but data is all zeros |
| **Context Sources Pie Chart** | ❌ Mock | Hardcoded source distribution |
| **Query Analysis** | ❌ Empty | No query tracking |
| **Patterns** | ⚠️ Partial | Shows RAG configs but has junk test data |
| **Optimization** | ❌ Stub | Just placeholder text |

---

## Goals

1. **Remove ALL mock data** from Context Engineering endpoints
2. **Implement real-time tracking** for context queries
3. **Calculate actual performance metrics** from RAG usage
4. **Display real document sources** from database
5. **Clean up test/junk data** from RAG patterns
6. **Implement optimization recommendations**

---

## Detailed Requirements

### 1. Context Stats Enhancement (2h)

**Current:**
```json
{
  "contextQueries": 0,
  "retrievalSuccess": 0.0,
  "avgResponseTime": "0.000s",
  "vectorEmbeddings": 292,
  "systemStatus": "operational",
  "lastQueryTime": null
}
```

**Required Changes:**

#### Backend: `/api/context/stats`

Track and aggregate:
- **Total context queries** from `document_usage` table where `event_type = 'rag_query'`
- **Retrieval success rate** = successful RAG queries / total RAG queries
- **Avg response time** = average `execution_time_ms` from tracked queries
- **Vector embeddings** = count from `document_chunks` where `embedding IS NOT NULL` ✅ (already working)
- **Last query time** = MAX(timestamp) from recent RAG queries

**SQL Implementation:**
```sql
-- Context query stats
SELECT 
    COUNT(*) as total_queries,
    COUNT(CASE WHEN results_count > 0 THEN 1 END) as successful_queries,
    AVG(execution_time_ms) as avg_response_time,
    MAX(timestamp) as last_query_time
FROM document_usage
WHERE event_type = 'rag_query'
    AND timestamp >= NOW() - INTERVAL '24 hours';

-- Vector embeddings count (already working)
SELECT COUNT(*) 
FROM document_chunks 
WHERE embedding IS NOT NULL;
```

---

### 2. Context Performance Real-Time Tracking (2h)

**Current:** Returns mock hourly data with all zeros

**Required Changes:**

#### Backend: `/api/context/performance`

Aggregate real RAG usage by time intervals:

```sql
SELECT 
    DATE_TRUNC('hour', timestamp) as time_bucket,
    COUNT(*) as queries,
    (COUNT(CASE WHEN results_count > 0 THEN 1 END)::float / COUNT(*)::float) * 100 as success_rate,
    AVG(execution_time_ms) / 1000.0 as avg_latency
FROM document_usage
WHERE event_type IN ('document_searched', 'rag_query')
    AND timestamp >= NOW() - INTERVAL '24 hours'
GROUP BY DATE_TRUNC('hour', timestamp)
ORDER BY time_bucket;
```

**Response Format:**
```json
[
  {
    "time": "14:00",
    "queries": 45,
    "success_rate": 95.5,
    "avg_latency": 0.810
  }
]
```

---

### 3. Context Sources from Real Documents (1h)

**Current:** Hardcoded mock data

**Required Changes:**

#### Backend: `/api/context/sources`

Query actual document types from database:

```sql
SELECT 
    CASE 
        WHEN file_type IN ('pdf', 'docx', 'txt', 'md') THEN 'Documents'
        WHEN file_type IN ('json', 'yaml', 'xml') THEN 'Configuration'
        WHEN file_type IN ('py', 'js', 'ts', 'java', 'go') THEN 'Code Files'
        WHEN file_type IN ('csv', 'xlsx') THEN 'Data Files'
        ELSE 'Other'
    END as category,
    COUNT(*) as count
FROM documents
WHERE status = 'completed'
GROUP BY category
ORDER BY count DESC;
```

**Response Format:**
```json
[
  {
    "name": "Documents",
    "value": 45,
    "color": "#ff6b35"
  },
  {
    "name": "Code Files",
    "value": 23,
    "color": "#72BF78"
  }
]
```

---

### 4. Recent Context Queries Tracking (1.5h)

**Current:** Returns empty array

**Required Changes:**

#### Backend Enhancement

**Step 1:** Update RAG endpoint to track queries
Already done in PRD-08! Just need to ensure `event_type = 'rag_query'` is used.

**Step 2:** Implement `/api/context/queries/recent` endpoint

```python
@router.get("/queries/recent")
async def get_recent_context_queries(
    limit: int = Query(10, ge=1, le=50),
    db: Session = Depends(get_db)
):
    """Get recent context/RAG queries with performance metrics"""
    try:
        query = text("""
            SELECT 
                id,
                query,
                results_count,
                execution_time_ms,
                timestamp,
                metadata
            FROM document_usage
            WHERE event_type IN ('document_searched', 'rag_query')
                AND query IS NOT NULL
            ORDER BY timestamp DESC
            LIMIT :limit
        """)
        
        result = db.execute(query, {"limit": limit}).fetchall()
        
        queries = []
        for row in result:
            # Calculate confidence from results
            confidence = min(1.0, (row.results_count / 10.0)) if row.results_count else 0.0
            
            queries.append({
                "id": f"query-{row.id}",
                "query": row.query,
                "agent": "System",  # TODO: Add agent tracking
                "confidence": confidence,
                "sources": row.results_count,
                "latency": row.execution_time_ms,
                "responseTime": f"{row.execution_time_ms}ms",
                "timestamp": row.timestamp.strftime("%H:%M:%S") if row.timestamp else "Unknown",
                "category": "Semantic Search" if row.query else "General"
            })
        
        return queries
        
    except Exception as e:
        logger.error(f"Error getting recent queries: {e}")
        raise HTTPException(status_code=500, detail=str(e))
```

---

### 5. Context Patterns Cleanup (30m)

**Current:** Returns RAG configs but includes test/junk/XSS data

**Required Changes:**

#### Backend: `/api/context/patterns`

Add filtering to exclude test data:

```python
@router.get("/patterns")
async def get_context_patterns(db: Session = Depends(get_db)):
    """Get RAG configuration patterns with real usage stats"""
    try:
        # Get RAG configs from database
        configs = db.query(RAGConfig).filter(
            RAGConfig.name.notlike('%test%'),  # Exclude test entries
            RAGConfig.name.notlike('%<script>%'),  # Exclude XSS attempts
            RAGConfig.name != ''  # Exclude empty names
        ).all()
        
        patterns = []
        for config in configs:
            # Get actual usage stats from document_usage
            usage_query = text("""
                SELECT 
                    COUNT(*) as usage_count,
                    AVG(CASE WHEN results_count > 0 THEN 1.0 ELSE 0.0 END) * 100 as accuracy,
                    AVG(results_count) as avg_sources
                FROM document_usage
                WHERE event_type = 'rag_query'
                    AND metadata->>'config_id' = :config_id
            """)
            
            stats = db.execute(usage_query, {"config_id": str(config.id)}).fetchone()
            
            patterns.append({
                "id": f"pattern-{config.id}",
                "name": config.name,
                "description": f"Retrieval pattern using {config.embedding_model} with {config.retrieval_strategy} strategy",
                "usage": stats.usage_count if stats else 0,
                "accuracy": stats.accuracy if stats else 0.0,
                "avgSources": int(stats.avg_sources) if stats and stats.avg_sources else config.top_k,
                "category": "RAG Configuration",
                "status": "active" if config.is_active else "inactive"
            })
        
        return patterns
        
    except Exception as e:
        logger.error(f"Error getting context patterns: {e}")
        raise HTTPException(status_code=500, detail=str(e))
```

**Manual Cleanup:**
```sql
-- Delete test/junk RAG configs
DELETE FROM rag_config 
WHERE name LIKE '%test%' 
    OR name LIKE '%<script>%' 
    OR name = 'Test Item';
```

---

### 6. Optimization Tab Implementation (1h)

**Current:** Just placeholder text

**Required Changes:**

#### Backend: `/api/context/optimize` (NEW)

Analyze system and provide recommendations:

```python
@router.get("/optimize")
async def get_optimization_recommendations(db: Session = Depends(get_db)):
    """Analyze context system and provide optimization recommendations"""
    try:
        recommendations = []
        
        # Check 1: Embeddings coverage
        embedding_query = text("""
            SELECT 
                COUNT(*) as total_docs,
                COUNT(CASE WHEN status = 'completed' THEN 1 END) as completed_docs,
                (SELECT COUNT(*) FROM document_chunks WHERE embedding IS NOT NULL) as embedded_chunks
            FROM documents
        """)
        
        stats = db.execute(embedding_query).fetchone()
        
        if stats.completed_docs < stats.total_docs:
            recommendations.append({
                "type": "warning",
                "category": "Coverage",
                "title": "Incomplete Document Processing",
                "description": f"{stats.total_docs - stats.completed_docs} documents not fully processed",
                "action": "Reprocess failed documents",
                "impact": "high"
            })
        
        # Check 2: Query performance
        perf_query = text("""
            SELECT AVG(execution_time_ms) as avg_time
            FROM document_usage
            WHERE event_type IN ('document_searched', 'rag_query')
                AND timestamp >= NOW() - INTERVAL '24 hours'
        """)
        
        perf = db.execute(perf_query).fetchone()
        
        if perf and perf.avg_time > 1000:
            recommendations.append({
                "type": "warning",
                "category": "Performance",
                "title": "Slow Query Performance",
                "description": f"Average query time is {perf.avg_time:.0f}ms (target: <1000ms)",
                "action": "Consider adding more vector indexes or reducing chunk size",
                "impact": "medium"
            })
        
        # Check 3: Usage patterns
        usage_query = text("""
            SELECT COUNT(*) as query_count
            FROM document_usage
            WHERE event_type IN ('document_searched', 'rag_query')
                AND timestamp >= NOW() - INTERVAL '24 hours'
        """)
        
        usage = db.execute(usage_query).fetchone()
        
        if usage and usage.query_count == 0:
            recommendations.append({
                "type": "info",
                "category": "Usage",
                "title": "No Recent Context Queries",
                "description": "System has not been used for context retrieval in the last 24 hours",
                "action": "Monitor usage patterns",
                "impact": "low"
            })
        else:
            recommendations.append({
                "type": "success",
                "category": "Health",
                "title": "System Healthy",
                "description": f"{usage.query_count} queries processed in last 24 hours",
                "action": "No action needed",
                "impact": "low"
            })
        
        return {
            "recommendations": recommendations,
            "last_analyzed": datetime.now().isoformat(),
            "system_health": "healthy" if len([r for r in recommendations if r["type"] == "warning"]) == 0 else "needs_attention"
        }
        
    except Exception as e:
        logger.error(f"Error getting optimization recommendations: {e}")
        raise HTTPException(status_code=500, detail=str(e))
```

#### Frontend: Update Optimization Tab

Replace placeholder with:
- Display recommendations cards
- Color-coded by type (success=green, warning=yellow, error=red)
- Action buttons for each recommendation
- System health indicator

---

## Implementation Plan

### Phase 1: Backend Data Sources (3h)
1. ✅ Update `/api/context/stats` with real query tracking
2. ✅ Update `/api/context/performance` with time-series data
3. ✅ Update `/api/context/sources` with real document types
4. ✅ Implement `/api/context/queries/recent` with tracking
5. ✅ Clean up and enhance `/api/context/patterns`
6. ✅ Implement new `/api/context/optimize` endpoint

### Phase 2: Database Cleanup (30m)
1. ✅ Delete test/junk RAG configs
2. ✅ Verify data integrity
3. ✅ Add constraints to prevent junk data

### Phase 3: Frontend Integration (2h)
1. ✅ Remove mock data fallbacks from components
2. ✅ Update chart components to handle real data
3. ✅ Implement optimization tab UI
4. ✅ Add loading states and error handling
5. ✅ Test all tabs with real data

### Phase 4: Testing (1h)
1. ✅ Manual API testing
2. ✅ UI testing with real data
3. ✅ Performance testing
4. ✅ Edge case testing (empty data, errors)

---

## Success Criteria

✅ All endpoints return ONLY real data (no hardcoded/mock values)  
✅ Context queries are tracked and displayed in real-time  
✅ Performance charts show actual RAG usage patterns  
✅ Source distribution reflects actual document types in database  
✅ Recent queries tab shows actual search/RAG queries  
✅ Patterns tab shows only valid RAG configurations with real usage stats  
✅ Optimization tab provides actionable recommendations  
✅ No test/junk/XSS data visible in UI  

---

## Technical Notes

### Database Schema Requirements

The `document_usage` table must support:
- `event_type` values: `document_searched`, `rag_query`
- `query` field for storing search terms
- `results_count` for tracking result size
- `execution_time_ms` for performance tracking
- `metadata` JSONB for additional context

**Already implemented in PRD-08!** ✅

### Performance Considerations

- Use indexes on `document_usage.event_type` and `document_usage.timestamp`
- Cache optimization recommendations for 5 minutes
- Aggregate time-series data in 1-hour buckets (not per-minute)
- Limit recent queries to 50 maximum

---

## Out of Scope (Future PRDs)

- Agent-specific query tracking
- Custom optimization rules
- Automated performance tuning
- RAG configuration A/B testing
- Real-time WebSocket updates for queries
- Advanced analytics (query clustering, trend analysis)

---

## Comparison: Before vs After

### Before
- Context Stats: 3/6 metrics show zeros
- Performance Charts: All zeros (mock data)
- Context Sources: Hardcoded fake sources
- Recent Queries: Empty
- Patterns: Polluted with test data
- Optimization: "Coming soon" placeholder

### After
- Context Stats: All metrics from real usage
- Performance Charts: Actual hourly RAG usage
- Context Sources: Real document type distribution
- Recent Queries: Live search/RAG queries with performance
- Patterns: Clean RAG configs with usage stats
- Optimization: Actionable recommendations based on analysis

---

**Estimated Total Effort:** 6-8 hours  
**Priority:** High (required for MVP demo)  
**Dependencies:** PRD-08 document system (completed ✅)

