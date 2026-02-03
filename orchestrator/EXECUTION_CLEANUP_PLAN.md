# Execution Cleanup & State Management Plan

## Problem
Executions can get stuck in "running" state if:
- Backend crashes/restarts mid-execution
- Network interruptions
- Timeout not enforced
- Process killed

This causes UI to hang trying to stream completed/stuck executions.

## Solution: Robust State Management

### 1. Auto-Cleanup Stuck Executions

**Background Task** (runs every 5 minutes):
```python
async def cleanup_stuck_executions():
    """Mark executions as failed if running > max_timeout"""

    # Find executions stuck in "running" for > 30 minutes
    stuck = db.query(WorkflowExecution).filter(
        WorkflowExecution.status == 'running',
        WorkflowExecution.started_at < datetime.now() - timedelta(minutes=30)
    ).all()

    for execution in stuck:
        execution.status = 'failed'
        execution.completed_at = datetime.now()
        execution.output_data = {
            "error": "Execution timeout - marked as failed by cleanup job",
            "auto_cleanup": True
        }
        db.commit()

        # Publish event to close any hanging streams
        redis_client.publish_workflow_event(
            workflow_id=execution.workflow_id,
            execution_id=execution.id,
            event_type="execution_failed",
            data={"reason": "timeout"}
        )
```

Add to `main.py` startup:
```python
@app.on_event("startup")
async def start_background_tasks():
    asyncio.create_task(cleanup_stuck_executions_loop())
```

### 2. Graceful Stream Handling

**Frontend Stream Proxy** (`/api/workflows/stream`):
```typescript
// Before opening stream, check execution status
const execution = await fetch(`/api/workflows/executions/${id}`)
const { status } = await execution.json()

if (status === 'completed' || status === 'failed') {
  // Don't open stream - show final state immediately
  return Response.json({
    error: 'Execution already finished',
    status,
    execution
  })
}

// Only stream if status is "pending" or "running"
```

**Backend Stream Endpoint** (`/api/workflows/executions/{id}/stream/aisdk`):
```python
@router.get("/{execution_id}/stream/aisdk")
async def stream_execution_aisdk(execution_id: int, db: Session = Depends(get_db)):
    execution = db.query(WorkflowExecution).filter_by(id=execution_id).first()

    # Check if execution is already done
    if execution.status in ['completed', 'failed', 'cancelled']:
        # Return immediately with final state
        return JSONResponse({
            "error": "Execution already finished",
            "status": execution.status,
            "completed_at": execution.completed_at.isoformat()
        })

    # Otherwise, proceed with SSE stream
    return EventSourceResponse(generate_events(execution_id))
```

### 3. UI "Re-run" Button

**Don't re-use execution IDs** - Create new execution:

```typescript
// ❌ BAD: Try to restart same execution
await fetch(`/api/workflows/executions/${executionId}/restart`)

// ✅ GOOD: Create new execution with same config
const workflow = executions.find(e => e.id === executionId)?.workflow
await fetch(`/api/workflows/${workflow.id}/execute-advanced`, {
  method: 'POST',
  body: JSON.stringify({ input_data: originalInputData })
})
```

Button in UI:
```tsx
<Button onClick={handleRerun}>
  <RotateCw className="w-4 h-4" />
  Re-run (New Execution)
</Button>
```

### 4. Execution Lifecycle States

```
PENDING → RUNNING → COMPLETED ✅
              ↓
            FAILED ❌
              ↓
         CANCELLED 🛑
```

**Never go backwards** - states are final.

### 5. Timeout Configuration

Add to WorkflowExecution model:
```python
max_execution_time = Column(Integer, default=1800)  # 30 minutes
timeout_at = Column(DateTime)  # Set when starting

# On execution start:
execution.timeout_at = datetime.now() + timedelta(seconds=execution.max_execution_time)
```

### 6. Execution Monitoring Dashboard

Show in UI:
```
⏱️  Running: 2 executions
✅ Completed (last hour): 45 executions
❌ Failed (last hour): 3 executions
🧹 Auto-cleaned: 1 execution
```

Alert if stuck executions > 5

## Implementation Priority

1. **HIGH**: Auto-cleanup background task (prevent accumulation)
2. **HIGH**: Stream endpoint checks (prevent UI hangs)
3. **MEDIUM**: UI re-run button (better UX)
4. **MEDIUM**: Timeout configuration (per-workflow limits)
5. **LOW**: Monitoring dashboard (nice to have)

## Database Cleanup Strategy

**Keep executions for audit:**
- Last 1000 executions per workflow
- OR executions from last 30 days
- Archive older executions to S3/cold storage

**DO NOT delete:**
- Failed executions (debugging)
- Executions with quality_score > 0.8 (learning data)
- Manual executions (user-initiated)

**Cleanup cron** (weekly):
```sql
-- Archive executions older than 90 days with low importance
SELECT * INTO archive_executions
FROM workflow_executions
WHERE created_at < NOW() - INTERVAL '90 days'
  AND status = 'completed'
  AND quality_score < 0.5;

DELETE FROM workflow_executions
WHERE created_at < NOW() - INTERVAL '90 days'
  AND status = 'completed'
  AND quality_score < 0.5;
```

## Testing

1. Start execution, kill backend mid-run, verify auto-cleanup marks as failed
2. Try to stream completed execution, verify immediate response (no hang)
3. Click "Re-run" on failed execution, verify new execution created
4. Let execution run for > 30 minutes, verify timeout cleanup

## Monitoring Metrics

- `executions_stuck_cleaned`: Counter of auto-cleaned executions
- `execution_duration_seconds`: Histogram of execution times
- `stream_connection_duration`: How long streams stay open
- `execution_state_transitions`: Count of state changes
