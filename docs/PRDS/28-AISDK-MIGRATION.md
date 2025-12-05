# PRD-28: Vercel AI SDK Migration

## 1. Executive Summary
This document outlines the plan to migrate the Automatos AI Platform's workflow streaming architecture from a custom Server-Sent Events (SSE) implementation to the **Vercel AI SDK**. This migration aims to resolve persistent UI latency issues ("chunking/lag"), standardize the streaming protocol, and enable advanced UX features like smooth token streaming, automatic reconnections, and rich UI interactions.

## 2. Problem Statement
The current workflow streaming implementation relies on a custom SSE setup (`WorkflowStageTracker` -> Redis/Memory -> SSE Endpoint -> `EventSource`). While functional, it suffers from:
-   **Latency & Jitter**: Events often arrive in bursts due to buffering at various network layers (proxies, Nginx, browser), causing a "laggy" feel.
-   **Complexity**: Maintaining custom connection management, heartbeats, and error recovery logic is error-prone.
-   **Limited UX**: Implementing "typing effects" or smooth token updates requires significant custom frontend logic.
-   **Synchronization Issues**: Disconnects between backend state and frontend UI (e.g., "stuck" stages) due to missing events or race conditions.

## 3. Proposed Solution: Vercel AI SDK
We will adopt the **Vercel AI SDK** (specifically the Data Stream Protocol) as the standard for all real-time communication between the Orchestrator and the Frontend.

### Key Benefits
-   **Standardized Protocol**: Uses a robust, text-based protocol for streaming text, data, and tool calls.
-   **Optimized Streaming**: Designed specifically to minimize latency and handle token-by-token updates smoothly.
-   **Resilience**: Built-in automatic reconnection and error handling.
-   **Developer Experience**: Simple hooks (`useChat`, `useCompletion`) replace complex `EventSource` management.

## 4. Architecture Design

### 4.1 Data Stream Protocol
The backend will emit events in the AI SDK's Data Stream Protocol format. Each chunk is a line of text:

```text
0:"Hello"
0:" world"
d:{"type":"stage_update", "stage": 1, "status": "complete"}
e:{"code": "error_code", "message": "Something went wrong"}
```

-   `0`: Text delta (for LLM tokens)
-   `d`: Data payload (for workflow stage updates, logs, JSON objects)
-   `e`: Error information

### 4.2 Backend Architecture (FastAPI)
We will create a generic `StreamAdapter` that converts our internal workflow events into the AI SDK format.

**Current Flow:**
`WorkflowStageTracker` -> `WorkflowStreamManager` -> `SSE Generator` (Custom JSON)

**New Flow:**
`WorkflowStageTracker` -> `AISDKStreamAdapter` -> `StreamingResponse` (AI SDK Protocol)

**Code Example (Adapter):**
```python
async def stream_workflow_as_aisdk(execution_id: int):
    # Yield initial connection data
    yield json.dumps({"type": "data", "data": {"status": "connected"}}) + "\n\n"
    
    async for event in event_generator(execution_id):
        if event["type"] == "token":
            # Text delta for chat/response
            yield json.dumps({"type": "text-delta", "textDelta": event["content"]}) + "\n\n"
        elif event["type"] == "stage_update":
            # Custom data for UI state
            yield json.dumps({
                "type": "data", 
                "data": {"type": "stage_update", "payload": event["data"]}
            }) + "\n\n"
```

### 4.3 Frontend Architecture (Next.js)
We will replace the custom `useWorkflowExecution` hook with the AI SDK's `useChat` or `useCompletion` hooks.

**Integration:**
```typescript
import { useChat } from 'ai/react';

const { messages, data } = useChat({
  api: '/api/workflow/stream',
  body: { workflowId: 123 },
  onResponse: (response) => {
    // Handle initial connection
  },
  onFinish: () => {
    // Handle completion
  }
});

// 'data' will contain our stage updates
// 'messages' will contain the chat/log stream
```

## 5. Implementation Plan

### Phase 1: Backend Adapter (1-2 Days)
1.  Install `ai` package (if Python SDK exists) or implement protocol manually (simple JSON wrapping).
2.  Create `AISDKStreamAdapter` class in `services/workflow_streaming_service.py`.
3.  Create new endpoint `/api/workflows/{id}/stream/aisdk` that uses this adapter.
4.  Ensure both old SSE and new AI SDK endpoints work in parallel (for safe migration).

### Phase 2: Frontend Integration (1-2 Days)
1.  Install `ai` package: `npm install ai`.
2.  Create a Next.js Route Handler (`app/api/chat/route.ts`) to proxy requests to the FastAPI backend (avoids CORS issues and handles streaming headers correctly).
3.  Create a new component `WorkflowStreamViewer` using `useChat`.
4.  Map the `data` stream to the existing `ExecutionTheater` state (stages, logs).

### Phase 3: Verification & Switchover (1 Day)
1.  Run side-by-side comparison of old vs. new streaming.
2.  Verify latency improvement (measure time-to-first-token and stage update delay).
3.  Deprecate old SSE endpoint.
4.  Remove legacy `WorkflowStreamManager` code.

## 6. Migration Strategy
-   **Dual-Stack**: We will keep the current SSE implementation running while building the AI SDK implementation.
-   **Feature Flag**: Use a feature flag or a separate URL route to toggle between the two streaming methods in the UI.
-   **Rollback**: If issues arise, we can instantly revert to the SSE implementation.

## 7. Success Metrics
-   **Latency**: < 50ms latency for token updates.
-   **Reliability**: Zero "stuck" stages due to missed events.
-   **Code Quality**: Reduction in custom streaming code (backend & frontend).
