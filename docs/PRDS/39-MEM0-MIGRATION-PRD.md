# PRD-39: Memory Migration to Mem0

**Status:** 🟡 Planning Phase
**Date:** January 27, 2026
**Author:** Automatos AI
**Related:** PRD-05 (Memory Systems)

---

## Executive Summary

The objective is to migrate Automatos AI's internal memory system to use **mem0**, a dedicated memory layer for AI agents. Mem0 offers a robust, open-source solution for managing long-term user memory, session context, and intelligent retrieval. We will utilize the self-hosted mem0 instance running on Railway (`automatos-mem0-server.railway.internal`) to provide a unified memory backend for all Automatos consumers (Chatbot, Workflows, Agents).

This migration aims to resolve issues with memory flakiness ("calling too often", "saying randomly") by leveraging mem0's production-ready features like intelligent fact extraction and structured retrieval.

## Goals

1.  **Migrate Memory Backend**: detailed in `modules/memory` to use mem0 as the primary storage and retrieval engine.
2.  **Unified Memory**: Ensure all consumers (Chatbot, Agents) read/write to the same mem0 instance.
3.  **Improved Stability**: addressing existing reliability issues by adopting mem0's proven architecture.
4.  **Simplicity**: Reduce custom code maintenance by offloading memory complexity to mem0.

## Architecture

### Current State
-   **Storage**: `pgvector` (Vectors) + Custom Knowledge Graph.
-   **Logic**: Custom `MemoryInjector`, `HierarchicalMemorySystem`.
-   **Issues**: Retrieval logic is custom and sometimes overly aggressive or irrelevant.

### Proposed State
-   **Backend**: Mem0 Server (Railway Enclave).
-   **Client**: `mem0ai` Python SDK (or direct API if SDK is too heavy/incompatible).
-   **Integration Point**: `modules/memory/integrations/mem0_client.py`.
-   **Pattern**: Hybrid approach initially (as recommended in research), but moving towards full mem0 adoption.

**Flow:**
1.  **Chatbot** receives message.
2.  **Memory Service** calls mem0 `v1/memories/search` (via SDK/API) with `user_id` (scoped to Workspace/User).
3.  **Context Injection**: Relevant memories injected into LLM context.
4.  **Storage**: Post-response, conversation is sent to mem0 `v1/memories/` (add) for async processing/fact extraction.

## Implementation Details

### 1. New Client Integration
Create `modules/memory/integrations/client.py`:
-   Wrapper around `mem0` SDK or `requests`.
-   Handles configuration (URL, API Key).
-   methods: `add()`, `search()`, `get_all()`.

### 2. Refactor Memory Service
Update `modules/memory/service.py`:
-   Replace/Augment `HierarchicalMemorySystem` calls with Mem0 client.
-   Ensure `user_id` mapping: `f"workspace_{workspace_id}_user_{user_id}"` to ensure isolation.

### 3. Dependencies
-   Add `mem0ai` to `requirements.txt`.
-   Env Var: `MEM0_API_URL=http://automatos-mem0-server.railway.internal`.

### 4. Verification
-   Use `scripts/test_mem0_railway.py` for connectivity checks.
-   New Integration Test: `tests/integration/test_memory_flow.py`.

## Migration Strategy
1.  **Add Mem0 Client**: Parallel to existing memory.
2.  **Dual Write**: Write to both systems (optional, or just switch).
3.  **Switch Read**: Toggle to read from mem0.
4.  **Deprecated**: Remove old memory code after validation.

## Success Metrics
-   **Connectivity**: 100% success rate connecting to Railway instance.
-   **Latency**: Retrieval < 200ms.
-   **Relevance**: Subjective improvement in chat context recall.
