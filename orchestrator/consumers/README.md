# ⚡ Consumers - The Background Powerhouse

> **"While your API sleeps, we're making magic happen."**

---

## 💡 What Are Consumers?

Think of consumers as **the backstage crew** of Automatos:

- 🎬 **API Layer:** The stage (where users interact)
- 🎭 **Consumers:** The crew (making it all work)

**When you:**
- Upload a 100-page PDF → **Document Processor** chunks it in the background
- Start a complex workflow → **Workflow Consumer** orchestrates it asynchronously  
- Send a chat message → **Chatbot Consumer** streams the response in real-time

**You see instant response. Consumers do the heavy lifting.** 💪

---

## 🏗️ Architecture

```
consumers/
├── chatbot/                  # Real-time chat streaming
│   ├── streaming.py          # StreamingChatService
│   ├── tool_router.py        # Tool execution routing
│   └── history.py            # Chat history management
│
├── document_processor/       # Async document processing
│   ├── processor.py          # Main processing logic
│   ├── chunker.py            # Smart text chunking
│   ├── embedder.py           # Generate embeddings
│   └── queue.py              # Task queue management
│
└── workflows/                # Workflow execution
    ├── streaming.py          # Workflow progress streaming
    ├── executor.py           # Background execution
    └── stage_tracker.py      # 9-stage tracking
```

---

## 🔥 Consumer Highlights

### 1️⃣ **Chatbot Consumer** - SSE Streaming Magic

**The Problem:** WebSockets are complex, break on network changes, require special infrastructure.

**The Solution:** Server-Sent Events (SSE) + AI SDK format

```python
# consumers/chatbot/streaming.py

async def stream_chat_response(
    message: str,
    session_id: str,
    stream_manager: SSEStreamManager
):
    """Stream chat responses in real-time"""
    
    # 1. LLM generates response
    async for chunk in llm_client.stream_completion(message):
        # Stream to client via SSE
        await stream_manager.send_event({
            "type": "chunk",
            "content": chunk.text
        })
    
    # 2. LLM requests tool use
    if chunk.function_call:
        # Execute tool in background
        result = await tool_router.execute_tool(
            tool_name=chunk.function_call.name,
            args=chunk.function_call.arguments
        )
        
        # Stream tool result
        await stream_manager.send_event({
            "type": "tool_result",
            "tool": chunk.function_call.name,
            "result": result
        })
    
    # 3. Save to history
    await save_chat_history(session_id, message, response)
```

**Why It's Cool:**
- ✅ **Auto-reconnect** built into browsers
- ✅ **HTTP/2 multiplexing** (1 connection, many streams)
- ✅ **Simpler** than WebSocket
- ✅ **Works everywhere** (no special infrastructure)

### 2️⃣ **Document Processor** - Intelligent Background Processing

**The Problem:** User uploads 500MB PDF. They don't want to wait 5 minutes for a response.

**The Solution:** Queue it, process it, notify when done.

```python
# consumers/document_processor/processor.py

async def process_document_async(document_id: int, db: Session):
    """Process document in background"""
    
    try:
        # 1. Update status
        document = db.query(Document).get(document_id)
        document.status = "processing"
        db.commit()
        
        # 2. Extract text
        text = await extract_text(document.file_path)
        
        # 3. Smart chunking (context-aware)
        chunks = await smart_chunker.chunk(
            text=text,
            max_chunk_size=1000,
            overlap=200,
            preserve_context=True  # Don't split mid-sentence!
        )
        
        # 4. Generate embeddings (batch for speed)
        embeddings = await batch_embed(chunks)
        
        # 5. Store in vector DB
        await vector_db.insert_many(
            documents=chunks,
            embeddings=embeddings,
            metadata={"source_id": document_id}
        )
        
        # 6. Update status
        document.status = "completed"
        document.chunk_count = len(chunks)
        db.commit()
        
        # 7. Notify user via Redis pub/sub
        await redis.publish("document_processed", {
            "document_id": document_id,
            "chunks": len(chunks)
        })
        
    except Exception as e:
        # Robust error handling
        document.status = "failed"
        document.error_message = str(e)
        db.commit()
        logger.error(f"Document processing failed: {e}")
```

**Why It's Cool:**
- ✅ **Non-blocking** - API returns immediately
- ✅ **Resilient** - Failures don't crash the API
- ✅ **Scalable** - Add more workers to process faster
- ✅ **Observable** - Track progress via Redis pub/sub

### 3️⃣ **Workflow Consumer** - Orchestration at Scale

**The Problem:** Workflows can be long-running and complex (multi-agent, multi-stage).

**The Solution:** Background execution with real-time progress streaming.

```python
# consumers/workflows/streaming.py

async def execute_workflow_with_streaming(
    execution_id: int,
    workflow: Workflow,
    stream_manager: SSEStreamManager
):
    """Execute workflow and stream progress"""
    
    tracker = WorkflowStageTracker(execution_id, stream_manager)
    
    # Stage 1: Initialize
    await tracker.emit_stage_start(1, "Initializing workflow...")
    context = await initialize_workflow_context(workflow)
    await tracker.emit_stage_complete(1)
    
    # Stage 2: Decompose into subtasks
    await tracker.emit_stage_start(2, "Breaking down work...")
    subtasks = await llm_decompose_tasks(workflow.goal, workflow.context)
    await tracker.emit_stage_complete(2, {"subtask_count": len(subtasks)})
    
    # Stage 3-7: Execute with different strategies
    for i, subtask in enumerate(subtasks):
        await tracker.emit_subtask_start(subtask.id, subtask.name)
        
        # Execute based on strategy (parallel, sequential, etc.)
        result = await execute_strategy.run(subtask, context)
        
        await tracker.emit_subtask_complete(subtask.id, result)
    
    # Stage 8: Aggregate results
    await tracker.emit_stage_start(8, "Combining results...")
    final_result = await aggregate_results(subtask_results)
    await tracker.emit_stage_complete(8)
    
    # Stage 9: Quality check
    await tracker.emit_stage_start(9, "Quality verification...")
    quality_score = await assess_quality(final_result)
    await tracker.emit_stage_complete(9, {"score": quality_score})
    
    # Done!
    await tracker.emit_workflow_complete(final_result)
```

**Why It's Cool:**
- ✅ **9-stage tracking** - Users see exactly what's happening
- ✅ **Real-time updates** - SSE streams progress
- ✅ **Multi-agent** - Coordinates agent collaboration
- ✅ **Resumable** - Can pause/resume workflows

---

## 🎯 Why Background Processing Matters

### **Without Consumers (Blocking)**
```
User Request → API → Process (5 minutes) → Response
                     ⏰ User waits... and waits...
```

❌ Terrible UX  
❌ Timeouts  
❌ Wasted connections  
❌ Can't scale  

### **With Consumers (Non-blocking)**
```
User Request → API → Queue Task → Immediate Response ✅
                     ↓
              Consumer → Process in background
                     ↓
              Notify user when done (Redis/SSE)
```

✅ **Instant response**  
✅ **No timeouts**  
✅ **Scalable** (add more workers)  
✅ **Resilient** (retry on failure)  

---

## 🚀 Creating a New Consumer

### **Step 1: Define the Service**

```python
# consumers/your_consumer/processor.py

async def process_your_thing(
    thing_id: int,
    db: Session,
    stream_manager: Optional[SSEStreamManager] = None
):
    """Process thing in background"""
    
    try:
        # Get thing from DB
        thing = db.query(Thing).get(thing_id)
        thing.status = "processing"
        db.commit()
        
        # Do the work
        result = await do_heavy_computation(thing)
        
        # Update DB
        thing.status = "completed"
        thing.result = result
        db.commit()
        
        # Stream update if connected
        if stream_manager:
            await stream_manager.send_event({
                "type": "thing_processed",
                "thing_id": thing_id,
                "result": result
            })
        
        # Publish to Redis for other services
        await redis_client.publish("thing_processed", {
            "thing_id": thing_id
        })
        
    except Exception as e:
        thing.status = "failed"
        thing.error = str(e)
        db.commit()
        logger.error(f"Processing failed: {e}")
```

### **Step 2: Add API Endpoint**

```python
# api/your_api.py

@router.post("/things/{thing_id}/process")
async def process_thing(
    thing_id: int,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """Trigger background processing"""
    
    # Queue the task
    background_tasks.add_task(
        process_your_thing,
        thing_id=thing_id,
        db=db
    )
    
    return {"message": "Processing started", "thing_id": thing_id}
```

### **Step 3: Add Streaming Endpoint (Optional)**

```python
@router.get("/things/{thing_id}/stream")
async def stream_thing_process(thing_id: int):
    """Stream processing progress"""
    
    stream_manager = SSEStreamManager()
    
    async def event_generator():
        # Listen to Redis for updates
        async for event in stream_manager.listen(f"thing_{thing_id}"):
            yield f"data: {json.dumps(event)}\n\n"
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream"
    )
```

---

## 🤝 Contributing to Consumers

### **High-Impact Ideas**

| Feature | Difficulty | Impact |
|---------|-----------|--------|
| Video processing consumer | 🟡 Medium | 🔥 High |
| Code compilation consumer | 🟡 Medium | ⭐ Medium |
| Image generation consumer | 🟢 Easy | 🔥 High |
| Email sending consumer | 🟢 Easy | ⭐ Medium |
| Scheduled tasks consumer | 🟡 Medium | 🔥 High |
| Multi-stage pipelines | 🔴 Hard | 🔥 High |

### **Consumer Best Practices**

1. **Always use explicit DB sessions** - Don't rely on context managers with async
2. **Handle errors gracefully** - Update status, log, notify
3. **Make it observable** - Redis pub/sub, SSE streaming
4. **Keep it idempotent** - Safe to retry
5. **Add progress tracking** - Users want to know what's happening

---

## 🌊 SSE Streaming Deep Dive

### **Why SSE > WebSocket for Automatos**

| Feature | SSE | WebSocket |
|---------|-----|-----------|
| **Direction** | Server → Client | Bidirectional |
| **Protocol** | HTTP | Custom |
| **Reconnect** | Automatic | Manual |
| **HTTP/2** | Yes (multiplexing) | No |
| **Complexity** | Low | Higher |
| **Use Case** | **Our use case!** | Real-time games |

**We chose SSE because:**
- We only need server → client (progress updates)
- Browser auto-reconnect is perfect
- HTTP/2 multiplexing = efficient
- Simpler to implement and debug

### **SSE Event Format**

```
data: {"type": "stage_start", "stage": 2, "message": "Processing..."}

data: {"type": "progress", "percent": 50}

data: {"type": "stage_complete", "stage": 2, "result": {...}}
```

### **Client-Side (JavaScript)**

```javascript
const eventSource = new EventSource('/api/workflows/executions/123/stream');

eventSource.onmessage = (event) => {
  const data = JSON.parse(event.data);
  
  if (data.type === 'stage_start') {
    updateUI(`Starting: ${data.message}`);
  } else if (data.type === 'progress') {
    updateProgressBar(data.percent);
  }
};

eventSource.onerror = () => {
  // Browser will auto-reconnect!
  console.log('Connection lost, reconnecting...');
};
```

---

## ⚡ Performance & Scaling

### **Worker Pools**

```bash
# Scale horizontally - add more workers
docker-compose up --scale worker=5

# Each worker processes tasks from queue
# Redis ensures no duplicate processing
```

### **Queue Priority**

```python
# High-priority tasks first
await task_queue.enqueue(
    task=process_vip_document,
    priority="high"  # Processed before "normal" tasks
)
```

### **Rate Limiting**

```python
# Prevent overwhelming external APIs
@rate_limit(requests=100, per_seconds=60)
async def call_external_api(...):
    # Max 100 calls per minute
```

---

## 🎯 The Vision

**Today:** Consumers handle async work  
**Tomorrow:** Intelligent task scheduling, priority learning, autonomous optimization

**Imagine:**
- Consumers that learn optimal batch sizes
- Auto-scaling based on queue depth
- Predictive pre-processing (process before user asks!)
- Cross-consumer collaboration (workflow triggers document processing)

---

## 📚 Learn More

- **[Chatbot Streaming](chatbot/README.md)** - How chat works
- **[Document Processing](document_processor/README.md)** - The pipeline
- **[Workflow Execution](workflows/README.md)** - Orchestration details

---

**Ready to build the consumer that changes everything?** ⚡

Start in `consumers/` and make background processing beautiful!
