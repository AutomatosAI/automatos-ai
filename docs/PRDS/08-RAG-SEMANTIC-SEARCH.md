# PRD 08: Document RAG & Semantic Search System
**Updated with UI Integration Strategy**

## 1. Overview

### Purpose
Complete the Document Management system with production-ready semantic search, RAG (Retrieval Augmented Generation) capabilities, usage analytics, and real-time processing monitoring. Transforms the platform from document storage to an intelligent knowledge retrieval system.

### Vision Alignment
Following the Context Engineering paradigm:
- **Atoms**: Individual document chunks with embeddings
- **Molecules**: Semantically related chunks grouped by similarity
- **Cells**: RAG-augmented agent context with retrieved knowledge
- **Organs**: Multi-agent systems sharing document knowledge
- **Organisms**: Complete knowledge-aware orchestration workflows

---

## 2. UI Integration Strategy

### 2.1 🔍 Semantic Search Integration

**Location**: `components/documents/document-management.tsx`

**Implementation**: Add new tab **"Search"** (position after "Library", before "Upload")

**Current Tabs**:
1. Library - Document Library
2. **→ NEW: Search - Semantic Search** ← Add here
3. Upload - Upload
4. Processing - Processing
5. Analytics - Analytics
6. CodeGraph - CodeGraph

**Code Integration**:
```typescript
// In document-management.tsx, add new tab
<TabsTrigger value="search" className="flex items-center space-x-2">
  <Search className="w-4 h-4" />
  <span className="hidden sm:inline">Semantic Search</span>
</TabsTrigger>

<TabsContent value="search">
  <DocumentSemanticSearch 
    documents={documents} 
    onResultSelect={handleDocumentSelect}
  />
</TabsContent>
```

**Reusability**: ✅ YES!

**Component Design for Maximum Reuse**:
```typescript
// Create: components/documents/semantic-search.tsx
// Reusable across:
// 1. Document Management page (search documents)
// 2. Chatbot interface (search for context)
// 3. Agent execution panel (search for RAG context)
// 4. Context Engineering page (search patterns)

interface SemanticSearchProps {
  context: 'documents' | 'chatbot' | 'agent' | 'patterns'
  onResultSelect?: (result: SearchResult) => void
  onResultsChange?: (results: SearchResult[]) => void
  showActions?: boolean
  maxResults?: number
}

export function SemanticSearch({
  context,
  onResultSelect,
  onResultsChange,
  showActions = true,
  maxResults = 10
}: SemanticSearchProps) {
  // Universal semantic search component
  // Adapts behavior based on context
}
```

**Usage Examples**:
```typescript
// 1. In document-management.tsx
<SemanticSearch 
  context="documents" 
  onResultSelect={handleViewDocument}
  showActions={true}
/>

// 2. In chatbot.tsx
<SemanticSearch 
  context="chatbot" 
  onResultSelect={handleUseInChat}
  maxResults={5}
/>

// 3. In agent-execution.tsx
<SemanticSearch 
  context="agent" 
  onResultsChange={handleAddToContext}
  maxResults={3}
/>
```

---

### 2.2 🤖 RAG Retrieval Integration

**Location**: `components/context/context-engineering.tsx`

**Implementation**: Add new tab **"RAG Context"** (position after "Optimization")

**Current Tabs**:
1. Performance - Performance
2. Queries - Query Analysis
3. Patterns - Patterns
4. Optimization - Optimization
5. **→ NEW: RAG - RAG Context Builder** ← Add here

**Why Context Engineering Page?**
- RAG is fundamentally about **context optimization**
- Fits the "Context Engineering" paradigm perfectly
- Users building prompts/context will naturally look here
- Can visualize: Query → Retrieval → Context Building → Optimization

**Code Integration**:
```typescript
// In context-engineering.tsx, add new tab
<TabsTrigger value="rag" className="flex items-center space-x-2">
  <Brain className="w-4 h-4" />
  <span className="hidden sm:inline">RAG Context</span>
</TabsTrigger>

<TabsContent value="rag">
  <RAGContextBuilder 
    onContextBuilt={handleContextReady}
    showVisualization={true}
  />
</TabsContent>
```

**Component Features**:
```typescript
// Create: components/context/rag-context-builder.tsx
export function RAGContextBuilder() {
  return (
    <div className="space-y-6">
      {/* Query Input */}
      <RAGQueryInput onQuerySubmit={handleQuery} />
      
      {/* Live Context Building Animation */}
      <RAGPipeline 
        stages={['Search', 'Diversity', 'Token Budget', 'Format']}
        currentStage={currentStage}
        progress={progress}
      />
      
      {/* Retrieved Chunks with Sources */}
      <RAGChunkViewer 
        chunks={retrievedChunks}
        showSimilarity={true}
        showTokenCount={true}
      />
      
      {/* Final Context Preview */}
      <RAGContextPreview 
        context={formattedContext}
        tokenCount={totalTokens}
        diversityScore={diversity}
      />
      
      {/* Actions */}
      <RAGActions>
        <Button>Copy Context</Button>
        <Button>Use in Agent</Button>
        <Button>Adjust Settings</Button>
      </RAGActions>
    </div>
  )
}
```

**Integration with Agents**:
```typescript
// In agent-execution.tsx
// Button: "Add RAG Context"
<Button onClick={() => setShowRAGBuilder(true)}>
  <Brain className="w-4 h-4 mr-2" />
  Add Document Knowledge
</Button>

<RAGContextBuilderModal 
  open={showRAGBuilder}
  onContextBuilt={(context) => {
    // Add to agent prompt
    setAgentPrompt(prev => `${prev}\n\n${context}`)
  }}
/>
```

---

### 2.3 📊 Usage Analytics Integration

**Location**: `components/documents/document-analytics.tsx` (EXTEND EXISTING)

**Implementation**: Enhance existing analytics component

**Current State** (from document-analytics.tsx):
- ✅ Already has: Overview stats, Document types, Processing status, Categories
- ✅ Already has: Time range selector, Card layouts, Progress bars
- ❌ Missing: Usage tracking, Search patterns, Popular documents

**Enhancement Strategy**:
```typescript
// EXTEND document-analytics.tsx (don't replace)
// Add new sections to existing component

export function DocumentAnalytics({ documents, documentStats }: DocumentAnalyticsProps) {
  // ... existing code ...
  
  // ADD: Fetch usage analytics
  const { data: usageData } = useDocumentUsageAnalytics(timeRange)
  
  return (
    <div className="space-y-6">
      {/* KEEP: Existing overview stats */}
      {/* KEEP: Existing document types */}
      {/* KEEP: Existing processing status */}
      
      {/* ADD: New Usage Analytics Section */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <Card className="glass-card">
          <CardHeader>
            <CardTitle>Popular Searches (Last 7 Days)</CardTitle>
          </CardHeader>
          <CardContent>
            <PopularSearches data={usageData?.popular_searches} />
          </CardContent>
        </Card>
        
        <Card className="glass-card">
          <CardHeader>
            <CardTitle>Most Accessed Documents</CardTitle>
          </CardHeader>
          <CardContent>
            <PopularDocuments data={usageData?.popular_documents} />
          </CardContent>
        </Card>
      </div>
      
      {/* ADD: Activity Timeline */}
      <Card className="glass-card">
        <CardHeader>
          <CardTitle>Usage Over Time</CardTitle>
        </CardHeader>
        <CardContent>
          <TimeSeriesChart data={usageData?.time_series} />
        </CardContent>
      </Card>
      
      {/* ADD: Live Activity Feed */}
      <Card className="glass-card">
        <CardHeader>
          <CardTitle>Recent Activity</CardTitle>
        </CardHeader>
        <CardContent>
          <LiveActivityFeed events={usageData?.recent_events} />
        </CardContent>
      </Card>
      
      {/* KEEP: Existing recent activity */}
    </div>
  )
}
```

**New Components to Create**:
```typescript
// components/documents/analytics/popular-searches.tsx
// components/documents/analytics/popular-documents.tsx
// components/documents/analytics/time-series-chart.tsx
// components/documents/analytics/live-activity-feed.tsx
```

---

### 2.4 ⏱️ Processing Queue Integration

**Location**: `components/documents/document-processing.tsx` (ENHANCE EXISTING)

**Implementation**: Add WebSocket real-time updates to existing component

**Current State** (from document-processing.tsx):
- ✅ Already has: Processing stats cards
- ✅ Already has: Tabs (Queue, Active, Completed, Failed)
- ✅ Already has: Progress bars, ETA display
- ❌ Missing: WebSocket live updates, Real queue status from backend

**Enhancement Strategy**:
```typescript
// ENHANCE document-processing.tsx
export function DocumentProcessing({ documents, onDocumentSelect }: DocumentProcessingProps) {
  // EXISTING: Static processing state
  const [processingStats, setProcessingStats] = useState({...})
  
  // ADD: Real-time WebSocket connection
  const { 
    queueStatus, 
    isConnected 
  } = useProcessingQueueWebSocket()
  
  // ADD: Real processing queue from backend
  const { data: backendQueue } = useProcessingQueue()
  
  // MERGE: Backend data with local state
  const processingDocuments = useMemo(() => {
    return mergeWithBackendQueue(documents, backendQueue)
  }, [documents, backendQueue])
  
  return (
    <div className="space-y-6">
      {/* ENHANCE: Stats with real-time data */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <Card className="glass-card">
          <CardContent className="p-6">
            {/* ADD: Live indicator */}
            {isConnected && <LiveIndicator />}
            {/* EXISTING: Stats display */}
          </CardContent>
        </Card>
      </div>
      
      {/* KEEP: Existing tabs structure */}
      <Tabs defaultValue="queue">
        {/* ENHANCE: Active Processing tab with live updates */}
        <TabsContent value="active">
          {processingDocuments.map(doc => (
            <motion.div key={doc.id}>
              <Card className="glass-card">
                <CardContent className="p-4">
                  {/* ENHANCE: Real-time progress from WebSocket */}
                  <LiveProgressBar 
                    progress={doc.live_progress} 
                    step={doc.current_step}
                    eta={doc.eta_seconds}
                  />
                  
                  {/* ENHANCE: Step-by-step visualization */}
                  <ProcessingSteps 
                    steps={['Upload', 'Extract', 'Chunk', 'Embed', 'Store']}
                    currentStep={doc.current_step}
                  />
                </CardContent>
              </Card>
            </motion.div>
          ))}
        </TabsContent>
      </Tabs>
    </div>
  )
}
```

**New Hook to Create**:
```typescript
// hooks/use-processing-queue-websocket.ts
export function useProcessingQueueWebSocket() {
  const [queueStatus, setQueueStatus] = useState<QueueStatus | null>(null)
  const [isConnected, setIsConnected] = useState(false)
  
  useEffect(() => {
    const ws = new WebSocket(`wss://${API_URL.replace('https://', '')}/ws/documents/processing`)
    
    ws.onopen = () => setIsConnected(true)
    ws.onmessage = (event) => {
      const update = JSON.parse(event.data)
      setQueueStatus(update)
    }
    ws.onclose = () => setIsConnected(false)
    
    return () => ws.close()
  }, [])
  
  return { queueStatus, isConnected }
}
```

---

## 3. Component Architecture Summary

### 3.1 New Components to Create

```
components/
├── documents/
│   ├── semantic-search.tsx                   ← NEW (reusable)
│   ├── semantic-search-results.tsx          ← NEW
│   ├── semantic-search-filters.tsx          ← NEW
│   ├── analytics/
│   │   ├── popular-searches.tsx             ← NEW
│   │   ├── popular-documents.tsx            ← NEW
│   │   ├── time-series-chart.tsx            ← NEW
│   │   └── live-activity-feed.tsx           ← NEW
│   └── processing/
│       ├── live-progress-bar.tsx            ← NEW
│       ├── processing-steps.tsx             ← NEW
│       └── live-indicator.tsx               ← NEW
├── context/
│   ├── rag-context-builder.tsx              ← NEW
│   ├── rag-query-input.tsx                  ← NEW
│   ├── rag-pipeline.tsx                     ← NEW
│   ├── rag-chunk-viewer.tsx                 ← NEW
│   ├── rag-context-preview.tsx              ← NEW
│   └── rag-actions.tsx                      ← NEW
└── shared/
    ├── similarity-badge.tsx                  ← NEW (reusable)
    ├── token-meter.tsx                       ← NEW (reusable)
    └── diversity-indicator.tsx               ← NEW (reusable)
```

### 3.2 Existing Components to Enhance

```
ENHANCE:
├── document-management.tsx    → Add "Search" tab
├── document-analytics.tsx     → Add usage analytics sections
├── document-processing.tsx    → Add WebSocket live updates
└── context-engineering.tsx    → Add "RAG" tab
```

### 3.3 New Hooks to Create

```
hooks/
├── use-semantic-search-api.ts              ← NEW
├── use-rag-retrieve-api.ts                 ← NEW
├── use-document-usage-analytics-api.ts     ← NEW
├── use-processing-queue-websocket.ts       ← NEW
└── use-processing-queue-api.ts             ← NEW
```

---

## 4. Implementation Phases with UI Integration

### Phase 1: Semantic Search (Week 1)

**Backend** (30 min):
- Implement `POST /api/documents/search` endpoint
- pgvector similarity query
- Query embedding generation

**Frontend** (2 hours):
1. Create `semantic-search.tsx` component (reusable)
2. Create `semantic-search-results.tsx` component
3. Create `semantic-search-filters.tsx` component
4. Add "Search" tab to `document-management.tsx`
5. Create `use-semantic-search-api.ts` hook
6. Style with existing glass-card patterns

**Testing** (30 min):
- Test search from document management
- Test reusability in chatbot context
- Verify similarity scoring display

---

### Phase 2: RAG Retrieval (Week 2)

**Backend** (20 min):
- Implement `POST /api/rag/retrieve` endpoint
- MMR diversity algorithm
- Token budget optimization

**Frontend** (2.5 hours):
1. Create `rag-context-builder.tsx` main component
2. Create `rag-pipeline.tsx` (animated steps)
3. Create `rag-chunk-viewer.tsx` (chunk display)
4. Create `rag-context-preview.tsx` (formatted preview)
5. Add "RAG" tab to `context-engineering.tsx`
6. Create `use-rag-retrieve-api.ts` hook
7. Add integration with agent execution

**Testing** (30 min):
- Test RAG context building
- Verify diversity calculation display
- Test agent integration

---

### Phase 3: Usage Analytics (Week 3)

**Backend** (45 min):
- Implement `POST /api/documents/analytics/track` endpoint
- Implement `GET /api/documents/analytics/usage` endpoint
- Create database tables
- Event aggregation logic

**Frontend** (3 hours):
1. Create analytics sub-components:
   - `popular-searches.tsx`
   - `popular-documents.tsx`
   - `time-series-chart.tsx`
   - `live-activity-feed.tsx`
2. Enhance `document-analytics.tsx` with new sections
3. Create `use-document-usage-analytics-api.ts` hook
4. Add event tracking to search/view actions
5. Integrate charts with recharts

**Testing** (30 min):
- Verify event tracking
- Test chart rendering
- Verify time-series aggregation

---

### Phase 4: Processing Queue (Week 3)

**Backend** (30 min):
- Implement `GET /api/documents/queue/status` endpoint
- Implement WebSocket `/ws/documents/processing`
- Real-time status updates

**Frontend** (2 hours):
1. Create processing sub-components:
   - `live-progress-bar.tsx`
   - `processing-steps.tsx`
   - `live-indicator.tsx`
2. Enhance `document-processing.tsx` with WebSocket
3. Create `use-processing-queue-websocket.ts` hook
4. Create `use-processing-queue-api.ts` hook
5. Add animated status transitions

**Testing** (30 min):
- Test WebSocket connection
- Verify real-time updates
- Test reconnection logic

---

## 5. Design System Consistency

### 5.1 Reuse Existing Patterns

**Cards**:
```typescript
// Use existing glass-card class
<Card className="glass-card">
  <CardHeader>
    <CardTitle>Semantic Search</CardTitle>
  </CardHeader>
  <CardContent>
    {/* Content */}
  </CardContent>
</Card>
```

**Stats Cards** (from existing components):
```typescript
<Card className="glass-card">
  <CardContent className="p-6">
    <div className="flex items-center justify-between">
      <div>
        <p className="text-sm font-medium text-muted-foreground">Label</p>
        <p className="text-2xl font-bold">{value}</p>
        <p className="text-xs text-green-600 mt-1">Change</p>
      </div>
      <div className="p-3 rounded-xl bg-gradient-to-br from-blue-500 to-blue-600">
        <Icon className="w-6 h-6 text-white" />
      </div>
    </div>
  </CardContent>
</Card>
```

**Progress Bars** (existing):
```typescript
<Progress value={percentage} className="h-2" />
```

**Badges** (existing):
```typescript
<Badge variant="outline">Status</Badge>
<Badge variant="default">Type</Badge>
```

### 5.2 New Reusable Components

**Similarity Badge**:
```typescript
// components/shared/similarity-badge.tsx
export function SimilarityBadge({ score }: { score: number }) {
  const color = score >= 0.8 ? 'green' : score >= 0.6 ? 'yellow' : 'orange'
  return (
    <Badge className={`bg-${color}-500/10 text-${color}-400`}>
      {(score * 100).toFixed(0)}% Match
    </Badge>
  )
}
```

**Token Meter**:
```typescript
// components/shared/token-meter.tsx
export function TokenMeter({ current, max }: { current: number; max: number }) {
  const percentage = (current / max) * 100
  return (
    <div className="space-y-2">
      <div className="flex justify-between text-sm">
        <span>Tokens Used</span>
        <span>{current} / {max}</span>
      </div>
      <Progress value={percentage} className="h-2" />
    </div>
  )
}
```

---

## 6. API Integration Summary

### 6.1 New API Endpoints

```typescript
// Backend endpoints to implement

// Semantic Search
POST /api/documents/search
{
  query: string
  limit?: number
  min_similarity?: number
  document_ids?: number[]
}

// RAG Retrieval
POST /api/rag/retrieve
{
  query: string
  max_chunks?: number
  max_tokens?: number
  diversity?: number
}

// Usage Analytics
POST /api/documents/analytics/track
{
  event_type: string
  document_id?: number
  metadata: object
}

GET /api/documents/analytics/usage
?period=7d&group_by=day

// Processing Queue
GET /api/documents/queue/status

WebSocket: /ws/documents/processing
```

### 6.2 Hook Integration

```typescript
// Frontend hooks (to create)

// Semantic Search
const { data, isLoading } = useSemanticSearch(query, options)

// RAG Retrieval
const { data, isLoading } = useRAGRetrieve(query, options)

// Usage Analytics
const { data, isLoading } = useDocumentUsageAnalytics(period)
const trackEvent = useTrackDocumentEvent()

// Processing Queue
const { data, isLoading } = useProcessingQueue()
const { queueStatus, isConnected } = useProcessingQueueWebSocket()
```

---

## 7. Success Criteria with UI

### 7.1 Functional

**Semantic Search**:
- [ ] New "Search" tab visible in document management
- [ ] Search results display with similarity scores
- [ ] Results clickable to view full document
- [ ] Reusable in chatbot and agent contexts

**RAG Retrieval**:
- [ ] New "RAG" tab visible in context engineering
- [ ] Pipeline animation shows stages
- [ ] Chunks display with source citations
- [ ] Context copyable for agent use

**Usage Analytics**:
- [ ] Popular searches display in analytics
- [ ] Popular documents show view counts
- [ ] Time-series chart renders correctly
- [ ] Live activity feed updates

**Processing Queue**:
- [ ] WebSocket connection status visible
- [ ] Live progress updates without refresh
- [ ] Step-by-step visualization animates
- [ ] ETA countdown shows remaining time

### 7.2 Non-Functional

- [ ] Search latency < 500ms
- [ ] RAG retrieval < 1s
- [ ] UI responsive on mobile
- [ ] Animations smooth (60fps)
- [ ] WebSocket reconnects automatically
- [ ] Error states handled gracefully

### 7.3 Integration

- [ ] Semantic search works in 3+ contexts (documents, chat, agents)
- [ ] RAG context integrates with agent execution
- [ ] Analytics track all search/view events
- [ ] Processing queue shows real-time updates

---

## 8. Updated Timeline

### Week 1: Semantic Search
**Backend**: 30 min  
**Frontend**: 2 hours  
**Testing**: 30 min  
**Total**: 3 hours

### Week 2: RAG Retrieval
**Backend**: 20 min  
**Frontend**: 2.5 hours  
**Testing**: 30 min  
**Total**: 3 hours

### Week 3: Analytics + Queue
**Backend**: 1.25 hours  
**Frontend**: 5 hours  
**Testing**: 1 hour  
**Total**: 7.25 hours

**Grand Total**: ~13 hours for complete implementation

---

## 9. Priority Recommendation

### Recommended Order:
1. **Semantic Search** (3 hours) - Immediate user value, reusable
2. **Processing Queue** (2.5 hours) - Better UX, quick win
3. **RAG Retrieval** (3 hours) - Enable agent knowledge
4. **Usage Analytics** (4.5 hours) - Long-term insights

**Why this order?**
- Semantic Search gives immediate "wow factor"
- Processing Queue improves UX with minimal effort
- RAG Retrieval builds on search foundation
- Analytics can collect data while you implement others

---

## 10. Answers to Your Questions

### Q1: Would semantic search be a new tab in the document page?
**A**: YES - Add "Search" tab in `document-management.tsx` (position 2, after "Library")

### Q2: Can it be used for chatbot and agents?
**A**: YES - Design as reusable component with `context` prop:
```typescript
<SemanticSearch context="documents" /> // Document page
<SemanticSearch context="chatbot" />   // Chatbot
<SemanticSearch context="agent" />     // Agent execution
```

### Q3: Would RAG go in Context Engineering page?
**A**: YES - Add "RAG" tab in `context-engineering.tsx` (position 5, after "Optimization")  
**Why**: RAG IS context engineering - perfect conceptual fit

### Q4: Usage analytics fits into existing document analytics?
**A**: YES - ENHANCE existing `document-analytics.tsx`:
- Keep existing overview, types, status sections
- Add new sections: Popular searches, popular docs, time-series, activity feed
- Don't replace, extend!

### Q5: Processing queue fits into document-processing.tsx?
**A**: YES - ENHANCE existing `document-processing.tsx`:
- Keep existing tab structure and stats
- Add WebSocket live updates
- Enhance progress bars with real-time data
- Add animated step visualization

---

**This PRD now provides complete UI integration guidance for seamless implementation into your existing component architecture!** 🚀
