# Drawings

## Figure 1: System Architecture — Shared Semantic Field

```
+------------------+     +------------------+     +------------------+
|    AGENT A       |     |    AGENT B       |     |    AGENT C       |
|  (Researcher)    |     |  (Analyst)       |     |  (Writer)        |
+--------+---------+     +--------+---------+     +--------+---------+
         |                        |                        |
    inject|                  inject|                   query|
         |                        |                        |
         v                        v                        v
+------------------------------------------------------------------------+
|                                                                        |
|                    SHARED SEMANTIC VECTOR FIELD                         |
|                  (Mission-Scoped Vector Collection)                     |
|                                                                        |
|  +------------+  +------------+  +------------+  +------------+        |
|  | Pattern 1  |  | Pattern 2  |  | Pattern 3  |  | Pattern 4  |       |
|  | agent: A   |  | agent: A   |  | agent: B   |  | agent: A   |       |
|  | key: "EU   |  | key: "risk |  | key: "gap  |  | key: "dead |       |
|  |  AI Act"   |  |  matrix"   |  |  analysis" |  |  line"     |       |
|  | str: 1.0   |  | str: 0.95  |  | str: 1.0   |  | str: 0.85  |       |
|  | cos: ----  |  | cos: ----  |  | cos: ----  |  | cos: ----  |       |
|  +------------+  +------------+  +------------+  +------------+        |
|                                                                        |
|  Retrieval: By SEMANTIC MEANING (not by agent, not by time)            |
|  Ranking:   By RESONANCE SCORE (not raw cosine similarity)             |
|                                                                        |
+------------------------------------------------------------------------+
         |                        |                        |
         v                        v                        v
+------------------+     +------------------+     +------------------+
|  VECTOR DATABASE |     | EMBEDDING MODEL  |     |    MISSION       |
|  (Qdrant)        |     | (2048-dim)       |     |  COORDINATOR     |
+------------------+     +------------------+     +------------------+
```

## Figure 2: Pattern Lifecycle

```
                    INJECTION
                       |
                       v
              +------------------+
              |  Compute SHA-256 |
              |  content hash    |
              +--------+---------+
                       |
              +--------v---------+
              | Duplicate check  |----YES----> Reinforce existing
              | (hash lookup)    |             (increment access_count,
              +--------+---------+              update last_accessed)
                       |
                      NO
                       |
              +--------v---------+
              | Generate semantic|
              | embedding vector |
              +--------+---------+
                       |
              +--------v---------+
              | Store pattern    |
              | strength = S_0   |
              | x permeability   |
              +--------+---------+
                       |
                       v
               PATTERN IN FIELD
                       |
          +------------+------------+
          |                         |
    QUERIED                   NOT QUERIED
          |                         |
    +-----v------+           +------v-----+
    | Resonance  |           | Temporal   |
    | scored     |           | decay      |
    | Retrieved  |           | continues  |
    | access++   |           |            |
    | co-access  |           | Eventually |
    | bonus      |           | falls below|
    +-----+------+           | archival   |
          |                  | threshold  |
    REINFORCED               +------+-----+
    (resists decay)                 |
                              ARCHIVED
                        (excluded from results)
```

## Figure 3: Resonance Scoring Pipeline

```
QUERY: "What compliance deadlines exist?"
         |
         v
+-------------------+
| Generate query    |
| embedding vector  |
| (2048 dimensions) |
+---------+---------+
          |
          v
+-------------------+
| Vector DB search  |
| Fetch 3 x top_k   |
| by ANN similarity |
+---------+---------+
          |
          v  (for each candidate)
+-------------------+
| Raw cosine sim    |-----> cos(theta) = 0.85
| query . pattern   |
+---------+---------+
          |
          v
+-------------------+
| Square cosine     |-----> cos^2(theta) = 0.7225
| (amplify strong   |
|  matches)         |
+---------+---------+
          |
          v
+-------------------+
| Temporal decay    |-----> e^(-0.1 x 2.0) = 0.8187
| (hours since last |       (2 hours old)
|  access)          |
+---------+---------+
          |
          v
+-------------------+
| Access boost      |-----> min(1 + 0.05 x 5, 2.0) = 1.25
| (5 prior accesses)|       (5 previous retrievals)
+---------+---------+
          |
          v
+-------------------+
| RESONANCE SCORE   |-----> R = 0.7225 x 1.0 x 0.8187 x 1.25
| R = cos^2 x S_0   |       R = 0.7394
|   x decay x boost |
+---------+---------+
          |
          v
+-------------------+
| Archival filter   |-----> S(t) = 1.0 x 0.8187 x 1.25 = 1.023
| S(t) > 0.05?     |       YES: include in results
+---------+---------+
          |
          v
+-------------------+
| Sort by resonance |
| Return top_k      |
+-------------------+
```

## Figure 4: Mission Lifecycle Integration

```
MISSION START
     |
     v
+--------------------+
| 1. Create Qdrant   |
|    collection       |
|    "field_{uuid}"   |
+----------+---------+
           |
           v
+--------------------+
| 2. Create payload  |
|    indexes:         |
|    - content_hash  |
|    - agent_id      |
|    - created_at    |
+----------+---------+
           |
           v
+--------------------+
| 3. Seed with       |
|    mission goal     |
|    (strength=1.0,   |
|     agent_id=0)     |
+----------+---------+
           |
           v
+========================+
|   MISSION EXECUTION    |
|                        |
|  Task 1 completes ---> inject output into field
|  Task 2 completes ---> inject output into field
|  Agent queries field <--- semantic retrieval
|  Agent injects finding --> dedup + store
|  Task 3 completes ---> inject output into field
|  ...                   |
+========================+
           |
           v
+--------------------+
| MISSION TERMINAL   |
| (completed/failed/ |
|  cancelled)        |
+----------+---------+
           |
           v
+--------------------+
| 4. Destroy Qdrant  |
|    collection       |
+--------------------+
           |
           v
+--------------------+
| 5. Tick-based      |
|    cleanup catches  |
|    any missed       |
|    destructions     |
+--------------------+
```

## Figure 5: Telephone Game vs Shared Field

```
=== MESSAGE PASSING (Telephone Game) ===

Agent A                    Agent B                    Agent C
produces 20 findings       receives 20 findings       receives 5 findings
                    ------>  references only 5   ----->  references only 2
                           (15 LOST)                   (3 more LOST)

Finding #7: directly relevant to Agent C's task
Status: LOST (Agent B didn't mention it)

Total information loss: 18 out of 20 findings (90%)


=== SHARED SEMANTIC FIELD ===

Agent A                    Agent B                    Agent C
produces 20 findings       queries field               queries field
injects all 20      -----> finds relevant findings     finds ALL relevant
into shared field          injects 3 analyses   -----> including finding #7
                           into field                  which was NEVER mentioned
                                                       by Agent B

Finding #7: directly relevant to Agent C's task
Status: FOUND (semantic similarity match in shared field)

Total information loss: 0 out of 20 findings (0%)
```

## Figure 6: A/B Comparison Results

```
Coverage by Scenario (% of findings discovered)

100% |
 90% |          [88]
 80% |           ##
 70% |  [71]     ##
 60% |   ##      ##              [62]
 50% |   ##      ##    [50]       ##
 40% |   ## [43] ## [38] ## [38]  ## [38] [38]
 30% |   ##  ||  ##  ||  ##  ||   ##  ||   ##
 20% |   ##  ||  ##  ||  ##  ||   ##  ||   ##
 10% |   ##  ||  ##  ||  ## [12]  ##  ||   ## [12]
  0% +---##--||--##--||--##--||---##--||---##--||---
       EU AI   Cyber  Market   Product  Incident
       Act     Sec    Research Launch   Response

     ## = Vector Field    || = Message Passing

Summary:
  Vector Field average:    62%
  Message Passing average: 29%
  Gap:                    +33%
  Scenarios won by VF:     5/5
  Findings lost by MP:     28
```

---

*These drawings are intended to illustrate the system architecture, data flow, and experimental results described in the specification. They should be converted to formal patent drawings for the non-provisional application.*
