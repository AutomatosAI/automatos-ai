# PRD-100 — Research: Autonomous Operating Layer & Neural Field Orchestration

**Version:** 1.0
**Type:** Research Master Document
**Status:** Active
**Priority:** P0
**Author:** Gerard Kavanagh + Claude
**Date:** 2026-03-14
**Scope:** Complete research roadmap from current platform through Mission Mode to Neural Field Orchestration

---

## 0. Why This Document Exists

On 2026-03-14, after 81 PRDs of building Automatos, we stopped to take stock. What we found:

1. The platform has real, working infrastructure — not prototypes
2. The gap to orchestration is narrower than assumed
3. The Context Engineering research repo contains the theoretical foundation for something nobody else is building
4. Previous attempts at big PRDs (82 original draft) tried to do everything at once

**This document is the research anchor.** Every stage of the roadmap gets:
- A research phase (understand the problem, study prior art)
- A design document (architecture decisions)
- An implementation PRD (buildable spec)

We are not going blind this time.

---

## 1. The Vision (Plain English)

Automatos is an **AI operating system for knowledge work**.

**Phase 1 (built):** The Lego platform — 340 LLMs, 850 tools, skills, heartbeats, recipes, 11 channels, memory, board, reports. Users assemble pieces manually.

**Phase 2 (next):** Mission Mode — a coordinator that decomposes complex goals into tasks, assigns agents (roster or ephemeral "contractors"), executes with verification, tracks everything on the board, and learns from outcomes. Users describe what they want; the system figures out how.

**Phase 3 (future):** Neural Field Orchestration — micro-agents on Kubernetes sharing a continuous semantic field instead of passing messages. Context resonates across agents. Irrelevant information decays naturally. Task completion detected through attractor convergence. Distributed cognition, not distributed chat.

**What makes this different from everyone else:**

Everyone else builds message-passing multi-agent systems:
```
Agent A → text → Agent B → text → Agent C
(context degrades at every hop)
```

We're building toward shared semantic spaces:
```
┌─────────── SHARED FIELD ───────────┐
│  Agent A injects research findings │
│  Agent B's marketing context       │
│    resonates with A's findings     │
│  Agent C reads amplified patterns  │
│    (relevant = loud, noise = faded)│
└────────────────────────────────────┘
```

No telephone game. No context degradation. Agents operate in a shared medium of meaning.

---

## 2. What's Built Today (Phase 1 — Honest Inventory)

### Working Foundation Systems

| System | Status | Description |
|--------|--------|-------------|
| **ContextService** | ✅ | 8 modes, 12 sections, token-budgeted, parallel rendering |
| **Tool Router** | ✅ | Single source of truth. ToolRegistry + ActionRegistry → agent tools |
| **Unified Executor** | ✅ | Prefix-based dispatch to 9 execution modules |
| **Agent Factory** | ✅ | Clean rewrite. Tool loop (10 iterations), retry, ContextService integration |
| **Universal Router** | ✅ | 7-tier routing (override → cache → rules → trigger → semantic → keyword → LLM) |
| **Heartbeat Service** | ✅ | APScheduler cron, orchestrator + agent ticks, active hours, rate limiting |
| **Memory (5-layer)** | ✅ | Redis sessions → Postgres short-term → Mem0 long-term → RAG → NL2SQL |
| **Chatbot Pipeline** | ✅ | SSE streaming, tool loop, dedup, intent classification |
| **Inter-Agent Comms** | ✅ | Redis pub/sub + consensus protocols |
| **Multi-Agent Coordination** | ⚠️ | Built (877 lines, networkx), untested at scale |
| **Recipes** | ✅ | Multi-step, multi-agent, scheduled, board-integrated |
| **Board Tasks** | ✅ | Kanban with auto-creation from recipes/heartbeats |
| **Reports** | ✅ | Agent output persistence, cross-agent access, grading |
| **Scheduled Tasks** | ✅ | PRD-77 agent self-scheduling via APScheduler |
| **Task Reconciler** | ✅ | Stall detection + auto-retry |
| **Channel Adapters** | ✅ | 11 platforms (Slack, Discord, Telegram, WhatsApp, etc.) |
| **FutureAGI** | ✅ | Prompt evaluation/scanning integration |
| **340 LLMs** | ✅ | OpenRouter + direct provider support |
| **850 Tools** | ✅ | Composio + platform + workspace + core |

### What Does NOT Exist

| System | Notes |
|--------|-------|
| orchestration_runs / orchestration_tasks tables | No migration, no model, no code |
| Task graph with dependencies | No DAG, no dependency resolution |
| Coordinator service | Heartbeat orchestrator is closest but different purpose |
| Verifier / critic loop | No output validation against success criteria |
| Budget enforcement (per-run) | Token budget exists in ContextService but not per-mission |
| Run trace / explainability | No structured execution trace |
| Ephemeral "contractor" agents | Can create agents but no mission-scoped lifecycle |
| Shared semantic field | Theory in Context Engineering repo, no implementation |
| Neural field operations | Conceptual Python classes only |
| K8s micro-agent infrastructure | Not started |

---

## 3. User-Facing Model: Three Modes

After extensive discussion, we settled on three user-facing modes. Not four. Not five. Three.

| Mode | User Mental Model | System Behavior |
|------|------------------|-----------------|
| **Task** | "Do this for me" | Single agent, bounded work, tools, report back |
| **Routine** | "Do this every day/week" | Scheduled, repeatable, can be multi-agent, recipe-backed |
| **Mission** | "I have a big idea / complex goal" | Coordinator decomposes, spawns agents, verifies, tracks on board |

**Heartbeat is the engine, not a user concept.** It powers Tasks and Routines behind the scenes.

**Mission is the new capability.** Everything else already works.

### Mission Execution Flow

```
User: "Research EU AI Act compliance for our product"
        │
        ▼
   COORDINATOR (organ-level)
   - Decomposes goal into subtasks
   - Each subtask → board task (traceable, with project label + ID)
   - Each task assigned to roster agent OR ephemeral contractor
        │
        ├─ Task 1: "Research EU AI Act requirements"
        │   → Researcher agent (roster) or spawned specialist
        │   → Board: PROJECT-001-T1
        │   → Tools: web search, document analysis
        │   → Model: user's choice for research
        │
        ├─ Task 2: "Analyze our product against requirements"
        │   → Depends on Task 1 output
        │   → Board: PROJECT-001-T2
        │   → Tools: workspace files, code analysis
        │
        ├─ Task 3: "Write compliance report"
        │   → Depends on Task 1 + Task 2
        │   → Board: PROJECT-001-T3
        │   → Model: user's choice for writing
        │
        └─ Task 4: "Review and score report"
            → Verifier (different model than writer)
            → Board: PROJECT-001-T4
            → Scores quality, completeness, accuracy
        │
        ▼
   HUMAN REVIEW
   - All board tasks visible with project label
   - Full trace: cost, tokens, time, memory, tools used
   - Accept / reject / send back specific tasks
   - "Save as routine?" → becomes a recipe
```

### Key Design Decisions

**Autonomy toggle (per-mission):**
- **Default (approve):** System shows plan → human approves → execution begins
- **Autonomy mode:** Human sets budget + success criteria → system runs → human reviews at end

**Agent sourcing:**
- Simple missions → picks from roster agents
- Complex missions → coordinator spawns ephemeral "contractor" agents
- User can set model preferences per role (planner, coder, reviewer)

**Contractor agents:**
| | Roster Agent | Contractor Agent |
|---|---|---|
| Lifecycle | Permanent, has heartbeat | Spawned for mission, destroyed after |
| Config | DB-backed, skills, tools, personality | Coordinator-defined: role + tools + model |
| Memory | Long-term (Mem0) | Mission-scoped only |
| Board visibility | Always visible | Appears under mission project label |
| Cost tracking | Ongoing | Per-mission attribution |

**"Done" = Human sign-off.** System marks mission as "ready for review" when all tasks pass verification. Human accepts, rejects, or sends back.

**Budget = Human sets, system guides:**
> "This mission will likely need ~15 LLM calls across 5 tasks. Estimated cost: $2-4. Set a hard cap?"

**Learning = Outcome telemetry first.** Every task records: agent, model, tools, tokens, cost, duration, verifier score, human acceptance. No fancy learning engine — just data. Query it for patterns later.

**Successful mission → routine conversion.** "Save as routine?" button converts a working mission structure into a repeatable recipe.

---

## 4. Context Engineering Foundation

### The Atom → Organ Hierarchy

The Context Engineering research repo (`/Users/gkavanagh/Development/Automatos-AI-Platform/Context-Engineering/`) defines a biological metaphor for context complexity:

| Level | Name | What It Is | Automatos Today | Automatos Target |
|-------|------|-----------|-----------------|-----------------|
| 1 | **Atom** | Single instruction + constraints + output format | Basic agent prompt | Same, but scored via FutureAGI |
| 2 | **Molecule** | Instruction + examples + context (few-shot) | ContextService sections | Dynamic section selection per task |
| 3 | **Cell** | Molecule + persistent memory + state | Agent + Mem0 + session history | Memory with resonance/decay |
| 4 | **Organ** | Coordinated cells + shared memory + specialist routing | Multi-agent recipes | **Mission mode** — coordinator + specialists + shared context |
| 5 | **Neural System** | Cognitive tools as structured reasoning | Tool router + cognitive programs | Structured reasoning protocols per agent role |
| 6 | **Neural Field** | Continuous semantic landscape, resonance, attractors | Not implemented | **Phase 3** — shared field across K8s micro-agents |

### Key Theoretical Concepts

**Neural Fields:** Context as a continuous medium, not discrete chunks. Information patterns resonate (reinforce when aligned), decay (fade when irrelevant), and form attractors (stable convergent states).

**Resonance formula:** `R(A, B) = cos(θ) × |A| × |B| × S(A, B)` — semantic alignment amplifies signal.

**Persistence:** `S(t) = S₀ × e^(-λt)` — information decays unless reinforced by resonance.

**Attractor dynamics:** Task completion = field convergence to stable state. When agent outputs stop diverging and settle into consistent patterns, the mission is converging.

**Symbolic mechanisms:** LLMs implement abstract reasoning through Symbol Abstraction → Symbolic Induction → Retrieval. Structured formats (SKILL.md, JSON schemas, protocols) align with these mechanisms.

### Prior Art That Validates the Approach

| Concept | Precedent | Status |
|---------|-----------|--------|
| Shared semantic workspace | Blackboard Architecture (1980s) | Proven pattern, needs modernization |
| Environmental communication | Stigmergy (swarm intelligence) | Well-studied, pheromone = resonance |
| Shared model serving | Google Pathways | Production at Google scale |
| Associative shared memory | Tuple Spaces (Linda model) | Implemented in many distributed systems |
| Vector space operations | Modern embedding models + vector DBs | Production everywhere |
| Semantic similarity amplification | Reranking, cross-attention | Standard ML technique |
| TTL-based relevance decay | Cache eviction, memory consolidation | Standard systems pattern |
| Consensus detection | Byzantine fault tolerance, voting | Well-studied distributed systems problem |
| K8s micro-services | Standard container orchestration | Production everywhere |

**What doesn't exist:** The combination — vector-space shared memory + semantic resonance/decay + attractor convergence detection + K8s micro-agents as a coherent orchestration system. Every piece is individually proven. The assembly is novel.

---

## 5. Competitive Landscape

### Detailed Analysis (from research session 2026-03-14)

**Agent Zero** — Hierarchical delegation, prompt-driven behavior, conversation sealing.
- Strengths: clean delegation model, memory consolidation with LLM, utility model separation
- Weaknesses: no persistence (crash = lost), no multi-tenancy, no verification, FAISS-only memory
- What we adopt: conversation sealing, utility model, on-demand skill loading

**OpenClaw** — Personal AI gateway, 15+ channels, hub-and-spoke.
- Strengths: massive channel coverage, 6-tier tool policy layering, context compaction
- Weaknesses: single-user, SQLite/JSONL, no scaling, no multi-tenancy
- What we adopt: tool policy layering pattern, context compaction with dedicated model

**OpenAI Symphony** — Issue tracker daemon, Linear → Codex agents → PRs.
- Strengths: WORKFLOW.md policy-as-code (brilliant), strict workspace isolation, reconciliation loop, lifecycle hooks
- Weaknesses: no multi-agent coordination (deliberately), in-memory only, Linear-only
- What we adopt: reconciliation loop extension, lifecycle hooks, tracker-as-coordinator, continuation vs retry distinction

**CrewAI / AutoGen / LangGraph** — Popular multi-agent frameworks.
- All use message passing between agents
- No persistent runs, no budget control, no shared semantic space
- Framework-only (bring your own everything)

**Automatos advantage:** Only platform with all pieces built (tools, models, channels, memory, scheduling, board, reports) AND a theoretical foundation (Context Engineering) for something beyond message passing.

---

## 6. Research Roadmap

Each stage produces three artifacts:
1. **Research document** — study prior art, understand the problem, identify risks
2. **Design document / ADR** — architecture decisions, data model, interfaces
3. **Implementation PRD** — buildable spec with acceptance criteria

### Phase 2: Mission Mode

| PRD | Title | Type | Depends On | Delivers |
|-----|-------|------|------------|----------|
| **100** | This document | Research Master | — | Roadmap, vision, competitive analysis |
| **101** | Mission Schema & Data Model | Research + Design | 100 | orchestration_runs, tasks, events, dependencies schema. Study: DAG resolution patterns, state machines for task lifecycle, event sourcing patterns |
| **102** | Coordinator Architecture | Research + Design | 101 | How the coordinator plans, assigns, monitors. Study: blackboard architecture, HTN planning, BDI agents, Symphony's WORKFLOW.md pattern |
| **103** | Verification & Quality | Research + Design | 102 | LLM-as-judge patterns, success criteria specification, scoring rubrics. Study: FutureAGI eval integration, constitutional AI critique patterns, LMSYS arena methodology |
| **104** | Ephemeral Agents & Model Selection | Research + Design | 102 | Contractor agent lifecycle, model-per-role selection, cost estimation. Study: Agent Zero delegation, model routing (Martian, Unify.ai), cost optimization patterns |
| **105** | Budget & Governance | Research + Design | 101 | Per-mission budget enforcement, tool policy layering, approval gates. Study: OpenClaw tool policies, AWS billing patterns, rate limiting strategies |
| **106** | Outcome Telemetry & Learning Foundation | Research + Design | 101, 103 | What to track, how to store, how to query for patterns. Study: ML experiment tracking (MLflow, W&B), A/B testing frameworks, recommendation systems |

**Implementation PRDs (82A-82D) follow from these research docs.**

### Phase 2 → Phase 3 Bridge

| PRD | Title | Type | Depends On | Delivers |
|-----|-------|------|------------|----------|
| **107** | Context Interface Abstraction | Research + Design | 102, Context Engineering repo | Define the interface between coordinator and context layer such that Phase 3 can swap the implementation without changing the coordinator. Study: hexagonal architecture, port/adapter pattern |
| **108** | Memory Field Prototype | Research + Design | 107, Context Engineering `08_neural_fields_foundations.md` | Prototype shared vector space with injection, decay, resonance scoring. Study: FAISS/Qdrant shared indices, Redis vector search, temporal decay algorithms |

### Phase 3: Neural Field Orchestration

| PRD | Title | Type | Depends On | Delivers |
|-----|-------|------|------------|----------|
| **110** | Neural Field Architecture | Research | 108, Context Engineering repo | Full architecture for shared semantic fields. Study: distributed shared memory (DSM), content-addressable memory, holographic reduced representations |
| **111** | Resonance & Decay Mechanisms | Research + Design | 110 | Implementation design for semantic amplification and decay. Study: attention mechanisms, TF-IDF-like relevance scoring, exponential decay with reinforcement, Hebbian learning |
| **112** | Attractor Dynamics for Task Completion | Research + Design | 110, 111 | How to detect convergence in a shared field. Study: consensus algorithms, convergence detection in iterative methods, Lyapunov stability, clustering stability metrics |
| **113** | K8s Micro-Agent Infrastructure | Research + Design | 110 | Container architecture for field-connected agents. Study: K8s operators, sidecar patterns, service mesh (Istio), shared memory volumes, gRPC streaming |
| **114** | Symbolic Mechanism Integration | Research + Design | 110, Context Engineering `12_symbolic_mechanisms.md` | How to leverage LLM symbolic heads in field operations. Study: Yang et al. ICML 2025, neurosymbolic AI, structured generation |
| **115** | Emergence Detection & Safety | Research + Design | 112, 113 | How to detect emergent behaviors, when to intervene, safety bounds. Study: swarm robotics safety, AI alignment research, anomaly detection, circuit breakers |
| **116** | Distributed Cognition Integration | Research | 110-115 | Putting it all together — the full neural field orchestration system. Implementation plan for Phase 3. |

---

## 7. Phase 2 Implementation PRDs (following research)

After research PRDs 101-106 are complete, implementation PRDs are written:

| PRD | Title | Based On Research | Delivers |
|-----|-------|-------------------|----------|
| **82A** | Mission Schema + Context Modes | 101 | Alembic migration, SQLAlchemy models, COORDINATOR/VERIFIER context modes, API endpoints |
| **82B** | Sequential Mission Coordinator | 102, 103, 107 | CoordinatorService, plan→assign→execute→verify→human review, board integration |
| **82C** | Parallel Execution + Budget + Contractors | 104, 105 | Bounded parallel tasks, per-run budget, ephemeral agent lifecycle, model-per-role |
| **82D** | Complexity Detection + Outcome Telemetry | 106 | "This should be a Mission" detection, telemetry capture, pattern queries |

---

## 8. Risk Register

| # | Risk | Impact | Likelihood | Mitigation |
|---|------|--------|------------|------------|
| 1 | Over-scoping again | High | Medium | Each PRD is research-first. No implementation without understanding. |
| 2 | Phase 3 is science fiction | High | Low | Every component has proven precedent. Novel part is assembly. Prototype in PRD-108. |
| 3 | Coordinator complexity | High | High | Start sequential-only. No parallel, no dynamic replanning. Get lifecycle right first. |
| 4 | Cost blowout in missions | Medium | High | Budget tracking from 82A schema. Enforcement in 82C. Cheap models for coordination. |
| 5 | User confusion (Task vs Routine vs Mission) | Medium | Medium | Three modes only. Clear UX. System suggests appropriate mode. |
| 6 | Context Engineering theory doesn't translate to code | High | Medium | PRD-108 is the prototype gate. If field prototype doesn't outperform message passing, reassess Phase 3. |
| 7 | Learned patterns are useless | Medium | Medium | Start with raw telemetry. Only build recommendation engine when data proves patterns exist. |
| 8 | Neural field "resonance" is just RAG with extra steps | Medium | Medium | Research PRD-111 must identify concrete advantages over standard RAG. If none, simplify. |
| 9 | K8s complexity for micro-agents | Medium | High | Evaluate serverless alternatives (Lambda, Cloud Run). K8s only if shared state requires it. |
| 10 | Phase 2 already delivers enough value, Phase 3 never starts | Low | High | This is actually fine. Phase 2 is valuable standalone. Phase 3 is the moonshot. |

---

## 9. Open Research Questions

These must be answered during the research phases:

### Phase 2 Questions
1. **DAG execution engine:** Build custom or adopt (Prefect, Temporal, Airflow patterns)?
2. **Coordinator prompt design:** How much planning capability do current LLMs actually have for task decomposition?
3. **Verification accuracy:** Can LLM-as-judge reliably assess task completion? What's the false positive rate?
4. **Ephemeral agent overhead:** How fast can we spin up a contractor agent? Is the latency acceptable?
5. **Board integration:** Can the existing board_tasks schema support mission task graphs, or does it need extension?
6. **Model routing economics:** What's the actual cost difference between routing research to a cheap model vs. using the best model for everything?

### Phase 3 Questions
7. **Field vs. message passing:** Can we demonstrate measurably better outcomes with shared fields vs. standard agent messaging?
8. **Resonance implementation:** Is cosine similarity sufficient, or do we need learned resonance functions?
9. **Decay calibration:** How do we set decay rates? Per-domain? Per-task? Learned from data?
10. **Attractor detection:** How do we know when a field has converged? Statistical tests? Embedding distance thresholds?
11. **K8s shared memory:** Can we efficiently share a vector store across pods? What's the latency profile?
12. **Emergence safety:** How do we detect unexpected emergent behaviors before they cause harm?
13. **Scale limits:** How many micro-agents can share a field before noise overwhelms signal?

---

## 10. Success Criteria for This Research Program

### Phase 2 Success (Missions work)
- [ ] User can create a mission from natural language
- [ ] Coordinator decomposes into tasks visible on board
- [ ] Tasks execute through existing agent infrastructure
- [ ] Verification catches bad outputs
- [ ] Human reviews and accepts/rejects
- [ ] Full cost/token/time trace per mission
- [ ] Successful mission can be saved as routine
- [ ] Outcome telemetry captures enough data for future learning

### Phase 3 Success (Neural fields outperform message passing)
- [ ] Prototype shared field that agents read/write to
- [ ] Demonstrable context quality improvement over message passing (measured, not vibes)
- [ ] Resonance amplifies relevant cross-agent context
- [ ] Decay removes stale information without manual cleanup
- [ ] Attractor convergence correlates with actual task completion
- [ ] K8s micro-agents connect to shared field with acceptable latency (<100ms read/write)
- [ ] System remains safe and observable during emergent behavior

---

## 11. Timeline Philosophy

No dates. No "2 weeks per PRD." That's how we end up with 81 PRDs and most still in draft.

Instead: **each research PRD is done when its questions are answered and its design doc is peer-reviewed** (by Auto, by Claude, by Gerard, by whoever's available). Then the implementation PRD gets written. Then it gets built.

Sequential, not parallel. One research doc at a time. Each one informs the next.

**Start:** PRD-101 (Mission Schema research) — because everything else depends on getting the data model right.

---

## 12. The Differentiator (Why This Matters)

Every AI platform is building chatbots with tools.
Some are building multi-agent message passing.
A few are building workflow orchestration.

**Nobody is building:**
1. A platform where the execution infrastructure (340 models, 850 tools, 11 channels, 5-layer memory) is already production-grade
2. With a mission coordination layer that decomposes, delegates, verifies, and learns
3. Moving toward a shared semantic field where agents don't pass messages — they share meaning
4. Grounded in published research on neural field theory, symbolic mechanisms, and attractor dynamics
5. With every step researched, designed, and validated before building

That's not a chatbot platform. That's an autonomous operating layer for knowledge work, evolving toward distributed cognition.

And every piece is individually buildable with existing technology.

---

## Appendix A: Key Files & References

### Automatos Codebase (what exists)
- `orchestrator/modules/context/service.py` — ContextService (8 modes, 12 sections)
- `orchestrator/modules/tools/tool_router.py` — Tool Router (single source of truth)
- `orchestrator/modules/tools/execution/unified_executor.py` — Unified Executor
- `orchestrator/modules/agents/factory/agent_factory.py` — Agent Factory (tool loop)
- `orchestrator/core/routing/engine.py` — Universal Router (7-tier)
- `orchestrator/services/heartbeat_service.py` — Heartbeat Service
- `orchestrator/modules/memory/unified_memory_service.py` — 5-layer memory
- `orchestrator/consumers/chatbot/service.py` — Chatbot pipeline
- `orchestrator/modules/agents/communication/inter_agent.py` — Inter-agent comms
- `orchestrator/services/task_reconciler.py` — Task reconciliation

### Context Engineering Repo
- `00_foundations/01_atoms_prompting.md` through `14_unified_field_theory.md` — Full theoretical hierarchy
- `00_foundations/08_neural_fields_foundations.md` — Neural field theory
- `00_foundations/09_persistence_and_resonance.md` — Resonance mechanics
- `00_foundations/10_field_orchestration.md` — Multi-field coordination
- `00_foundations/11_emergence_and_attractor_dynamics.md` — Attractor theory
- `00_foundations/12_symbolic_mechanisms.md` — Symbolic reasoning in LLMs
- `00_foundations/13_quantum_semantics.md` — Quantum interpretation
- `00_foundations/14_unified_field_theory.md` — Capstone integration

### Competitive Research (conducted 2026-03-14)
- Agent Zero: hierarchical delegation, conversation sealing, FAISS memory, no persistence
- OpenClaw: personal gateway, 6-tier tool policies, 15+ channels, single-user
- Symphony: WORKFLOW.md policy-as-code, reconciliation loops, issue-tracker coordination
- Full analysis in PRD-82 Research (`docs/PRDS/82-RESEARCH-ORCHESTRATION-READINESS.md`)

### Academic References (from Context Engineering repo)
- Yang et al., ICML 2025 — Symbolic mechanisms in transformers
- Agostino et al., Indiana University 2025 — Quantum semantics
- IBM Zurich — Cognitive tools research (GPT-4.1 26.7% → 43.3% on AIME2024)
- Columbia, Shanghai AI Lab — Attractor dynamics in neural networks
- Full citations in `Context-Engineering/CITATIONS.pdf`

---

## Appendix B: Glossary

| Term | Definition |
|------|-----------|
| **Atom** | Simplest context unit: instruction + constraints + output format |
| **Molecule** | Atom + examples + context (few-shot learning) |
| **Cell** | Molecule + persistent memory + state |
| **Organ** | Coordinated cells with shared memory and specialist routing |
| **Neural Field** | Continuous semantic landscape where information resonates and decays |
| **Resonance** | Reinforcement of aligned information patterns in a shared field |
| **Attractor** | Stable state that a field naturally converges toward |
| **Decay** | Natural fading of unreinforced information in a field |
| **Mission** | User-initiated complex goal that requires coordinator decomposition |
| **Routine** | Scheduled, repeatable task or recipe (can be multi-agent) |
| **Task** | Bounded single-agent work unit |
| **Coordinator** | LLM-powered service that plans, assigns, and monitors mission execution |
| **Contractor** | Ephemeral agent spawned for a specific mission, destroyed after |
| **Roster Agent** | Permanent agent with DB config, skills, tools, personality |
| **Telemetry** | Per-task outcome data: agent, model, tools, tokens, cost, score, acceptance |
| **Stigmergy** | Indirect communication through shared environment (biological precedent for neural fields) |
| **Blackboard Architecture** | Shared workspace pattern where multiple knowledge sources read/write (CS precedent for neural fields) |
