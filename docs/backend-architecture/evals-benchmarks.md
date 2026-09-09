# Evals & Benchmarks

<details>
<summary>Relevant source files</summary>

The following files were used as context for generating this wiki page:

- [docs/PRDS/PRD-206-MEMORY-CONTINUITY-PERSONAL-CONTEXT.md](docs/PRDS/PRD-206-MEMORY-CONTINUITY-PERSONAL-CONTEXT.md)
- [docs/runbooks/S10-MEMORY-BASELINE-FREEZE.md](docs/runbooks/S10-MEMORY-BASELINE-FREEZE.md)
- [frontend/components/__tests__/prd205-auto-speaks.test.ts](frontend/components/__tests__/prd205-auto-speaks.test.ts)
- [orchestrator/core/llm/clients/openrouter_embedding.py](orchestrator/core/llm/clients/openrouter_embedding.py)
- [orchestrator/evals/graphiti_vs_baseline.py](orchestrator/evals/graphiti_vs_baseline.py)
- [orchestrator/scripts/eval/tool_routing/eval_seed.yaml](orchestrator/scripts/eval/tool_routing/eval_seed.yaml)
- [orchestrator/scripts/eval/tool_routing/eval_set.jsonl](orchestrator/scripts/eval/tool_routing/eval_set.jsonl)
- [orchestrator/scripts/eval/tool_routing/prompt_builder.py](orchestrator/scripts/eval/tool_routing/prompt_builder.py)
- [orchestrator/scripts/eval/tool_routing/run_eval.py](orchestrator/scripts/eval/tool_routing/run_eval.py)
- [orchestrator/scripts/eval/tool_routing/score.py](orchestrator/scripts/eval/tool_routing/score.py)
- [orchestrator/scripts/eval/tool_routing/seed_eval_set.py](orchestrator/scripts/eval/tool_routing/seed_eval_set.py)
- [orchestrator/tests/test_openrouter_embedding_routing.py](orchestrator/tests/test_openrouter_embedding_routing.py)
- [orchestrator/tests/test_prd198_graphiti_gate.py](orchestrator/tests/test_prd198_graphiti_gate.py)
- [orchestrator/tests/test_prd205_auto_speaks.py](orchestrator/tests/test_prd205_auto_speaks.py)
- [tools/benchmark_field_memory.py](tools/benchmark_field_memory.py)
- [tools/benchmark_results/benchmark_redis_20260330_135424.json](tools/benchmark_results/benchmark_redis_20260330_135424.json)
- [tools/benchmark_results/benchmark_redis_parallel_20260330_164310.json](tools/benchmark_results/benchmark_redis_parallel_20260330_164310.json)
- [tools/benchmark_results/benchmark_vector_field_20260330_133126.json](tools/benchmark_results/benchmark_vector_field_20260330_133126.json)
- [tools/benchmark_results/benchmark_vector_field_parallel_20260330_145313.json](tools/benchmark_results/benchmark_vector_field_parallel_20260330_145313.json)
- [tools/benchmark_results/benchmark_vector_field_parallel_20260330_152355.json](tools/benchmark_results/benchmark_vector_field_parallel_20260330_152355.json)
- [tools/benchmark_results/benchmark_vector_field_parallel_20260330_155229.json](tools/benchmark_results/benchmark_vector_field_parallel_20260330_155229.json)
- [tools/compare_benchmarks.py](tools/compare_benchmarks.py)

</details>



This page details the evaluation and benchmarking infrastructure within the Automatos AI codebase. It covers various evaluation harnesses for core functionalities like retrieval recall, NL2SQL, and tool routing, along with benchmark scripts for performance comparison. The goal is to ensure quality, track improvements, and provide gates for new feature adoption in the continuous integration (CI) pipeline.

## Retrieval Recall Evals (Graphiti vs. Baseline)

The `orchestrator/evals` directory contains the evaluation framework for retrieval recall, specifically comparing the Graphiti knowledge graph approach against a baseline. This is a critical component for ensuring the effectiveness of the RAG system.

### Purpose and Gating

The `graphiti_vs_baseline.py` script [orchestrator/evals/graphiti_vs_baseline.py:1-30]() implements a gate for the adoption of the Graphiti knowledge graph. This gate is designed to prevent false positives and ensure that Graphiti only gets adopted if it demonstrably improves retrieval recall by a specified margin. The evaluation is structured around three key inputs:

1.  **Retrieval Baseline**: A frozen baseline artifact (`kg_retrieval_2026-07.json`) [orchestrator/evals/graphiti_vs_baseline.py:45]() representing the performance of the existing retrieval system. This baseline is frozen and landed with PRD-186 (#547).
2.  **Memory Baseline (S10)**: A memory recall baseline (`memory_recall_2026-07.json`) [orchestrator/evals/graphiti_vs_baseline.py:46]() which is not yet frozen. Its absence results in a `PENDING` gate status, blocking further development until it's established. The process for freezing this baseline is documented in `docs/runbooks/S10-MEMORY-BASELINE-FREEZE.md` [docs/runbooks/S10-MEMORY-BASELINE-FREEZE.md]().
3.  **Graphiti Treatment**: The results of a live retrieval-recall run with the Graphiti feature enabled (`graphiti_recall.json`) [orchestrator/evals/graphiti_vs_baseline.py:47](). This artifact has the same shape as the frozen baseline.

The gate's verdict is based on an "uplift points" calculation, defined as `(treatment - best_baseline) * 100` [orchestrator/evals/graphiti_vs_baseline.py:91-93](). A minimum uplift margin (e.g., `UPLIFT_MARGIN_POINTS = 5.0`) [orchestrator/evals/graphiti_vs_baseline.py:42]() is required for adoption. Additionally, specific "capability slices" are measured, with the `contradicted_fact_resolution` slice being a gating factor [orchestrator/evals/graphiti_vs_baseline.py:53-58](). If this slice fails, adoption is blocked even if the overall recall margin is met.

### Implementation Details

The `compute_gate` function [orchestrator/evals/graphiti_vs_baseline.py:96-102]() is the core logic for determining the gate's verdict. It checks for missing artifacts and calculates the uplift.

-   `load_artifact(path: Path)`: Loads a JSON artifact from the specified path, returning `None` if the file doesn't exist [orchestrator/evals/graphiti_vs_baseline.py:61-66]().
-   `best_baseline_recall(artifact: Dict[str, Any], tenant_alias: str)`: Identifies the highest `mean_recall_at_5` among all non-Graphiti variants in a given artifact for a specific tenant [orchestrator/evals/graphiti_vs_baseline.py:69-79]().
-   `treatment_recall(artifact: Dict[str, Any], tenant_alias: str)`: Retrieves the `mean_recall_at_5` for the `graphiti` variant [orchestrator/evals/graphiti_vs_baseline.py:82-89]().

The `test_prd198_graphiti_gate.py` [orchestrator/tests/test_prd198_graphiti_gate.py:1-8]() provides unit tests for this gate, ensuring its logic is sound and that it correctly identifies `PENDING` states when inputs are missing or `DO_NOT_ADOPT` when the margin is not met or a gating slice fails.

### Diagram: Graphiti Eval Gate Data Flow

```mermaid
graph TD
    subgraph "Input Artifacts"
        A[Retrieval Baseline (kg_retrieval_2026-07.json)] --> C
        B[Memory Baseline S10 (memory_recall_2026-07.json)] --> C
        D[Graphiti Treatment (graphiti_recall.json)] --> C
    end

    subgraph "Evaluation Logic"
        C{compute_gate()} --> E{Check Missing Artifacts}
        E -- Missing --> F[Verdict: PENDING]
        E -- All Present --> G{Calculate best_baseline_recall()}
        G --> H{Calculate treatment_recall()}
        H --> I{Calculate uplift_points()}
        I --> J{Check uplift_points >= UPLIFT_MARGIN_POINTS}
        J -- False --> K[Verdict: DO_NOT_ADOPT]
        J -- True --> L{Check Gating Slice (contradicted_fact_resolution)}
        L -- PENDING / Failed --> K
        L -- Passed --> M[Verdict: ADOPT_UNBLOCKED]
    end

    subgraph "Output"
        F --> N[CI Gate Result]
        K --> N
        M --> N
    end

    style A fill:#fff,stroke:#333,stroke-width:2px
    style B fill:#fff,stroke:#333,stroke-width:2px
    style D fill:#fff,stroke:#333,stroke-width:2px
    style C fill:#fff,stroke:#333,stroke-width:2px
    style E fill:#fff,stroke:#333,stroke-width:2px
    style F fill:#fff,stroke:#333,stroke-width:2px
    style G fill:#fff,stroke:#333,stroke-width:2px
    style H fill:#fff,stroke:#333,stroke-width:2px
    style I fill:#fff,stroke:#333,stroke-width:2px
    style J fill:#fff,stroke:#333,stroke-width:2px
    style K fill:#fff,stroke:#333,stroke-width:2px
    style L fill:#fff,stroke:#333,stroke-width:2px
    style M fill:#fff,stroke:#333,stroke-width:2px
    style N fill:#fff,stroke:#333,stroke-width:2px
```
Sources:
- [orchestrator/evals/graphiti_vs_baseline.py:1-30]()
- [orchestrator/evals/graphiti_vs_baseline.py:45]()
- [orchestrator/evals/graphiti_vs_baseline.py:46]()
- [docs/runbooks/S10-MEMORY-BASELINE-FREEZE.md]()
- [orchestrator/evals/graphiti_vs_baseline.py:47]()
- [orchestrator/evals/graphiti_vs_baseline.py:91-93]()
- [orchestrator/evals/graphiti_vs_baseline.py:42]()
- [orchestrator/evals/graphiti_vs_baseline.py:53-58]()
- [orchestrator/evals/graphiti_vs_baseline.py:96-102]()
- [orchestrator/evals/graphiti_vs_baseline.py:61-66]()
- [orchestrator/evals/graphiti_vs_baseline.py:69-79]()
- [orchestrator/evals/graphiti_vs_baseline.py:82-89]()
- [orchestrator/tests/test_prd198_graphiti_gate.py:1-8]()

## NL2SQL Eval

The `modules/nl2sql` service includes a `benchmarks runner` [25.5. NL2SQL & Database Knowledge]() for evaluating its performance. While specific details of the NL2SQL eval harness are not provided in the given files, its existence indicates a mechanism to test the accuracy and reliability of the natural language to SQL conversion capabilities. This typically involves a dataset of natural language queries and their corresponding correct SQL queries, with metrics like exact match accuracy and semantic similarity.

Sources:
- [25.5. NL2SQL & Database Knowledge]()

## Tool-Routing Eval Harness

The `orchestrator/scripts/eval/tool_routing` directory contains a comprehensive evaluation harness for the tool routing mechanism. This harness is crucial for assessing how effectively the system identifies and selects the correct tool based on a user's natural language query.

### Purpose

The tool-routing eval harness iterates through a cartesian product of models, routing modes, and queries [orchestrator/scripts/eval/tool_routing/run_eval.py:4-5](). It calls the LLM with appropriate tool schemas and system prompts, captures the chosen action, and logs the results to `results.jsonl` [orchestrator/scripts/eval/tool_routing/run_eval.py:6-7](). This allows for detailed analysis of:

-   **Top-1 accuracy**: Whether the chosen action is among the correct actions [orchestrator/scripts/eval/tool_routing/score.py:5]().
-   **In-set hit rate**: Whether any correct action was surfaced to the LLM [orchestrator/scripts/eval/tool_routing/score.py:6]().
-   **Token usage**: Prompt, completion, and total tokens [orchestrator/scripts/eval/tool_routing/score.py:7-9]().
-   **Cost**: Mean cost per call and per correct call [orchestrator/scripts/eval/tool_routing/score.py:10-11]().
-   **Latency**: P50 and P95 latency [orchestrator/scripts/eval/tool_routing/score.py:12]().
-   **Error rate** [orchestrator/scripts/eval/tool_routing/score.py:13]().

### Modes of Evaluation

The `prompt_builder.py` [orchestrator/scripts/eval/tool_routing/prompt_builder.py:1-47]() defines four modes for prompt assembly, each testing a different aspect of tool routing:

1.  **`full`**: Dumps every `ActionDefinition` known to the registry, matching production's `ActionRegistry.build_prompt_summary()` [orchestrator/scripts/eval/tool_routing/prompt_builder.py:6-7](). The tool list is exhaustive, and `platform_execute.action.enum` is unset, allowing free-form string selection.
2.  **`filtered`**: Uses prompt-only narrowing. It delegates to `ActionSemanticIndex` (PRD-138 US-003) to rank actions and `ActionRegistry.build_filtered_prompt_summary()` (US-002) to render only the top-K actions in the prompt. The dispatcher schema remains unchanged, meaning the LLM *could* still pick any action, but the prompt steers it [orchestrator/scripts/eval/tool_routing/prompt_builder.py:10-16]().
3.  **`filtered_schema`**: Combines prompt and schema narrowing. The prompt is the same as `filtered`, but the `platform_execute` tool's `action.enum` is also set to the same top-K ranked action names. This reflects the production behavior after Phase 1b (US-008..US-010) [orchestrator/scripts/eval/tool_routing/prompt_builder.py:18-23]().
4.  **`graph`**: Employs graph-based ranking (PRD-139 US-007) by delegating to `GraphRouter`. This mode uses edges and affinities over embedding entry nodes. It includes chain hints for multi-action chains and falls back to `filtered` if the graph is empty [orchestrator/scripts/eval/tool_routing/prompt_builder.py:25-29]().

### Key Components

-   **`run_eval.py`**: The main script for running the evaluation [orchestrator/scripts/eval/tool_routing/run_eval.py:1-38](). It loads models from `models.yaml` [orchestrator/scripts/eval/tool_routing/run_eval.py:62-63]() and evaluation queries from `eval_set.jsonl` [orchestrator/scripts/eval/tool_routing/run_eval.py:193-195](). It also defines `PLATFORM_EXECUTE_TOOL` [orchestrator/scripts/eval/tool_routing/run_eval.py:88-113]() and other `TOP_LEVEL_TOOLS` [orchestrator/scripts/eval/tool_routing/run_eval.py:133-170]() that bypass the action catalog.
-   **`score.py`**: Processes the `results.jsonl` output from `run_eval.py` to compute and report metrics. It generates a markdown report (`report.md`) and a CSV summary (`summary.csv`) [orchestrator/scripts/eval/tool_routing/score.py:38-40]().
-   **`prompt_builder.py`**: Responsible for constructing the system prompt and tool schemas based on the chosen evaluation mode [orchestrator/scripts/eval/tool_routing/prompt_builder.py:1-47](). It uses `_format_action_line` [orchestrator/scripts/eval/tool_routing/prompt_builder.py:90-99]() and `_render_catalog` [orchestrator/scripts/eval/tool_routing/prompt_builder.py:102-115]() to format the action catalog.
-   **`eval_seed.yaml`**: A hand-curated seed evaluation set containing natural language queries mapped to correct platform actions, categorized by difficulty and type (e.g., easy, ambiguous, paraphrase, cross) [orchestrator/scripts/eval/tool_routing/eval_seed.yaml:1-14](). This is converted to `eval_set.jsonl` by `seed_eval_set.py`.

### Diagram: Tool Routing Eval Harness Flow

```mermaid
graph TD
    subgraph "Configuration & Data"
        A[eval_seed.yaml] --> B(seed_eval_set.py)
        B --> C[eval_set.jsonl]
        D[models.yaml] --> E
    end

    subgraph "Evaluation Loop (run_eval.py)"
        C --> E[Load Queries]
        D --> E
        E --> F{For each (Model, Mode, Query)}
        F --> G[PromptBuilder.build()]
        G -- System Prompt + Surfaced Actions --> H[Build Tools (platform_execute, composio_execute, etc.)]
        H -- Tool Schemas --> I[Call LLM (OpenRouter)]
        I -- Chosen Action, Tokens, Latency --> J[Log Result to results.jsonl]
        J -- Appends --> K[results.jsonl]
    end

    subgraph "Scoring & Reporting (score.py)"
        K --> L[Load results.jsonl]
        D --> L
        L --> M{Aggregate Metrics (Accuracy, Cost, Latency, etc.)}
        M --> N[Generate report.md]
        M --> O[Generate summary.csv]
    end

    style A fill:#fff,stroke:#333,stroke-width:2px
    style B fill:#fff,stroke:#333,stroke-width:2px
    style C fill:#fff,stroke:#333,stroke-width:2px
    style D fill:#fff,stroke:#333,stroke-width:2px
    style E fill:#fff,stroke:#333,stroke-width:2px
    style F fill:#fff,stroke:#333,stroke-width:2px
    style G fill:#fff,stroke:#333,stroke-width:2px
    style H fill:#fff,stroke:#333,stroke-width:2px
    style I fill:#fff,stroke:#333,stroke-width:2px
    style J fill:#fff,stroke:#333,stroke-width:2px
    style K fill:#fff,stroke:#333,stroke-width:2px
    style L fill:#fff,stroke:#333,stroke-width:2px
    style M fill:#fff,stroke:#333,stroke-width:2px
    style N fill:#fff,stroke:#333,stroke-width:2px
    style O fill:#fff,stroke:#333,stroke-width:2px
```
Sources:
- [orchestrator/scripts/eval/tool_routing/run_eval.py:4-5]()
- [orchestrator/scripts/eval/tool_routing/run_eval.py:6-7]()
- [orchestrator/scripts/eval/tool_routing/score.py:5]()
- [orchestrator/scripts/eval/tool_routing/score.py:6]()
- [orchestrator/scripts/eval/tool_routing/score.py:7-9]()
- [orchestrator/scripts/eval/tool_routing/score.py:10-11]()
- [orchestrator/scripts/eval/tool_routing/score.py:12]()
- [orchestrator/scripts/eval/tool_routing/score.py:13]()
- [orchestrator/scripts/eval/tool_routing/prompt_builder.py:1-47]()
- [orchestrator/scripts/eval/tool_routing/prompt_builder.py:6-7]()
- [orchestrator/scripts/eval/tool_routing/prompt_builder.py:10-16]()
- [orchestrator/scripts/eval/tool_routing/prompt_builder.py:18-23]()
- [orchestrator/scripts/eval/tool_routing/prompt_builder.py:25-29]()
- [orchestrator/scripts/eval/tool_routing/run_eval.py:1-38]()
- [orchestrator/scripts/eval/tool_routing/run_eval.py:62-63]()
- [orchestrator/scripts/eval/tool_routing/run_eval.py:193-195]()
- [orchestrator/scripts/eval/tool_routing/run_eval.py:88-113]()
- [orchestrator/scripts/eval/tool_routing/run_eval.py:133-170]()
- [orchestrator/scripts/eval/tool_routing/score.py:38-40]()
- [orchestrator/scripts/eval/tool_routing/prompt_builder.py:90-99]()
- [orchestrator/scripts/eval/tool_routing/prompt_builder.py:102-115]()
- [orchestrator/scripts/eval/tool_routing/eval_seed.yaml:1-14]()

## Tools/ Benchmark Scripts and Results

The `tools/` directory contains various benchmark scripts, notably `benchmark_field_memory.py` [tools/benchmark_field_memory.py:1-6]() for evaluating shared semantic fields.

### Field Memory Benchmark

The `benchmark_field_memory.py` script [tools/benchmark_field_memory.py:1-6]() performs an A/B test of shared semantic fields versus traditional message-passing for multi-agent context coverage. It's a standalone script that interacts with the platform API, making real LLM calls and using Qdrant for vector storage.

**Usage:**
The script can be run with different backends (e.g., `vector_field` or `redis`) and modes (`sequential` or `parallel`) [tools/benchmark_field_memory.py:10-19](). Results are stored in JSON files under `tools/benchmark_results/` [tools/benchmark_field_memory.py:19]().

**Modes:**
-   **`sequential`**: A 3-phase pipeline (Research → Analysis → Synthesis) [tools/benchmark_field_memory.py:23]().
-   **`parallel`**: Four parallel research agents plus a synthesis agent, designed to stress shared memory [tools/benchmark_field_memory.py:24]().

**Seed Facts:**
The benchmark uses `SEED_FACTS` [tools/benchmark_field_memory.py:50-166]() which are specific, verifiable facts categorized by difficulty (easy, medium, hard) and domain (e.g., EU AI Act, Cybersecurity, Market Research, Incident Response). This allows for granular analysis of how well different memory approaches handle varying levels of complexity and domain specificity.

**Comparison:**
The `compare_benchmarks.py` script [tools/compare_benchmarks.py:1-6]() is used to compare the results from different benchmark runs. It loads the latest results per label (e.g., `vector_field` vs. `redis`) and prints a formatted comparison table, including metrics like average coverage, coverage range, and per-difficulty/per-domain coverage [tools/compare_benchmarks.py:26-165]().

### Example Benchmark Results

The `tools/benchmark_results/` directory contains example JSON files from previous runs, such as:
-   `benchmark_vector_field_parallel_20260330_155229.json` [tools/benchmark_results/benchmark_vector_field_parallel_20260330_155229.json]()
-   `benchmark_redis_parallel_20260330_164310.json` [tools/benchmark_results/benchmark_redis_parallel_20260330_164310.json]()

These files contain detailed information about each trial, including the query, expected facts, retrieved facts, coverage score, tokens used, and telemetry data (e.g., field queries, field injects).

### Diagram: Field Memory Benchmark Flow

```mermaid
graph TD
    subgraph "Setup"
        A[SEED_FACTS (JSON)] --> B(benchmark_field_memory.py)
        C[AUTOMATOS_API_URL, AUTH_TOKEN, WORKSPACE] --> B
        D[OPENROUTER_API_KEY (for LLM judge)] --> B
    end

    subgraph "Benchmark Execution (benchmark_field_memory.py)"
        B --> E{Select Backend (vector_field / redis)}
        E --> F{Select Mode (sequential / parallel)}
        F --> G[Initialize Agents & LLM Client]
        G --> H{Run Trials (e.g., Research -> Analysis -> Synthesis)}
        H -- Agent Interactions, LLM Calls, Qdrant/Redis Ops --> I[Collect Metrics (Coverage, Tokens, Telemetry)]
        I --> J[Save Results to benchmark_results/benchmark_*.json]
    end

    subgraph "Analysis"
        J --> K(compare_benchmarks.py)
        K --> L[Formatted Comparison Report (Console Output)]
    end

    style A fill:#fff,stroke:#333,stroke-width:2px
    style B fill:#fff,stroke:#333,stroke-width:2px
    style C fill:#fff,stroke:#333,stroke-width:2px
    style D fill:#fff,stroke:#333,stroke-width:2px
    style E fill:#fff,stroke:#333,stroke-width:2px
    style F fill:#fff,stroke:#333,stroke-width:2px
    style G fill:#fff,stroke:#333,stroke-width:2px
    style H fill:#fff,stroke:#333,stroke-width:2px
    style I fill:#fff,stroke:#333,stroke-width:2px
    style J fill:#fff,stroke:#333,stroke-width:2px
    style K fill:#fff,stroke:#333,stroke-width:2px
    style L fill:#fff,stroke:#333,stroke-width:2px
```
Sources:
- [tools/benchmark_field_memory.py:1-6]()
- [tools/benchmark_field_memory.py:10-19]()
- [tools/benchmark_field_memory.py:23]()
- [tools/benchmark_field_memory.py:24]()
- [tools/benchmark_field_memory.py:50-166]()
- [tools/compare_benchmarks.py:1-6]()
- [tools/compare_benchmarks.py:26-165]()
- [tools/benchmark_results/benchmark_vector_field_parallel_20260330_155229.json]()
- [tools/benchmark_results/benchmark_redis_parallel_20260330_164310.json]()

## CI Eval Gates

Evaluation gates are integrated into the CI pipeline to ensure that new features or changes meet specific performance and quality thresholds before being merged.

### PRD-198 Graphiti Gate

As described in the Retrieval Recall Evals section, the PRD-198 gate [orchestrator/evals/graphiti_vs_baseline.py:1-30]() is a prime example of a CI eval gate. It prevents the adoption of the Graphiti knowledge graph unless it demonstrates a significant uplift in retrieval recall and passes specific capability slice tests. The `test_prd198_graphiti_gate.py` [orchestrator/tests/test_prd198_graphiti_gate.py:1-8]() ensures this gate functions correctly.

### PRD-205 Auto Speaks

PRD-205, "Auto Speaks," focuses on background-to-chat delivery. While not a direct performance benchmark, its tests in `orchestrator/tests/test_prd205_auto_speaks.py` [orchestrator/tests/test_prd205_auto_speaks.py:1-17]() act as a quality gate in CI. These tests ensure:

-   `ChatMessenger` correctly posts assistant messages from background producers.
-   Clerk-string resolution and workspace-scoped chat validation work.
-   Per-(workspace,user) Auto-thread fallback is robust.
-   Origin capture is server-injected and unspoofable.
-   Watcher verdicts/actions/escalations land in the conversation.
-   Scheduled-task output is delivered, not discarded.
-   `chat_changed` frames route by name and `notify_chat_event` emits.
-   `/vote` and `/agents` resolve correctly.

The frontend counterpart `frontend/components/__tests__/prd205-auto-speaks.test.ts` [frontend/components/__tests__/prd205-auto-speaks.test.ts:1-8]() verifies that SSE `chat_changed` frames are parsed correctly and that `messages.source.label` is mapped to the metadata badge slot, ensuring persistence across reloads.

These tests, run in CI, act as gates to ensure the reliability and correctness of critical communication pathways.

### Diagram: CI Eval Gate Integration

```mermaid
graph TD
    subgraph "Developer Workflow"
        A[Code Change] --> B(Pull Request)
    end

    subgraph "CI Pipeline"
        B --> C{Run Unit Tests}
        C -- Pass --> D{Run Eval Gates}
        D -- Graphiti Eval Gate --> E{Retrieval Recall Check (PRD-198)}
        D -- Auto Speaks Eval Gate --> F{Background Chat Delivery Check (PRD-205)}
        D -- Other Evals --> G{Other Quality Checks}
        E -- ADOPT_UNBLOCKED --> H[Eval Gate Passed]
        E -- PENDING / DO_NOT_ADOPT --> I[Eval Gate Failed]
        F -- Pass --> H
        F -- Fail --> I
        G -- Pass --> H
        G -- Fail --> I
    end

    subgraph "Outcome"
        H --> J[Merge Allowed]
        I --> K[Merge Blocked]
    end

    style A fill:#fff,stroke:#333,stroke-width:2px
    style B fill:#fff,stroke:#333,stroke-width:2px
    style C fill:#fff,stroke:#333,stroke-width:2px
    style D fill:#fff,stroke:#333,stroke-width:2px
    style E fill:#fff,stroke:#333,stroke-width:2px
    style F fill:#fff,stroke:#333,stroke-width:2px
    style G fill:#fff,stroke:#333,stroke-width:2px
    style H fill:#fff,stroke:#333,stroke-width:2px
    style I fill:#fff,stroke:#333,stroke-width:2px
    style J fill:#fff,stroke:#333,stroke-width:2px
    style K fill:#fff,stroke:#333,stroke-width:2px
```
Sources:
- [orchestrator/evals/graphiti_vs_baseline.py:1-30]()
- [orchestrator/tests/test_prd198_graphiti_gate.py:1-8]()
- [orchestrator/tests/test_prd205_auto_speaks.py:1-17]()
- [frontend/components/__tests__/prd205-auto-speaks.test.ts:1-8]()

---