# 🧠 Orchestrator Module - The Brain of the Operation

> **"Plans are nothing. Planning is everything."**

---

## 💡 The Problem

**Agents are chaotic.**
- They forget what step they're on.
- They get stuck in loops.
- They lose context between steps.
- They don't know how to handle failure.

You need a **system** to manage the chaos. You need an Orchestrator.

---

## ✨ The Solution

The **Orchestrator Module** is the state machine that drives Automatos.

It is responsible for:
1.  **Decomposition**: Breaking big goals into small tasks.
2.  **Execution**: Running those tasks (sequentially or parallel).
3.  **State Management**: Remembering exactly where we are.
4.  **Resilience**: Handling crashes, retries, and timeouts.

---

## 🏗️ Architecture

```
modules/orchestrator/
├── engine.py            # The core execution loop
├── state.py             # Workflow state management
├── planner.py           # LLM-based task decomposition
├── tracker.py           # Progress tracking & events
└── strategies/          # Execution strategies
    ├── sequential.py    # Step-by-step
    ├── parallel.py      # Map-reduce style
    └── dynamic.py       # Adaptive execution
```

---

## 🚀 Key Concepts

### 1️⃣ **The Workflow Lifecycle**

Every workflow goes through a strict lifecycle:

1.  **`PENDING`**: Created but not started.
2.  **`PLANNING`**: The LLM is breaking down the goal.
3.  **`RUNNING`**: Tasks are executing.
4.  **`PAUSED`**: Waiting for user input.
5.  **`COMPLETED`**: Success!
6.  **`FAILED`**: Something went wrong (and we couldn't fix it).

![Workflow Execution](../../../docs/assets/images/workflow_execution.png)

### 2️⃣ **Smart Decomposition**

We don't just run a prompt. We **plan**.

**User Goal**: "Build a React app that displays stock prices."

**Orchestrator Plan**:
1.  [Task] Initialize React project (Tool: `create_react_app`)
2.  [Task] Create API client (Tool: `write_file`)
3.  [Task] Build UI components (Tool: `write_file`)
4.  [Task] Verify build (Tool: `run_command`)

### 3️⃣ **Resumable Execution**

The Orchestrator persists state to the database after **every step**.
- Server crash? **Resume exactly where you left off.**
- API timeout? **Retry just that step.**
- User intervention? **Pause, edit state, resume.**

---

## ⚡ How It Works

```python
from modules.orchestrator import OrchestratorEngine

# 1. Initialize
engine = OrchestratorEngine(workflow_id=123)

# 2. Run
await engine.start()

# Inside the engine:
# - Loads workflow state
# - If no plan, calls Planner to create tasks
# - Picks next PENDING task
# - Assigns to an Agent
# - Executes
# - Updates DB
# - Emits SSE event
# - Repeats until done
```

---

## 🛠️ Execution Strategies

### **Sequential (Default)**
Strict dependency order. Task B cannot start until Task A finishes.
*Best for: Coding, deployment, step-by-step guides.*

### **Parallel (Fan-Out/Fan-In)**
Run multiple independent tasks at once.
*Best for: Research, scraping, batch processing.*

### **Dynamic (The "Auto-Pilot")**
The Orchestrator re-evaluates the plan after every step.
*Best for: Debugging, exploration, open-ended research.*

---

## 🔮 The Future

- **Sub-Workflows**: Workflows that call other workflows.
- **Human-in-the-Loop**: Explicit "Approval" steps in the plan.
- **Time Travel**: Revert a workflow to a previous state and try a different path.

**The Orchestrator turns "chaos" into "process".** ⚙️
