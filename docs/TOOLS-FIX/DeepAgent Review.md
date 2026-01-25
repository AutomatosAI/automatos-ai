# Architectural Analysis and Redesign Recommendations for Automatos AI Composio Integration

**Report Date:** 2026-01-25
**Prepared For:** Automatos AI Engineering Leadership
**Objective:** This report provides a deep architectural analysis of the current Automatos AI integration with the Composio platform. It examines the existing backend and frontend code, identifies architectural gaps and misalignments with Composio SDK best practices, and offers concrete recommendations for a strategic redesign. The goal is to create a more robust, scalable, and maintainable integration that fully leverages the capabilities of the Composio tool ecosystem.

## Executive Summary

The current integration of Composio into the Automatos AI platform is functional at a surface level but is architecturally suboptimal. It treats Composio as a simple collection of API endpoints rather than the sophisticated, SDK-driven tool orchestration layer it is designed to be. Our analysis reveals that the Automatos team has expended significant effort to custom-build functionalities that are already provided as core features within the Composio Python SDK. This has resulted in a brittle, complex, and inefficient implementation that fails to unlock the full potential of Composio's extensive tool library.

Key findings indicate a pattern of manual reimplementation in several critical areas:
1.  **Tool Discovery and Registration:** The system manually fetches lists of actions and painstakingly constructs tool definitions, a process the Composio SDK is designed to handle automatically.
2.  **Semantic Tool Search:** Automatos has developed an impressive internal semantic search engine for its own agents and tools. However, this capability has not been applied to the vast library of Composio tools, leaving a significant gap in dynamic tool discovery. The LLM is unable to find the best tool for a job from the 500+ available actions unless it is explicitly pre-loaded.
3.  **Execution and Abstraction:** Multiple layers of custom wrappers and executors have been built around the Composio SDK, adding unnecessary complexity and obscuring the SDK's native, more efficient execution pathways.
4.  **Data and State Management:** The platform uses its own database tables to manage Composio entity mappings and connection states, a task that, while necessary, is part of a larger pattern of building bespoke solutions for problems the SDK ecosystem helps to solve.

This report proposes a strategic architectural redesign centered on deep adoption of the Composio SDK's built-in features. Our recommendations include replacing the manual tool registration process with the SDK's native tool-loading mechanisms, implementing dynamic semantic tool discovery using Composio's `find_actions_by_use_case()` method, and streamlining the execution flow by removing redundant wrappers. Adopting these changes will dramatically simplify the codebase, reduce maintenance overhead, improve performance through features like SDK-level caching, and, most importantly, empower Automatos agents to intelligently and dynamically leverage the full spectrum of Composio's connected tools. We have provided a clear implementation roadmap to guide this transition, ensuring a phased and manageable migration to a more powerful and aligned architecture.

---

## 1. Deep Code and Architecture Analysis

A thorough review of the `orchestrator` and `frontend` codebases, supplemented by an analysis of database migrations and previous reports, reveals a consistent architectural approach to the Composio integration. The implementation is functional but follows a pattern of manual control and reimplementation of features inherent to the Composio SDK.

### 1.1. Orchestrator Backend Implementation

The backend architecture forms the core of the integration, managing everything from entity mapping and tool registration to execution. The implementation is spread across the API layer, core services, and modular tool components.

**Database Schema and Entity Management:**
The foundation of the integration is established in the database, as seen in the Alembic migration file `20260121_add_composio_integration.py`. This migration introduces two key tables: `composio_entities` and `composio_connections`. The `composio_entities` table creates a crucial link between an Automatos `workspace_id` and a `composio_entity_id`. This mapping is managed by the `core/composio/entity_manager.py` module, which generates a unique entity ID for each workspace (e.g., `automatos_your_workspace_id`). This is a sound and necessary practice for isolating data and tool access on a per-tenant basis within Composio. The `composio_connections` table is used to track the status of each application connection for a given entity. This indicates a design choice to mirror connection state within the Automatos database rather than relying solely on the Composio platform as the single source of truth, which adds a layer of synchronization complexity.

**Custom Composio Client Wrapper:**
At the heart of the interaction logic is a custom-built client wrapper located in `core/composio/client.py`. This `ComposioClient` class wraps the official `composio-core` SDK. It handles the initialization of the `Composio` object with the appropriate API key and exposes a set of bespoke methods for interacting with the service. These methods include `get_available_apps`, `get_app_actions`, and `execute_action`. A notable observation is that this wrapper appears to bypass the SDK for certain operations, making direct HTTP requests to the Composio backend (e.g., for fetching `triggers_types`). This hybrid approach of using both the SDK and direct API calls is a significant architectural concern, as it can lead to inconsistencies, bypass SDK-level improvements (like caching and error handling), and increase the maintenance burden when the Composio API evolves.

**Tool Registration and Discovery:**
The most significant area of architectural misalignment is the tool registration process, implemented in `modules/tools/registry/tool_registry.py`. The current system queries the database for tools where the provider is `composio` or the server URL is `composio://`. For each of these "base" app tools, it makes an API call via `composio_client.get_app_actions()` to fetch a list of all available actions for that application. The code then iterates through this list and manually constructs a unique tool definition for *every single action*. A helper function, `_build_composio_tool_name`, is used to generate sanitized and unique names (e.g., `composio_github_create_issue`).

This entire process is a manual and cumbersome reimplementation of a core Composio SDK feature. It forces the Automatos platform to manage the complexities of tool definition, schema formatting, and name sanitization. Furthermore, by loading every action for a connected app, it creates a bloated in-memory tool registry, which can negatively impact performance and increase the token count when presenting these tools to a Large Language Model (LLM).

**Tool Execution Flow:**
The execution of Composio tools is handled by the `modules/tools/execution/unified_executor.py`. This central executor identifies a tool as a Composio tool by checking for an `adapter_type` of `composio` in its metadata. When a match is found, it delegates the execution to a specialized `ComposioToolExecutor` defined in `core/composio/tool_executor.py`. This executor's primary role is to retrieve the correct `composio_entity_id` for the workspace, format the request, and call the `client.execute_action()` method from the custom wrapper. While this layered approach is organized, it introduces several layers of abstraction that are largely redundant. The SDK's native execution methods are designed to be called more directly, and these additional wrappers add unnecessary complexity to the call stack.

**Custom Semantic Search Infrastructure:**
The codebase contains a highly sophisticated, custom-built semantic search capability. Files such as `core/llm/semantic_skill_matcher.py`, `modules/rag/chunking/semantic_chunker.py`, and `modules/search/vector_store/store.py` demonstrate a significant investment in creating an engine that can find relevant internal tools and agents based on natural language queries. This system uses embeddings and vector similarity to match a task's description to an agent's skills. While impressive, this powerful engine is currently limited to internal Automatos resources. It is not being used to search within the vast library of Composio tools, representing a major missed opportunity for intelligent tool discovery.

### 1.2. Frontend Implementation

The frontend, built with Next.js and React, provides the user interface for managing Composio integrations. The implementation is clean and logically structured, reflecting the architecture of the backend.

**API Interaction and State Management:**
All communication with the backend's Composio endpoints is centralized in the `hooks/use-composio-api.ts` custom hook. This hook leverages the `react-query` library to manage server state, including caching lists of available and connected apps, and handling mutations for connecting or disconnecting applications. This is a robust and standard approach for managing asynchronous data in a modern React application.

**User Interface Components:**
The user-facing components are located in `components/composio/` and `components/tools/`. Key components include:
-   `ComposioAppsSection`: A dedicated UI for displaying Composio-related applications.
-   `AppConnectionButton`: A reusable button that triggers the connection flow for a specific app. The connection process correctly uses a callback URL to return the user to the Automatos platform after authenticating with the third-party service.
-   `ManageAppsModal` and `ToolConfigModal`: These components provide a user experience for managing connected applications and their actions. The `ToolConfigModal`, in particular, uses the `useAppActions` hook to fetch and display the individual actions associated with a connected Composio app. This confirms that the frontend is designed around the backend's approach of fetching all actions for an app and allowing the user to interact with them.

The frontend architecture is a logical extension of the backend's design. It effectively presents the data and functionality exposed by the backend API. However, because the backend is architecturally misaligned with the Composio SDK, the frontend is consequently limited to this suboptimal model of interaction.

---

## 2. Composio SDK Deep Dive: Unutilized Features

A deep investigation into the Composio Python SDK documentation and common usage patterns reveals a suite of powerful, built-in features that the current Automatos integration is not leveraging. Adopting these features is key to simplifying the architecture and unlocking advanced capabilities.

### 2.1. Intelligent Tool Discovery and Filtering

The current implementation fetches all actions for an app and relies on the LLM to find the right one from a potentially large list. The Composio SDK provides far more intelligent and efficient methods for tool discovery.

**Semantic Search with `find_actions_by_use_case()`:**
This is arguably the most critical and underutilized feature. The SDK includes a method specifically designed for semantic tool discovery. Instead of requiring an LLM to sift through dozens or hundreds of tool definitions, the developer can ask Composio for the most relevant tools in plain English.

For example, if an agent needs to address a task like "Notify the on-call engineer about a critical production error," the system can call `composio.find_actions_by_use_case(use_case="Notify the on-call engineer about a critical production error")`. Composio's backend will perform a semantic search across all available actions from connected apps and return a ranked list of the most relevant action names, such as `pagerduty.trigger_incident`, `slack.send_message`, and `twilio.send_sms`. This allows the Automatos orchestrator to dynamically identify a small, highly relevant subset of tools to present to the LLM, dramatically improving efficiency, accuracy, and reducing prompt token count. This feature directly obviates the need to extend the custom Automatos semantic search engine to cover Composio tools.

**Advanced Filtering with `get_tools()`:**
The primary method for fetching tool definitions, `composio.get_tools()`, is far more powerful than the simple `get_app_actions()` being used. It allows for server-side filtering by a variety of criteria, such as application name, action names, or categories. This enables the retrieval of only the necessary tool definitions. For instance, after using `find_actions_by_use_case()` to identify relevant actions, the orchestrator can fetch their full schemas with a single call: `composio.get_tools(actions=['pagerduty.trigger_incident', 'slack.send_message'])`. This is vastly more efficient than fetching all actions for the PagerDuty and Slack apps and filtering them in the application code.

### 2.2. Automated Tool Definition and LLM Integration

The current architecture involves a complex, manual process for creating tool definitions. The `composio-openai` package, listed in the `requirements.txt`, is designed to eliminate this entirely.

**The `ComposioToolSet` Helper:**
The `composio-openai` library provides a `ComposioToolSet` class (or a similarly named utility) that acts as an intelligent tool provider. It can be initialized with a list of desired applications (e.g., `apps=["github", "google"]`). The toolset object then handles the fetching of all relevant actions and, crucially, formats them into the precise JSON schema that the OpenAI API expects for its `tools` parameter. The developer simply calls a method like `toolset.get_tools()`, and the result is a list of perfectly formatted, ready-to-use tool definitions. This completely replaces the manual loops and dictionary construction found in `modules/tools/registry/tool_registry.py`. This not only simplifies the code but also ensures that the tool definitions are always up-to-date with the latest specifications from both Composio and the LLM provider.

### 2.3. Streamlined Execution and Entity Management

The current execution flow is obscured by multiple layers of custom wrappers. The SDK is designed for more direct interaction.

**Direct Execution with `composio.execute()`:**
The SDK provides a straightforward execution method, `composio.execute(tool_name, args, entity_id)`. This single method is the intended pathway for running any Composio action. It requires the unique name of the tool, a dictionary of arguments, and the `entity_id` of the user or workspace performing the action. The SDK internally handles the authentication, connection state, and API call logic. The custom `ComposioToolExecutor` and the various methods in the `ComposioClient` wrapper are largely redundant abstractions over this fundamental SDK function. Simplifying the execution path to call `composio.execute()` directly from the main orchestrator logic would reduce code complexity and remove potential points of failure.

**Implicit Entity Management:**
The `entity_id` is the core concept for tenancy and user management in Composio. The Automatos approach of creating a unique ID based on the workspace ID (`automatos_<workspace_id>`) is correct. The SDK consistently uses this `entity_id` across its methods—for connecting apps, finding use cases, and executing actions—to ensure all operations are performed in the context of the correct user with the right permissions and connections. While the Automatos database mapping is necessary, the application logic should focus on retrieving this ID and passing it to the SDK, allowing Composio to manage the underlying session and connection details.

### 2.4. Built-in Caching Mechanisms

The presence of a database migration for `cached_metadata` (`20260120_add_cached_metadata.py`) strongly suggests that Automatos has built its own system for caching tool schemas. High-performance SDKs like Composio's typically include a sophisticated, built-in caching layer (often in-memory with configurable persistence). This SDK-level caching is designed to minimize latency by avoiding repeated network requests for tool definitions, which rarely change. By relying on a custom database-backed cache, the Automatos platform is likely introducing unnecessary database I/O and reimplementing a performance optimization that is already available and likely more efficient within the SDK itself.

---

## 3. Gap Analysis: Reimplementations and Misalignments

The deep code analysis and SDK research reveal a clear and consistent gap: the Automatos integration is characterized by the manual reimplementation of core Composio SDK features. This has led to an architecture that is more complex, less efficient, and less capable than it could be.

### 3.1. What Automatos is Building that Composio Already Provides

**Manual Tool Registry vs. SDK Tool Loading:**
The most significant redundant effort is in `modules/tools/registry/tool_registry.py`. The entire process of fetching action names, looping through them, sanitizing names with `_build_composio_tool_name`, and manually constructing tool definition dictionaries is a reimplementation of the functionality provided by the `composio-openai` library's `ComposioToolSet`. The SDK is designed to deliver ready-to-use, LLM-compatible tool objects, eliminating the need for this entire subsystem. This manual process is not only redundant but also error-prone and difficult to maintain as tool schemas evolve.

**Custom Caching Layer vs. SDK Caching:**
The database migration `20260120_add_cached_metadata.py` points to the creation of a custom caching solution for tool metadata. The Composio SDK is engineered for performance and almost certainly includes its own optimized in-memory caching for these schemas to reduce network latency. The custom database cache adds an unnecessary layer of complexity and I/O operations, likely resulting in a less performant solution than the one provided out-of-the-box by the SDK.

**Complex Execution Wrappers vs. Direct SDK Execution:**
The chain of command for execution—from `UnifiedExecutor` to `_execute_composio_tool` to `ComposioToolExecutor` and finally to a custom client wrapper—is an overly complex abstraction. The SDK's `composio.execute()` method is designed for direct use. These additional layers in the Automatos codebase obscure the simple, underlying SDK call, increase the cognitive load for developers, and create more potential points of failure.

### 3.2. Features Automatos Should Be Using but Isn't

**Semantic Tool Discovery (`find_actions_by_use_case()`):**
This is the largest functional gap. Automatos has a powerful internal semantic search engine but has not applied this concept to the hundreds of external tools available via Composio. The current model requires the system to guess which apps to load or to load all tools from all connected apps, which is inefficient and unscalable. By not using `find_actions_by_use_case()`, Automatos is failing to empower its agents to dynamically discover the best tool for a given task from the entire Composio ecosystem. This severely limits the autonomy and problem-solving capability of the agents. An agent cannot use a tool it doesn't know exists; this feature is the bridge to that knowledge.

**Server-Side Filtering:**
The integration currently uses `get_app_actions()`, which retrieves all actions for a given application. This "fetch-everything-then-filter" approach is inefficient. The SDK's `get_tools()` method supports server-side filtering (e.g., by a list of action names). Using this feature, especially in combination with `find_actions_by_use_case()`, would allow the system to request only the specific tool definitions it needs, reducing network overhead and processing time.

**Automated LLM Tool Formatting:**
The `composio-openai` package exists specifically to bridge the gap between Composio tools and the OpenAI API. By manually creating tool definitions, Automatos is ignoring the primary purpose of this dependency. Using the provided helpers would ensure that the tool schemas are always correctly formatted and compatible with the target LLM, reducing the risk of integration errors when either the Composio tool schema or the LLM's function-calling API changes.

### 3.3. Architectural Misalignments

The core architectural misalignment is one of perspective. The current Automatos architecture treats Composio as a passive data source—a place to get a list of actions and a URL to call for execution. This "data-source" mindset has led to the development of custom systems to process and manage this data.

The recommended architectural perspective is to treat Composio as an **intelligent tool provider and execution engine**. In this model, the Automatos orchestrator's responsibility is not to manage the tools itself, but to query the provider (Composio) for the right tools for a given task, and then delegate execution to that provider. This aligns with modern principles of microservices and service-oriented architecture, where each component is trusted to perform its specialized function. Composio's specialty is discovering, formatting, and executing tools. The Automatos architecture should leverage this specialty, not replicate it. This shift in perspective from "managing tool data" to "consuming a tool service" is the fundamental change required to align the two systems effectively.

---

## 4. Concrete Redesign Recommendations

To bridge the identified gaps and align the Automatos platform with Composio SDK best practices, we recommend a series of targeted architectural changes. These changes are designed to replace custom-built, redundant components with more efficient, maintainable, and powerful SDK-native functionalities.

### 4.1. Recommendation 1: Overhaul Tool Registration with SDK-Native Loading

The current manual tool registration process is the primary source of complexity and should be the first target for redesign. The goal is to eliminate the manual fetching and formatting of actions.

**Current Implementation (`modules/tools/registry/tool_registry.py`):**
The existing code manually fetches actions and builds tool definitions in a loop. It involves custom name sanitization and dictionary construction, which is brittle and inefficient.

```python
# Simplified representation of the current approach
class ToolRegistry:
    def load_tools(self):
        # ...
        mcp_tools = db.query(MCPTool).filter(
            MCPTool.provider.ilike("composio")
        ).all()

        for mcp_tool in mcp_tools:
            base_tool_id = mcp_tool.name
            composio_client = get_composio_client()
            actions = composio_client.get_app_actions(base_tool_id) # Network call per app

            for action in actions:
                action_name = action.get("name")
                tool_name = self._build_composio_tool_name(action_name) # Custom naming logic

                # Manual construction of the tool definition dictionary
                tool_spec = {
                    "name": tool_name,
                    "description": action.get("description"),
                    "parameters": action.get("parameters"),
                    "metadata": {"adapter_type": "composio", ...}
                }
                self.register_tool(tool_spec)
```

**Proposed Implementation (using `composio-openai`):**
The redesigned approach leverages the `ComposioToolSet` from the `composio-openai` library to handle all discovery and formatting automatically.

```python
# Proposed redesign for the tool loading logic
from composio import Composio
from composio_openai import ComposioToolSet

class NewToolRegistry:
    def __init__(self, db, workspace_id):
        self.db = db
        self.workspace_id = workspace_id
        self.composio = Composio(api_key=get_composio_api_key())
        self.entity_id = self._get_entity_id(workspace_id)

    def _get_entity_id(self, workspace_id):
        # Logic to fetch 'automatos_<workspace_id>' from the database
        # ...
        return f"automatos_{workspace_id}"

    def load_tools_for_llm(self):
        """
        Loads and formats Composio tools directly for the LLM.
        """
        # 1. Get the list of apps the user has actually connected
        connected_apps = self.composio.get_connected_apps(entity_id=self.entity_id)
        app_names = [app['name'] for app in connected_apps]

        if not app_names:
            return []

        # 2. Initialize the toolset with the connected apps
        toolset = ComposioToolSet(apps=app_names)

        # 3. Get OpenAI-compatible tool definitions with a single method call
        # The toolset handles fetching, formatting, and SDK-level caching.
        openai_tools = toolset.get_tools(entity_id=self.entity_id)

        # 'openai_tools' is now a list of tool definitions ready to be passed
        # directly to the OpenAI client's 'tools' parameter.
        # No manual loops, no custom formatting, no redundant network calls.
        return openai_tools
```
**Why this is better:** This approach reduces hundreds of lines of complex, custom code to a few simple SDK calls. It eliminates redundant network requests, leverages the SDK's built-in caching, and guarantees that the tool definitions are always correctly formatted for the target LLM.

### 4.2. Recommendation 2: Implement Semantic Tool Discovery

This is the most impactful functional improvement. Instead of relying on static tool lists, the orchestrator should dynamically find the best tools for a given task using Composio's semantic search.

**Current State:**
No semantic search is performed for Composio tools. The system can only use tools that have been explicitly loaded into the agent's context.

**Proposed Design:**
Integrate a "tool-finding" step into the orchestrator's main workflow, before the final execution planning. This step will use `find_actions_by_use_case` to identify a small set of highly relevant tools.

```python
# Proposed logic within the main orchestrator service
from composio import Composio

async def orchestrate_task(task_description: str, workspace_id: str):
    # ... (initial setup)

    composio = Composio(api_key=get_composio_api_key())
    entity_id = get_entity_id_for_workspace(workspace_id)

    # --- NEW SEMANTIC DISCOVERY STEP ---
    # Find the top 3 most relevant Composio actions for the task
    try:
        relevant_actions = composio.find_actions_by_use_case(
            use_case=task_description,
            entity_id=entity_id,
            count=3  # Limit to the most relevant actions
        )
        print(f"Found relevant Composio actions: {relevant_actions}")
    except Exception as e:
        print(f"Could not perform semantic tool search: {e}")
        relevant_actions = []
    # --- END OF NEW STEP ---

    # Fetch full tool definitions for ONLY the relevant actions
    tools_for_llm = []
    if relevant_actions:
        # The ComposioToolSet can also be filtered to specific actions
        toolset = ComposioToolSet(actions=relevant_actions)
        tools_for_llm = toolset.get_tools(entity_id=entity_id)

    # Combine with internal Automatos tools
    all_available_tools = get_internal_tools() + tools_for_llm

    # Provide this small, highly relevant toolset to the LLM for planning
    llm_response = await llm_client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[{"role": "user", "content": task_description}],
        tools=all_available_tools,
        tool_choice="auto"
    )

    # ... (process LLM response and execute chosen tool)
```
**Why this is better:** This transforms the agent from a static tool user into a dynamic problem solver. It can now discover and utilize tools from the entire Composio library that are relevant to the user's immediate need, without requiring them to be pre-loaded. This dramatically increases the agent's capabilities and autonomy.

### 4.3. Recommendation 3: Simplify and Unify Tool Execution

The current multi-layered execution path should be flattened to remove redundant wrappers and call the SDK directly.

**Current Implementation (`modules/tools/execution/unified_executor.py`):**
The execution path involves multiple classes and methods, adding unnecessary overhead.

```python
# Simplified representation of the current execution path
class UnifiedExecutor:
    async def execute_tool(self, tool_spec, params, workspace_id):
        if tool_spec.metadata.get("adapter_type") == "composio":
            # Delegation to a specific Composio executor
            return await self._execute_composio_tool(tool_spec, params, workspace_id)
        # ...

    async def _execute_composio_tool(self, tool_spec, params, workspace_id):
        # Lazily initializes and calls another executor class
        if not self.composio_executor:
            self.composio_executor = ComposioToolExecutor(self.db)
        
        action_name = tool_spec.name.replace("composio_", "", 1)
        return await self.composio_executor.execute(
            action=action_name,
            params=params,
            workspace_id=workspace_id
        )
```

**Proposed Implementation (Direct SDK Call):**
The `UnifiedExecutor` should handle Composio execution directly, calling the SDK without intermediate wrappers.

```python
# Proposed redesign of the UnifiedExecutor
from composio import Composio

class UnifiedExecutor:
    def __init__(self, db):
        self.db = db
        self.composio = Composio(api_key=get_composio_api_key())

    async def execute_tool(self, tool_call, workspace_id):
        tool_name = tool_call.function.name
        tool_args = json.loads(tool_call.function.arguments)

        # Check if it's a Composio tool (e.g., by prefix or another marker)
        if tool_name.startswith("composio_"):
            entity_id = get_entity_id_for_workspace(workspace_id)
            
            # Execute directly using the SDK
            try:
                execution_result = self.composio.execute(
                    tool_name=tool_name,
                    args=tool_args,
                    entity_id=entity_id
                )
                return execution_result
            except Exception as e:
                # Handle execution errors
                return f"Error executing Composio tool {tool_name}: {e}"

        # ... (handle internal tools)
```
**Why this is better:** This change significantly simplifies the codebase by removing the `ComposioToolExecutor` class and the `_execute_composio_tool` method. The execution logic is clearer, more direct, and easier to debug. It treats Composio tools as first-class citizens within the main execution loop, relying on the SDK to handle the underlying complexity.

---

## 5. Implementation Roadmap

We propose a phased approach to implementing these architectural changes. This allows for incremental improvements and testing, minimizing disruption to the existing system.

### Phase 1: Simplify Execution and Remove Wrappers (1-2 Sprints)

The first step is to refactor the execution flow, as it provides immediate code simplification with minimal functional changes.

1.  **Refactor `UnifiedExecutor`:** Modify `modules/tools/execution/unified_executor.py` to call the Composio SDK's `execute` method directly, as detailed in Recommendation 4.3.
2.  **Deprecate `ComposioToolExecutor`:** Once the `UnifiedExecutor` is updated, the `core/composio/tool_executor.py` file and the `ComposioToolExecutor` class can be safely removed from the codebase.
3.  **Consolidate `ComposioClient`:** Review the custom `core/composio/client.py` wrapper. Any methods that are simple pass-throughs to the SDK should be deprecated. The goal is to reduce this file to only essential functions that cannot be handled by the SDK, such as API key management or highly specific custom logic.
4.  **Testing:** Implement comprehensive integration tests to ensure that tool execution continues to function correctly across different apps and argument types after the refactoring.

### Phase 2: Overhaul Tool Registration (2-3 Sprints)

This is a more significant architectural change that replaces the manual tool registry.

1.  **Integrate `ComposioToolSet`:** Create a new service or modify the existing `ToolRegistry` to use `ComposioToolSet` from the `composio-openai` library, as shown in Recommendation 4.1. This new logic will fetch connected apps for a user and use the toolset to get LLM-ready tool definitions.
2.  **Adapt LLM Integration:** Update the part of the code that calls the LLM to accept the tool format produced by `ComposioToolSet` directly. This may involve removing any custom formatting steps that were previously required.
3.  **Deprecate Manual Registration:** Once the new SDK-based loading mechanism is validated, the old code in `modules/tools/registry/tool_registry.py` responsible for looping through `get_app_actions` and building tool specs can be removed. The custom `_build_composio_tool_name` function will also become obsolete.
4.  **Frontend Validation:** Ensure the frontend continues to function correctly. While this is primarily a backend change, testing should confirm that the user's ability to see and manage tools is unaffected.

### Phase 3: Implement Semantic Tool Discovery (2-3 Sprints)

This phase introduces the most powerful new capability to the platform.

1.  **Create a Tool Discovery Service:** Implement a new service or a pre-processing step within the main orchestrator that calls `composio.find_actions_by_use_case()`, as detailed in Recommendation 4.2. This service will take a natural language task description and return a list of relevant Composio action names.
2.  **Integrate into Orchestration Flow:** Wire this new service into the main task execution pipeline. The flow should be: Task Description -> Semantic Discovery Service -> Get Relevant Action Names -> Fetch Full Tool Definitions -> Provide to LLM.
3.  **Update Prompting Strategy:** The prompts sent to the LLM for planning should be updated to reflect this new dynamic capability. For example, the prompt could state, "Based on a semantic search, the most relevant tools for your task are [tool1, tool2]. Please formulate a plan using one of these tools."
4.  **UI for Discovery (Optional Extension):** Consider adding a feature to the frontend chat or workflow builder that shows the user which tools were dynamically discovered for their specific query, increasing transparency and trust in the system.

By following this roadmap, Automatos AI can systematically transition from its current custom integration to a more streamlined, powerful, and maintainable architecture that fully harnesses the intelligent capabilities of the Composio platform.