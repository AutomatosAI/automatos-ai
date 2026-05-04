# Import all models from core.py
# This allows imports like: from models import Agent, Skill, Workflow, Base
from .core import *  # noqa: includes LLMModel, WorkspaceModel, UserApiKey, LLMUsage

# PRD-37: Workspace model (required for FK resolution)
from .workspaces import *

# Also import specialized models from submodules
# Note: Import order matters - tools.py extends ToolUsageLog from core.py
from .credentials import *
from .system_settings import *
from .database_knowledge import *
from .context_policy import *
from .tools import *  # Import last - may extend models from core.py
from .tool_assignments import *  # PRD-35: Tool catalog and assignments
from .composio_cache import *  # Redesign: Composio metadata cache
from .tool_routing import *  # PRD-139: Tool Routing Graph (edges, affinities, intent clusters)
from .routing import *  # PRD-50: Universal Orchestrator Router
from .channels import *  # PRD-55: Channel Connections (US-019)
from .marketplace_plugins import *  # PRD-42: Plugin Marketplace
from .personas import *  # PRD-42: Agent Personas
from .system_prompts import *  # PRD-58: System Prompt Management
from .voice_profiles import *  # PRD-74: Voice Profiles
from .business_profiles import BusinessProfile  # noqa: F401  # PRD-130: Business Intake Wizard
from .blueprints import *  # Governance: Agent Blueprints

# PRD-82A: Orchestration (Sequential Mission Coordinator)
from .orchestration import (  # noqa: F401
    OrchestrationRun,
    OrchestrationTask,
    OrchestrationTaskDependency,
    OrchestrationEvent,
)
from .orchestration_enums import (  # noqa: F401
    StateType,
    RunState,
    TaskState,
    EventType,
    ActorType,
    TaskType,
    TriggerRule,
    FailureReasonCode,
    ALLOWED_TASK_TRANSITIONS,
    ALLOWED_RUN_TRANSITIONS,
    TERMINAL_RUN_STATES,
    TERMINAL_TASK_STATES,
    DONE_TASK_STATES,
    BOARD_STATUS_MAP,
)

# PRD-72: Board Tasks — explicit import so `from core.models import BoardTask` works
from .core import BoardTask  # noqa: F811
from .core import BlogPost  # noqa: F811

# PRD-38.4: SDK API Keys (safe — standalone table, no FK deps on workspaces)
try:
    from .sdk_api_keys import *
except ImportError:
    pass

try:
    from .channels import *  # PRD-55: Channel Connections
except ImportError:
    pass

# Optional model packs (may not exist in all deployments/branches)
try:
    from .composio import *  # PRD-36: Composio integration
except ImportError:
    pass

try:
    from .cloud_sync import *  # PRD-42: Cloud Document Sync
except ImportError:
    pass

try:
    from core.workspaces.models import *  # PRD-37: Workspaces
    from core.workspaces import *  # Import all workspace module components
except ImportError:
    pass
