# Import all models from core.py
# This allows imports like: from models import Agent, Skill, Workflow, Base
from .core import *

# PRD-37: Workspace model (required for FK resolution)
from .workspaces import *

# Also import specialized models from submodules
# Note: Import order matters - tools.py extends ToolUsageLog from core.py
from .credentials import *
from .system_settings import *
from .database_knowledge import *
from .enhanced import *
from .code_graph import *
from .context_policy import *
from .tools import *  # Import last - may extend models from core.py
from .tool_assignments import *  # PRD-35: Tool catalog and assignments
from .composio_cache import *  # Redesign: Composio metadata cache
from .routing import *  # PRD-50: Universal Orchestrator Router

# Optional model packs (may not exist in all deployments/branches)
try:
    from .composio import *  # PRD-36: Composio integration
except ImportError:
    pass

try:
    from core.workspaces.models import *  # PRD-37: Workspaces
    from core.workspaces import *  # Import all workspace module components
except ImportError:
    pass
