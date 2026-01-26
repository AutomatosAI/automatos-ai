
import sys
import os
import logging
from datetime import datetime
from dotenv import load_dotenv

# Add orchestrator to path
sys.path.append(os.path.join(os.getcwd(), "orchestrator"))

# Load environment variables explicitly
load_dotenv(os.path.join(os.getcwd(), "orchestrator", ".env"))

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from core.database.database import get_database_url
from core.models.tools import Tool, ToolConfiguration
from core.models.tools import ToolCategory # Enum
try:
    from core.composio.client import get_composio_client
except ImportError:
    # Fallback/Mock for client if import fails (though it should exist)
    get_composio_client = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def sync_connections():
    url = get_database_url()
    try:
        engine = create_engine(url)  
        Session = sessionmaker(bind=engine)
        db = Session()
    except Exception as e:
        logger.error(f"Failed to connect to DB: {e}")
        return

    try:
        if not get_composio_client:
            logger.error("Could not import get_composio_client")
            return

        client = get_composio_client()
        # Fetch connections
        # The SDK method might vary. Usually client.get_connections() or similar.
        # Based on logs: GET /api/v3/connected_accounts
        connections = client.get_connections() 
        # If SDK doesn't have get_connections, we might need to use 'connected_accounts' endpoint via http.
        # Let's inspect active connections.
        
        # Actually client.profiles.get_list() or something?
        # Let's assume we can get a list of connected App names.
        # Safest is to iterate all tools and check connection status? No, too slow.
        
        # Let's try to get connected accounts.
        # Note: Composio SDK methods are: client.connected_accounts.get() ?
        # I'll wrap in try/except.
        
        connected_apps = set()
        try:
             # This returns a list of connected account objects
             # We want the 'app_name' from them.
             acts = client.connected_accounts.get()
             if isinstance(acts, list):
                 for act in acts:
                     if hasattr(act, 'app_name'):
                         connected_apps.add(act.app_name.lower())
                     elif isinstance(act, dict) and 'app_name' in act:
                         connected_apps.add(act['app_name'].lower())
             logger.info(f"Found connected apps in Composio: {connected_apps}")
        except Exception as e:
            logger.error(f"Failed to fetch connections via SDK: {e}")
            # Fallback for dev: Assume 'github' and 'composio' if user said "I have two"
            # But better to fail gracefully.
            return

        if not connected_apps:
            logger.warning("No connected apps found.")
            # For testing, if user said he has 2, and we find 0, maybe API key issue?
            pass

        # Now update DB
        # Find tools with provider in connected_apps
        # Note: 'composio_GITHUB_...' -> provider usually 'github' or 'composio'
        # In my sync script, I set provider="system" or "composio".
        # Wait, I set provider="system" if "composio" not in name...
        # Let's check the sync script I wrote:
        # provider="system" if "composio" not in tool_spec.name else "composio"
        # AND `ToolSpec` from `_register_mcp_tools` sets provider correctly?
        # Let's check `_register_mcp_tools`:
        # provider = mcp_tool.provider or mcp_tool.name (e.g. 'github')
        
        # Wait, if `sync_tools_to_db` used `registry.get_all_tools()`, 
        # the `ToolSpec` in registry HAS provider set.
        # But in my sync script I overrode it?
        # `provider="system" if "composio" not in tool_spec.name else "composio"`
        # This was a HUGE MISTAKE.
        # I forced everything to "composio" or "system".
        # I lost the actual provider (e.g. "github", "slack").
        # The `ToolSpec` object likely has the correct metadata.
        # I need to fix `sync_tools_to_db.py` to use `tool_spec.metadata.get('provider')`!
        
        # BUT FIRST, let's fix the provider in DB for existing tools? 
        # Or re-run sync?
        # Re-running sync with correct provider logic is better.
        
        logger.info("Script finish - this was just a check.")

    except Exception as e:
        logger.error(f"Error: {e}")
        db.rollback()
    finally:
        db.close()

if __name__ == "__main__":
    sync_connections()
