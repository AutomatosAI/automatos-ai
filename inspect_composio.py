import os
import sys
import json
import logging

# Add orchestrator to python path
sys.path.append(os.path.join(os.getcwd(), 'orchestrator'))

from core.composio.client import ComposioClient

# Manually load .env from orchestrator directory
try:
    with open('orchestrator/.env', 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'): continue
            if '=' in line:
                k, v = line.split('=', 1)
                os.environ[k.strip()] = v.strip().strip("'").strip('"')
except FileNotFoundError:
    print("Warning: orchestrator/.env file not found")


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def inspect_toolkit(app_name):
    client = ComposioClient()
    
    data = {}
    
    try:
        print(f"Fetching toolkit '{app_name}'...")
        toolkits = client.composio.toolkits.get()
        target_toolkit = next((t for t in toolkits if t.slug.lower() == app_name.lower()), None)
        
        if not target_toolkit:
            print(f"Toolkit '{app_name}' not found.")
            return

        # Serialize Toolkit
        toolkit_data = vars(target_toolkit).copy()
        if 'meta' in toolkit_data:
             toolkit_data['meta'] = vars(toolkit_data['meta'])
        
        # Remove private members
        toolkit_data = {k: v for k, v in toolkit_data.items() if not k.startswith('_')}
        data['toolkit'] = toolkit_data

        print(f"Fetching actions for {app_name}...")
        actions = client.composio.tools.get(user_id="placeholder", toolkits=[app_name], limit=5)
        
        # Serialize Actions
        data['actions'] = actions  # These are already dicts or objects we can serialize

        print("\n" + "="*30)
        print("JSON METADATA OUTPUT")
        print("="*30)
        import json
        print(json.dumps(data, indent=2, default=str))

    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    app_name = "gmail"
    if len(sys.argv) > 1:
        app_name = sys.argv[1]
    inspect_toolkit(app_name)
