#!/usr/bin/env python3
"""
Update Main Router Script
==========================

Adds the new v2 API routers to main.py

Usage:
    python scripts/update_main_router.py
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

main_py_path = project_root / "main.py"

# Imports to add
new_imports = """
# NEW: Tools Redesign v2 API Routers
from api.composio_v2 import router as composio_v2_router
from api.agent_tools import router as agent_tools_router
"""

# Router includes to add
new_routes = """
# NEW: Tools Redesign v2 Routes
app.include_router(composio_v2_router)
app.include_router(agent_tools_router)
"""


def main():
    print("=" * 60)
    print("Updating main.py with new API routers...")
    print("=" * 60)
    
    if not main_py_path.exists():
        print(f"❌ Error: {main_py_path} not found")
        sys.exit(1)
    
    # Read current main.py
    with open(main_py_path, 'r') as f:
        content = f.read()
    
    # Check if already updated
    if "composio_v2_router" in content:
        print("✅ main.py already updated - no changes needed")
        return
    
    # Find where to insert imports (after other imports)
    import_marker = "from api."
    import_pos = content.rfind(import_marker)
    if import_pos == -1:
        print("❌ Error: Could not find import section")
        sys.exit(1)
    
    # Find end of that line
    import_line_end = content.find('\n', import_pos)
    
    # Insert new imports
    new_content = (
        content[:import_line_end + 1] +
        new_imports +
        content[import_line_end + 1:]
    )
    
    # Find where to insert routes (after app.include_router calls)
    router_marker = "app.include_router("
    last_router_pos = new_content.rfind(router_marker)
    if last_router_pos == -1:
        print("❌ Error: Could not find router section")
        sys.exit(1)
    
    # Find end of that line
    router_line_end = new_content.find('\n', last_router_pos)
    
    # Insert new routes
    final_content = (
        new_content[:router_line_end + 1] +
        new_routes +
        new_content[router_line_end + 1:]
    )
    
    # Backup original
    backup_path = main_py_path.with_suffix('.py.backup')
    with open(backup_path, 'w') as f:
        f.write(content)
    print(f"✅ Created backup: {backup_path}")
    
    # Write updated file
    with open(main_py_path, 'w') as f:
        f.write(final_content)
    
    print("✅ main.py updated successfully!")
    print("\nNext steps:")
    print("1. Restart your backend server")
    print("2. Test the new endpoints")
    print("3. Run: python scripts/sync_composio_metadata.py")
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
