"""
Bulk Load MCP Servers from JSON Library
========================================

Loads MCP server definitions from JSON file into the database.
Supports both initial load and updates of existing servers.

Usage:
    python scripts/load_mcp_servers.py [json_file]
    
    Default: mcp_servers_library_30.json

Example:
    python scripts/load_mcp_servers.py mcp_servers_library_30.json
    python scripts/load_mcp_servers.py mcp_servers_library_400.json
"""

import sys
import os
import json
from pathlib import Path
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy.orm import Session
from sqlalchemy import func
from database.database import SessionLocal
from database.models import MCPTool
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_mcp_servers_from_json(json_file: str, db: Session):
    """
    Load MCP servers from JSON file into database
    
    Args:
        json_file: Path to JSON file containing MCP server definitions
        db: Database session
    """
    
    # Read JSON file
    logger.info(f"📚 Reading MCP servers from: {json_file}")
    
    if not os.path.exists(json_file):
        logger.error(f"❌ File not found: {json_file}")
        return
    
    with open(json_file, 'r', encoding='utf-8') as f:
        servers = json.load(f)
    
    logger.info(f"📦 Found {len(servers)} MCP server definitions")
    logger.info("=" * 80)
    
    loaded = 0
    updated = 0
    skipped = 0
    errors = 0
    
    for idx, server_data in enumerate(servers, 1):
        try:
            server_name = server_data.get('name', f'Unknown-{idx}')
            logger.info(f"\n[{idx}/{len(servers)}] Processing: {server_name}")
            
            # Check if server already exists (by name)
            existing = db.query(MCPTool).filter(
                MCPTool.name == server_data['name']
            ).first()
            
            if existing:
                # Update existing server
                logger.info(f"  🔄 Server exists, updating...")
                
                existing.description = server_data.get('description')
                existing.provider = server_data.get('provider')
                existing.category = server_data.get('category')
                existing.icon = server_data.get('icon')
                existing.version = server_data.get('version')
                existing.status = server_data.get('status', 'inactive')
                existing.mcp_server_url = server_data.get('mcp_server_url')
                existing.capabilities = server_data.get('capabilities', {})
                existing.credentials_schema = server_data.get('credentials_schema', {})
                existing.tags = server_data.get('tags', [])
                existing.tool_metadata = server_data.get('metadata', {})
                existing.updated_at = datetime.now()
                
                updated += 1
                logger.info(f"  ✅ Updated: {server_name}")
                logger.info(f"     Category: {existing.category} | Provider: {existing.provider}")
                logger.info(f"     Auto-enable: {existing.tool_metadata.get('auto_enable_on_credential', 'N/A')}")
                
            else:
                # Create new server
                logger.info(f"  ➕ Creating new server...")
                
                mcp_tool = MCPTool(
                    name=server_data['name'],
                    description=server_data.get('description'),
                    provider=server_data.get('provider'),
                    category=server_data.get('category'),
                    icon=server_data.get('icon'),
                    version=server_data.get('version', '1.0.0'),
                    status=server_data.get('status', 'inactive'),
                    mcp_server_url=server_data.get('mcp_server_url'),
                    capabilities=server_data.get('capabilities', {}),
                    credentials_schema=server_data.get('credentials_schema', {}),
                    tags=server_data.get('tags', []),
                    tool_metadata=server_data.get('metadata', {}),
                    created_by='mcp_loader'
                )
                
                db.add(mcp_tool)
                loaded += 1
                logger.info(f"  ✅ Created: {server_name}")
                logger.info(f"     Category: {mcp_tool.category} | Provider: {mcp_tool.provider}")
                logger.info(f"     Tags: {', '.join(mcp_tool.tags[:3])}{'...' if len(mcp_tool.tags) > 3 else ''}")
                logger.info(f"     Auto-enable: {mcp_tool.tool_metadata.get('auto_enable_on_credential', 'N/A')}")
        
        except KeyError as e:
            errors += 1
            logger.error(f"  ❌ Missing required field in {server_name}: {e}")
        
        except Exception as e:
            errors += 1
            logger.error(f"  ❌ Error loading {server_name}: {e}")
    
    # Commit all changes
    try:
        db.commit()
        logger.info("\n" + "=" * 80)
        logger.info("💾 Database commit successful")
    except Exception as e:
        db.rollback()
        logger.error(f"\n❌ Database commit failed: {e}")
        logger.error("All changes rolled back")
        return
    
    # Print summary
    logger.info("\n" + "=" * 80)
    logger.info("📊 LOADING SUMMARY:")
    logger.info(f"  ✅ Loaded (new):     {loaded}")
    logger.info(f"  🔄 Updated:          {updated}")
    logger.info(f"  ⏭️  Skipped:          {skipped}")
    logger.info(f"  ❌ Errors:           {errors}")
    logger.info(f"  📦 Total in DB:      {loaded + updated}")
    logger.info("=" * 80)
    
    # Show category breakdown
    category_stats = db.query(
        MCPTool.category,
        func.count(MCPTool.id).label('count')
    ).group_by(MCPTool.category).all()
    
    logger.info("\n📋 Category Breakdown:")
    for category, count in category_stats:
        logger.info(f"  {category:20s}: {count:3d} servers")
    
    # Show status breakdown
    status_stats = db.query(
        MCPTool.status,
        func.count(MCPTool.id).label('count')
    ).group_by(MCPTool.status).all()
    
    logger.info("\n📊 Status Breakdown:")
    for status, count in status_stats:
        logger.info(f"  {status:20s}: {count:3d} servers")
    
    logger.info("\n🎉 MCP servers loading complete!")


if __name__ == "__main__":
    # Get JSON file from command line or use default
    if len(sys.argv) > 1:
        json_file = sys.argv[1]
    else:
        # Try to find the file in the workspace root
        workspace_root = Path(__file__).parent.parent.parent.parent
        json_file = workspace_root / 'mcp_servers_library_30.json'
        
        if not json_file.exists():
            logger.error(f"❌ Default file not found: {json_file}")
            logger.info("Usage: python scripts/load_mcp_servers.py <json_file>")
            sys.exit(1)
    
    logger.info("🚀 MCP Server Bulk Loader")
    logger.info(f"📁 JSON File: {json_file}")
    logger.info("")
    
    # Create database session
    db = SessionLocal()
    
    try:
        load_mcp_servers_from_json(str(json_file), db)
    except Exception as e:
        logger.error(f"\n💥 Fatal error: {e}", exc_info=True)
        sys.exit(1)
    finally:
        db.close()

