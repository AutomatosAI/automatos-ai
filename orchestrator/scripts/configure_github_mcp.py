"""
Configure GitHub MCP Tool in Database
=====================================

This script properly configures the GitHub MCP tool with correct JSON schema
and credentials for repository operations.
"""

import sys
sys.path.append('/root/automatos-ai/orchestrator')

from database.database import get_db
from models import MCPTool

# GitHub MCP Configuration based on MCP standard
GITHUB_MCP_CONFIG = {
    "name": "GitHub MCP",
    "description": "GitHub repository management - clone repos, create PRs, manage issues, read files",
    "provider": "GitHub",
    "version": "1.0.0",
    "category": "Development",
    "status": "active",
    "icon": "github",
    # Tags as list (PostgreSQL ARRAY type)
    "tags_list": ["git", "github", "repository", "version-control", "code"],
    
    # MCP Server URL - using GitHub API
    "mcp_server_url": "https://api.github.com",
    
    # Tool capabilities - GitHub API methods
    "capabilities": {
        "methods": [
            # Repository operations
            "repos.get",              # Get repository details
            "repos.listForAuthenticatedUser",  # List user's repos
            "repos.getContent",       # Read file contents
            "repos.createOrUpdateFileContents",  # Write/update files
            
            # Pull Request operations
            "pulls.create",           # Create PR
            "pulls.list",             # List PRs
            "pulls.get",              # Get PR details
            "pulls.merge",            # Merge PR
            
            # Issue operations
            "issues.create",          # Create issue
            "issues.list",            # List issues
            "issues.get",             # Get issue details
            "issues.update",          # Update issue
            
            # Branch operations  
            "git.createRef",          # Create branch
            "git.getRef",             # Get branch
            "git.listMatchingRefs",   # List branches
            
            # Commit operations
            "repos.listCommits",      # List commits
            "git.getCommit",          # Get commit details
        ],
        
        # MCP standard format
        "mcp_version": "1.0",
        "protocol": "http",
        "authentication": "bearer_token",
        
        # GitHub-specific configuration
        "base_url": "https://api.github.com",
        "headers": {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28"
        },
        
        # Common parameters
        "common_params": {
            "owner": "Repository owner (username or organization)",
            "repo": "Repository name",
            "ref": "Branch or tag reference",
            "path": "File path in repository"
        }
    },
    
    # Credentials schema - what's needed to use this tool
    "credentials_schema": {
        "type": "object",
        "required": ["github_token"],
        "properties": {
            "github_token": {
                "type": "string",
                "description": "GitHub Personal Access Token with repo scope",
                "pattern": "^ghp_[a-zA-Z0-9]{36}$",
                "sensitive": True
            },
            "default_owner": {
                "type": "string",
                "description": "Default repository owner (optional)",
                "default": "AutomatosAI"
            }
        },
        
        # Token scopes required
        "required_scopes": [
            "repo",           # Full repository access
            "read:org",       # Read org data
            "write:org",      # Manage org (for creating repos)
            "delete_repo"     # Delete repositories (optional)
        ],
        
        # Example usage (DO NOT include real tokens in code)
        "example": {
            "github_token": "YOUR_GITHUB_TOKEN_HERE",
            "default_owner": "AutomatosAI"
        }
    },
    
    # Tool metadata
    "metadata": {
        "docs_url": "https://docs.github.com/en/rest",
        "rate_limits": {
            "authenticated": "5000 requests/hour",
            "unauthenticated": "60 requests/hour"
        },
        "usage_examples": [
            {
                "name": "Clone Repository",
                "method": "repos.get",
                "params": {"owner": "AutomatosAI", "repo": "automatos-ai"},
                "description": "Get repository details before cloning"
            },
            {
                "name": "Read File",
                "method": "repos.getContent",
                "params": {"owner": "AutomatosAI", "repo": "automatos-ai", "path": "README.md"},
                "description": "Read file contents from repository"
            },
            {
                "name": "Create Pull Request",
                "method": "pulls.create",
                "params": {
                    "owner": "AutomatosAI",
                    "repo": "automatos-ai",
                    "title": "Feature: New capability",
                    "head": "feature-branch",
                    "base": "main",
                    "body": "Description of changes"
                },
                "description": "Create a new pull request"
            }
        ]
    }
}

def main():
    """Configure GitHub MCP tool in database"""
    db = next(get_db())
    
    try:
        # Check if GitHub MCP already exists
        github_tool = db.query(MCPTool).filter(MCPTool.name == "GitHub MCP").first()
        
        if github_tool:
            print(f"✅ GitHub MCP tool already exists (ID: {github_tool.id})")
            print("   Updating configuration...")
            
            # Update existing tool
            github_tool.description = GITHUB_MCP_CONFIG["description"]
            github_tool.provider = GITHUB_MCP_CONFIG["provider"]
            github_tool.version = GITHUB_MCP_CONFIG["version"]
            github_tool.category = GITHUB_MCP_CONFIG["category"]
            github_tool.status = GITHUB_MCP_CONFIG["status"]
            github_tool.icon = GITHUB_MCP_CONFIG["icon"]
            github_tool.tags = GITHUB_MCP_CONFIG["tags_list"]  # PostgreSQL ARRAY
            github_tool.mcp_server_url = GITHUB_MCP_CONFIG["mcp_server_url"]
            github_tool.capabilities = GITHUB_MCP_CONFIG["capabilities"]
            github_tool.credentials_schema = GITHUB_MCP_CONFIG["credentials_schema"]
            github_tool.tool_metadata = GITHUB_MCP_CONFIG["metadata"]
            
            action = "Updated"
        else:
            print("📝 Creating new GitHub MCP tool...")
            
            # Create new tool
            github_tool = MCPTool(
                name=GITHUB_MCP_CONFIG["name"],
                description=GITHUB_MCP_CONFIG["description"],
                provider=GITHUB_MCP_CONFIG["provider"],
                version=GITHUB_MCP_CONFIG["version"],
                category=GITHUB_MCP_CONFIG["category"],
                status=GITHUB_MCP_CONFIG["status"],
                icon=GITHUB_MCP_CONFIG["icon"],
                tags=GITHUB_MCP_CONFIG["tags_list"],  # PostgreSQL ARRAY
                mcp_server_url=GITHUB_MCP_CONFIG["mcp_server_url"],
                capabilities=GITHUB_MCP_CONFIG["capabilities"],
                credentials_schema=GITHUB_MCP_CONFIG["credentials_schema"],
                tool_metadata=GITHUB_MCP_CONFIG["metadata"],
                created_by="system"
            )
            db.add(github_tool)
            action = "Created"
        
        db.commit()
        db.refresh(github_tool)
        
        print(f"\n✅ {action} GitHub MCP tool successfully!")
        print(f"   ID: {github_tool.id}")
        print(f"   Name: {github_tool.name}")
        print(f"   Provider: {github_tool.provider}")
        print(f"   Status: {github_tool.status}")
        print(f"   URL: {github_tool.mcp_server_url}")
        print(f"   Methods: {len(github_tool.capabilities.get('methods', []))}")
        print(f"\n📋 Available Methods:")
        for method in github_tool.capabilities.get('methods', []):
            print(f"   - {method}")
        
        print(f"\n🔐 Required Credentials:")
        for cred in github_tool.credentials_schema.get('required', []):
            print(f"   - {cred}")
        
        print("\n✅ GitHub MCP tool is ready to use!")
        print("   Toggle it ON in the Tools Dashboard to make it available to agents.")
        
        return github_tool.id
        
    except Exception as e:
        db.rollback()
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None
    finally:
        db.close()

if __name__ == "__main__":
    tool_id = main()
    if tool_id:
        print(f"\n🎯 Next steps:")
        print(f"   1. Go to Tools Dashboard UI")
        print(f"   2. Toggle 'GitHub MCP' to ENABLED")
        print(f"   3. Create a test workflow with task type: 'code_review' or 'repository_management'")
        print(f"   4. Agent will automatically receive GitHub MCP tool access!")

