# Product Requirements Document (PRD)

# Automatos Marketplace: Curated Skills & Plugins Platform

**Version**: 1.0  
**Date**: February 2026  
**Author**: Automatos Team  
**Status**: Draft

---

## Executive Summary

Automatos Marketplace is a centralized, admin-curated repository of skills, plugins, personas, and commands that Automatos users can browse, enable, and assign to their AI agents. Unlike open marketplaces where users install arbitrary code, Automatos pre-installs vetted plugins from the Claude ecosystem (buildwithclaude.com, official marketplaces, etc.) and exposes them as a managed catalog to users.

### Key Differentiators
- **Admin-Curated**: Only Automatos admins can add plugins to the shared catalog
- **No User Code Execution**: Users select from pre-installed, vetted plugins
- **Per-Agent Assignment**: Users assign specific skills/plugins to specific agents
- **Token Optimization**: Agents only load assigned skills, not the entire catalog
- **Multi-Model Support**: Skills work across 400+ LLM models, not just Claude

---

## Goals & Objectives

### Business Goals
1. Reduce support burden by preventing users from installing broken/malicious plugins
2. Create differentiation through curated, high-quality skill library
3. Enable upsell opportunities (premium skills, skill packs)
4. Increase platform stickiness through rich ecosystem

### User Goals
1. Easily discover relevant skills for their use cases
2. Assign skills to agents without technical knowledge
3. Trust that all available skills are vetted and functional
4. Understand what each skill does before enabling

### Technical Goals
1. Sync plugins from external sources (GitHub, buildwithclaude.com)
2. Store skills efficiently (S3 + PostgreSQL metadata)
3. Load skills on-demand into agent pods
4. Support versioning and rollbacks
5. Track usage and performance per skill

---

## User Personas

### 1. Automatos Admin (Internal)
- Curates the marketplace catalog
- Syncs plugins from external sources
- Vets and approves skills for user access
- Monitors skill usage and performance
- Manages premium/free skill tiers

### 2. Workspace Owner (Customer)
- Browses available skills in marketplace
- Enables skills for their workspace
- Assigns skills to specific agents
- Views skill documentation and examples

### 3. Agent Builder (Customer)
- Creates and configures AI agents
- Selects persona, skills, and tools for each agent
- Tests agent behavior with different skill combinations
- Optimizes token usage through targeted skill selection

---

## System Architecture

### High-Level Overview

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│                           EXTERNAL SOURCES                                   │
│                                                                             │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐             │
│  │ buildwithclaude │  │ GitHub Markets  │  │ Official Claude │             │
│  │     .com        │  │                 │  │   Marketplace   │             │
│  │  (7,900+ items) │  │ (Community)     │  │   (Anthropic)   │             │
│  └────────┬────────┘  └────────┬────────┘  └────────┬────────┘             │
│           │                    │                    │                       │
│           └────────────────────┼────────────────────┘                       │
│                                │                                            │
│                                ▼                                            │
└────────────────────────────────┼────────────────────────────────────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │    SYNC SERVICE         │
                    │                         │
                    │  • Scheduled sync jobs  │
                    │  • Manual import        │
                    │  • Validation pipeline  │
                    │  • Transform to format  │
                    └────────────┬────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                       AUTOMATOS MARKETPLACE                                  │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                         STORAGE LAYER                                │   │
│  │                                                                      │   │
│  │  S3 Bucket                           PostgreSQL                      │   │
│  │  ─────────                           ──────────                      │   │
│  │  /skills/{slug}/{version}/           • marketplace_skills            │   │
│  │    ├── manifest.json                 • marketplace_plugins           │   │
│  │    ├── instructions.md               • marketplace_personas          │   │
│  │    ├── scripts/                      • skill_categories              │   │
│  │    └── assets/                       • sync_history                  │   │
│  │                                      • usage_analytics               │   │
│  │  /plugins/{slug}/{version}/                                          │   │
│  │  /personas/{slug}.json                                               │   │
│  │                                                                      │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                 │                                           │
│                                 ▼                                           │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                      MARKETPLACE SERVICE                             │   │
│  │                                                                      │   │
│  │  API Endpoints:                                                      │   │
│  │  • GET  /marketplace/skills                (list catalog)           │   │
│  │  • GET  /marketplace/skills/{id}           (get skill details)      │   │
│  │  • GET  /marketplace/skills/{id}/content   (get full content)       │   │
│  │  • GET  /marketplace/plugins                                         │   │
│  │  • GET  /marketplace/personas                                        │   │
│  │  • GET  /marketplace/categories                                      │   │
│  │  • POST /agents/{id}/assembled-context     (for agent pods)         │   │
│  │                                                                      │   │
│  │  Internal (Admin Only):                                              │   │
│  │  • POST /admin/sync                        (trigger sync)           │   │
│  │  • POST /admin/skills                      (manual add)             │   │
│  │  • PUT  /admin/skills/{id}                 (update)                 │   │
│  │  • DELETE /admin/skills/{id}               (remove)                 │   │
│  │                                                                      │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                 │
          ┌──────────────────────┼──────────────────────┐
          │                      │                      │
          ▼                      ▼                      ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   WORKSPACE A   │    │   WORKSPACE B   │    │   WORKSPACE C   │
│                 │    │                 │    │                 │
│ Enabled Skills: │    │ Enabled Skills: │    │ Enabled Skills: │
│ • social-media  │    │ • code-review   │    │ • seo-expert    │
│ • copywriting   │    │ • devops        │    │ • analytics     │
│                 │    │                 │    │                 │
│ Agents:         │    │ Agents:         │    │ Agents:         │
│ └─ SalesBot     │    │ └─ CodeReviewer │    │ └─ SEOBot       │
│    └─ skills:   │    │    └─ skills:   │    │    └─ skills:   │
│      • social   │    │      • code-rev │    │      • seo      │
│      • copy     │    │      • devops   │    │      • analytics│
└─────────────────┘    └─────────────────┘    └─────────────────┘
```text

---

## Sync Pipeline

### Source Integration

The sync service pulls from multiple sources and normalizes to Automatos format:

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│                           SYNC PIPELINE                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  1. DISCOVERY                                                               │
│     │                                                                       │
│     ├─► buildwithclaude.com API/scrape                                     │
│     │   └─► Extract: name, description, category, source_url               │
│     │                                                                       │
│     ├─► GitHub Marketplaces                                                │
│     │   └─► Clone repo, read marketplace.json                              │
│     │   └─► Extract plugin entries                                         │
│     │                                                                       │
│     └─► Manual admin upload                                                │
│                                                                             │
│  2. FETCH                                                                   │
│     │                                                                       │
│     ├─► Clone/download plugin source                                       │
│     ├─► Read plugin.json manifest                                          │
│     ├─► Read SKILL.md files                                                │
│     ├─► Collect scripts, assets                                            │
│     └─► Read agents/, commands/, hooks/                                    │
│                                                                             │
│  3. TRANSFORM                                                               │
│     │                                                                       │
│     ├─► Normalize to Automatos schema                                      │
│     ├─► Extract instructions text                                          │
│     ├─► Parse frontmatter metadata                                         │
│     ├─► Generate token estimates                                           │
│     └─► Categorize (auto + manual override)                                │
│                                                                             │
│  4. VALIDATE                                                                │
│     │                                                                       │
│     ├─► Schema validation                                                  │
│     ├─► Security scan (no malicious code)                                  │
│     ├─► Quality check (has description, instructions)                      │
│     └─► Duplicate detection                                                │
│                                                                             │
│  5. STORE                                                                   │
│     │                                                                       │
│     ├─► Upload to S3 (versioned)                                           │
│     ├─► Insert/update PostgreSQL metadata                                  │
│     ├─► Update Redis cache                                                 │
│     └─► Log sync event                                                     │
│                                                                             │
│  6. NOTIFY (optional)                                                       │
│     │                                                                       │
│     └─► Webhook to admin dashboard                                         │
│     └─► Slack notification for new skills                                  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```text

### Sync Configuration

```yaml
# config/sync_sources.yaml

sources:
  - name: buildwithclaude
    type: api
    url: https://www.buildwithclaude.com/api/plugins
    schedule: "0 0 * * *"  # Daily at midnight
    enabled: true
    filters:
      categories:
        - development
        - productivity
        - marketing
      min_stars: 10  # If available
    
  - name: claude-plugins-official
    type: github_marketplace
    repo: anthropics/claude-plugins-official
    schedule: "0 */6 * * *"  # Every 6 hours
    enabled: true
    auto_approve: true  # Official = trusted
    
  - name: awesome-claude-plugins
    type: github_marketplace
    repo: Chat2AnyLLM/awesome-claude-plugins
    schedule: "0 0 * * 0"  # Weekly
    enabled: true
    auto_approve: false  # Requires manual review
    
  - name: superpowers
    type: github_plugin
    repo: obra/superpowers
    schedule: "0 0 * * *"
    enabled: true
    auto_approve: true

validation:
  require_description: true
  min_description_length: 50
  require_instructions: true
  max_skill_size_kb: 500
  blocked_patterns:
    - "eval("
    - "exec("
    - "__import__"
    - "subprocess.call"
```text

---

## Data Models

### Database Schema

```sql
-- ============================================
-- MARKETPLACE CATALOG (Admin-managed)
-- ============================================

-- Categories for organization
CREATE TABLE skill_categories (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    slug VARCHAR(50) UNIQUE NOT NULL,
    name VARCHAR(100) NOT NULL,
    description TEXT,
    icon VARCHAR(50),  -- emoji or icon name
    sort_order INT DEFAULT 0,
    parent_id UUID REFERENCES skill_categories(id),
    created_at TIMESTAMP DEFAULT NOW()
);

-- Skills in the marketplace
CREATE TABLE marketplace_skills (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    
    -- Identity
    slug VARCHAR(100) UNIQUE NOT NULL,
    name VARCHAR(255) NOT NULL,
    version VARCHAR(20) NOT NULL,
    
    -- Content location
    s3_path VARCHAR(500) NOT NULL,  -- s3://bucket/skills/{slug}/{version}/
    
    -- Metadata
    description TEXT NOT NULL,
    long_description TEXT,
    category_id UUID REFERENCES skill_categories(id),
    tags VARCHAR(50)[] DEFAULT '{}',
    
    -- Source tracking
    source_type VARCHAR(50) NOT NULL,  -- 'buildwithclaude', 'github', 'manual'
    source_url VARCHAR(500),
    source_repo VARCHAR(200),
    source_marketplace VARCHAR(100),
    original_author VARCHAR(200),
    
    -- Token optimization
    token_estimate INT,  -- Estimated tokens when loaded
    recommended_models VARCHAR(100)[] DEFAULT '{}',
    
    -- Admin controls
    is_active BOOLEAN DEFAULT true,
    is_premium BOOLEAN DEFAULT false,
    is_featured BOOLEAN DEFAULT false,
    approval_status VARCHAR(20) DEFAULT 'pending',  -- pending, approved, rejected
    approved_by UUID REFERENCES admin_users(id),
    approved_at TIMESTAMP,
    
    -- Analytics
    enable_count INT DEFAULT 0,
    last_synced_at TIMESTAMP,
    
    -- Timestamps
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Plugins (bundles of skills, commands, agents, etc.)
CREATE TABLE marketplace_plugins (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    
    slug VARCHAR(100) UNIQUE NOT NULL,
    name VARCHAR(255) NOT NULL,
    version VARCHAR(20) NOT NULL,
    
    s3_path VARCHAR(500) NOT NULL,
    
    description TEXT NOT NULL,
    category_id UUID REFERENCES skill_categories(id),
    tags VARCHAR(50)[] DEFAULT '{}',
    
    -- Plugin contents summary
    skills_count INT DEFAULT 0,
    commands_count INT DEFAULT 0,
    agents_count INT DEFAULT 0,
    hooks_count INT DEFAULT 0,
    
    -- Source
    source_type VARCHAR(50) NOT NULL,
    source_url VARCHAR(500),
    source_repo VARCHAR(200),
    
    -- Admin
    is_active BOOLEAN DEFAULT true,
    is_premium BOOLEAN DEFAULT false,
    approval_status VARCHAR(20) DEFAULT 'pending',
    
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Personas (system prompts)
CREATE TABLE marketplace_personas (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    
    slug VARCHAR(100) UNIQUE NOT NULL,
    name VARCHAR(255) NOT NULL,
    
    -- Content
    system_prompt TEXT NOT NULL,
    description TEXT NOT NULL,
    voice_description VARCHAR(500),  -- 'professional, friendly, concise'
    
    category_id UUID REFERENCES skill_categories(id),
    
    -- Model hints
    temperature FLOAT DEFAULT 0.7,
    recommended_models VARCHAR(100)[] DEFAULT '{}',
    
    is_active BOOLEAN DEFAULT true,
    is_premium BOOLEAN DEFAULT false,
    
    created_at TIMESTAMP DEFAULT NOW()
);

-- Sync history for audit trail
CREATE TABLE sync_history (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    
    source_name VARCHAR(100) NOT NULL,
    started_at TIMESTAMP NOT NULL,
    completed_at TIMESTAMP,
    
    status VARCHAR(20) NOT NULL,  -- running, completed, failed
    
    items_discovered INT DEFAULT 0,
    items_added INT DEFAULT 0,
    items_updated INT DEFAULT 0,
    items_skipped INT DEFAULT 0,
    items_failed INT DEFAULT 0,
    
    error_message TEXT,
    details JSONB,
    
    triggered_by VARCHAR(50)  -- 'scheduled', 'manual', 'admin:{id}'
);


-- ============================================
-- WORKSPACE LAYER (Per-customer)
-- ============================================

-- What skills a workspace has enabled
CREATE TABLE workspace_enabled_skills (
    workspace_id UUID REFERENCES workspaces(id) ON DELETE CASCADE,
    skill_id UUID REFERENCES marketplace_skills(id),
    
    enabled_at TIMESTAMP DEFAULT NOW(),
    enabled_by UUID REFERENCES users(id),
    
    PRIMARY KEY (workspace_id, skill_id)
);

-- What plugins a workspace has enabled
CREATE TABLE workspace_enabled_plugins (
    workspace_id UUID REFERENCES workspaces(id) ON DELETE CASCADE,
    plugin_id UUID REFERENCES marketplace_plugins(id),
    
    enabled_at TIMESTAMP DEFAULT NOW(),
    enabled_by UUID REFERENCES users(id),
    
    PRIMARY KEY (workspace_id, plugin_id)
);


-- ============================================
-- AGENT LAYER (Per-agent assignments)
-- ============================================

-- Skills assigned to specific agents
CREATE TABLE agent_assigned_skills (
    agent_id UUID REFERENCES agents(id) ON DELETE CASCADE,
    skill_id UUID REFERENCES marketplace_skills(id),
    
    priority INT DEFAULT 0,  -- Load order (higher = earlier in context)
    assigned_at TIMESTAMP DEFAULT NOW(),
    
    PRIMARY KEY (agent_id, skill_id)
);

-- Personas assigned to agents
CREATE TABLE agent_assigned_personas (
    agent_id UUID REFERENCES agents(id) ON DELETE CASCADE,
    persona_id UUID REFERENCES marketplace_personas(id),
    
    -- Or custom override
    custom_system_prompt TEXT,
    use_custom BOOLEAN DEFAULT false,
    
    assigned_at TIMESTAMP DEFAULT NOW(),
    
    PRIMARY KEY (agent_id, persona_id)
);


-- ============================================
-- ANALYTICS
-- ============================================

CREATE TABLE skill_usage_events (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    
    skill_id UUID REFERENCES marketplace_skills(id),
    agent_id UUID REFERENCES agents(id),
    workspace_id UUID REFERENCES workspaces(id),
    
    event_type VARCHAR(50) NOT NULL,  -- 'loaded', 'invoked', 'error'
    
    tokens_used INT,
    latency_ms INT,
    model VARCHAR(100),
    
    created_at TIMESTAMP DEFAULT NOW()
);

-- Index for analytics queries
CREATE INDEX idx_skill_usage_created ON skill_usage_events(created_at);
CREATE INDEX idx_skill_usage_skill ON skill_usage_events(skill_id);
```text

### S3 Object Structure

```text
s3://automatos-marketplace/

├── skills/
│   ├── social-media-marketing/
│   │   ├── v1.0.0/
│   │   │   ├── manifest.json          # Automatos-normalized metadata
│   │   │   ├── instructions.md        # Main skill instructions
│   │   │   ├── scripts/
│   │   │   │   └── analyze.py
│   │   │   └── assets/
│   │   │       └── templates/
│   │   └── v1.1.0/
│   │       └── ...
│   │
│   ├── code-review/
│   │   └── v2.0.0/
│   │       ├── manifest.json
│   │       ├── instructions.md
│   │       └── references/
│   │           ├── security-patterns.md
│   │           └── performance-tips.md
│   │
│   └── seo-expert/
│       └── ...

├── plugins/
│   ├── pr-review-toolkit/
│   │   └── v1.5.0/
│   │       ├── manifest.json
│   │       ├── skills/
│   │       │   ├── comments-review/
│   │       │   │   └── SKILL.md
│   │       │   └── test-review/
│   │       │       └── SKILL.md
│   │       ├── agents/
│   │       │   └── pr-reviewer.md
│   │       └── commands/
│   │           └── review.md
│   │
│   └── devops-automation/
│       └── ...

└── personas/
    ├── sales-expert.json
    ├── support-agent.json
    ├── technical-writer.json
    └── ...
```text

### Manifest Schema (Automatos Format)

```json
// skills/{slug}/{version}/manifest.json
{
  "schema_version": "1.0",
  "type": "skill",
  
  "identity": {
    "slug": "social-media-marketing",
    "name": "Social Media Marketing Expert",
    "version": "1.2.0",
    "description": "Expert-level social media marketing, content strategy, and engagement optimization",
    "long_description": "This skill transforms your agent into a social media marketing expert..."
  },
  
  "source": {
    "type": "github",
    "url": "https://github.com/example/social-media-skill",
    "marketplace": "awesome-claude-code-plugins",
    "original_author": "John Doe",
    "license": "MIT",
    "synced_at": "2026-02-03T12:00:00Z"
  },
  
  "content": {
    "instructions_file": "instructions.md",
    "scripts": [
      {
        "name": "analyze_engagement",
        "file": "scripts/analyze.py",
        "language": "python",
        "description": "Analyze engagement metrics"
      }
    ],
    "references": [],
    "assets": []
  },
  
  "optimization": {
    "token_estimate": 1500,
    "recommended_models": ["gpt-4o", "claude-sonnet-4-20250514", "gemini-1.5-pro"],
    "min_context_tokens": 8000
  },
  
  "metadata": {
    "category": "marketing",
    "tags": ["social-media", "content", "engagement", "marketing"],
    "use_cases": [
      "Create social media content calendars",
      "Optimize posting schedules",
      "Analyze engagement metrics"
    ]
  }
}
```text

---

## User Interface Flows

### 1. Admin: Sync Management

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│  AUTOMATOS ADMIN > MARKETPLACE > SYNC                                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  SYNC SOURCES                                               [+ Add] │   │
│  │                                                                      │   │
│  │  ┌────────────────────────────────────────────────────────────────┐ │   │
│  │  │ 🟢 buildwithclaude.com                                         │ │   │
│  │  │    Last sync: 2 hours ago | 7,901 items | Next: in 22 hours    │ │   │
│  │  │    [Sync Now] [Configure] [Disable]                            │ │   │
│  │  └────────────────────────────────────────────────────────────────┘ │   │
│  │                                                                      │   │
│  │  ┌────────────────────────────────────────────────────────────────┐ │   │
│  │  │ 🟢 anthropics/claude-plugins-official                          │ │   │
│  │  │    Last sync: 6 hours ago | 36 items | Next: in 30 min         │ │   │
│  │  │    [Sync Now] [Configure] [Disable]                            │ │   │
│  │  └────────────────────────────────────────────────────────────────┘ │   │
│  │                                                                      │   │
│  │  ┌────────────────────────────────────────────────────────────────┐ │   │
│  │  │ 🟡 Chat2AnyLLM/awesome-claude-plugins                          │ │   │
│  │  │    Last sync: 3 days ago | 816 items | 12 pending review       │ │   │
│  │  │    [Sync Now] [Review Pending] [Configure]                     │ │   │
│  │  └────────────────────────────────────────────────────────────────┘ │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  SYNC HISTORY                                                       │   │
│  │                                                                      │   │
│  │  ┌──────────┬────────────────────┬────────┬──────┬────────┬──────┐ │   │
│  │  │ Time     │ Source             │ Status │ Add  │ Update │ Skip │ │   │
│  │  ├──────────┼────────────────────┼────────┼──────┼────────┼──────┤ │   │
│  │  │ 2h ago   │ buildwithclaude    │ ✓ Done │ 3    │ 12     │ 7886 │ │   │
│  │  │ 6h ago   │ claude-official    │ ✓ Done │ 0    │ 2      │ 34   │ │   │
│  │  │ 1d ago   │ awesome-plugins    │ ✓ Done │ 15   │ 8      │ 793  │ │   │
│  │  │ 2d ago   │ buildwithclaude    │ ✗ Fail │ -    │ -      │ -    │ │   │
│  │  └──────────┴────────────────────┴────────┴──────┴────────┴──────┘ │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```text

### 2. Admin: Approval Queue

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│  AUTOMATOS ADMIN > MARKETPLACE > PENDING APPROVAL (12)                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  [Select All] [Approve Selected] [Reject Selected]                         │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ □ ai-code-reviewer v2.1.0                              [View Source] │   │
│  │   Source: awesome-claude-plugins | Category: Development            │   │
│  │   "AI-powered code review with security analysis..."                │   │
│  │   Token est: 2,100 | Scripts: 3 | References: 2                     │   │
│  │                                                                      │   │
│  │   ⚠️ Warnings:                                                       │   │
│  │   • Contains subprocess calls in scripts/scan.py                    │   │
│  │                                                                      │   │
│  │   [Approve] [Reject] [Review Code]                                  │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ □ tiktok-viral-generator v1.0.0                        [View Source] │   │
│  │   Source: buildwithclaude | Category: Marketing                     │   │
│  │   "Generate viral TikTok content ideas..."                          │   │
│  │   Token est: 800 | Scripts: 0 | References: 0                       │   │
│  │                                                                      │   │
│  │   ✓ No warnings                                                     │   │
│  │                                                                      │   │
│  │   [Approve] [Reject]                                                │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```text

### 3. User: Marketplace Browse

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│  AUTOMATOS > MARKETPLACE                                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  [🔍 Search skills, plugins, personas...                              ]    │
│                                                                             │
│  Categories: [All] [Development] [Marketing] [Sales] [Support] [Analytics] │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  FEATURED                                                           │   │
│  │                                                                      │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐              │   │
│  │  │ 📊 Analytics │  │ 💼 Sales     │  │ 🔧 DevOps    │              │   │
│  │  │    Expert    │  │    Expert    │  │    Master    │              │   │
│  │  │              │  │              │  │              │              │   │
│  │  │ ⭐ Featured  │  │ ⭐ Featured  │  │ ⭐ Featured  │              │   │
│  │  │ 2.1k enabled │  │ 1.8k enabled │  │ 1.5k enabled │              │   │
│  │  └──────────────┘  └──────────────┘  └──────────────┘              │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  ALL SKILLS (847)                                    Sort: Popular ▼ │   │
│  │                                                                      │   │
│  │  ┌────────────────────────────────────────────────────────────────┐ │   │
│  │  │ 📱 Social Media Marketing                          [+ Enable]  │ │   │
│  │  │                                                                 │ │   │
│  │  │ Expert-level social media marketing, content strategy, and     │ │   │
│  │  │ engagement optimization. Includes templates for major platforms.│ │   │
│  │  │                                                                 │ │   │
│  │  │ 🏷️ Marketing, Social Media    📊 ~1,500 tokens    👥 2.1k uses │ │   │
│  │  └────────────────────────────────────────────────────────────────┘ │   │
│  │                                                                      │   │
│  │  ┌────────────────────────────────────────────────────────────────┐ │   │
│  │  │ 🔍 SEO Expert                                       ✓ Enabled  │ │   │
│  │  │                                                                 │ │   │
│  │  │ Comprehensive SEO analysis, keyword research, and content      │ │   │
│  │  │ optimization for search engines.                               │ │   │
│  │  │                                                                 │ │   │
│  │  │ 🏷️ Marketing, SEO            📊 ~1,200 tokens    👥 1.8k uses │ │   │
│  │  └────────────────────────────────────────────────────────────────┘ │   │
│  │                                                                      │   │
│  │  [Load More...]                                                     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```text

### 4. User: Skill Detail

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│  AUTOMATOS > MARKETPLACE > Social Media Marketing                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                                                                      │   │
│  │  📱 Social Media Marketing Expert                                   │   │
│  │                                                                      │   │
│  │  Expert-level social media marketing, content strategy, and         │   │
│  │  engagement optimization. Includes templates for major platforms.   │   │
│  │                                                                      │   │
│  │  ┌──────────────┬──────────────┬──────────────┬──────────────┐     │   │
│  │  │ Version      │ Category     │ Est. Tokens  │ Workspaces   │     │   │
│  │  │ 1.2.0        │ Marketing    │ ~1,500       │ 2,147        │     │   │
│  │  └──────────────┴──────────────┴──────────────┴──────────────┘     │   │
│  │                                                                      │   │
│  │  [+ Enable for Workspace]                                           │   │
│  │                                                                      │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  WHAT THIS SKILL DOES                                               │   │
│  │                                                                      │   │
│  │  When assigned to an agent, this skill enables:                     │   │
│  │                                                                      │   │
│  │  • Creating content calendars for multiple platforms                │   │
│  │  • Optimizing posting schedules based on engagement data            │   │
│  │  • Writing platform-specific copy (Twitter, LinkedIn, Instagram)    │   │
│  │  • Analyzing engagement metrics and suggesting improvements         │   │
│  │  • Generating hashtag strategies                                    │   │
│  │                                                                      │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  RECOMMENDED MODELS                                                 │   │
│  │                                                                      │   │
│  │  Works best with:                                                   │   │
│  │  • GPT-4o, GPT-4o-mini (excellent)                                  │   │
│  │  • Claude Sonnet 4 (excellent)                                      │   │
│  │  • Gemini 1.5 Pro (good)                                            │   │
│  │                                                                      │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  SOURCE                                                             │   │
│  │                                                                      │   │
│  │  From: awesome-claude-code-plugins                                  │   │
│  │  Author: Michael Galpert                                            │   │
│  │  License: MIT                                                       │   │
│  │  Last synced: 2 hours ago                                           │   │
│  │                                                                      │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```text

### 5. User: Agent Configuration with Skills

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│  AUTOMATOS > AGENTS > Edit: LinkedIn Sales Bot                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  BASIC INFO                                                         │   │
│  │                                                                      │   │
│  │  Name:  [LinkedIn Sales Bot                              ]          │   │
│  │  Model: [gpt-4o-mini ▼]                                             │   │
│  │                                                                      │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  PERSONA                                                            │   │
│  │                                                                      │   │
│  │  ○ Use marketplace persona:  [Sales Expert ▼]                       │   │
│  │  ● Use custom persona:                                              │   │
│  │    ┌───────────────────────────────────────────────────────────┐   │   │
│  │    │ You are a LinkedIn sales expert. You help users find      │   │   │
│  │    │ prospects, write personalized outreach, and track...      │   │   │
│  │    └───────────────────────────────────────────────────────────┘   │   │
│  │                                                                      │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  ASSIGNED SKILLS                                          [+ Add]   │   │
│  │                                                                      │   │
│  │  Only skills enabled in your workspace are shown.                   │   │
│  │  [Browse Marketplace] to enable more.                               │   │
│  │                                                                      │   │
│  │  ┌────────────────────────────────────────────────────────────────┐ │   │
│  │  │ ☑ 📱 Social Media Marketing              ~1,500 tokens    [×] │ │   │
│  │  │ ☑ 🎯 Lead Generation                      ~1,200 tokens    [×] │ │   │
│  │  │ ☐ 🔍 SEO Expert                           ~1,200 tokens        │ │   │
│  │  │ ☐ 📊 Analytics Expert                     ~1,800 tokens        │ │   │
│  │  └────────────────────────────────────────────────────────────────┘ │   │
│  │                                                                      │   │
│  │  Estimated context usage: ~2,700 tokens (skills only)              │   │
│  │                                                                      │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  ASSIGNED TOOLS                                           [+ Add]   │   │
│  │                                                                      │   │
│  │  ┌────────────────────────────────────────────────────────────────┐ │   │
│  │  │ ☑ 🔗 LinkedIn API                         ✓ Connected          │ │   │
│  │  │ ☑ 📊 HubSpot CRM                          ✓ Connected          │ │   │
│  │  │ ☐ 🐦 Twitter API                          ✓ Connected          │ │   │
│  │  └────────────────────────────────────────────────────────────────┘ │   │
│  │                                                                      │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  [Cancel]                                              [Save Agent]         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```text

---

## API Specifications

### Marketplace Service API

```yaml
openapi: 3.0.0
info:
  title: Automatos Marketplace API
  version: 1.0.0

paths:
  # ============================================
  # PUBLIC ENDPOINTS (User-facing)
  # ============================================
  
  /marketplace/skills:
    get:
      summary: List all available skills
      parameters:
        - name: category
          in: query
          schema:
            type: string
        - name: search
          in: query
          schema:
            type: string
        - name: tags
          in: query
          schema:
            type: array
            items:
              type: string
        - name: page
          in: query
          schema:
            type: integer
            default: 1
        - name: limit
          in: query
          schema:
            type: integer
            default: 20
        - name: sort
          in: query
          schema:
            type: string
            enum: [popular, newest, name]
      responses:
        200:
          description: List of skills
          content:
            application/json:
              schema:
                type: object
                properties:
                  skills:
                    type: array
                    items:
                      $ref: '#/components/schemas/SkillSummary'
                  pagination:
                    $ref: '#/components/schemas/Pagination'

  /marketplace/skills/{skill_id}:
    get:
      summary: Get skill details
      parameters:
        - name: skill_id
          in: path
          required: true
          schema:
            type: string
      responses:
        200:
          description: Skill details
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/SkillDetail'

  /marketplace/skills/{skill_id}/content:
    get:
      summary: Get full skill content (instructions, scripts)
      description: Used by agent pods to load skill content
      parameters:
        - name: skill_id
          in: path
          required: true
          schema:
            type: string
      responses:
        200:
          description: Full skill content
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/SkillContent'

  # ============================================
  # WORKSPACE ENDPOINTS
  # ============================================
  
  /workspaces/{workspace_id}/skills:
    get:
      summary: List skills enabled for workspace
      responses:
        200:
          description: Enabled skills
    
    post:
      summary: Enable a skill for workspace
      requestBody:
        content:
          application/json:
            schema:
              type: object
              properties:
                skill_id:
                  type: string
      responses:
        201:
          description: Skill enabled

  /workspaces/{workspace_id}/skills/{skill_id}:
    delete:
      summary: Disable a skill for workspace
      responses:
        204:
          description: Skill disabled

  # ============================================
  # AGENT ENDPOINTS
  # ============================================
  
  /agents/{agent_id}/skills:
    get:
      summary: List skills assigned to agent
      responses:
        200:
          description: Assigned skills
    
    put:
      summary: Update skills assigned to agent
      requestBody:
        content:
          application/json:
            schema:
              type: object
              properties:
                skill_ids:
                  type: array
                  items:
                    type: string
      responses:
        200:
          description: Skills updated

  /agents/{agent_id}/assembled-context:
    get:
      summary: Get fully assembled context for agent pod
      description: Returns system prompt with persona + skills baked in, plus tool definitions
      responses:
        200:
          description: Assembled context
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/AssembledContext'

  # ============================================
  # ADMIN ENDPOINTS
  # ============================================
  
  /admin/sync:
    post:
      summary: Trigger sync from external sources
      security:
        - AdminAuth: []
      requestBody:
        content:
          application/json:
            schema:
              type: object
              properties:
                source:
                  type: string
                  description: Source name or "all"
      responses:
        202:
          description: Sync started

  /admin/skills:
    post:
      summary: Manually add a skill
      security:
        - AdminAuth: []
      requestBody:
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/CreateSkillRequest'
      responses:
        201:
          description: Skill created

  /admin/skills/{skill_id}/approve:
    post:
      summary: Approve a pending skill
      security:
        - AdminAuth: []
      responses:
        200:
          description: Skill approved

  /admin/skills/{skill_id}/reject:
    post:
      summary: Reject a pending skill
      security:
        - AdminAuth: []
      requestBody:
        content:
          application/json:
            schema:
              type: object
              properties:
                reason:
                  type: string
      responses:
        200:
          description: Skill rejected

components:
  schemas:
    SkillSummary:
      type: object
      properties:
        id:
          type: string
        slug:
          type: string
        name:
          type: string
        description:
          type: string
        category:
          type: string
        tags:
          type: array
          items:
            type: string
        token_estimate:
          type: integer
        enable_count:
          type: integer
        is_premium:
          type: boolean
        is_featured:
          type: boolean

    SkillDetail:
      allOf:
        - $ref: '#/components/schemas/SkillSummary'
        - type: object
          properties:
            long_description:
              type: string
            use_cases:
              type: array
              items:
                type: string
            recommended_models:
              type: array
              items:
                type: string
            source:
              type: object
              properties:
                type:
                  type: string
                url:
                  type: string
                author:
                  type: string
            version:
              type: string

    SkillContent:
      type: object
      properties:
        id:
          type: string
        slug:
          type: string
        name:
          type: string
        instructions:
          type: string
          description: Full markdown instructions
        scripts:
          type: array
          items:
            type: object
            properties:
              name:
                type: string
              language:
                type: string
              code:
                type: string
        references:
          type: array
          items:
            type: object
            properties:
              name:
                type: string
              content:
                type: string

    AssembledContext:
      type: object
      properties:
        agent_id:
          type: string
        model:
          type: string
        temperature:
          type: number
        system_prompt:
          type: string
          description: Complete system prompt with persona + all skills
        tools:
          type: array
          items:
            type: object
            description: OpenAI-format tool definitions
        token_estimate:
          type: integer
```text

---

## Implementation Phases

### Phase 1: Foundation (Weeks 1-2)
- [ ] Set up S3 bucket structure
- [ ] Create PostgreSQL schema
- [ ] Build basic Marketplace Service API
- [ ] Implement skill storage and retrieval
- [ ] Build admin UI for manual skill upload

### Phase 2: Sync Pipeline (Weeks 3-4)
- [ ] Build sync service framework
- [ ] Implement buildwithclaude.com connector
- [ ] Implement GitHub marketplace connector
- [ ] Add validation pipeline
- [ ] Build admin approval queue UI

### Phase 3: User Experience (Weeks 5-6)
- [ ] Build marketplace browse UI
- [ ] Build skill detail page
- [ ] Implement workspace skill enabling
- [ ] Build agent skill assignment UI
- [ ] Add search and filtering

### Phase 4: Agent Integration (Weeks 7-8)
- [ ] Implement `/agents/{id}/assembled-context` endpoint
- [ ] Update agent pods to fetch context on startup
- [ ] Add Redis caching for hot skills
- [ ] Implement context refresh on skill changes
- [ ] Add usage analytics

### Phase 5: Polish & Launch (Weeks 9-10)
- [ ] Performance optimization
- [ ] Admin analytics dashboard
- [ ] Documentation
- [ ] Beta testing with select users
- [ ] Production deployment

---

## Success Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Skills in catalog | 500+ | Count of approved skills |
| User skill enables | 5+ per workspace | Avg enabled skills |
| Agent skill assignments | 2-3 per agent | Avg skills per agent |
| Sync success rate | >99% | Successful syncs / total |
| Skill load time | <100ms | P95 latency |
| User satisfaction | >4.0/5 | Survey rating |

---

## Open Questions

1. **Premium Skills**: How to handle paid skills? Stripe integration? Skill packs?
2. **User Submissions**: Should users be able to submit skills for review?
3. **Versioning**: How to handle breaking changes in skill versions?
4. **Dependencies**: Can skills depend on other skills?
5. **Localization**: Should skills support multiple languages?

---

## Appendix

### A. Example Sync Job Output

```json
{
  "sync_id": "sync_abc123",
  "source": "buildwithclaude",
  "started_at": "2026-02-03T12:00:00Z",
  "completed_at": "2026-02-03T12:05:32Z",
  "status": "completed",
  "summary": {
    "discovered": 7901,
    "added": 15,
    "updated": 42,
    "skipped": 7832,
    "pending_review": 12,
    "failed": 0
  },
  "new_skills": [
    {"slug": "ai-code-reviewer", "name": "AI Code Reviewer"},
    {"slug": "tiktok-viral", "name": "TikTok Viral Generator"}
  ],
  "pending_review": [
    {"slug": "crypto-trader", "reason": "Contains financial advice"},
    {"slug": "web-scraper", "reason": "Contains subprocess calls"}
  ]
}
```text

### B. Claude Marketplace Format Reference

```json
// Standard Claude marketplace.json format
{
  "name": "my-marketplace",
  "version": "1.0.0",
  "description": "My custom marketplace",
  "owner": {
    "name": "Automatos",
    "url": "https://automatos.ai"
  },
  "plugins": [
    {
      "name": "my-skill",
      "description": "Does something useful",
      "version": "1.0.0",
      "source": "./plugins/my-skill",
      "category": "development",
      "strict": false
    }
  ]
}
```text

---

### C. Plugin Runtime Flow (Implemented)

The plugin system runtime loads plugin context into agent system prompts at execution time. Agents with assigned plugins **skip legacy skill loading entirely** — plugins replace skills as the primary capability mechanism.

**Runtime chain:**

```text
Agent assigned plugins (DB: agent_assigned_plugins)
  │
  ▼
PluginContextService.get_assigned_plugins(agent_id)
  │  Joins AgentAssignedPlugin ↔ MarketplacePlugin, ordered by priority ASC
  │  Returns: List[(AgentAssignedPlugin, MarketplacePlugin)]
  │
  ├──► Tier 1: build_tier1_summary(plugin_rows)
  │      • DB fields only — zero S3/Redis calls
  │      • ~200 tokens per plugin
  │      • Markdown output: name, slug, description, tags, skill/command counts
  │      • ALWAYS included in system prompt for ALL assigned plugins
  │
  └──► Tier 2: build_tier2_content(plugin_rows, task_context, max_plugins=2)
         │
         ├─ _select_relevant_plugins(rows, task_context, max_plugins)
         │    • Keyword scoring: name, slug, description, tags vs task_context
         │    • Tags weighted 2x (high-signal)
         │    • Words < 4 chars skipped from description
         │    • Fallback: priority order when no keyword matches
         │    • Returns top N most relevant plugins
         │
         └─ For each selected plugin:
              ├─ PluginContentCache.get_file(slug, version, "SKILL.md")
              │    Redis → S3 fallback → Redis populate
              │    Graceful: S3 failure logs warning, returns partial content
              │
              └─ PluginContentCache.get_manifest(slug, version)
                   Returns parsed manifest.json
                   Tools rendered as "**tool_name**: description"
```text

**Cache architecture:**

```text
PluginContentCache (singleton via get_plugin_content_cache())
  │
  ├─ Redis layer
  │    Key prefixes: plugin_content: / plugin_manifest: / plugin_files:
  │    TTL: PLUGIN_CACHE_TTL_SECONDS (default 3600s)
  │    Graceful: RedisError → falls back to S3 directly
  │
  ├─ S3 layer (MarketplaceS3Service)
  │    Path: plugins/{slug}/{version}/
  │
  └─ invalidate_plugin(slug, version)
       Deletes: content key + manifest key + scans plugin_files:* keys
```text

**Skills-vs-Plugins decision (caller responsibility):**

```python
plugin_rows = plugin_svc.get_assigned_plugins(agent_id)
if plugin_rows:
    # Plugins present → tier1 + tier2, SKIP legacy skills
    prompt += plugin_svc.build_tier1_summary(plugin_rows)
    prompt += await plugin_svc.build_tier2_content(plugin_rows, task)
else:
    # No plugins → load legacy skills as before
    prompt += load_skills(agent)
```text

**API endpoints (agent_plugins.py):**

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/agents/{id}/plugins` | GET | List assigned plugins with marketplace details |
| `/api/agents/{id}/plugins` | PUT | Replace assignments (dedup, workspace-enabled validation, priority = list order) |
| `/api/agents/{id}/assembled-context` | GET | Full assembled context: persona + tier1 + tier2 + tools |

**Agent response (agents.py `_build_agent_response`):**

The `plugins` array is included in every agent GET response via `AgentResponse` Pydantic model:
```json
{
  "plugins": [{
    "plugin_id": "uuid", "slug": "email-automation", "name": "Email Automation",
    "version": "1.2.0", "description": "Send and manage emails",
    "skills_count": 5, "commands_count": 3
  }]
}
```text

**Source files:**

| File | Purpose |
|------|---------|
| `orchestrator/core/services/plugin_context_service.py` | Core engine: query, tier1, tier2, relevance scoring |
| `orchestrator/core/services/plugin_cache.py` | Redis+S3 cache: get_file, get_manifest, invalidate |
| `orchestrator/api/agent_plugins.py` | REST endpoints: list, update, assembled-context |
| `orchestrator/api/agents.py` | `_build_agent_response()` — plugins in agent GET |
| `orchestrator/core/models/marketplace_plugins.py` | DB models: MarketplacePlugin, AgentAssignedPlugin, WorkspaceEnabledPlugin |
| `orchestrator/core/models/core.py` | AgentResponse Pydantic model with `plugins` field |

**Key design decisions:**

| Decision | Rationale |
|----------|-----------|
| Two-tier loading | Tier 1 (~200 tok/plugin) always cheap; Tier 2 (~2000 tok) only for relevant plugins — controls prompt budget |
| Keyword scoring (not embeddings) | Deterministic, zero-latency — plugin tags and slugs are high-signal enough |
| `max_plugins=2` default | 2 full SKILL.md files ~ 4000 tokens, reasonable context budget |
| Graceful S3 failure | Plugin header still appears if SKILL.md fails — LLM knows plugin exists |
| Priority-ordered fallback | When no keyword matches, first N by priority — admin-controlled ordering |

### D. Plugin System Test Suite (38 tests)

All tests use mocked DB/S3/Redis — no live services needed. Runs in ~0.8s.

**Test files:**

| File | Tests | Coverage |
|------|-------|----------|
| `orchestrator/tests/conftest.py` | — | Shared fixtures: mock_db, sample_plugin, three_plugins, mock_agent, _make_plugin() |
| `orchestrator/tests/test_plugin_context_service.py` | 17 | get_assigned_plugins (3), build_tier1_summary (4), build_tier2_content (5), _select_relevant_plugins (5) |
| `orchestrator/tests/test_plugin_assignment_api.py` | 8 | list_agent_plugins (4: details, empty, 404, 403), update_agent_plugins (4: replace, dedup, workspace validation, priority) |
| `orchestrator/tests/test_agents_api_plugins.py` | 4 | _build_agent_response (3: plugins present, empty, field values), AgentResponse model schema (1) |
| `orchestrator/tests/test_plugin_runtime_integration.py` | 9 | Skills-skip-when-plugins (2), tier1/tier2 prompt assembly (3), PluginContentCache (4: hit, miss+populate, Redis fallback, invalidation) |

**Run:**
```bash
cd orchestrator && python -m pytest tests/ -v --tb=short
```text

**Mock strategy:**
- DB: `MagicMock(spec=Session)` with chainable `.query().join().filter().order_by().all()`
- S3: Mock `PluginContentCache` via `patch("core.services.plugin_cache.get_plugin_content_cache")`
- Redis: Direct injection `cache._redis = MagicMock()` with `.get()`, `.setex()`, `.delete()`, `.scan_iter()`
- Auth deps: `sys.modules` stubs for `jwt`/`core.auth.clerk`/`core.auth.hybrid` (avoids full dep chain)

---

*Document End*
