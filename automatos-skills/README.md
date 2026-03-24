# Automatos Skills Library

Professional agent skills for the Automatos AI Platform marketplace.

## Directory Structure

```
automatos-skills/
  skills/
    engineering/       # Software engineering, DevOps, architecture
    design/            # UI/UX, brand, design systems
    marketing/         # Content, SEO, social, growth
    sales/             # Outbound, discovery, pipeline, deals
    product/           # Strategy, prioritization, feedback
    project-management/ # Scheduling, tracking, coordination
    testing/           # QA, performance, API testing
    support/           # Customer support, analytics, compliance
    paid-media/        # PPC, ad creative, analytics
    specialized/       # Document generation, recruitment, etc.
```

Each skill lives in its own directory: `skills/{category}/{slug}/SKILL.md`

## SKILL.md Format

Every skill file uses YAML frontmatter followed by structured Markdown sections.

### Frontmatter Fields

```yaml
---
name: Skill Display Name
version: 1.0.0
category: engineering
tags: [backend, api, architecture]
description: >-
  One-paragraph description of what this skill enables an agent to do.
recommended_tools:
  - workspace_read_file
  - workspace_write_file
  - workspace_exec
  - GITHUB
recommended_model: sonnet-4.6
---
```

| Field | Required | Description |
|-------|----------|-------------|
| `name` | Yes | Human-readable skill name |
| `version` | Yes | Semver version string |
| `category` | Yes | One of the 10 category directories |
| `tags` | Yes | Array of searchable tags |
| `description` | Yes | 1-2 sentence description |
| `recommended_tools` | Yes | Array of Automatos tool names |
| `recommended_model` | Yes | One of: `haiku-4.5`, `sonnet-4.6`, `opus-4.6` |

### Markdown Sections

```markdown
## Identity

Who this agent is and their expertise domain.

## Core Mission

Primary objective and value proposition.

## Workflow

Step-by-step process the agent follows.

## Deliverables

What outputs the agent produces.

## Rules

Constraints, guardrails, and behavioral boundaries.
```

### Tool Name Mapping

| Automatos Tool | Description |
|----------------|-------------|
| `workspace_read_file` | Read files from agent workspace |
| `workspace_write_file` | Write files to agent workspace |
| `workspace_exec` | Execute shell commands in workspace |
| `workspace_list_dir` | List directory contents |
| `workspace_grep` | Search file contents |
| `workspace_git` | Git operations |
| `GITHUB` | GitHub integration (Composio) |
| `SLACK` | Slack integration (Composio) |
| `GOOGLE_SHEETS` | Google Sheets integration (Composio) |
| `GOOGLE_DOCS` | Google Docs integration (Composio) |

### Model Recommendations

| Model | Use For |
|-------|---------|
| `haiku-4.5` | Simple/repetitive tasks, high-volume agents |
| `sonnet-4.6` | Standard work, most agent tasks |
| `opus-4.6` | Complex reasoning, architecture, strategy |

## Import Script

Skills were imported from [agency-agents](https://github.com/msitarzewski/agency-agents) using:

```bash
python scripts/import_agency_skills.py \
  --source-dir scripts/vendor/agency-agents \
  --output-dir automatos-skills/skills \
  --category engineering
```

See `scripts/import_agency_skills.py --help` for full usage.
