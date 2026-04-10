# Shopify Agent Design

> Building a professional e-commerce operations agent using Automatos AI Platform

## Executive Summary

Shopify released their AI Toolkit (Winter '26) to help developers build Shopify apps. But Automatos can deliver something more powerful: a **business operations agent** that doesn't just help write code — it actually runs the store.

| Approach | Target User | What It Does |
|----------|-------------|--------------|
| Shopify AI Toolkit | Developers | Generates validated GraphQL, Liquid templates |
| Automatos Shopify Agent | Merchants/Operators | Manages inventory, orders, pricing, customer issues |

---

## Shopify AI Toolkit: What It Actually Is

### Overview

The Shopify AI Toolkit is **not** an SDK or traditional MCP server. It's a collection of:

- **Agent plugins/extensions** for AI coding clients (Claude Code, Cursor, VS Code, etc.)
- **16 skill files** teaching agents Shopify API patterns
- **Validation scripts** that check GraphQL against live Shopify schemas
- **CLI wrapper** that executes operations via `shopify store execute`

### Core Capabilities

| Capability | Description |
|------------|-------------|
| Live Docs + Schemas | Query canonical Shopify schemas (Admin GraphQL, Storefront, Liquid, UI extensions) |
| Code Validation | Validate GraphQL queries, Liquid templates, UI extensions before execution |
| Store Execution | Run operations through Shopify CLI's `store execute` command |
| Auto-updates | New capabilities land without manual upgrades |

### Installation

```bash
# Claude Code
/plugin marketplace add Shopify/shopify-ai-toolkit
/plugin install shopify-plugin@shopify-ai-toolkit

# Cursor
# Via Cursor Marketplace

# Gemini CLI
gemini extensions install https://github.com/Shopify/shopify-ai-toolkit
```

**Requirements:** Node.js 18+, Shopify CLI 3.93.0+

### Limitation

The toolkit generates CLI commands — it doesn't execute API calls directly. Users must run `shopify store auth` + `shopify store execute` manually.

---

## Automatos Shopify Agent: The Better Solution

### Why Automatos Is Different

Automatos already has the infrastructure to build a superior Shopify agent:

| Component | Automatos Capability | Benefit |
|-----------|---------------------|---------|
| **Composio** | ~400 Shopify APIs | Direct execution, no CLI needed |
| **Skills** | Import official toolkit skills | Domain knowledge + validation |
| **RAG** | Company documents | Business context (SOPs, contracts, pricing) |
| **Graphify** | Knowledge graph | Product → Supplier → Warehouse relationships |
| **NL2SQL** | Database queries | Internal data (margins, history, forecasts) |
| **Memory** | Persistent context | Learns patterns, preferences, seasonality |
| **Missions** | Automated workflows | Scheduled tasks, multi-step operations |
| **Multi-Agent** | Coordination | Works with Marketing, Support, Finance agents |

### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    SHOPIFY AGENT                                │
├─────────────────────────────────────────────────────────────────┤
│  PERSONA                                                        │
│  "Senior e-commerce operations specialist. Manages inventory,   │
│   orders, pricing, and customer issues. Proactive about stock   │
│   levels and fulfillment SLAs."                                 │
├─────────────────────────────────────────────────────────────────┤
│  TOOLS (Composio)                                               │
│  ~400 Shopify APIs: products, orders, customers, inventory,     │
│  fulfillment, discounts, metafields, webhooks, analytics...     │
├─────────────────────────────────────────────────────────────────┤
│  SKILLS (Official Toolkit)                                      │
│  shopify-admin, shopify-liquid, shopify-storefront-graphql,     │
│  shopify-functions, shopify-hydrogen, shopify-payments-apps...  │
├─────────────────────────────────────────────────────────────────┤
│  CONTEXT                                                        │
│  ├─ RAG: SOPs, product specs, vendor contracts, pricing rules  │
│  ├─ Graph: Product→Supplier→Warehouse relationships            │
│  ├─ NL2SQL: Internal orders DB, inventory history, margins     │
│  └─ Memory: Past decisions, customer preferences, seasonality  │
├─────────────────────────────────────────────────────────────────┤
│  MISSIONS                                                       │
│  • Daily: Low inventory alert → reorder recommendation          │
│  • Hourly: Unfulfilled orders > 24h → escalate                 │
│  • Weekly: Price competitiveness check vs competitors          │
│  • On-demand: "Prep for Black Friday" multi-step workflow      │
└─────────────────────────────────────────────────────────────────┘
```

---

## Comparison: Toolkit vs Automatos

| Dimension | Shopify AI Toolkit | Automatos Shopify Agent |
|-----------|-------------------|------------------------|
| **Target User** | Developers coding apps | Merchants running stores |
| **Knowledge** | Generic Shopify docs | YOUR business context (RAG + Graph) |
| **Interaction** | Single-turn Q&A | Multi-step missions |
| **Execution** | Generates CLI commands | Direct API execution |
| **Memory** | None | Learns your patterns |
| **Automation** | Manual | Scheduled missions |
| **Coordination** | Solo | Multi-agent (Marketing, Support, etc.) |
| **Data Access** | Shopify only | Shopify + internal DBs + documents |

---

## Use Cases

### Use Case 1: Order Delay Investigation

**User Query:** "We're getting complaints about delayed orders. What's going on?"

**Toolkit Response:**
> "Here's how to query unfulfilled orders in GraphQL..."
> ```graphql
> query { orders(first: 50, query: "fulfillment_status:unfulfilled") { ... } }
> ```

**Automatos Agent Response:**
1. Queries Shopify orders API (Composio) → finds 47 unfulfilled > 48h
2. Checks internal DB (NL2SQL) → correlates with supplier shipments
3. Finds pattern: 80% are SKUs from Supplier X
4. Checks RAG → finds Supplier X contract has 48h SLA
5. Drafts email to Supplier X with specific PO numbers
6. Creates mission: "Monitor Supplier X orders for next 7 days"
7. Notifies Support Agent to proactively reach out to affected customers

---

### Use Case 2: Black Friday Preparation

**User Query:** "Prep the store for Black Friday"

**Toolkit Response:**
> "Here's how to create a discount code..."

**Automatos Agent Response:**
1. Analyzes last year's Black Friday data (NL2SQL)
2. Identifies top 20 products by revenue
3. Checks current inventory levels (Shopify API)
4. Flags 5 products at risk of stockout
5. Generates reorder recommendations with supplier lead times (RAG)
6. Creates discount rules for slow-moving inventory
7. Updates product descriptions with holiday messaging
8. Schedules price changes for midnight Thursday
9. Creates monitoring mission for real-time sales dashboard

---

### Use Case 3: Customer Lifetime Value Optimization

**User Query:** "Which customers should we focus retention efforts on?"

**Automatos Agent Response:**
1. Queries customer data (Shopify API) — order history, frequency, AOV
2. Cross-references with support tickets (NL2SQL) — satisfaction signals
3. Checks purchase patterns against product graph (Graphify)
4. Identifies 150 high-LTV customers with declining engagement
5. Segments by product category preference
6. Drafts personalized re-engagement campaigns
7. Passes list to Marketing Agent for execution

---

### Use Case 4: Inventory Optimization

**User Query:** "Set up automatic inventory management"

**Automatos Agent Response:**
1. Creates daily mission: Check inventory levels at 6 AM
2. Applies reorder rules from uploaded SOP document (RAG)
3. Considers supplier lead times from vendor contracts (RAG)
4. Factors in seasonality from historical data (NL2SQL)
5. Auto-generates PO drafts when thresholds hit
6. Alerts on dead stock (no sales in 90 days)
7. Suggests markdowns based on margin rules

---

## Official Shopify AI Toolkit Skills

Import these skills to give the agent deep Shopify domain knowledge:

| Skill | Description | Link |
|-------|-------------|------|
| `shopify-admin` | Admin GraphQL API patterns, queries, mutations | [View](https://github.com/Shopify/shopify-ai-toolkit/tree/main/skills/shopify-admin) |
| `shopify-admin-execution` | Execute validated GraphQL against stores via CLI | [View](https://github.com/Shopify/shopify-ai-toolkit/tree/main/skills/shopify-admin-execution) |
| `shopify-storefront-graphql` | Storefront API for headless commerce | [View](https://github.com/Shopify/shopify-ai-toolkit/tree/main/skills/shopify-storefront-graphql) |
| `shopify-liquid` | Liquid templating for themes | [View](https://github.com/Shopify/shopify-ai-toolkit/tree/main/skills/shopify-liquid) |
| `shopify-hydrogen` | Hydrogen framework for headless storefronts | [View](https://github.com/Shopify/shopify-ai-toolkit/tree/main/skills/shopify-hydrogen) |
| `shopify-functions` | Shopify Functions for custom backend logic | [View](https://github.com/Shopify/shopify-ai-toolkit/tree/main/skills/shopify-functions) |
| `shopify-custom-data` | Metafields and metaobjects | [View](https://github.com/Shopify/shopify-ai-toolkit/tree/main/skills/shopify-custom-data) |
| `shopify-customer` | Customer accounts and authentication | [View](https://github.com/Shopify/shopify-ai-toolkit/tree/main/skills/shopify-customer) |
| `shopify-dev` | General Shopify development patterns | [View](https://github.com/Shopify/shopify-ai-toolkit/tree/main/skills/shopify-dev) |
| `shopify-partner` | Partner API and app management | [View](https://github.com/Shopify/shopify-ai-toolkit/tree/main/skills/shopify-partner) |
| `shopify-payments-apps` | Payment app development | [View](https://github.com/Shopify/shopify-ai-toolkit/tree/main/skills/shopify-payments-apps) |
| `shopify-polaris-admin-extensions` | Admin UI extensions with Polaris | [View](https://github.com/Shopify/shopify-ai-toolkit/tree/main/skills/shopify-polaris-admin-extensions) |
| `shopify-polaris-app-home` | App home page extensions | [View](https://github.com/Shopify/shopify-ai-toolkit/tree/main/skills/shopify-polaris-app-home) |
| `shopify-polaris-checkout-extensions` | Checkout UI extensions | [View](https://github.com/Shopify/shopify-ai-toolkit/tree/main/skills/shopify-polaris-checkout-extensions) |
| `shopify-polaris-customer-account-extensions` | Customer account page extensions | [View](https://github.com/Shopify/shopify-ai-toolkit/tree/main/skills/shopify-polaris-customer-account-extensions) |
| `shopify-pos-ui` | Point of Sale UI extensions | [View](https://github.com/Shopify/shopify-ai-toolkit/tree/main/skills/shopify-pos-ui) |

**Repository:** https://github.com/Shopify/shopify-ai-toolkit

---

## Implementation Steps

No code changes required. Pure configuration:

### Step 1: Create Agent (5 min)
- Name: "Shopify Operations"
- Persona: E-commerce operations specialist
- Model: Claude Sonnet or GPT-4

### Step 2: Assign Tools (2 min)
- Connect Shopify via Composio (OAuth2)
- ~400 APIs automatically available

### Step 3: Import Skills (10 min)
- Download SKILL.md files from toolkit repo
- Upload to Automatos Skills marketplace
- Assign to agent

### Step 4: Add Context (30 min)
- Upload business documents → RAG
- Product catalogs, SOPs, vendor contracts
- Run Graphify on uploaded docs

### Step 5: Create Missions (15 min)
- Daily inventory check
- Order fulfillment monitor
- Weekly pricing review

---

## Vertical Agent Template

This pattern extends to other platforms:

| Vertical | Composio App | Skills Source | Key Missions |
|----------|--------------|---------------|--------------|
| **Shopify** | SHOPIFY (~400 APIs) | shopify-ai-toolkit | Inventory, orders, pricing |
| **HubSpot** | HUBSPOT | hubspot-mcp | Lead scoring, campaigns |
| **QuickBooks** | QUICKBOOKS | accounting-* | Invoice reminders, reconciliation |
| **Zendesk** | ZENDESK | support-* | Ticket triage, escalation |
| **Salesforce** | SALESFORCE | salesforce-* | Pipeline, forecasting |

Each vertical agent combines:
- **Platform APIs** (via Composio) — direct execution
- **Domain Skills** — specialized knowledge
- **Business Context** (RAG + Graph + NL2SQL) — your data
- **Automation** (Missions) — scheduled workflows
- **Coordination** (Multi-agent) — cross-functional work

---

## Summary

Shopify's AI Toolkit is designed for developers building apps. Automatos delivers something more valuable: an AI agent that actually **operates** the business — with full context, direct execution, and automated workflows.

**No PRD needed. No code changes. Just configuration.**
