# Design proposed org structure, teams, titles, and reporting lines

## Proposed Org Chart

```
CTO Auto
├── Engineering
│   ├── PATCHER (Bug Fix Engineer)
│   ├── QA ENGINEER (QA Engineer)
│   ├── CODER (Full-Stack Developer)
│   └── FORGE (Workflow Architect)
├── Operations
│   ├── JIRA ADMIN (Project Operations)
│   ├── SENTINEL (Infrastructure Watchdog)
│   └── ATLAS (Business Intelligence)
├── Research/Knowledge
│   ├── ORACLE (Knowledge Curator)
│   └── WebSearch (Research Specialist)
├── Marketing/Content
│   ├── HARPER (Content Engine)
│   ├── QUILL (Long-Form Content Writer)
│   ├── EDITOR (Blog Editor)
│   └── CANVAS (Visual Content Specialist)
├── Growth/Sales
│   ├── SCOUT (Lead Intelligence)
│   └── RALLY (Community Growth)
├── Support
│   ├── ECHO (Customer Support Chatbot)
│   └── COMMS (Communications Operations)
└── SCRIBE (Document Generation Specialist)
```

## Team Assignments

1. **Engineering**:
   - **PATCHER**: Focuses on bug fixes and code improvements.
   - **QA ENGINEER**: Ensures quality through automated testing.
   - **CODER**: Develops clean code across frontend and backend.
   - **FORGE**: Designs and optimizes workflows.

2. **Operations**:
   - **JIRA ADMIN**: Manages project workflows and Jira operations.
   - **SENTINEL**: Monitors infrastructure and platform health.
   - **ATLAS**: Tracks business metrics and generates reports.

3. **Research/Knowledge**:
   - **ORACLE**: Maintains knowledge base health and retrieval quality.
   - **WebSearch**: Provides research capabilities for various domains.

4. **Marketing/Content**:
   - **HARPER**: Creates social media and changelog content.
   - **QUILL**: Writes long-form blog posts and technical articles.
   - **EDITOR**: Reviews and improves blog drafts.
   - **CANVAS**: Generates visual and multimedia assets.

5. **Growth/Sales**:
   - **SCOUT**: Qualifies leads and researches prospects.
   - **RALLY**: Engages with communities and grows the waiting list.

6. **Support**:
   - **ECHO**: Handles customer-facing support queries.
   - **COMMS**: Manages internal and external communications.

7. **Document Generation**:
   - **SCRIBE**: Produces polished documents and reports.

## Titles and Reporting Lines

- **CTO Auto**: Oversees all teams and strategic direction.
  - **Engineering Manager**: Reports to CTO Auto.
    - PATCHER (Bug Fix Engineer)
    - QA ENGINEER (QA Engineer)
    - CODER (Full-Stack Developer)
    - FORGE (Workflow Architect)
  - **Operations Manager**: Reports to CTO Auto.
    - JIRA ADMIN (Project Operations)
    - SENTINEL (Infrastructure Watchdog)
    - ATLAS (Business Intelligence)
  - **Research/Knowledge Lead**: Reports to CTO Auto.
    - ORACLE (Knowledge Curator)
    - WebSearch (Research Specialist)
  - **Marketing/Content Lead**: Reports to CTO Auto.
    - HARPER (Content Engine)
    - QUILL (Long-Form Content Writer)
    - EDITOR (Blog Editor)
    - CANVAS (Visual Content Specialist)
  - **Growth/Sales Lead**: Reports to CTO Auto.
    - SCOUT (Lead Intelligence)
    - RALLY (Community Growth)
  - **Support Lead**: Reports to CTO Auto.
    - ECHO (Customer Support Chatbot)
    - COMMS (Communications Operations)
  - **SCRIBE (Document Generation Specialist)**: Reports directly to CTO Auto.

## Rationale

1. **Functional Alignment**:
   - The structure groups agents by their primary functions, ensuring clarity in roles and responsibilities.
   - Engineering, Operations, and Research/Knowledge form the core technical teams, while Marketing/Content, Growth/Sales, and Support focus on external and internal engagement.

2. **Scalability**:
   - Teams like Growth/Sales and Marketing/Content are currently small but can be expanded with additional agents as needed.
   - Engineering and Operations are well-staffed, ensuring robust technical and operational support.

3. **Efficiency**:
   - Reporting lines are designed to minimize bottlenecks, with direct lines to the CTO for critical functions like Document Generation.
   - Cross-functional collaboration is encouraged, especially between Research/Knowledge and Marketing/Content for content creation.

4. **Preservation of Agents**:
   - Every existing agent is retained and assigned a role that aligns with their description and capabilities.
   - No deletions or replacements are suggested, adhering to the task requirements.

5. **Future Augmentation**:
   - Understaffed teams (e.g., Growth/Sales) are noted for future expansion, ensuring the structure remains flexible and adaptable. 

This proposed structure leverages the strengths of each agent while maintaining a clear hierarchy and functional focus, ensuring the Automatos platform operates efficiently and effectively.

---

# Audit current agent roster and capabilities

Now I have all the data needed. Let me compile the comprehensive audit.

---

# Agent Roster Audit — Automatos AI

**Audit Date:** 2026-03-27 | **Auditor:** ATLAS (BI Agent) | **Workspace Agents:** 18 active | **7-Day LLM Spend:** $31.41

---

## Agent Inventory

### 1. COMMS (ID: 19)
| Attribute | Detail |
|---|---|
| **Role/Purpose** | Communications operations — manages email, Slack, and calendar workflows; triages messages; drafts replies; coordinates team comms |
| **Type** | Custom |
| **Model** | `openrouter/free` (OpenRouter) |
| **Temperature** | 0.7 |
| **Tools Assigned** | 3 |
| **Has System Prompt** | Yes |
| **Tags** | communications, email, slack, calendar, comms |
| **Department Fit** | Operations / Internal Comms |
| **Primary Business Function** | Internal communication triage and response drafting |
| **Notable Strengths** | Active heartbeat (runs every ~1 hour); zero-cost model keeps overhead at $0; multi-channel coverage (email, Slack, calendar) |
| **7-Day Usage** | 195 requests, $0.00 cost |
| **Overlaps** | Partial overlap with RALLY on outreach messaging |
| **Gaps** | No CRM integration; no external customer communication capability; relies on free-tier model which may limit quality for complex drafting |

---

### 2. PATCHER (ID: 89)
| Attribute | Detail |
|---|---|
| **Role/Purpose** | Automated bug fixer — reads tickets, reproduces issues, writes minimal fixes, validates tests, opens draft PRs |
| **Type** | Custom |
| **Model** | `anthropic/claude-sonnet-4.6` (Anthropic) |
| **Temperature** | 0.2 (low — precision-focused) |
| **Tools Assigned** | 3 (includes GitHub, Jira integrations) |
| **Has System Prompt** | Yes |
| **Tags** | bug-fix, patcher, coding, github, jira |
| **Department Fit** | Engineering |
| **Primary Business Function** | Automated bug triage and patch generation |
| **Notable Strengths** | Premium model for code accuracy; low temperature for deterministic output; integrated with both GitHub and Jira for end-to-end bug workflow |
| **7-Day Usage** | 116 requests, $0.16 cost (very cost-efficient) |
| **Overlaps** | Overlaps with CODER on code writing; overlaps with JIRA ADMIN on ticket management |
| **Gaps** | Scoped to bug fixes only — no feature development; no deployment capability |

---

### 3. JIRA ADMIN (ID: 93)
| Attribute | Detail |
|---|---|
| **Role/Purpose** | Project operations — creates, updates, transitions, and comments on Jira issues from QA reports and engineering workflows |
| **Type** | Support |
| **Model** | `deepseek/deepseek-chat` (OpenRouter) |
| **Temperature** | 0.2 |
| **Tools Assigned** | 2 |
| **Has System Prompt** | ⚠️ **No** |
| **Tags** | jira, project-management, ops, jira-admin |
| **Department Fit** | Engineering / Operations |
| **Primary Business Function** | Issue lifecycle management in Jira |
| **Notable Strengths** | Active heartbeat for periodic checks; budget model keeps costs at $0.007/week; tightly scoped to Jira operations |
| **7-Day Usage** | 4 requests, $0.007 cost |
| **Overlaps** | Overlaps with PATCHER on Jira ticket handling |
| **Gaps** | No system prompt — behavior is undefined beyond description; no sprint planning or roadmap capability; no cross-tool project management (only Jira) |

---

### 4. SCRIBE (ID: 102)
| Attribute | Detail |
|---|---|
| **Role/Purpose** | Document generation specialist — turns structured inputs into polished PDF, DOCX, and XLSX outputs |
| **Type** | Custom |
| **Model** | `openai/gpt-4.1-mini` (OpenAI) |
| **Temperature** | 0.7 |
| **Tools Assigned** | 6 (highest tool count on the roster) |
| **Has System Prompt** | Yes |
| **Tags** | documents, reports, pdf, docx, xlsx, scribe |
| **Department Fit** | Operations / Cross-functional |
| **Primary Business Function** | Report and document generation for all teams |
| **Notable Strengths** | Most tools of any agent (6); multi-format output (PDF, DOCX, XLSX); serves as a shared service for the entire org |
| **7-Day Usage** | 6 requests, $0.06 cost |
| **Overlaps** | Could overlap with QUILL on written content, but SCRIBE focuses on formatted documents while QUILL focuses on blog/editorial |
| **Gaps** | Low utilization suggests underuse or manual workarounds; no template management capability mentioned; no presentation (PPTX) support noted |

---

### 5. CANVAS (ID: 137)
| Attribute | Detail |
|---|---|
| **Role/Purpose** | Visual content specialist — generates and refines images and multimedia assets from creative briefs |
| **Type** | Custom |
| **Model** | `google/gemini-3-pro-image-preview` (Google) |
| **Temperature** | 0.6 |
| **Tools Assigned** | 0 |
| **Has System Prompt** | Yes |
| **Tags** | graphics, images, video, design, creative, canvas |
| **Department Fit** | Content / Marketing / Brand |
| **Primary Business Function** | Visual asset creation for product, marketing, and brand |
| **Notable Strengths** | Dedicated image-generation model; creative-focused temperature; serves marketing, product, and brand teams |
| **7-Day Usage** | 20 requests, $0.93 cost |
| **Overlaps** | Complements HARPER (social content) and QUILL (blog content) — provides the visual layer |
| **Gaps** | Zero tools assigned — cannot access file storage, brand guidelines docs, or asset libraries; no video editing capability despite "video" tag; no design system or template enforcement |

---

### 6. QA ENGINEER (ID: 138)
| Attribute | Detail |
|---|---|
| **Role/Purpose** | Runs automated test suites, classifies failures (P0–P3), correlates with platform logs, produces structured QA reports |
| **Type** | Custom |
| **Model** | `deepseek/deepseek-chat` (OpenRouter) |
| **Temperature** | 0.2 |
| **Tools Assigned** | 1 |
| **Has System Prompt** | Yes |
| **Tags** | testing, devops, nightly, qa-engineer, pytest, playwright |
| **Department Fit** | Engineering / QA |
| **Primary Business Function** | Automated testing and quality assurance reporting |
| **Notable Strengths** | Structured severity classification (P0–P3); integrates with nightly test suite (visible in activity feed); feeds into JIRA ADMIN and PATCHER pipeline |
| **7-Day Usage** | 62 requests, $0.15 cost |
| **Overlaps** | Works in tandem with PATCHER (QA finds bugs → PATCHER fixes) and JIRA ADMIN (QA reports → tickets) |
| **Gaps** | Only 1 tool assigned; no performance/load testing capability; no security testing; no visual regression testing |

---

### 7. WebSearch (ID: 141)
| Attribute | Detail |
|---|---|
| **Role/Purpose** | Universal search agent — flights, hotels, e-commerce, news, academic, and location data via natural language |
| **Type** | Custom |
| **Model** | `openai/gpt-5.4` (OpenAI) |
| **Temperature** | 1.1 (highest on roster — creative/exploratory) |
| **Tools Assigned** | 1 (Composio Search) |
| **Has System Prompt** | Yes |
| **Tags** | search, web, travel, shopping, flights, news |
| **Department Fit** | Cross-functional utility / Research |
| **Primary Business Function** | Real-time web intelligence and research |
| **Notable Strengths** | Most premium model on the roster (GPT-5.4); broad search coverage; high utilization (137 requests/week) |
| **7-Day Usage** | 137 requests, $6.84 cost (**2nd highest spender**) |
| **Overlaps** | Overlaps with SCOUT on research; overlaps with RALLY on community/market research |
| **Gaps** | Very high temperature (1.1) may produce inconsistent results; expensive model for search tasks; no result caching or deduplication |

---

### 8. ECHO (ID: 145)
| Attribute | Detail |
|---|---|
| **Role/Purpose** | Customer-facing chatbot widget for Automatos product support; RAG-only, no external tools |
| **Type** | Support |
| **Model** | `qwen/qwen3-vl-30b-a3b-thinking` (Qwen) |
| **Temperature** | 0.8 |
| **Tools Assigned** | 0 (by design — RAG-only) |
| **Has System Prompt** | Yes |
| **Tags** | budstacks, customer service, helper, chatbot, widget |
| **Department Fit** | Customer Support |
| **Primary Business Function** | Self-service customer support via website widget |
| **Notable Strengths** | Intentionally tool-free for safety (RAG-only prevents unintended actions); customer-facing with professional prompt; vision-capable model |
| **7-Day Usage** | Not in top-5 agents — low or no direct requests this period |
| **Overlaps** | None significant — only customer-facing agent |
| **Gaps** | No escalation path to human support; no ticket creation capability; no multi-language support mentioned; cannot access external knowledge beyond RAG |

---

### 9. SENTINEL (ID: 184)
| Attribute | Detail |
|---|---|
| **Role/Purpose** | Infrastructure watchdog — monitors platform health, error spikes, deploy status, service availability, LLM cost anomalies |
| **Type** | Support |
| **Model** | `openrouter/free` (OpenRouter) |
| **Temperature** | 0.3 |
| **Tools Assigned** | 1 |
| **Has System Prompt** | Yes |
| **Tags** | devops, monitoring, sentinel, infra |
| **Department Fit** | Engineering / DevOps |
| **Primary Business Function** | Platform health monitoring and alerting |
| **Notable Strengths** | Highest request count on the roster (415 requests/week); zero cost on free model; active heartbeat running every ~1 hour; stores baselines for comparison |
| **7-Day Usage** | 415 requests, $0.00 cost |
| **Overlaps** | Partial overlap with ATLAS on cost anomaly detection; both monitor LLM spend |
| **Gaps** | Free-tier model may miss nuanced anomaly patterns; no incident response capability (detect-only); no integration with PagerDuty/OpsGenie-style alerting; no log aggregation |

---

### 10. SCOUT (ID: 185)
| Attribute | Detail |
|---|---|
| **Role/Purpose** | Lead intelligence — qualifies prospects, scores leads, researches company fit, drafts outreach |
| **Type** | Custom |
| **Model** | `google/gemini-2.5-pro` (Google) |
| **Temperature** | 0.4 |
| **Tools Assigned** | ⚠️ **0** |
| **Has System Prompt** | Yes |
| **Tags** | growth, sales, lead-gen, scout |
| **Department Fit** | Go-to-Market / Sales |
| **Primary Business Function** | Lead qualification and outreach drafting |
| **Notable Strengths** | Strong model (Gemini 2.5 Pro); structured lead scoring rubric in prompt; low temperature for consistent evaluations |
| **7-Day Usage** | 27 requests, $0.31 cost |
| **Overlaps** | Overlaps with RALLY on outreach and growth activities |
| **Gaps** | **Zero tools assigned** — cannot access CRM, email, or web search to actually research leads; effectively limited to prompt-based analysis of provided data; no pipeline tracking |

---

### 11. HARPER (ID: 186)
| Attribute | Detail |
|---|---|
| **Role/Purpose** | Content engine — turns platform activity, milestones, and metrics into drafts for LinkedIn, changelogs, and X posts |
| **Type** | Custom |
| **Model** | `gpt-4` (OpenAI) |
| **Temperature** | 0.7 |
| **Tools Assigned** | 5 |
| **Has System Prompt** | Yes |
| **Tags** | marketing, content, social, harper |
| **Department Fit** | Marketing / Content |
| **Primary Business Function** | Social media and changelog content creation |
| **Notable Strengths** | Well-tooled (5 tools); active in Blog Pipeline recipe; channel-appropriate content formatting; 3rd highest spender indicating heavy use |
| **7-Day Usage** | 44 requests, $5.58 cost (**3rd highest spender**) |
| **Overlaps** | Overlaps with QUILL on content creation; overlaps with RALLY on social/community messaging |
| **Gaps** | Uses older GPT-4 model (not GPT-4.1 or 5.4); no analytics on post performance; no scheduling/publishing capability for social platforms; high input cost ratio ($5.53 input vs $0.04 output) suggests large context but short outputs |

---

### 12. ATLAS (ID: 188) — *This agent*
| Attribute | Detail |
|---|---|
| **Role/Purpose** | Business intelligence — tracks costs, usage trends, agent activity, operational metrics; produces weekly reports; flags anomalies |
| **Type** | Custom |
| **Model** | `anthropic/claude-opus-4.6` (Anthropic) |
| **Temperature** | 0.3 |
| **Tools Assigned** | 1 |
| **Has System Prompt** | Yes |
| **Tags** | analytics, bi, atlas, ops |
| **Department Fit** | Analytics / Operations |
| **Primary Business Function** | Operational intelligence and cost tracking |
| **Notable Strengths** | Premium reasoning model (Claude Opus); low temperature for analytical precision; structured weekly report format; anomaly detection thresholds defined |
| **7-Day Usage** | 59 requests, $4.43 cost (**4th highest spender**) |
| **Overlaps** | Partial overlap with SENTINEL on cost monitoring |
| **Gaps** | Only 1 tool; no direct database query tool for revenue/customer metrics; no Stripe integration for financial data; expensive model for routine metric gathering |

---

### 13. FORGE (ID: 189)
| Attribute | Detail |
|---|---|
| **Role/Purpose** | Workflow architect — converts natural-language automation requests into Automatos recipes with step design, data flow, and agent assignments |
| **Type** | Custom |
| **Model** | `google/gemini-2.5-pro` (Google) |
| **Temperature** | 0.3 |
| **Tools Assigned** | ⚠️ **0** |
| **Has System Prompt** | Yes |
| **Tags** | operations, automation, forge, workflow |
| **Department Fit** | Operations / Automation |
| **Primary Business Function** | Workflow and recipe design |
| **Notable Strengths** | Strong reasoning model; low temperature for precise recipe construction; dedicated automation architect role |
| **7-Day Usage** | 2 requests, $0.06 cost |
| **Overlaps** | None significant — unique role |
| **Gaps** | **Zero tools** — cannot test or deploy the recipes it designs; very low utilization (2 requests/week); no access to existing recipe library for reference |

---

### 14. ORACLE (ID: 190)
| Attribute | Detail |
|---|---|
| **Role/Purpose** | Knowledge curator — audits document health, reprocesses failed documents, tests retrieval quality, flags stale knowledge |
| **Type** | Custom |
| **Model** | `deepseek/deepseek-chat` (OpenRouter) |
| **Temperature** | 0.2 |
| **Tools Assigned** | ⚠️ **0** |
| **Has System Prompt** | Yes |
| **Tags** | knowledge, rag, oracle, documents |
| **Department Fit** | Knowledge Management |
| **Primary Business Function** | RAG knowledge base health and quality assurance |
| **Notable Strengths** | Dedicated knowledge quality role; low temperature for consistent auditing; budget model appropriate for document scanning |
| **7-Day Usage** | 2 requests, $0.003 cost |
| **Overlaps** | None significant — unique role |
| **Gaps** | **Zero tools** — cannot actually reprocess documents or access the knowledge base programmatically; extremely low utilization; no automated scheduling for audits |

---

### 15. QUILL (ID: 191)
| Attribute | Detail |
|---|---|
| **Role/Purpose** | Long-form content — blog posts, tutorials, case studies, release notes, founder-led articles |
| **Type** | Custom |
| **Model** | `deepseek/deepseek-chat` (DeepSeek direct) |
| **Temperature** | 0.6 |
| **Tools Assigned** | 2 |
| **Has System Prompt** | Yes |
| **Tags** | content, blog, long-form, writing, editorial, quill |
| **Department Fit** | Content / Marketing |
| **Primary Business Function** | Blog and long-form content production |
| **Notable Strengths** | **Highest request count among non-free agents** (259 requests); active in Blog Pipeline; cost-effective model for high-volume writing; balanced temperature for creative yet consistent output |
| **7-Day Usage** | 259 requests, $7.74 cost (**#1 highest spender**) |
| **Overlaps** | Overlaps with HARPER on content creation; overlaps with EDITOR on blog quality |
| **Gaps** | High cost driven by volume — may benefit from output caching; no SEO keyword research capability; no content calendar management |

---

### 16. RALLY (ID: 192)
| Attribute | Detail |
|---|---|
| **Role/Purpose** | Community growth — finds AI/automation communities, identifies influencers, maps outreach opportunities, grows waiting list |
| **Type** | Custom |
| **Model** | `google/gemini-2.5-pro` (Google) |
| **Temperature** | 0.5 |
| **Tools Assigned** | 2 |
| **Has System Prompt** | Yes |
| **Tags** | community, growth, ecosystem, outreach, waiting-list, rally |
| **Department Fit** | Go-to-Market / Community |
| **Primary Business Function** | Community engagement and ecosystem growth |
| **Notable Strengths** | Strong reasoning model; balanced temperature; dedicated community focus that no other agent covers |
| **7-Day Usage** | Not in top cost agents — moderate or low usage |
| **Overlaps** | Overlaps with SCOUT on outreach; overlaps with HARPER on messaging |
| **Gaps** | No Discord/Slack community management tools; no analytics on community engagement metrics; no event management capability |

---

### 17. EDITOR (ID: 305)
| Attribute | Detail |
|---|---|
| **Role/Purpose** | Blog editor — reviews drafts for clarity, engagement, grammar, SEO, factual accuracy; improves without changing voice; never publishes |
| **Type** | Custom |
| **Model** | `deepseek/deepseek-chat` (DeepSeek direct) |
| **Temperature** | 0.7 |
| **Tools Assigned** | 0 |
| **Has System Prompt** | Yes |
| **Tags** | blog, editing, review, seo, quality, draft |
| **Department Fit** | Content / Editorial |
| **Primary Business Function** | Content quality assurance and editorial review |
| **Notable Strengths** | Clear guardrail — never publishes, only improves; SEO-aware editing; works in Blog Pipeline with QUILL; cost-effective model |
| **7-Day Usage** | 62 requests, $0.56 cost |
| **Overlaps** | Tightly coupled with QUILL (writer → editor pipeline); partial overlap with HARPER on content quality |
| **Gaps** | Zero tools — cannot access SEO tools, analytics, or competitor content for benchmarking; no style guide enforcement mechanism |

---

### 18. CODER (ID: 308)
| Attribute | Detail |
|---|---|
| **Role/Purpose** | General-purpose developer — clean code in Next.js, Python, Java; thinks clearly and tests code |
| **Type** | Custom |
| **Model** | `anthropic/claude-sonnet-4.6` (Anthropic) |
| **Temperature** | 0.51 |
| **Tools Assigned** | 1 |
| **Has System Prompt** | ⚠️ **No** |
| **Tags** | develop, code, frontend, backend |
| **Department Fit** | Engineering |
| **Primary Business Function** | Feature development and general coding |
| **Notable Strengths** | Premium coding model (Claude Sonnet 4.6); multi-language (Next.js, Python, Java); covers both frontend and backend |
| **7-Day Usage** | 11 requests, $0.58 cost |
| **Overlaps** | Overlaps with PATCHER on code writing (CODER = features, PATCHER = bug fixes) |
| **Gaps** | **No system prompt** — behavior is undefined; very new agent (created today); no architecture/design documentation capability; no CI/CD integration |

---

## Capability Coverage

### Coverage Matrix by Department

| Department | Agents | Coverage Level | Key Capabilities |
|---|---|---|---|
| **Engineering** | CODER, PATCHER, QA ENGINEER | 🟢 Good | Feature dev, bug fixing, automated testing, severity classification, PR creation |
| **Go-to-Market** | SCOUT, RALLY | 🟡 Moderate | Lead scoring, prospect research, community mapping, outreach drafting — but SCOUT has zero tools |
| **Content** | QUILL, EDITOR, HARPER, CANVAS | 🟢 Strong | Blog writing, editorial review, social media drafts, visual assets, changelog generation. Full pipeline exists (write → edit → publish) |
| **Operations** | FORGE, JIRA ADMIN, COMMS | 🟡 Moderate | Workflow design, Jira management, internal comms — but FORGE has zero tools and low utilization |
| **Support** | ECHO | 🟠 Basic | RAG-only customer chatbot; no escalation, no ticketing, no multi-channel support |
| **Analytics** | ATLAS | 🟡 Moderate | Cost tracking, usage trends, anomaly detection, weekly reports — single agent, limited tool access |
| **Knowledge** | ORACLE | 🟠 Basic | Document health auditing — but zero tools and near-zero utilization |
| **Automation** | FORGE | 🟠 Basic | Recipe design only — no execution, testing, or monitoring of automations |
| **DevOps / Infra** | SENTINEL | 🟡 Moderate | Health monitoring, error spikes, cost anomalies — but free model and detect-only (no response) |
| **Research** | WebSearch | 🟢 Good | Broad web search across news, academic, travel, e-commerce — premium model |

### Model Distribution

| Provider | Agents | Models Used |
|---|---|---|
| **OpenRouter (free)** | COMMS, SENTINEL | `openrouter/free` — $0 cost |
| **DeepSeek** | QUILL, EDITOR | `deepseek/deepseek-chat` — budget tier |
| **OpenRouter (DeepSeek)** | JIRA ADMIN, QA ENGINEER, ORACLE | `deepseek/deepseek-chat` — budget tier |
| **Google** | SCOUT, FORGE, RALLY, CANVAS | `gemini-2.5-pro`, `gemini-3-pro-image-preview` — mid-premium |
| **OpenAI** | WebSearch, HARPER, SCRIBE | `gpt-5.4`, `gpt-4`, `gpt-4.1-mini` — mixed tiers |
| **Anthropic** | PATCHER, CODER, ATLAS | `claude-sonnet-4.6`, `claude-opus-4.6` — premium |
| **Qwen** | ECHO | `qwen3-vl-30b-a3b-thinking` — mid tier |

### Cost Concentration (7-Day)

The top 4 agents account for **$24.55 of $31.41 total spend (78%)**:
1. **QUILL** — $7.74 (24.6%) — 259 requests
2. **WebSearch** — $6.84 (21.8%) — 137 requests
3. **HARPER** — $5.58 (17.8%) — 44 requests
4. **ATLAS** — $4.43 (14.1%) — 59 requests

Seven agents cost less than $0.20/week combined. Two agents (COMMS, SENTINEL) run at $0.00 on free models.

### Active Heartbeats Observed
- **COMMS** — ~every 1 hour
- **SENTINEL** — ~every 1 hour
- **JIRA ADMIN** — ~every 15 minutes (most frequent)
- **Orchestrator** — ~every 30 minutes (system-level)

### Active Recipes Observed
- **Blog Pipeline** — multi-step (3–4 steps), heavily used today (12+ executions), involves QUILL → EDITOR → HARPER chain
- **Nightly Self-Test Suite** — 1-step, involves QA ENGINEER, multiple runs today

---

## Notable Overlaps

### 1. Content Creation Cluster (QUILL ↔ HARPER ↔ EDITOR)
- **QUILL** writes long-form blog content; **HARPER** writes social media and changelog content; **EDITOR** reviews blog drafts.
- This is a **healthy, intentional pipeline** — not redundancy. QUILL drafts → EDITOR refines → HARPER repurposes for social channels.
- **Risk:** HARPER's description says it "turns platform activity into drafts" which could overlap with QUILL's release-note and changelog writing. The boundary between "blog post about a release" (QUILL) and "changelog post" (HARPER) may blur.

### 2. Code Writing (CODER ↔ PATCHER)
- Both write code using the same model (`claude-sonnet-4.6`).
- **Intentional split:** PATCHER is scoped to bug fixes from tickets; CODER handles feature development.
- **Risk:** Without a system prompt, CODER's scope is undefined — it could drift into bug-fix territory or vice versa.

### 3. Jira/Ticket Management (PATCHER ↔ JIRA ADMIN)
- PATCHER reads and interacts with Jira tickets as part of its bug-fix workflow.
- JIRA ADMIN is a dedicated Jira operations agent.
- **Risk:** Both can create/update Jira issues, potentially causing duplicate comments or conflicting status transitions.

### 4. Cost & Health Monitoring (ATLAS ↔ SENTINEL)
- ATLAS tracks LLM costs and flags anomalies in weekly reports.
- SENTINEL monitors LLM cost anomalies in real-time.
- **Intentional split:** SENTINEL is real-time alerting; ATLAS is periodic analysis and reporting.
- **Risk:** Both may flag the same cost spike independently without coordination.

### 5. Growth & Outreach (SCOUT ↔ RALLY)
- SCOUT qualifies leads and drafts outreach.
- RALLY finds communities and maps outreach opportunities.
- **Risk:** Both could target the same prospects/communities without a shared pipeline or CRM.

### 6. Research Capability (WebSearch ↔ SCOUT ↔ RALLY)
- WebSearch is the general research utility, but SCOUT and RALLY both need research capabilities.
- Neither SCOUT nor RALLY has WebSearch-style tools assigned — they can't independently research prospects or communities.

---

## Preliminary Gaps

### Critical Gaps (No agent covers these at all)

| Gap Area | Impact | Notes |
|---|---|---|
| **Security & Compliance** | High | No agent monitors for security vulnerabilities, dependency CVEs, data privacy compliance, or access control audits |
| **Data Engineering / ETL** | Medium-High | No agent handles data pipeline management, database migrations, or data transformation workflows |
| **Product Management** | Medium | No agent tracks feature requests, manages roadmap prioritization, or synthesizes user feedback into PRD-style documents |
| **HR / People Ops** | Low (for AI company) | No agent manages hiring pipelines, onboarding, or team coordination — acceptable if team is small |
| **Financial Operations** | Medium | ATLAS tracks costs but has no Stripe/billing integration; no agent handles invoicing, revenue recognition, or financial reporting |
| **Customer Success** | Medium-High | ECHO handles support but no agent tracks customer health scores, onboarding completion, or churn risk |

### Configuration Gaps (Agents exist but are under-equipped)

| Agent | Issue | Severity |
|---|---|---|
| **SCOUT** | 0 tools assigned — cannot research leads, access CRM, or send emails | 🔴 High |
| **FORGE** | 0 tools assigned — cannot test or deploy the recipes it designs | 🔴 High |
| **ORACLE** | 0 tools assigned — cannot access document APIs to audit or reprocess | 🔴 High |
| **CANVAS** | 0 tools assigned — cannot save assets to storage or access brand guidelines | 🟡 Medium |
| **EDITOR** | 0 tools assigned — cannot access SEO tools or analytics | 🟡 Medium |
| **CODER** | No system prompt — behavior undefined | 🟡 Medium |
| **JIRA ADMIN** | No system prompt — behavior undefined | 🟡 Medium |
| **ECHO** | 0 tools by design (RAG-only), but no escalation mechanism | 🟡 Medium |

### Utilization Gaps (Agents with very low activity)

| Agent | 7-Day Requests | 7-Day Cost | Assessment |
|---|---|---|---|
| **FORGE** | 2 | $0.06 | Near-dormant — workflow design capability is underutilized |
| **ORACLE** | 2 | $0.003 | Near-dormant — knowledge auditing not happening regularly |
| **JIRA ADMIN** | 4 | $0.007 | Low activity — may be underutilized or tasks handled manually |
| **SCRIBE** | 6 | $0.06 | Low — document generation capability available but rarely invoked |

### Model Efficiency Gaps

| Agent | Concern |
|---|---|
| **WebSearch** | GPT-5.4 at temperature 1.1 for search tasks — premium model may be overkill; high temperature may reduce result consistency |
| **HARPER** | Uses older `gpt-4` — $5.53 input cost for only $0.04 output suggests very large context windows with minimal generation |
| **ATLAS** | Claude Opus 4.6 for metric gathering — premium reasoning model may be over-specified for routine data collection tasks |
| **COMMS / SENTINEL** | Free-tier models — appropriate for cost but may miss nuanced patterns or produce lower-quality outputs for complex tasks |

---

### Summary

The Automatos agent roster is **strongest in content production and engineering**, with well-defined pipelines (Blog Pipeline, QA → Bug Fix → PR). **Go-to-market and operations are structurally present but under-tooled** — SCOUT, FORGE, and ORACLE all lack the tools needed to fulfill their stated missions. The biggest organizational blind spots are **security/compliance, customer success beyond basic support, and financial operations**. Cost is concentrated in 4 agents (78% of spend), with content production (QUILL + HARPER) alone consuming 42% of the weekly LLM budget.

---

# Research marketplace opportunities to fill org gaps

## Market Scan

The current roster is already strong in a few areas:

- **Engineering execution:** CODER, PATCHER, QA ENGINEER, JIRA ADMIN
- **Content and communications:** HARPER, QUILL, EDITOR, SCRIBE, COMMS
- **Growth and research:** SCOUT, RALLY, WebSearch
- **Ops/analytics/platform support:** ATLAS, FORGE, ORACLE, SENTINEL
- **Customer-facing support:** ECHO
- **Creative/media:** CANVAS

From the shared field and roster audit, the biggest likely org gaps are:

- **Security / compliance**
- **Dedicated product/leadership support**
- **Customer success / retention**
- **Finance / revops**
- **Recruiting / HR**
- **Data engineering / data quality**
- **Sales enablement / account expansion**
- **Design/UX quality governance**

That matches what the marketplace and web scan suggest is becoming standard for AI-native orgs in 2026: companies are filling not just “build” roles, but also **operational control roles** around GTM, compliance, retention, and data reliability. Sources point to AI agent ecosystems expanding most quickly in:
- enterprise operations and leadership support
- customer success and RevOps
- compliance/governance
- data pipeline and remediation workflows  
  — [Source: StackOne](https://www.stackone.com/blog/ai-agent-tools-landscape-2026), [Fullcast](https://www.fullcast.com/content/agentic-ai-platforms), [OpenAI Frontier](https://openai.com/business/frontier), [Salesforce Agentforce](https://www.salesforce.com/agentforce)

### Relevant marketplace options already visible
Marketplace candidates that appear most relevant to current gaps include:

- **Executive Summary Generator** — leadership reporting
- **Client Success Manager** — retention / renewal motion
- **Bookkeeper** — finance operations
- **Recruitment Sourcer** — talent pipeline support
- **AI Data Remediation Engineer** — data quality / remediation
- **Account Strategist** — expansion / post-sale sales motion
- **Accessibility Auditor** — UX / enterprise readiness

Relevant plugins:
- **12-Factor Agents - Security Hardening**
- **hr-legal-compliance**
- **business-analytics**
- **customer-sales-automation**

Given the current roster, the practical opportunity is **not** adding more content or general coding capacity. It is strengthening the operating system around the company.

---

## Recommended Additions

### 1. Executive Summary Generator
- **Category:** Agent
- **Problem solved:** Leadership likely lacks a dedicated agent for turning raw reports, activity, and research into concise decision-ready summaries.
- **Likely team placement:** CTO / Leadership Ops / Chief of Staff function
- **Expected value:** High leverage for founder/exec bandwidth; reduces time spent reading long outputs from ATLAS, QUILL, WebSearch, and ops agents.
- **Urgency:** **High**
- **Overlap / risk:** Some overlap with **SCRIBE** and **ATLAS**, but those generate documents and analytics; this role is more about **decision synthesis**.

### 2. Client Success Manager
- **Category:** Agent
- **Problem solved:** There is support coverage via ECHO, but not a dedicated **post-sale retention / health / renewal** owner.
- **Likely team placement:** Customer Success / Revenue
- **Expected value:** Improves retention, early churn detection, renewal readiness, and structured follow-up on account risk.
- **Urgency:** **High**
- **Overlap / risk:** Partial overlap with **ECHO** and **SCOUT**, but ECHO is reactive support and SCOUT is lead intelligence; this fills the missing **post-sale** layer.

### 3. Bookkeeper
- **Category:** Agent
- **Problem solved:** No dedicated finance operations agent exists for reconciliation, cash visibility, anomaly flagging, and recurring summaries.
- **Likely team placement:** Finance / RevOps / Operations
- **Expected value:** Better operating discipline, cleaner revenue/cost visibility, faster close processes, stronger founder control over financial hygiene.
- **Urgency:** **High**
- **Overlap / risk:** Some overlap with **ATLAS**, but ATLAS is BI/reporting-oriented. Bookkeeper is more transactional and finance-ops focused.

### 4. Recruitment Sourcer
- **Category:** Agent
- **Problem solved:** No dedicated talent acquisition support exists despite likely future hiring needs across engineering, GTM, and operations.
- **Likely team placement:** People / Recruiting
- **Expected value:** Speeds top-of-funnel hiring, reduces manual sourcing load, helps build candidate pipeline before a full HR function exists.
- **Urgency:** **Medium-High**
- **Overlap / risk:** Minimal overlap with current roster. Could share some research patterns with SCOUT/WebSearch, but the workflow and outcome are distinct.

### 5. 12-Factor Agents - Security Hardening
- **Category:** Plugin
- **Problem solved:** The roster currently lacks a clear security/compliance specialist. This plugin directly addresses guardrails, secrets, policy, and security infrastructure patterns.
- **Likely team placement:** Platform Engineering / Security / CTO Office
- **Expected value:** High enterprise-readiness value; supports governance, safer scaling, and customer trust.
- **Urgency:** **Very High**
- **Overlap / risk:** Could introduce process overhead if installed without a clear owner. Best paired with a defined security/compliance charter.

### 6. hr-legal-compliance
- **Category:** Plugin
- **Problem solved:** No dedicated compliance/documentation layer for HR, employment policies, and common frameworks like GDPR/SOC2/HIPAA.
- **Likely team placement:** Operations / People / Security-Compli­ance
- **Expected value:** Strong leverage for policy maturity, audit prep, and customer/vendor diligence responses.
- **Urgency:** **High**
- **Overlap / risk:** Some overlap with SCRIBE/QUILL for drafting, but this is more valuable as a **structured compliance knowledge layer** than a writing tool.

### 7. AI Data Remediation Engineer
- **Category:** Agent or Skill
- **Problem solved:** The company has BI and knowledge curation, but no dedicated owner for **data quality failures, broken pipelines, anomaly correction, and remediation workflows**.
- **Likely team placement:** Data / Platform Engineering
- **Expected value:** Improves trust in analytics, downstream reporting, automations, and RAG inputs.
- **Urgency:** **High**
- **Overlap / risk:** Some overlap with **ORACLE** on knowledge quality, but ORACLE is RAG/document-focused; this is broader operational data engineering.

### 8. Account Strategist
- **Category:** Agent or Skill
- **Problem solved:** SCOUT covers lead intelligence, but there is no clear role for **account planning, expansion mapping, and retention-linked upsell support**.
- **Likely team placement:** Sales / Customer Success
- **Expected value:** Better expansion motion, more structured enterprise account development, stronger net revenue retention support.
- **Urgency:** **Medium-High**
- **Overlap / risk:** Moderate overlap with SCOUT. Needs clean boundary: **SCOUT = pre-sale qualification**, **Account Strategist = post-sale expansion / account planning**.

### 9. Accessibility Auditor
- **Category:** Agent or Skill
- **Problem solved:** No dedicated UX quality/accessibility role exists. This is important for enterprise readiness, usability, and frontend quality.
- **Likely team placement:** Product / Design / Frontend Engineering
- **Expected value:** Better product polish, reduced accessibility debt, stronger procurement readiness for larger customers.
- **Urgency:** **Medium**
- **Overlap / risk:** Some overlap with **QA ENGINEER**, but QA is broad testing; this adds depth in accessibility and UX compliance.

---

## Rationale

The shortlist is designed to fit the **existing roster**, not restart it.

### What the org already has enough of
The company does **not** appear short on:
- content production
- general coding
- workflow design
- research
- reactive support
- monitoring

In fact, those are the strongest parts of the roster.

### What the org is missing
The gaps are mostly in the layers that make an AI company operationally durable:

1. **Leadership support**
   - Founders and execs need synthesized decision support, not just more raw output.
   - That makes **Executive Summary Generator** unusually high leverage.

2. **Customer retention**
   - Support alone does not create retention.
   - A **Client Success Manager** fills the missing post-sale motion.

3. **Finance discipline**
   - BI without bookkeeping leaves a gap between insight and operational truth.
   - A **Bookkeeper** is a practical, non-glamorous, high-value addition.

4. **Security / compliance**
   - This is the clearest enterprise gap in the current org.
   - Security-hardening and compliance plugins are likely more urgent than another creative or engineering agent.

5. **Talent acquisition**
   - Recruiting support becomes necessary before a full people team exists.
   - A sourcer is a lightweight way to add capability.

6. **Data reliability**
   - AI companies degrade quickly if the data layer is noisy.
   - A remediation-focused role supports ATLAS, ORACLE, and all analytics/automation downstream.

7. **Sales enablement beyond lead gen**
   - SCOUT covers top-of-funnel, but expansion and account development appear uncovered.
   - That makes **Account Strategist** a practical add.

8. **UX/accessibility maturity**
   - Accessibility often becomes a blocker later in enterprise sales.
   - Adding that early is a defensive quality move.

### Best sequencing
If only a few additions should be prioritized, the best order is:

1. **12-Factor Agents - Security Hardening**
2. **Client Success Manager**
3. **Bookkeeper**
4. **Executive Summary Generator**
5. **AI Data Remediation Engineer**
6. **Recruitment Sourcer**
7. **Account Strategist**
8. **Accessibility Auditor**

That sequence balances immediate operational risk with growth enablement.

---

## Risks and Caveats

- **Do not add everything at once.** Too many new agents/plugins without routing discipline creates confusion instead of leverage.
- **Overlap risk is real.**
  - **ATLAS vs Bookkeeper**
  - **SCOUT vs Account Strategist**
  - **QA ENGINEER vs Accessibility Auditor**
  - **SCRIBE/QUILL vs Executive Summary Generator / hr-legal-compliance**
  - **ORACLE vs AI Data Remediation Engineer**
- **Clear charters matter more than installation count.** Each new addition should have a crisp ownership boundary.
- **Some needs may be solved faster by plugins or skills than full agents.**
  - Security/compliance is a good example: a plugin may add value faster than a whole new autonomous role.
- **Current connected tool coverage is uneven.** Some future value depends on downstream integrations being available and assigned properly.
- **Product management is still only partially covered.** None of the visible marketplace candidates is a perfect PM owner; Executive Summary Generator helps leadership, but a dedicated PM/roadmap function may still need a future addition if one appears in the marketplace.

### Bottom line
The most practical additions are the ones that strengthen the company’s **operating spine**:
- **security/compliance**
- **customer success**
- **finance ops**
- **leadership synthesis**
- **data reliability**
- **recruiting**
- **account expansion**
- **UX/accessibility**

Those are the gaps least covered by the current roster and most likely to compound value quickly.

---

# Synthesize audit, market scan, and org design into a detailed planning brief

Here is the unified, detailed planning brief synthesizing the audit, marketplace gap research, and proposed org structure for Automatos AI:

---

# Automatos AI - Planning Brief for Organizational Design and Capability Augmentation

**Date:** 2026-03-27  
**Prepared by:** SCRIBE (Document Generation Specialist)  
**Based on inputs from:** ATLAS (BI Agent), Audit Reports, Marketplace Research, Org Design Proposal

---

## Executive Summary

Automatos AI currently operates with a strong foundational roster of 18 active agents spanning engineering, operations, content, research, growth, and support functions. The existing structure effectively covers core technical execution (coding, bug fixing, QA), content production (writing, editing, visual assets), research, platform monitoring, and customer support. However, strategic gaps remain in areas critical to scaling and enterprise readiness, including security and compliance, dedicated leadership and product support, customer success and retention, finance and revenue operations, recruiting, data engineering, sales enablement, and design/UX governance.

This planning brief synthesizes a recommended organizational structure that preserves all existing agents, groups them into clear functional teams with defined reporting lines under the CTO, and proposes a prioritized shortlist of new agent additions and plugins from the marketplace. These additions emphasize strengthening operational control layers and governance without duplicating existing strengths.

---

## Current-State Roster Review

### Agent Inventory Summary

- **Engineering:** PATCHER (bug fixing), QA ENGINEER (automated testing), CODER (full-stack development), FORGE (workflow architecture)  
  *Strengths:* High-quality coding and testing coverage, integrated Jira/GitHub workflows.  
  *Gaps:* No feature development scope for PATCHER; no deployment automation; CODER and JIRA ADMIN lack custom system prompts.

- **Operations:** JIRA ADMIN (project ops), SENTINEL (infrastructure monitoring), ATLAS (business intelligence)  
  *Strengths:* Strong operational monitoring and BI reporting.  
  *Gaps:* Limited cross-tool project management; no sprint or roadmap planning capabilities.

- **Research/Knowledge:** ORACLE (knowledge curation), WebSearch (real-time web research)  
  *Strengths:* Broad domain research with premium GPT-5.4 model; knowledge base health.  
  *Gaps:* WebSearch’s high creativity temperature may reduce consistency; no result caching.

- **Marketing/Content:** HARPER (social content), QUILL (long-form writing), EDITOR (blog editing), CANVAS (visual assets)  
  *Strengths:* Comprehensive content pipeline across formats and channels.  
  *Gaps:* CANVAS lacks tools for asset management and video editing; no design system enforcement.

- **Growth/Sales:** SCOUT (lead intelligence), RALLY (community growth)  
  *Strengths:* Lead qualification and community engagement.  
  *Gaps:* SCOUT has no assigned tools; no account expansion or post-sale sales motion.

- **Support:** ECHO (customer chatbot), COMMS (internal comms)  
  *Strengths:* Multi-channel communication coverage; proactive internal comms.  
  *Gaps:* COMMS lacks CRM and external customer communication; ECHO is reactive only, no post-sale retention ownership.

- **Document Generation:** SCRIBE (document/report generation)  
  *Strengths:* Multi-format polished outputs; highest tool count; cross-functional service.  
  *Gaps:* Low utilization; no template or presentation support; partial overlap with QUILL on content but distinct focus.

### Usage and Cost Insights

- Top cost agents: QUILL, WebSearch, HARPER, ATLAS  
- High volume usage agents: SENTINEL, QUILL, COMMS  
- Several agents operate on zero-cost/free-tier models (COMMS, SENTINEL) keeping overhead low.

---

## Proposed Organizational Structure

```
CTO Auto
├── Engineering Manager
│   ├── PATCHER (Bug Fix Engineer)
│   ├── QA ENGINEER (QA Engineer)
│   ├── CODER (Full-Stack Developer)
│   └── FORGE (Workflow Architect)
├── Operations Manager
│   ├── JIRA ADMIN (Project Operations)
│   ├── SENTINEL (Infrastructure Watchdog)
│   └── ATLAS (Business Intelligence)
├── Research/Knowledge Lead
│   ├── ORACLE (Knowledge Curator)
│   └── WebSearch (Research Specialist)
├── Marketing/Content Lead
│   ├── HARPER (Content Engine)
│   ├── QUILL (Long-Form Content Writer)
│   ├── EDITOR (Blog Editor)
│   └── CANVAS (Visual Content Specialist)
├── Growth/Sales Lead
│   ├── SCOUT (Lead Intelligence)
│   └── RALLY (Community Growth)
├── Support Lead
│   ├── ECHO (Customer Support Chatbot)
│   └── COMMS (Communications Operations)
└── SCRIBE (Document Generation Specialist) — reports directly to CTO Auto
```

### Reporting Lines & Rationale

- CTO Auto holds overall strategic and operational oversight.
- Functional teams organized by primary roles: Engineering, Operations, Research/Knowledge, Marketing/Content, Growth/Sales, Support.
- SCRIBE as a cross-functional specialist reports directly to CTO for broad organizational utility.
- Reporting lines minimize bottlenecks and encourage cross-team collaboration, especially between Research and Marketing for content generation.
- The structure supports scalability by allowing expansion within smaller teams like Growth/Sales and Marketing.

---

## Team-by-Team Assignments Summary

- **Engineering:** Bug fixes, testing, coding, workflow design  
- **Operations:** Project ops, infrastructure monitoring, BI analytics  
- **Research/Knowledge:** Knowledge curation, web research  
- **Marketing/Content:** Social and long-form content creation, editorial review, visual asset production  
- **Growth/Sales:** Lead qualification, community engagement  
- **Support:** Customer chatbot, internal/external communications  
- **Document Generation:** Polished report and document production

---

## Marketplace Addition Recommendations

To address key operational and strategic gaps, the following agents and plugins are recommended for prioritized addition:

1. **Executive Summary Generator**  
   - Role: Leadership reporting synthesis  
   - Team: CTO / Leadership Ops / Chief of Staff  
   - Value: High leverage for exec bandwidth; distills reports and data into decision-ready briefings  
   - Urgency: High

2. **Client Success Manager**  
   - Role: Post-sale retention, renewal, and account health  
   - Team: Customer Success / Revenue  
   - Value: Improves churn management and renewal readiness  
   - Urgency: High

3. **Bookkeeper**  
   - Role: Finance operations, reconciliation, cash visibility  
   - Team: Finance / RevOps / Operations  
   - Value: Enhances financial discipline and founder control  
   - Urgency: High

4. **Recruitment Sourcer**  
   - Role: Talent pipeline support and candidate sourcing  
   - Team: People / Recruiting  
   - Value: Accelerates hiring and reduces manual sourcing burden  
   - Urgency: Medium-High

5. **12-Factor Agents - Security Hardening Plugin**  
   - Role: Security guardrails, policy enforcement, secrets management  
   - Team: Platform Engineering / Security / CTO Office  
   - Value: Enterprise readiness and governance  
   - Urgency: Very High

6. **hr-legal-compliance Plugin**  
   - Role: HR compliance documentation, policy frameworks (GDPR, SOC2, HIPAA)  
   - Team: Operations / People / Security-Compliance  
   - Value: Audit readiness and policy maturity  
   - Urgency: High

7. **AI Data Remediation Engineer**  
   - Role: Data quality, anomaly correction, pipeline remediation  
   - Team: Data / Platform Engineering  
   - Value: Improves trust in analytics and automation  
   - Urgency: High

8. **Account Strategist**  
   - Role: Account planning, expansion, upsell support  
   - Team: Sales / Customer Success  
   - Value: Strengthens net revenue retention and expansion motion  
   - Urgency: Medium-High

9. **Accessibility Auditor**  
   - Role: UX quality, accessibility compliance  
   - Team: Product / Design / Frontend Engineering  
   - Value: Enhances enterprise readiness and product polish  
   - Urgency: Medium

---

## Principles for Roster Preservation and Growth

- **Preserve all existing agents**: No deletions or replacements; all current capabilities remain intact and assigned clear roles.  
- **Add new capabilities only**: Focus on filling gaps in operational control, compliance, leadership support, and customer success.  
- **Avoid content or coding redundancy**: The current roster is strong in content creation, coding, and research; additions should complement rather than duplicate.  
- **Ensure clear ownership**: New agents and plugins should have well-defined team placements and reporting lines to avoid overlap.  
- **Prioritize enterprise readiness**: Security, compliance, data integrity, and retention are key for scaling and customer trust.  
- **Support scalability and flexibility**: The proposed org structure and additions allow for team expansion and evolving business needs.

---

This brief provides a unified, detailed view of Automatos AI's current agent capabilities, organizational design, and prioritized marketplace augmentations to support sustained growth, operational excellence, and enterprise readiness.

Please review and advise on next steps.

---

---

# Create operational playbook ideas and phased no-change build plan

## Playbook Ideas

Here is a practical set of playbook ideas (as Automatos recipes) that can be implemented using the current agent roster to improve operational efficiency and establish clear, automated workflows.

### 1. Engineering & QA

*   **Playbook: Automated Bug Triage and Patching**
    *   **Trigger:** `event` (New bug reported in Jira or via a dedicated channel monitored by ECHO).
    *   **Goal:** Triage, assign, fix, and verify bug reports with minimal human intervention.
    *   **Steps:**
        1.  **Triage Report** - **Agent: JIRA ADMIN**
            *   **Prompt:** "A new bug report has been received. Analyze the report content, add the 'Needs-Triage' label, and extract key details: summary, steps to reproduce, and severity."
            *   **Input:** `trigger.event_data`
            *   **Output:** `bug_summary`, `repro_steps`, `severity_guess`
        2.  **Attempt Reproduction** - **Agent: QA ENGINEER**
            *   **Prompt:** "Attempt to reproduce the bug using these steps: `{{repro_steps}}`. Report back with success or failure and any relevant logs."
            *   **Input:** `repro_steps`
            *   **Output:** `reproduction_confirmed` (boolean), `logs`
        3.  **Assign & Schedule Fix** - **Agent: JIRA ADMIN**
            *   **Prompt:** "If `reproduction_confirmed` is true, assign the Jira ticket to PATCHER. If false, label the ticket 'Cannot-Reproduce' and add a comment with the `{{logs}}`."
            *   **Input:** `reproduction_confirmed`, `logs`
            *   **Output:** `ticket_id`
        4.  **Develop Patch** - **Agent: PATCHER**
            *   **Prompt:** "Develop a code patch for the bug described in ticket `{{ticket_id}}`. Create a pull request when complete."
            *   **Input:** `ticket_id`
            *   **Output:** `pull_request_url`
        5.  **Verify Fix** - **Agent: QA ENGINEER**
            *   **Prompt:** "The bug in ticket `{{ticket_id}}` has a proposed fix at `{{pull_request_url}}`. Pull the branch and run automated tests to verify the fix and check for regressions. Merge if all tests pass."
            *   **Input:** `ticket_id`, `pull_request_url`
            *   **Output:** `merge_status`

### 2. Content & Marketing

*   **Playbook: End-to-End Content Pipeline**
    *   **Trigger:** `manual` (With a topic brief).
    *   **Goal:** Streamline the creation of a blog post from research to publication and promotion.
    *   **Steps:**
        1.  **Initial Research** - **Agent: WebSearch**
            *   **Prompt:** "Conduct initial research on the topic: '`{{topic_brief}}`'. Gather key statistics, expert opinions, and competing articles."
            *   **Input:** `topic_brief`
            *   **Output:** `research_summary`
        2.  **Write Draft** - **Agent: QUILL**
            *   **Prompt:** "Write a long-form blog post on '`{{topic_brief}}`' using the following research: `{{research_summary}}`. Focus on a clear narrative and actionable advice."
            *   **Input:** `topic_brief`, `research_summary`
            *   **Output:** `draft_content`
        3.  **Create Visuals** - **Agent: CANVAS**
            *   **Prompt:** "Create a header image and one in-article diagram for a blog post titled '`{{topic_brief}}`'. The content is about `{{draft_content[:200]}}...`"
            *   **Input:** `topic_brief`, `draft_content`
            *   **Output:** `image_urls`
        4.  **Edit & Finalize** - **Agent: EDITOR**
            *   **Prompt:** "Review and edit this draft: `{{draft_content}}`. Check for grammar, style, and clarity. Integrate the images: `{{image_urls}}`. Prepare the final markdown for publishing."
            *   **Input:** `draft_content`, `image_urls`
            *   **Output:** `final_markdown`
        5.  **Publish & Promote** - **Agent: HARPER**
            *   **Prompt:** "The blog post is live. Write three social media posts (Twitter, LinkedIn) to promote it, linking to the new article. Tag relevant influencers from the research."
            *   **Input:** `research_summary` (for influencers)
            *   **Output:** `social_post_ids`

### 3. Operations & Reporting

*   **Playbook: Weekly Business Intelligence Report**
    *   **Trigger:** `cron` (`0 9 * * 1` - Every Monday at 9 AM).
    *   **Goal:** Automatically generate and distribute a weekly performance report.
    *   **Steps:**
        1.  **Gather Metrics** - **Agent: ATLAS**
            *   **Prompt:** "Generate the standard weekly business intelligence report. Query for: new users, agent usage breakdown, recipe run successes vs. failures, and top 5 most active agents for the last 7 days."
            *   **Input:** (none)
            *   **Output:** `bi_data` (structured JSON)
        2.  **Generate Document** - **Agent: SCRIBE**
            *   **Prompt:** "Create a polished PDF report titled 'Weekly Performance Review' using the following data: `{{bi_data}}`. Use the 'Weekly Report' template."
            *   **Input:** `bi_data`
            *   **Output:** `report_url`
        3.  **Distribute Report** - **Agent: COMMS**
            *   **Prompt:** "Distribute the weekly report. Send an email to 'leadership@automatos.ai' with the subject 'Weekly Performance Report is Ready' and a link to `{{report_url}}`."
            *   **Input:** `report_url`
            *   **Output:** `distribution_confirmation`

### 4. Growth & Support

*   **Playbook: New Lead Qualification & Outreach**
    *   **Trigger:** `manual` (With a lead's email or LinkedIn profile).
    *   **Goal:** Enrich a new lead and prepare a personalized outreach draft.
    *   **Steps:**
        1.  **Enrich Lead** - **Agent: SCOUT**
            *   **Prompt:** "Find information on the company and role for this lead: `{{lead_profile}}`. Identify company size, industry, and recent news."
            *   **Input:** `lead_profile`
            *   **Output:** `lead_enrichment_data`
        2.  **Draft Outreach** - **Agent: COMMS**
            *   **Prompt:** "Draft a personalized, cold outreach email to the lead based on this data: `{{lead_enrichment_data}}`. The goal is to book a 15-minute discovery call. Do not send it; save it as a draft."
            *   **Input:** `lead_enrichment_data`
            *   **Output:** `draft_email_content`

## Phased Roadmap

This roadmap outlines a phased approach to implementation, starting with review and approval, and progressively building capabilities without disrupting current operations.

### Phase 0: Review and Approval (Current State)
This phase is dedicated to planning and governance. No technical changes will be made.
*   **Activities:**
    *   Review this playbook and roadmap document with all stakeholders, including the CTO and functional leads.
    *   Gather feedback on playbook designs and priorities.
    *   Secure formal approval to proceed to Phase 1.
    *   Finalize the budget for any marketplace plugins or agent model upgrades identified as high-priority.
*   **Exit Criteria:** Written approval from CTO Auto to begin Phase 1.

### Phase 1: High-Priority Additions & Foundational Workflows
This phase focuses on addressing the most critical gaps and implementing high-value, low-risk automations.
*   **Activities:**
    *   **Capability Hardening:**
        *   Assign the **12-Factor Agents - Security Hardening Plugin** to relevant engineering and operations agents (PATCHER, CODER, SENTINEL).
        *   Assign the **hr-legal-compliance Plugin** to COMMS and SCRIBE to assist with policy and document generation.
        *   Assign necessary tools (e.g., a Sales Navigator or CRM tool via Composio) to **SCOUT** to enable its core function.
    *   **Playbook Implementation:**
        *   Build and test the **"Automated Bug Triage and Patching"** playbook to improve engineering velocity.
        *   Build and test the **"Weekly Business Intelligence Report"** playbook to establish a reliable reporting cadence.
*   **Exit Criteria:** Successful, documented execution of both playbooks for two consecutive weeks. Security plugins confirmed as active and configured.

### Phase 2: Workflow Refinement & Expansion
This phase builds on the stable foundation of Phase 1 to expand automation into more complex, revenue-impacting areas.
*   **Activities:**
    *   **New Agent Introduction (Post-Approval):**
        *   Introduce the **Client Success Manager** and **Account Strategist** agents to address post-sale and account expansion gaps.
    *   **Playbook Implementation:**
        *   Build and test the **"End-to-End Content Pipeline"** to accelerate marketing efforts.
        *   Build and test the **"New Lead Qualification & Outreach"** playbook to empower the Growth team.
        *   Design and implement a **"Support Escalation"** playbook, integrating ECHO with JIRA ADMIN and COMMS.
    *   **Refinement:** Analyze performance data from Phase 1 playbooks and make adjustments to prompts, agent assignments, or step logic.
*   **Exit Criteria:** Content and Lead playbooks are fully operational. New agents demonstrate value in pilot runs.

### Phase 3: Scaling and Proactive Automation
This phase focuses on optimizing and scaling the AI workforce, moving from reactive workflows to proactive, intelligent operations.
*   **Activities:**
    *   **New Agent Introduction (Post-Approval):**
        *   Introduce the **Recruitment Sourcer** to support team growth.
        *   Introduce the **AI Data Remediation Engineer** to improve data quality proactively.
    *   **Proactive Playbooks:**
        *   Upgrade the Bug Triage playbook to be triggered proactively by **SENTINEL** based on anomaly detection, not just manual reports.
        *   Create a **"Knowledge Maintenance"** playbook where **ORACLE** periodically scans for outdated documentation and suggests updates.
    *   **Scaling:** Increase the autonomy of agents in existing playbooks, reducing the need for human approval on routine tasks (e.g., auto-publishing social media posts for certain content types).
*   **Exit Criteria:** At least one major workflow is now fully proactive. New scaling-focused agents are integrated and handling production tasks.

## Approval Gates

Progression between phases will be governed by explicit approval gates to ensure control, alignment, and measurable progress.

1.  **Gate (Phase 0 -> 1):**
    *   **Requirement:** Formal sign-off on this entire plan from CTO Auto.
    *   **Evidence:** A documented approval (e.g., a signed-off task in Jira, an email confirmation).

2.  **Gate (Phase 1 -> 2):**
    *   **Requirement:** Performance review of Phase 1 playbooks and a go/no-go decision from the Engineering and Operations Managers.
    *   **Evidence:** A dashboard from ATLAS showing at least a 15% reduction in bug triage time and 99% reliability in weekly report generation.

3.  **Gate (Phase 2 -> 3):**
    *   **Requirement:** A business case review demonstrating the ROI of the Content and Growth playbooks, approved by the Marketing and Sales Leads.
    *   **Evidence:** A report from SCRIBE showing metrics like "time-to-publish" for content and "leads qualified per week" that meet or exceed targets.

## Guardrails

To ensure this initiative remains a controlled, non-disruptive planning and execution exercise, the following guardrails will be strictly enforced:

*   **Planning, Not Execution:** This document and all initial activities are for planning purposes only. No recipes will be built or agents modified until Phase 1 is formally approved.
*   **Preservation of Existing Roster:** All 18 existing agents will be preserved. No agents will be deleted. New agents will only be added after passing the specified approval gates for Phases 2 and 3.
*   **Configuration Before Creation:** We will always prioritize modifying an existing agent's configuration (e.g., updating a system prompt, assigning a new tool) over creating a new agent to fill a capability gap.
*   **Incremental Rollout:** Playbooks will be built, tested, and deployed one at a time. We will not attempt a "big bang" release of all proposed automations simultaneously.
*   **Human-in-the-Loop by Default:** All new playbooks, especially those involving external communication or system changes, will initially be designed with a final human approval step. The level of autonomy will only be increased after a period of demonstrated reliability.