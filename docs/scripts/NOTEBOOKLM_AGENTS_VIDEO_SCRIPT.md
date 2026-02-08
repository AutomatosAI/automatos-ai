# Automatos AI — Your Complete Digital Workforce

## Video Script for NotebookLM Source Document

---

### Introduction: The Problem

Every business runs on repetitive work. Emails pile up. Invoices arrive and sit in an inbox. Calendar invites get missed. Customer tickets go stale. You hire people to handle this — assistants, analysts, account managers — but there's never enough hours in the day.

What if you could hire a digital employee that works 24/7, never takes a sick day, and costs a fraction of what a human hire does?

That's what an Automatos AI Agent is.

---

### What Is an Agent?

An agent is a digital worker. Not a chatbot that gives generic answers — an actual assistant that does real work on your behalf.

Think of it like hiring a new employee. You give them a role, access to the tools they need, and train them on how your business works. Then you set them loose.

Here are some examples of what agents can do:

- **Email Assistant**: Reads your inbox every morning, summarises what's important, drafts replies in your voice, and flags anything urgent
- **Invoice Processor**: Watches for incoming invoices, extracts the data, matches it against purchase orders, and updates your accounting system
- **Sales Assistant**: Monitors new leads, researches the company, drafts a personalised outreach email, and books a follow-up in your calendar
- **Support Triage Agent**: Picks up new customer tickets from Jira or Zendesk, categorises them by urgency, and routes them to the right team
- **Meeting Prep Agent**: Before every calendar event, pulls relevant documents, summarises recent conversations with that contact, and prepares talking points

These aren't hypothetical. You can build every one of these in Automatos in under 10 minutes.

---

### Building an Agent: 5 Simple Steps

Creating an agent in Automatos follows a guided wizard. No code required. Just five decisions.

#### Step 1: Name and Category

Give your agent a name and pick a category. We have 15 categories — Communication, Analytics, Business, Development, Sales, Support, and more. This helps organise your team of agents as it grows.

#### Step 2: Choose a Persona

This is where you give your agent a personality. A persona defines their role and how they communicate.

Are they a formal executive assistant? A friendly customer support rep? A no-nonsense data analyst?

You can pick from our library of pre-built personas — we've crafted these for common roles so you don't have to think about prompt engineering. Or, if you want full control, write your own custom persona. You can even start from a pre-built one and tweak it.

The persona is what makes your agent feel like part of your team, not a generic AI.

#### Step 3: Pick the Right Brain (LLM Model)

This is one of the most powerful features of Automatos — every agent gets its own brain. You're not locked into one AI model for everything.

Why does this matter? Because different models are better at different tasks, and they have very different costs.

Here's how to think about it:

- **Claude by Anthropic**: Exceptional at coding, complex analysis, and following detailed instructions. If your agent needs to process structured data, write technical content, or handle nuanced tasks — Claude is your pick.
- **GPT-4o or GPT-5 by OpenAI**: Strong all-rounders. Great for general reasoning, research, creative writing, and conversational tasks. Perfect for customer-facing agents.
- **Open-source models like Llama or Mistral**: Extremely cost-effective. If your agent just needs to read emails and summarise them, or classify incoming tickets — you don't need the most expensive brain in the room. An open-source model handles this brilliantly at a fraction of the cost.

The key insight: match the brain to the job. Your email summariser doesn't need the same horsepower as your data analyst. Automatos lets you optimise for both quality and cost, per agent.

You also set parameters here like temperature (how creative vs. consistent the agent should be) and you can configure a fallback model — if your primary model is unavailable, the agent automatically switches to a backup so work never stops.

#### Step 4: Connect Their Tools

Now give your agent access to the apps it needs to do its job. Just like giving a new employee logins to company software.

Automatos integrates with over 850 apps and services. Gmail, Outlook, Slack, Microsoft Teams, Google Drive, Dropbox, Jira, Linear, Salesforce, HubSpot, Notion, GitHub — the list goes on.

But here's the important part: you only assign the specific tools each agent needs. Why?

Imagine walking into a hardware store with 850 tools on the wall and being told "use whatever you need." You'd be paralysed. But if someone hands you a hammer, a drill, and a tape measure and says "build this shelf" — you get straight to work.

That's exactly how it works for agents. Your email assistant gets Gmail and Google Calendar. Your support agent gets Jira and Slack. Your sales agent gets HubSpot and LinkedIn. They know exactly which tools to reach for, and they act fast.

Connecting tools is simple — just click to authorise via OAuth (the same "Sign in with Google" flow you're already used to).

#### Step 5: Add Capabilities

This is the final step, and it's where agents go from generic to genuinely yours.

Capabilities are like training courses for your agent. They teach your agent specific skills and knowledge that make them experts at their particular job.

Some examples:

- **Company Writing Style**: Upload your brand guidelines and tone of voice. Now every email, report, and message your agent writes sounds like it came from your team.
- **Email Response Patterns**: Show your agent how you typically reply to different types of emails. It learns your patterns and drafts responses the way you would.
- **Sales Playbook**: Feed it your pitch deck, objection handling guide, and case studies. Your sales agent now sells like your best rep.
- **Code Standards**: If you have development agents, give them your coding standards, architecture patterns, and review checklist.
- **Domain Knowledge**: Industry-specific knowledge, compliance requirements, internal processes — anything your agent needs to know to do its job properly.

Capabilities are what separate a generic AI from an agent that truly works for your business. They're the institutional knowledge that makes your team effective — now packaged up and given to your digital workforce.

You can browse and install capabilities from the Automatos Marketplace, or build your own.

---

### Using Your Agents: Three Ways to Work

Once your agent is built, there are three ways to put it to work.

#### 1. Chat With Them

The simplest way. Open the Automatos chat interface, pick an agent, and just talk to them.

"Hey, summarise my emails from this morning and add any meetings to my calendar."

"Draft a follow-up email to the client we met yesterday. Keep it warm but professional."

"Review this document and highlight any compliance issues."

It's like messaging a colleague on Slack — except this colleague has read every document, remembers every conversation, and responds in seconds.

You can also chat with your agents through other channels — Slack, WhatsApp, Microsoft Teams. Meet them where you already work.

#### 2. Schedule Recurring Tasks

Some work needs to happen every day, every week, or every month. Set it up once and forget about it.

Through our recipe system, you can schedule agents to run on a timetable:

- Every morning at 8am: Email agent summarises your inbox and sends you a brief
- Every Friday at 5pm: Analytics agent compiles your weekly performance report
- First of every month: Finance agent reconciles invoices against payments

Your agents work while you sleep, while you're in meetings, while you're on holiday. The work just gets done.

#### 3. Set Up Triggers

This is where agents become truly autonomous. Instead of you telling them what to do, they respond to events as they happen.

- **Invoice arrives** in your email → Finance agent extracts the data and logs it
- **New Jira ticket created** → Support agent triages it, adds context, assigns priority
- **Customer order placed** → Onboarding agent sends a welcome sequence
- **GitHub pull request opened** → Code review agent analyses the changes
- **Calendar event in 30 minutes** → Prep agent sends you a briefing document

Triggers turn your agents from assistants into autonomous workers. They watch, they react, they act — without you lifting a finger.

---

### Recipes: Multiple Agents, One Workflow

One agent is powerful. Multiple agents working together? That's a digital workforce.

Recipes let you chain agents together into multi-step workflows. Each agent does their part and passes the results to the next — like an assembly line where every station has a specialist.

Example — a **Weekly Client Report** recipe with four steps: a Data Agent pulls numbers from your connected tools, an Analysis Agent spots trends and flags anomalies, a Writing Agent formats it into a branded report, and a Delivery Agent emails it to clients and posts it to Slack. Four agents, one seamless pipeline.

Agents can run sequentially (relay race — one finishes, next starts) or in parallel (two agents work different parts of the job at the same time, a third combines the results). For a lead qualification recipe, one agent researches the company on LinkedIn while another pulls CRM history — simultaneously. A third agent merges both and drafts personalised outreach.

Every step has error handling: stop the recipe, skip and continue, or retry with backoff. Recipes run on cron schedules or fire from webhook triggers. Set it and forget it.

---

### Knowledge Bases: Give Your Agents a Memory

Your agents are smart, but they don't know your business. Knowledge bases fix that.

Upload documents — PDF, Word, Markdown, plain text — and Automatos reads them, breaks them into intelligent chunks, and builds a searchable index. When an agent needs information, it finds the exact paragraph, not just the right file. This is RAG (Retrieval-Augmented Generation) — your agent grounds its answers in your actual business data.

Even better: connect your cloud storage. Link Google Drive, Dropbox, OneDrive, or Box through secure OAuth, and Automatos syncs your files automatically. Documents added or updated in the cloud are processed and indexed without you lifting a finger.

This is where it gets powerful. Tell your agent: "Find last quarter's revenue from the board report and send it to Sarah in finance." It searches your Drive, finds the report, extracts the numbers, and emails Sarah. Or: "Pull up everything we have on the Henderson account" — it searches across emails, proposals, meeting notes, and contracts and gives you a full briefing in seconds.

Without a knowledge base, your agent says "I don't have that information." With one, it knows your business as well as your longest-serving employee. Every workspace gets its own isolated knowledge base — your documents are never shared with other users.

---

### The Automatos Marketplace

If all this sounds like a lot — which agent, which model, which tools, how to train it — we've got you covered.

The Automatos Marketplace has five sections:

1. **Applications** — 850+ app integrations (Gmail, Slack, Salesforce, Jira, HubSpot, Notion, GitHub, Stripe, Shopify and hundreds more). Browse and connect in one click.
2. **Agents** — Pre-built by the Automatos team or experienced community members. Each comes with a persona, recommended model, and tools already configured. Install with one click, customise to fit.
3. **Recipes** — Ready-made multi-agent workflows. Install a recipe and it sets up the full pipeline — agents, tools, logic, scheduling, everything.
4. **LLMs** — Browse and compare all available AI models side by side. Strengths, pricing, speed, ideal use cases.
5. **Capabilities** — Specialist skills like "Professional Email Writing" or "Financial Report Analysis." Install and assign to any agent — instant expertise.

Community members can submit their own agents, recipes, and capabilities. Earn trusted creator status after five approved submissions and publish instantly. All capabilities go through security scanning before they're available.

Don't know where to start? Browse what others have built, install what fits, customise for your business. Up and running in minutes.

---

### Bringing It All Together

Say you're running a growing e-commerce business. Here's what you set up in an afternoon:

1. **Morning Briefing Agent** — scans email, Slack, and Shopify. Daily summary at 7:30am.
2. **Customer Support Recipe** — three agents: triage tickets, search your knowledge base to draft a response, send the reply and update the ticket. Runs every time a new ticket comes in.
3. **Invoice Agent** — watches Gmail for invoices, extracts data, matches to purchase orders, flags anything unusual.
4. **Weekly Report Recipe** — Friday at 5pm: data agent pulls numbers, analysis agent spots trends, writing agent formats the report, delivery agent emails it to leadership and posts to Slack.
5. **Sales Follow-Up Agent** — new lead fills out a form, agent researches the company, pulls case studies from your knowledge base, drafts outreach, books a calendar reminder.

A mix of solo agents and multi-agent recipes. Powered by your documents and cloud storage. Half of them installed from the marketplace and customised in five minutes.

Build your agents. Teach them with your documents. Chain them together with recipes. When you don't know where to start, the marketplace has you covered. Your digital workforce is ready to hire.

## Automatos Brand Reference

**Brand**: Automatos AI
**Tagline**: Your Digital Workforce
**Primary Color**: Orange (#FF6B2C / HSL 16 100% 60%)
**Background**: Dark theme (#0F0F0F)
**Visual Style**: Modern, clean, glass morphism effects with orange accents. Professional but approachable. Dark backgrounds with subtle glass-panel effects.
**Tone**: Confident, clear, jargon-free. We explain complex AI concepts in everyday language. Our audience is 70-80% non-technical business users who want results, not theory.
**Logo**: Automatos mark — a stylised "A" in brand orange.
**Platform Pillars**: Agents (build & train), Knowledge Bases (your data), Recipes (multi-agent workflows), Marketplace (community resources), Analytics (monitor & optimise).
