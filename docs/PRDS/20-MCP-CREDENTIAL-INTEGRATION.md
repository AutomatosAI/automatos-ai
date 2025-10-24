# PRD-20: MCP Server Library & Credential Integration - The 400+ Integration Explosion

**Status**: Active Development  
**Priority**: P0 - CRITICAL Platform Differentiator  
**Effort**: Phase 1: 8-12 hours | Phase 2: 20-30 hours  
**Dependencies**: PRD-17 (Dynamic Tool Assignment), PRD-18 (Credential Management)

---

## Executive Summary

Transform Automatos AI from a platform with a few tools to the **most comprehensive AI orchestration platform** with 400+ pre-integrated MCP servers, each perfectly linked to credentials. This creates a Netflix-style marketplace where users enable integrations by simply adding credentials.

### The Vision: "One Credential Away from Any Integration"

> **User adds AWS credentials** → 15 AWS MCP servers instantly available  
> **User adds GitHub token** → 8 GitHub MCP servers enabled  
> **User adds Slack token** → 6 Slack MCP servers ready  
> **Total: 400+ integrations, zero configuration**

### Current State → Target State

| Aspect | Current | After Phase 1 | After Phase 2 |
|--------|---------|---------------|---------------|
| **Credential Types** | 30 | 30 | 400+ |
| **MCP Servers** | 8 manual | 30 pre-loaded | 400+ pre-loaded |
| **Auto-Activation** | Manual | ✅ Working | ✅ Working |
| **UI Pagination** | No | ✅ Yes | ✅ Yes |
| **Credential Linking** | Partial | ✅ Complete | ✅ Complete |
| **Search/Filter** | Basic | ✅ Enhanced | ✅ Enhanced |

---

## Part 1: Two-Phase Strategy

### Phase 1: Proof of Concept (30 Servers) - THIS WEEK ✅

**Goal**: Prove the system works end-to-end with 30 MCP servers

**Tasks**:
1. ✅ Create 30 MCP server metadata (matching 30 credential types)
2. ✅ Implement credential-based auto-activation
3. ✅ Add pagination to Tools & Settings pages
4. ✅ Test complete flow: Add credential → Enable MCP server → Assign to agent → Execute
5. ✅ Verify performance with 30 items
6. ✅ Fix any bugs or UX issues

**Success Criteria**:
- [ ] All 30 MCP servers load in UI with pagination
- [ ] Adding AWS credential auto-enables AWS MCP servers
- [ ] Agent can execute AWS tools successfully
- [ ] UI remains performant with 30 items
- [ ] Pagination works smoothly
- [ ] Search and filters work correctly

**Deliverables**:
- `mcp_servers_library_30.json` - 30 MCP server definitions
- `load_mcp_servers.py` - Script to bulk-load MCP servers
- Pagination for Tools page
- Pagination for Credentials page
- Enhanced search/filter UI
- Complete test results

**Timeline**: 8-12 hours (1-2 days)

---

### Phase 2: Full Integration (400+ Servers) - NEXT WEEK 🚀

**Goal**: Clone ALL 400+ credential types and MCP servers from n8n

**Tasks**:
1. ⏳ Scrape/clone n8n GitHub credentials directory
2. ⏳ Convert all 400+ credential types to our format
3. ⏳ Generate 400+ matching MCP server definitions
4. ⏳ Bulk load into database
5. ⏳ Stress test pagination with 400+ items
6. ⏳ Optimize database queries for scale
7. ⏳ Add advanced filtering (by category, provider, tags)
8. ⏳ Add bulk operations (enable/disable multiple)

**Success Criteria**:
- [ ] 400+ credential types in database
- [ ] 400+ MCP servers pre-loaded
- [ ] Pagination handles 400+ items smoothly
- [ ] Search returns results in <500ms
- [ ] UI loads pages of 20-50 items instantly
- [ ] Filters reduce results effectively
- [ ] Zero performance degradation

**Deliverables**:
- `all_credential_types_400.py` - Complete n8n credential types
- `mcp_servers_library_400.json` - 400+ MCP servers
- Advanced pagination component
- Enhanced search/filter system
- Performance optimization report
- Comprehensive documentation

**Timeline**: 20-30 hours (3-4 days)

---

## Part 2: Phase 1 Implementation Details (30 Servers)

### 2.1 The 30 MCP Server Library

**Matching Our 30 Credential Types**:

```json
[
  // AI & ML (3 servers)
  {
    "name": "OpenAI Assistant MCP",
    "description": "Access OpenAI's Assistant API, DALL-E, and GPT models",
    "provider": "OpenAI",
    "category": "ai",
    "icon": "🧠",
    "version": "1.0.0",
    "status": "inactive",
    "mcp_server_url": null,
    "capabilities": {
      "methods": ["chat.completions", "images.generate", "assistants.create", "embeddings.create"],
      "features": ["function_calling", "vision", "embeddings", "image_generation"]
    },
    "credentials_schema": {
      "required": ["api_key"],
      "credential_type": "openai_api"
    },
    "tags": ["ai", "llm", "image-generation", "embeddings"],
    "metadata": {
      "documentation": "https://platform.openai.com/docs",
      "auto_enable_on_credential": "openai_api",
      "usable_as_tool": true
    }
  },
  {
    "name": "Anthropic Claude MCP",
    "description": "Access Claude models for advanced reasoning and analysis",
    "provider": "Anthropic",
    "category": "ai",
    "icon": "🤖",
    "version": "1.0.0",
    "status": "inactive",
    "capabilities": {
      "methods": ["messages.create", "messages.stream"],
      "features": ["long_context", "reasoning", "analysis"]
    },
    "credentials_schema": {
      "required": ["api_key"],
      "credential_type": "anthropic_api"
    },
    "tags": ["ai", "llm", "reasoning", "long-context"],
    "metadata": {
      "documentation": "https://docs.anthropic.com",
      "auto_enable_on_credential": "anthropic_api"
    }
  },
  {
    "name": "HuggingFace Models MCP",
    "description": "Access HuggingFace model hub and inference API",
    "provider": "HuggingFace",
    "category": "ai",
    "icon": "🤗",
    "version": "1.0.0",
    "status": "inactive",
    "capabilities": {
      "methods": ["inference.text", "inference.image", "models.list"],
      "features": ["open_models", "inference_api", "model_hub"]
    },
    "credentials_schema": {
      "required": ["api_token"],
      "credential_type": "huggingface_api"
    },
    "tags": ["ai", "ml", "open-source", "inference"],
    "metadata": {
      "documentation": "https://huggingface.co/docs",
      "auto_enable_on_credential": "huggingface_api"
    }
  },
  
  // Database (5 servers)
  {
    "name": "PostgreSQL Query MCP",
    "description": "Execute SQL queries, manage schemas, and analyze PostgreSQL databases",
    "provider": "PostgreSQL",
    "category": "database",
    "icon": "🐘",
    "version": "1.0.0",
    "status": "inactive",
    "capabilities": {
      "methods": ["query.execute", "schema.inspect", "table.analyze", "backup.create"],
      "features": ["sql_execution", "schema_management", "query_optimization"]
    },
    "credentials_schema": {
      "required": ["host", "port", "database", "user", "password"],
      "credential_type": "postgres_credentials"
    },
    "tags": ["database", "sql", "postgres", "analytics"],
    "metadata": {
      "documentation": "https://www.postgresql.org/docs/",
      "auto_enable_on_credential": "postgres_credentials",
      "usable_as_tool": true
    }
  },
  {
    "name": "MySQL Database MCP",
    "description": "MySQL database operations, queries, and management",
    "provider": "MySQL",
    "category": "database",
    "icon": "🐬",
    "version": "1.0.0",
    "status": "inactive",
    "capabilities": {
      "methods": ["query.execute", "schema.manage", "backup.export"],
      "features": ["sql_queries", "schema_ops", "data_export"]
    },
    "credentials_schema": {
      "required": ["host", "port", "database", "user", "password"],
      "credential_type": "mysql_credentials"
    },
    "tags": ["database", "sql", "mysql"],
    "metadata": {
      "auto_enable_on_credential": "mysql_credentials"
    }
  },
  {
    "name": "MongoDB Operations MCP",
    "description": "NoSQL document operations, aggregations, and indexes",
    "provider": "MongoDB",
    "category": "database",
    "icon": "🍃",
    "version": "1.0.0",
    "status": "inactive",
    "capabilities": {
      "methods": ["find", "insert", "update", "aggregate", "index.create"],
      "features": ["nosql_queries", "aggregations", "geospatial"]
    },
    "credentials_schema": {
      "required": ["connection_string"],
      "credential_type": "mongodb_credentials"
    },
    "tags": ["database", "nosql", "mongodb", "documents"],
    "metadata": {
      "auto_enable_on_credential": "mongodb_credentials"
    }
  },
  {
    "name": "Redis Cache MCP",
    "description": "Redis key-value operations, pub/sub, and caching",
    "provider": "Redis",
    "category": "database",
    "icon": "⚡",
    "version": "1.0.0",
    "status": "inactive",
    "capabilities": {
      "methods": ["get", "set", "delete", "publish", "subscribe", "scan"],
      "features": ["kv_store", "pubsub", "caching", "rate_limiting"]
    },
    "credentials_schema": {
      "required": ["host", "port"],
      "optional": ["password"],
      "credential_type": "redis_credentials"
    },
    "tags": ["database", "cache", "redis", "pubsub"],
    "metadata": {
      "auto_enable_on_credential": "redis_credentials",
      "usable_as_tool": true
    }
  },
  {
    "name": "Elasticsearch Search MCP",
    "description": "Full-text search, analytics, and log aggregation",
    "provider": "Elasticsearch",
    "category": "database",
    "icon": "🔍",
    "version": "1.0.0",
    "status": "inactive",
    "capabilities": {
      "methods": ["search", "index", "bulk", "aggregations"],
      "features": ["full_text_search", "analytics", "logging"]
    },
    "credentials_schema": {
      "required": ["base_url"],
      "optional": ["username", "password"],
      "credential_type": "elasticsearch_credentials"
    },
    "tags": ["database", "search", "elasticsearch", "analytics"],
    "metadata": {
      "auto_enable_on_credential": "elasticsearch_credentials"
    }
  },
  
  // Cloud (3 servers)
  {
    "name": "AWS Cloud MCP",
    "description": "Comprehensive AWS services: S3, EC2, Lambda, DynamoDB, SQS, SNS, and more",
    "provider": "Amazon Web Services",
    "category": "cloud",
    "icon": "☁️",
    "version": "1.0.0",
    "status": "inactive",
    "capabilities": {
      "methods": ["s3.upload", "s3.download", "ec2.list", "lambda.invoke", "dynamodb.query"],
      "features": ["object_storage", "compute", "serverless", "messaging"]
    },
    "credentials_schema": {
      "required": ["access_key_id", "secret_access_key", "region"],
      "credential_type": "aws_credentials"
    },
    "tags": ["cloud", "aws", "storage", "compute", "serverless"],
    "metadata": {
      "documentation": "https://docs.aws.amazon.com/",
      "auto_enable_on_credential": "aws_credentials",
      "usable_as_tool": true
    }
  },
  {
    "name": "Azure Cloud MCP",
    "description": "Microsoft Azure services: Storage, VMs, Functions, Cosmos DB",
    "provider": "Microsoft Azure",
    "category": "cloud",
    "icon": "☁️",
    "version": "1.0.0",
    "status": "inactive",
    "capabilities": {
      "methods": ["storage.blob", "compute.vm", "functions.deploy", "cosmosdb.query"],
      "features": ["blob_storage", "compute", "serverless", "database"]
    },
    "credentials_schema": {
      "required": ["client_id", "client_secret", "tenant_id"],
      "credential_type": "azure_credentials"
    },
    "tags": ["cloud", "azure", "microsoft", "storage"],
    "metadata": {
      "auto_enable_on_credential": "azure_credentials"
    }
  },
  {
    "name": "Google Cloud MCP",
    "description": "GCP services: Cloud Storage, Compute Engine, BigQuery, Cloud Functions",
    "provider": "Google Cloud",
    "category": "cloud",
    "icon": "☁️",
    "version": "1.0.0",
    "status": "inactive",
    "capabilities": {
      "methods": ["storage.upload", "compute.instance", "bigquery.query", "functions.deploy"],
      "features": ["object_storage", "compute", "analytics", "serverless"]
    },
    "credentials_schema": {
      "required": ["service_account_key"],
      "credential_type": "google_cloud_credentials"
    },
    "tags": ["cloud", "gcp", "google", "analytics"],
    "metadata": {
      "auto_enable_on_credential": "google_cloud_credentials"
    }
  },
  
  // Communication (5 servers)
  {
    "name": "Slack Communication MCP",
    "description": "Send messages, create channels, manage Slack workspace",
    "provider": "Slack",
    "category": "communication",
    "icon": "💬",
    "version": "1.0.0",
    "status": "inactive",
    "capabilities": {
      "methods": ["chat.postMessage", "channels.create", "users.list", "files.upload"],
      "features": ["messaging", "channels", "file_sharing", "notifications"]
    },
    "credentials_schema": {
      "required": ["access_token"],
      "credential_type": "slack_api"
    },
    "tags": ["communication", "slack", "messaging", "team"],
    "metadata": {
      "documentation": "https://api.slack.com/",
      "auto_enable_on_credential": "slack_api",
      "usable_as_tool": true
    }
  },
  {
    "name": "Discord Bot MCP",
    "description": "Send messages via webhook or bot, manage Discord servers",
    "provider": "Discord",
    "category": "communication",
    "icon": "🎮",
    "version": "1.0.0",
    "status": "inactive",
    "capabilities": {
      "methods": ["webhook.send", "message.create", "embed.create"],
      "features": ["webhooks", "messaging", "embeds"]
    },
    "credentials_schema": {
      "required": ["webhook_url"],
      "credential_type": "discord_webhook"
    },
    "tags": ["communication", "discord", "gaming", "community"],
    "metadata": {
      "auto_enable_on_credential": "discord_webhook"
    }
  },
  {
    "name": "Telegram Bot MCP",
    "description": "Telegram bot API for messages, commands, and inline queries",
    "provider": "Telegram",
    "category": "communication",
    "icon": "✈️",
    "version": "1.0.0",
    "status": "inactive",
    "capabilities": {
      "methods": ["sendMessage", "sendPhoto", "sendDocument", "getUpdates"],
      "features": ["messaging", "media", "commands", "inline_mode"]
    },
    "credentials_schema": {
      "required": ["bot_token"],
      "credential_type": "telegram_api"
    },
    "tags": ["communication", "telegram", "messaging", "bot"],
    "metadata": {
      "auto_enable_on_credential": "telegram_api"
    }
  },
  {
    "name": "Twilio SMS/Voice MCP",
    "description": "Send SMS, make calls, and manage Twilio communications",
    "provider": "Twilio",
    "category": "communication",
    "icon": "📱",
    "version": "1.0.0",
    "status": "inactive",
    "capabilities": {
      "methods": ["messages.create", "calls.create", "verify.check"],
      "features": ["sms", "voice", "verification", "whatsapp"]
    },
    "credentials_schema": {
      "required": ["account_sid", "auth_token"],
      "credential_type": "twilio_api"
    },
    "tags": ["communication", "sms", "voice", "twilio"],
    "metadata": {
      "auto_enable_on_credential": "twilio_api"
    }
  },
  {
    "name": "SendGrid Email MCP",
    "description": "Send emails, manage templates, and track email analytics",
    "provider": "SendGrid",
    "category": "communication",
    "icon": "📧",
    "version": "1.0.0",
    "status": "inactive",
    "capabilities": {
      "methods": ["mail.send", "templates.create", "stats.get"],
      "features": ["email", "templates", "analytics", "marketing"]
    },
    "credentials_schema": {
      "required": ["api_key"],
      "credential_type": "sendgrid_api"
    },
    "tags": ["communication", "email", "sendgrid", "marketing"],
    "metadata": {
      "auto_enable_on_credential": "sendgrid_api"
    }
  },
  
  // Code & Version Control (2 servers)
  {
    "name": "GitHub Integration MCP",
    "description": "Manage repos, PRs, issues, actions, and GitHub API",
    "provider": "GitHub",
    "category": "code",
    "icon": "🐙",
    "version": "1.0.0",
    "status": "inactive",
    "capabilities": {
      "methods": ["repos.create", "pulls.create", "issues.create", "actions.trigger", "code.search"],
      "features": ["repository_management", "pull_requests", "issues", "ci_cd", "code_search"]
    },
    "credentials_schema": {
      "required": ["access_token"],
      "credential_type": "github_api"
    },
    "tags": ["code", "github", "version-control", "ci-cd"],
    "metadata": {
      "documentation": "https://docs.github.com/rest",
      "auto_enable_on_credential": "github_api",
      "usable_as_tool": true
    }
  },
  {
    "name": "GitLab Integration MCP",
    "description": "GitLab repos, merge requests, CI/CD pipelines",
    "provider": "GitLab",
    "category": "code",
    "icon": "🦊",
    "version": "1.0.0",
    "status": "inactive",
    "capabilities": {
      "methods": ["projects.create", "merge_requests.create", "pipelines.trigger"],
      "features": ["repository_management", "merge_requests", "ci_cd"]
    },
    "credentials_schema": {
      "required": ["access_token"],
      "credential_type": "gitlab_api"
    },
    "tags": ["code", "gitlab", "version-control", "devops"],
    "metadata": {
      "auto_enable_on_credential": "gitlab_api"
    }
  },
  
  // Infrastructure (3 servers)
  {
    "name": "SSH Remote Execution MCP",
    "description": "Execute commands on remote servers via SSH",
    "provider": "SSH",
    "category": "infrastructure",
    "icon": "🖥️",
    "version": "1.0.0",
    "status": "inactive",
    "capabilities": {
      "methods": ["command.execute", "file.upload", "file.download", "tunnel.create"],
      "features": ["remote_execution", "file_transfer", "tunneling"]
    },
    "credentials_schema": {
      "required": ["host", "port", "username", "auth_method"],
      "credential_type": "ssh_credentials"
    },
    "tags": ["infrastructure", "ssh", "remote", "deployment"],
    "metadata": {
      "documentation": "https://www.ssh.com/academy/ssh",
      "auto_enable_on_credential": "ssh_credentials",
      "usable_as_tool": true
    }
  },
  {
    "name": "Docker Container MCP",
    "description": "Manage Docker containers, images, and registries",
    "provider": "Docker",
    "category": "infrastructure",
    "icon": "🐳",
    "version": "1.0.0",
    "status": "inactive",
    "capabilities": {
      "methods": ["containers.run", "images.build", "compose.up", "registry.push"],
      "features": ["container_management", "image_building", "compose", "registry"]
    },
    "credentials_schema": {
      "required": ["registry_url", "username", "password"],
      "credential_type": "docker_credentials"
    },
    "tags": ["infrastructure", "docker", "containers", "devops"],
    "metadata": {
      "auto_enable_on_credential": "docker_credentials"
    }
  },
  {
    "name": "Kubernetes Cluster MCP",
    "description": "Deploy and manage Kubernetes resources and workloads",
    "provider": "Kubernetes",
    "category": "infrastructure",
    "icon": "⎈",
    "version": "1.0.0",
    "status": "inactive",
    "capabilities": {
      "methods": ["pods.create", "services.expose", "deployments.scale", "logs.stream"],
      "features": ["orchestration", "scaling", "monitoring", "deployments"]
    },
    "credentials_schema": {
      "required": ["kubeconfig"],
      "credential_type": "kubernetes_credentials"
    },
    "tags": ["infrastructure", "kubernetes", "k8s", "orchestration"],
    "metadata": {
      "auto_enable_on_credential": "kubernetes_credentials"
    }
  },
  
  // Payment (2 servers)
  {
    "name": "Stripe Payment MCP",
    "description": "Process payments, manage subscriptions, and handle Stripe webhooks",
    "provider": "Stripe",
    "category": "payment",
    "icon": "💳",
    "version": "1.0.0",
    "status": "inactive",
    "capabilities": {
      "methods": ["charges.create", "subscriptions.create", "customers.manage", "webhooks.verify"],
      "features": ["payments", "subscriptions", "invoices", "webhooks"]
    },
    "credentials_schema": {
      "required": ["secret_key"],
      "credential_type": "stripe_api"
    },
    "tags": ["payment", "stripe", "subscriptions", "billing"],
    "metadata": {
      "documentation": "https://stripe.com/docs/api",
      "auto_enable_on_credential": "stripe_api",
      "usable_as_tool": true
    }
  },
  {
    "name": "PayPal Payment MCP",
    "description": "PayPal payments, checkout, and subscription management",
    "provider": "PayPal",
    "category": "payment",
    "icon": "💰",
    "version": "1.0.0",
    "status": "inactive",
    "capabilities": {
      "methods": ["orders.create", "payments.capture", "subscriptions.create"],
      "features": ["checkout", "payments", "subscriptions"]
    },
    "credentials_schema": {
      "required": ["client_id", "client_secret"],
      "credential_type": "paypal_api"
    },
    "tags": ["payment", "paypal", "checkout"],
    "metadata": {
      "auto_enable_on_credential": "paypal_api"
    }
  },
  
  // CRM (2 servers)
  {
    "name": "Salesforce CRM MCP",
    "description": "Manage Salesforce contacts, leads, opportunities, and workflows",
    "provider": "Salesforce",
    "category": "crm",
    "icon": "📊",
    "version": "1.0.0",
    "status": "inactive",
    "capabilities": {
      "methods": ["contacts.create", "leads.convert", "opportunities.update", "reports.run"],
      "features": ["crm", "sales", "reporting", "automation"]
    },
    "credentials_schema": {
      "required": ["access_token", "instance_url"],
      "credential_type": "salesforce_oauth2"
    },
    "tags": ["crm", "salesforce", "sales", "enterprise"],
    "metadata": {
      "auto_enable_on_credential": "salesforce_oauth2"
    }
  },
  {
    "name": "HubSpot CRM MCP",
    "description": "HubSpot contacts, deals, marketing automation, and analytics",
    "provider": "HubSpot",
    "category": "crm",
    "icon": "🎯",
    "version": "1.0.0",
    "status": "inactive",
    "capabilities": {
      "methods": ["contacts.create", "deals.update", "emails.send", "analytics.get"],
      "features": ["crm", "marketing", "automation", "analytics"]
    },
    "credentials_schema": {
      "required": ["api_key"],
      "credential_type": "hubspot_api"
    },
    "tags": ["crm", "hubspot", "marketing", "sales"],
    "metadata": {
      "auto_enable_on_credential": "hubspot_api"
    }
  },
  
  // Monitoring (1 server)
  {
    "name": "Datadog Monitoring MCP",
    "description": "Infrastructure monitoring, APM, logs, and analytics",
    "provider": "Datadog",
    "category": "monitoring",
    "icon": "📈",
    "version": "1.0.0",
    "status": "inactive",
    "capabilities": {
      "methods": ["metrics.submit", "logs.send", "monitors.create", "dashboards.get"],
      "features": ["monitoring", "apm", "logging", "alerting"]
    },
    "credentials_schema": {
      "required": ["api_key", "app_key"],
      "credential_type": "datadog_api"
    },
    "tags": ["monitoring", "datadog", "observability", "metrics"],
    "metadata": {
      "auto_enable_on_credential": "datadog_api"
    }
  },
  
  // Storage (1 server)
  {
    "name": "Amazon S3 Storage MCP",
    "description": "S3 object storage: upload, download, manage buckets and objects",
    "provider": "Amazon S3",
    "category": "storage",
    "icon": "🗄️",
    "version": "1.0.0",
    "status": "inactive",
    "capabilities": {
      "methods": ["upload", "download", "list", "delete", "presign", "bucket.create"],
      "features": ["object_storage", "bucket_management", "presigned_urls", "versioning"]
    },
    "credentials_schema": {
      "required": ["access_key_id", "secret_access_key", "region"],
      "credential_type": "s3_credentials"
    },
    "tags": ["storage", "s3", "aws", "object-storage"],
    "metadata": {
      "documentation": "https://docs.aws.amazon.com/s3/",
      "auto_enable_on_credential": "s3_credentials",
      "usable_as_tool": true
    }
  },
  
  // Generic/Utility (3 servers)
  {
    "name": "Generic REST API MCP",
    "description": "Call any REST API with customizable authentication",
    "provider": "Generic",
    "category": "api",
    "icon": "🔌",
    "version": "1.0.0",
    "status": "inactive",
    "capabilities": {
      "methods": ["request.get", "request.post", "request.put", "request.delete"],
      "features": ["rest_api", "custom_auth", "flexible"]
    },
    "credentials_schema": {
      "required": ["api_key"],
      "credential_type": "generic_api"
    },
    "tags": ["api", "rest", "generic", "flexible"],
    "metadata": {
      "auto_enable_on_credential": "generic_api",
      "usable_as_tool": true
    }
  },
  {
    "name": "OAuth2 Token Manager MCP",
    "description": "Manage OAuth2 tokens, refresh, and handle token expiry",
    "provider": "OAuth2",
    "category": "api",
    "icon": "🔐",
    "version": "1.0.0",
    "status": "inactive",
    "capabilities": {
      "methods": ["token.refresh", "token.validate", "token.revoke"],
      "features": ["oauth2", "token_management", "auto_refresh"]
    },
    "credentials_schema": {
      "required": ["access_token"],
      "credential_type": "oauth2_token"
    },
    "tags": ["api", "oauth2", "authentication"],
    "metadata": {
      "auto_enable_on_credential": "oauth2_token"
    }
  },
  {
    "name": "HTTP Basic Auth MCP",
    "description": "Simple HTTP basic authentication for APIs",
    "provider": "HTTP",
    "category": "api",
    "icon": "🔑",
    "version": "1.0.0",
    "status": "inactive",
    "capabilities": {
      "methods": ["request.authenticated"],
      "features": ["basic_auth", "simple"]
    },
    "credentials_schema": {
      "required": ["username", "password"],
      "credential_type": "http_basic_auth"
    },
    "tags": ["api", "authentication", "basic-auth"],
    "metadata": {
      "auto_enable_on_credential": "http_basic_auth"
    }
  }
]
```

**Total: 30 MCP Servers** across 9 categories, perfectly matched to our 30 credential types!

---

### 2.2 Auto-Activation System

**How It Works**:

```python
# services/mcp_auto_activation.py

class MCPAutoActivationService:
    """
    Automatically enables MCP servers when credentials are added
    """
    
    async def activate_mcp_servers_for_credential(
        self,
        credential_id: int,
        credential_type_name: str,
        db: Session
    ):
        """
        When user adds a credential, auto-enable matching MCP servers
        
        Example:
        - User adds OpenAI credential
        - System finds MCP servers with metadata.auto_enable_on_credential = "openai_api"
        - Enables those MCP servers (status: inactive → active)
        - Logs activation
        """
        
        # Find MCP servers that match this credential type
        matching_servers = db.query(MCPTool).filter(
            MCPTool.status == 'inactive',
            MCPTool.tool_metadata['auto_enable_on_credential'].astext == credential_type_name
        ).all()
        
        logger.info(f"🔌 Activating {len(matching_servers)} MCP servers for credential type: {credential_type_name}")
        
        activated_count = 0
        for server in matching_servers:
            try:
                # Update status to active
                server.status = 'active'
                
                # Link to credential
                server.tool_metadata = {
                    **server.tool_metadata,
                    'linked_credential_id': credential_id,
                    'activated_at': datetime.now().isoformat(),
                    'activation_method': 'auto'
                }
                
                activated_count += 1
                logger.info(f"  ✅ Activated: {server.name}")
                
            except Exception as e:
                logger.error(f"  ❌ Failed to activate {server.name}: {e}")
        
        db.commit()
        
        logger.info(f"🎉 Auto-activation complete: {activated_count}/{len(matching_servers)} servers enabled")
        
        return {
            'activated_count': activated_count,
            'total_matching': len(matching_servers),
            'servers': [s.name for s in matching_servers[:activated_count]]
        }
```

**Trigger Point**:
```python
# In api/credentials.py - create_credential endpoint

@router.post("/")
async def create_credential(...):
    # ... existing code ...
    
    # Create credential
    credential = credential_store.create_credential(...)
    
    # AUTO-ACTIVATE matching MCP servers
    activation_service = MCPAutoActivationService()
    activation_result = await activation_service.activate_mcp_servers_for_credential(
        credential_id=credential.id,
        credential_type_name=credential_type.name,
        db=db
    )
    
    return {
        "credential": credential,
        "auto_activated_servers": activation_result
    }
```

---

### 2.3 Pagination Implementation

#### Backend: API Pagination

```python
# Update api/mcp_tools.py

@router.get("/")
async def list_mcp_tools(
    skip: int = Query(0, ge=0, description="Number of items to skip"),
    limit: int = Query(20, ge=1, le=100, description="Number of items per page"),
    category: Optional[str] = Query(None, description="Filter by category"),
    status: Optional[str] = Query(None, description="Filter by status"),
    search: Optional[str] = Query(None, description="Search in name/description"),
    provider: Optional[str] = Query(None, description="Filter by provider"),
    db: Session = Depends(get_db)
):
    """
    List MCP tools with pagination, search, and filters
    """
    
    # Build base query
    query = db.query(MCPTool)
    
    # Apply filters
    if category:
        query = query.filter(MCPTool.category == category)
    if status:
        query = query.filter(MCPTool.status == status)
    if provider:
        query = query.filter(MCPTool.provider == provider)
    if search:
        search_pattern = f"%{search}%"
        query = query.filter(
            or_(
                MCPTool.name.ilike(search_pattern),
                MCPTool.description.ilike(search_pattern)
            )
        )
    
    # Get total count (before pagination)
    total = query.count()
    
    # Apply pagination
    tools = query.order_by(MCPTool.name).offset(skip).limit(limit).all()
    
    return {
        "items": [format_mcp_tool(tool) for tool in tools],
        "total": total,
        "skip": skip,
        "limit": limit,
        "pages": (total + limit - 1) // limit  # Ceiling division
    }

# Similarly update api/credentials.py
@router.get("/")
async def list_credentials(
    skip: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
    credential_type_id: Optional[int] = None,
    environment: Optional[str] = None,
    search: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """List credentials with pagination"""
    # ... same pagination pattern
```

#### Frontend: Pagination Component

```typescript
// components/shared/pagination.tsx

interface PaginationProps {
  currentPage: number
  totalPages: number
  totalItems: number
  itemsPerPage: number
  onPageChange: (page: number) => void
  onItemsPerPageChange?: (itemsPerPage: number) => void
}

export function Pagination({ 
  currentPage, 
  totalPages, 
  totalItems,
  itemsPerPage,
  onPageChange,
  onItemsPerPageChange
}: PaginationProps) {
  const startItem = (currentPage - 1) * itemsPerPage + 1
  const endItem = Math.min(currentPage * itemsPerPage, totalItems)
  
  return (
    <div className="flex items-center justify-between py-4">
      {/* Left: Items info */}
      <div className="text-sm text-muted-foreground">
        Showing {startItem}-{endItem} of {totalItems} items
      </div>
      
      {/* Center: Page navigation */}
      <div className="flex items-center gap-2">
        <Button
          variant="outline"
          size="sm"
          onClick={() => onPageChange(currentPage - 1)}
          disabled={currentPage === 1}
        >
          <ChevronLeft className="h-4 w-4" />
          Previous
        </Button>
        
        {/* Page numbers */}
        <div className="flex gap-1">
          {Array.from({ length: Math.min(totalPages, 7) }, (_, i) => {
            // Smart page number display (show current +/- 3)
            let pageNum
            if (totalPages <= 7) {
              pageNum = i + 1
            } else if (currentPage <= 4) {
              pageNum = i + 1
            } else if (currentPage >= totalPages - 3) {
              pageNum = totalPages - 6 + i
            } else {
              pageNum = currentPage - 3 + i
            }
            
            return (
              <Button
                key={pageNum}
                variant={currentPage === pageNum ? "default" : "outline"}
                size="sm"
                className="w-10"
                onClick={() => onPageChange(pageNum)}
              >
                {pageNum}
              </Button>
            )
          })}
        </div>
        
        <Button
          variant="outline"
          size="sm"
          onClick={() => onPageChange(currentPage + 1)}
          disabled={currentPage === totalPages}
        >
          Next
          <ChevronRight className="h-4 w-4" />
        </Button>
      </div>
      
      {/* Right: Items per page */}
      {onItemsPerPageChange && (
        <Select 
          value={itemsPerPage.toString()} 
          onValueChange={(value) => onItemsPerPageChange(parseInt(value))}
        >
          <SelectTrigger className="w-32">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="20">20 per page</SelectItem>
            <SelectItem value="50">50 per page</SelectItem>
            <SelectItem value="100">100 per page</SelectItem>
          </SelectContent>
        </Select>
      )}
    </div>
  )
}
```

#### Update Tools Dashboard with Pagination

```typescript
// components/tools/tools-dashboard.tsx

export function ToolsDashboard() {
  const [currentPage, setCurrentPage] = useState(1)
  const [itemsPerPage, setItemsPerPage] = useState(20)
  const [searchQuery, setSearchQuery] = useState('')
  const [selectedCategory, setSelectedCategory] = useState('all')
  
  // Fetch with pagination
  const { data: toolsData, isLoading } = useMCPTools({
    skip: (currentPage - 1) * itemsPerPage,
    limit: itemsPerPage,
    category: selectedCategory !== 'all' ? selectedCategory : undefined,
    search: searchQuery || undefined
  })
  
  const tools = toolsData?.items || []
  const totalTools = toolsData?.total || 0
  const totalPages = Math.ceil(totalTools / itemsPerPage)
  
  return (
    <div className="space-y-6">
      {/* ... existing header and stats ... */}
      
      {/* Search and filters */}
      <div className="flex gap-4">
        <Input
          placeholder="Search tools..."
          value={searchQuery}
          onChange={(e) => {
            setSearchQuery(e.target.value)
            setCurrentPage(1) // Reset to page 1 on search
          }}
        />
        {/* ... category filters ... */}
      </div>
      
      {/* Tools grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {tools.map((tool) => (
          <ToolCard key={tool.id} tool={tool} />
        ))}
      </div>
      
      {/* NEW: Pagination */}
      <Pagination
        currentPage={currentPage}
        totalPages={totalPages}
        totalItems={totalTools}
        itemsPerPage={itemsPerPage}
        onPageChange={setCurrentPage}
        onItemsPerPageChange={(newLimit) => {
          setItemsPerPage(newLimit)
          setCurrentPage(1) // Reset to page 1
        }}
      />
    </div>
  )
}
```

#### Update Credentials Tab with Pagination

```typescript
// components/settings/CredentialsTab.tsx

export function CredentialsTab() {
  const [currentPage, setCurrentPage] = useState(1)
  const [itemsPerPage, setItemsPerPage] = useState(20)
  const [searchTerm, setSearchTerm] = useState('')
  const [environmentFilter, setEnvironmentFilter] = useState<string>('all')
  
  // Fetch with pagination
  const { data: credentialsData, isLoading } = useCredentials({
    skip: (currentPage - 1) * itemsPerPage,
    limit: itemsPerPage,
    environment: environmentFilter !== 'all' ? environmentFilter : undefined,
    search: searchTerm || undefined
  })
  
  const credentials = credentialsData?.items || []
  const totalCredentials = credentialsData?.total || 0
  const totalPages = Math.ceil(totalCredentials / itemsPerPage)
  
  return (
    <div className="space-y-6">
      {/* ... existing content ... */}
      
      {/* Credentials grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {credentials.map((cred) => (
          <CredentialCard key={cred.id} credential={cred} />
        ))}
      </div>
      
      {/* NEW: Pagination */}
      <Pagination
        currentPage={currentPage}
        totalPages={totalPages}
        totalItems={totalCredentials}
        itemsPerPage={itemsPerPage}
        onPageChange={setCurrentPage}
        onItemsPerPageChange={setItemsPerPage}
      />
    </div>
  )
}
```

---

### 2.4 Database Bulk Load Script

```python
# scripts/load_mcp_servers.py

"""
Bulk load MCP servers from JSON library
"""

import sys
import os
import json
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy.orm import Session
from database.database import SessionLocal, engine
from database.models import MCPTool
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_mcp_servers_from_json(json_file: str, db: Session):
    """Load MCP servers from JSON file"""
    
    with open(json_file, 'r') as f:
        servers = json.load(f)
    
    logger.info(f"📚 Loading {len(servers)} MCP servers...")
    
    loaded = 0
    updated = 0
    errors = 0
    
    for server_data in servers:
        try:
            # Check if server already exists
            existing = db.query(MCPTool).filter(
                MCPTool.name == server_data['name']
            ).first()
            
            if existing:
                # Update existing
                existing.description = server_data.get('description')
                existing.provider = server_data.get('provider')
                existing.category = server_data.get('category')
                existing.icon = server_data.get('icon')
                existing.version = server_data.get('version')
                existing.status = server_data.get('status', 'inactive')
                existing.capabilities = server_data.get('capabilities', {})
                existing.credentials_schema = server_data.get('credentials_schema', {})
                existing.tags = server_data.get('tags', [])
                existing.tool_metadata = server_data.get('metadata', {})
                existing.updated_at = datetime.now()
                
                updated += 1
                logger.info(f"  🔄 Updated: {server_data['name']}")
            else:
                # Create new
                mcp_tool = MCPTool(
                    name=server_data['name'],
                    description=server_data.get('description'),
                    provider=server_data.get('provider'),
                    category=server_data.get('category'),
                    icon=server_data.get('icon'),
                    version=server_data.get('version'),
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
                logger.info(f"  ✅ Created: {server_data['name']}")
        
        except Exception as e:
            errors += 1
            logger.error(f"  ❌ Error loading {server_data.get('name', 'unknown')}: {e}")
    
    db.commit()
    
    logger.info(f"\n📊 Summary:")
    logger.info(f"  ✅ Loaded: {loaded}")
    logger.info(f"  🔄 Updated: {updated}")
    logger.info(f"  ❌ Errors: {errors}")
    logger.info(f"  📦 Total: {loaded + updated}")

if __name__ == "__main__":
    db = SessionLocal()
    try:
        json_file = sys.argv[1] if len(sys.argv) > 1 else 'mcp_servers_library_30.json'
        load_mcp_servers_from_json(json_file, db)
    finally:
        db.close()
```

---

## Part 3: Phase 2 Implementation (400+ Servers)

### 3.1 n8n Credential Scraping Strategy

**Source**: https://github.com/n8n-io/n8n/tree/master/packages/nodes-base/credentials

**Approach**:
```python
# scripts/scrape_n8n_credentials.py

"""
Scrape ALL credential types from n8n GitHub repository
"""

import requests
import json
import re
from typing import List, Dict, Any

class N8nCredentialScraper:
    """Scrape n8n credentials from GitHub"""
    
    GITHUB_API = "https://api.github.com"
    N8N_REPO = "n8n-io/n8n"
    CREDS_PATH = "packages/nodes-base/credentials"
    
    def __init__(self, github_token: str = None):
        self.token = github_token
        self.headers = {}
        if github_token:
            self.headers['Authorization'] = f'token {github_token}'
    
    async def scrape_all_credentials(self) -> List[Dict[str, Any]]:
        """
        Scrape all .credentials.ts files from n8n repo
        """
        
        # Step 1: List all credential files
        url = f"{self.GITHUB_API}/repos/{self.N8N_REPO}/contents/{self.CREDS_PATH}"
        response = requests.get(url, headers=self.headers)
        files = response.json()
        
        # Filter .credentials.ts files
        cred_files = [
            f for f in files 
            if f['name'].endswith('.credentials.ts')
        ]
        
        logger.info(f"📚 Found {len(cred_files)} credential files in n8n")
        
        credentials = []
        
        for file_info in cred_files:
            try:
                # Step 2: Fetch file content
                file_url = file_info['download_url']
                file_content = requests.get(file_url).text
                
                # Step 3: Parse TypeScript to extract schema
                parsed = self._parse_credential_file(file_content, file_info['name'])
                
                if parsed:
                    credentials.append(parsed)
                    logger.info(f"  ✅ Parsed: {parsed['name']}")
                
            except Exception as e:
                logger.error(f"  ❌ Failed to parse {file_info['name']}: {e}")
        
        logger.info(f"\n🎉 Successfully parsed {len(credentials)} credentials!")
        
        return credentials
    
    def _parse_credential_file(self, content: str, filename: str) -> Dict[str, Any]:
        """
        Parse TypeScript credential file to extract:
        - Credential name
        - Display name
        - Properties/fields
        - Test endpoint (if exists)
        - Documentation URL
        """
        
        # Extract class name (credential name)
        class_match = re.search(r'export class (\w+) implements', content)
        if not class_match:
            return None
        
        name = class_match.group(1)
        
        # Extract display name
        display_name_match = re.search(r'displayName\s*=\s*[\'"](.+?)[\'"]', content)
        display_name = display_name_match.group(1) if display_name_match else name
        
        # Extract properties (schema)
        properties_match = re.search(r'properties:\s*\[(.*?)\]', content, re.DOTALL)
        schema = []
        
        if properties_match:
            # Parse each property definition
            # This is complex - would need full TypeScript parser
            # For MVP: Use regex patterns to extract basic info
            pass
        
        # Extract category from filename or content
        category = self._infer_category(name, content)
        
        return {
            "name": self._to_snake_case(name),
            "display_name": display_name,
            "category": category,
            "icon": self._infer_icon(category),
            "description": f"{display_name} credentials",
            "schema_definition": schema,
            "is_system": False
        }
    
    def _infer_category(self, name: str, content: str) -> str:
        """Infer category from name and content"""
        name_lower = name.lower()
        
        if any(x in name_lower for x in ['aws', 'azure', 'gcp', 'google cloud']):
            return 'cloud'
        elif any(x in name_lower for x in ['postgres', 'mysql', 'mongo', 'redis', 'database']):
            return 'database'
        elif any(x in name_lower for x in ['slack', 'discord', 'telegram', 'email', 'sms']):
            return 'communication'
        elif any(x in name_lower for x in ['github', 'gitlab', 'bitbucket', 'git']):
            return 'code'
        elif any(x in name_lower for x in ['stripe', 'paypal', 'payment']):
            return 'payment'
        elif any(x in name_lower for x in ['salesforce', 'hubspot', 'crm']):
            return 'crm'
        else:
            return 'api'
```

### 3.2 MCP Server Generation

**For each credential type, generate corresponding MCP server**:

```python
# scripts/generate_mcp_servers_from_credentials.py

def generate_mcp_server_from_credential(credential_type: Dict) -> Dict:
    """
    Auto-generate MCP server definition from credential type
    """
    
    return {
        "name": f"{credential_type['display_name']} MCP",
        "description": f"Integration for {credential_type['display_name']}",
        "provider": credential_type.get('provider', credential_type['display_name']),
        "category": credential_type['category'],
        "icon": credential_type['icon'],
        "version": "1.0.0",
        "status": "inactive",  # Start disabled
        "mcp_server_url": None,  # Will be populated when implemented
        "capabilities": {
            "methods": infer_methods(credential_type),
            "features": infer_features(credential_type)
        },
        "credentials_schema": {
            "required": extract_required_fields(credential_type['schema_definition']),
            "credential_type": credential_type['name']
        },
        "tags": generate_tags(credential_type),
        "metadata": {
            "documentation": credential_type.get('documentation_url'),
            "auto_enable_on_credential": credential_type['name'],
            "usable_as_tool": True,
            "source": "n8n",
            "generated": True
        }
    }
```

### 3.3 Advanced Search & Filtering

```typescript
// components/shared/advanced-filter.tsx

interface AdvancedFilterProps {
  onFilterChange: (filters: FilterState) => void
  categories: string[]
  providers: string[]
  tags: string[]
}

export function AdvancedFilter({ onFilterChange, categories, providers, tags }: AdvancedFilterProps) {
  const [filters, setFilters] = useState<FilterState>({
    search: '',
    categories: [],
    providers: [],
    tags: [],
    status: 'all',
    sortBy: 'name',
    sortOrder: 'asc'
  })
  
  return (
    <Card className="glass-card">
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Filter className="h-5 w-5" />
          Advanced Filters
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* Search */}
        <div>
          <Label>Search</Label>
          <Input
            placeholder="Search by name, description, or tags..."
            value={filters.search}
            onChange={(e) => updateFilter('search', e.target.value)}
          />
        </div>
        
        {/* Multi-select categories */}
        <div>
          <Label>Categories</Label>
          <MultiSelect
            options={categories.map(c => ({ label: c, value: c }))}
            selected={filters.categories}
            onChange={(selected) => updateFilter('categories', selected)}
          />
        </div>
        
        {/* Multi-select providers */}
        <div>
          <Label>Providers</Label>
          <MultiSelect
            options={providers.map(p => ({ label: p, value: p }))}
            selected={filters.providers}
            onChange={(selected) => updateFilter('providers', selected)}
          />
        </div>
        
        {/* Tag filter */}
        <div>
          <Label>Tags</Label>
          <MultiSelect
            options={tags.map(t => ({ label: t, value: t }))}
            selected={filters.tags}
            onChange={(selected) => updateFilter('tags', selected)}
          />
        </div>
        
        {/* Status filter */}
        <div>
          <Label>Status</Label>
          <Select value={filters.status} onValueChange={(v) => updateFilter('status', v)}>
            <SelectTrigger>
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="all">All</SelectItem>
              <SelectItem value="active">Active</SelectItem>
              <SelectItem value="inactive">Inactive</SelectItem>
            </SelectContent>
          </Select>
        </div>
        
        {/* Sort */}
        <div className="grid grid-cols-2 gap-4">
          <div>
            <Label>Sort By</Label>
            <Select value={filters.sortBy} onValueChange={(v) => updateFilter('sortBy', v)}>
              <SelectTrigger>
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="name">Name</SelectItem>
                <SelectItem value="provider">Provider</SelectItem>
                <SelectItem value="category">Category</SelectItem>
                <SelectItem value="created_at">Date Added</SelectItem>
              </SelectContent>
            </Select>
          </div>
          <div>
            <Label>Order</Label>
            <Select value={filters.sortOrder} onValueChange={(v) => updateFilter('sortOrder', v)}>
              <SelectTrigger>
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="asc">A → Z</SelectItem>
                <SelectItem value="desc">Z → A</SelectItem>
              </SelectContent>
            </Select>
          </div>
        </div>
        
        {/* Clear filters */}
        <Button 
          variant="outline" 
          className="w-full"
          onClick={() => {
            const defaultFilters = { /* ... */ }
            setFilters(defaultFilters)
            onFilterChange(defaultFilters)
          }}
        >
          Clear Filters
        </Button>
      </CardContent>
    </Card>
  )
}
```

---

## Part 4: Testing Plan

### Phase 1 Testing (30 Servers)

**Test 1: Load MCP Servers**
```bash
# Load 30 servers
python scripts/load_mcp_servers.py mcp_servers_library_30.json

# Verify in database
psql -U postgres -d orchestrator_db -c "SELECT COUNT(*) FROM mcp_tools;"
# Expected: 30+

# Verify by category
psql -U postgres -d orchestrator_db -c "SELECT category, COUNT(*) FROM mcp_tools GROUP BY category;"
```

**Test 2: Pagination**
```bash
# Test API pagination
curl "http://localhost:8000/api/mcp-tools/?skip=0&limit=10"
curl "http://localhost:8000/api/mcp-tools/?skip=10&limit=10"
curl "http://localhost:8000/api/mcp-tools/?skip=20&limit=10"

# Verify UI loads pages
# Open: http://localhost:3000/tools
# Navigate through pages 1, 2, 3
```

**Test 3: Auto-Activation**
```bash
# Add AWS credential via UI
# Watch backend logs for:
# "🔌 Activating X MCP servers for credential type: aws_credentials"
# "✅ Activated: AWS Cloud MCP"

# Verify in database
psql -U postgres -d orchestrator_db -c "SELECT name, status FROM mcp_tools WHERE status = 'active';"
```

**Test 4: Search & Filter**
```
# Search: "aws"
# Expected: AWS Cloud MCP, Amazon S3 Storage MCP

# Filter: category=database
# Expected: PostgreSQL, MySQL, MongoDB, Redis, Elasticsearch

# Filter: provider=OpenAI
# Expected: OpenAI Assistant MCP
```

**Test 5: Agent Tool Assignment**
```
# Via UI:
1. Go to Tools page
2. Enable "GitHub Integration MCP"
3. Go to Agent Assignment modal
4. Assign GitHub MCP to Code Architect agent
5. Verify assignment saved

# Via API:
curl -X POST http://localhost:8000/api/agents/5/tools/assign \
  -d '{"tool_id": 16, "credential_id": 7}'
```

**Test 6: End-to-End Workflow**
```
USER FLOW:
1. User adds GitHub credential in Settings
2. System auto-enables GitHub MCP servers
3. User assigns GitHub tools to Code Architect agent
4. User runs workflow with Code Architect
5. Agent executes GitHub tool (create PR)
6. Credential is automatically injected
7. Success! PR created on GitHub

VERIFY:
- Credential encrypted in DB ✅
- MCP server activated ✅
- Tool assigned to agent ✅
- Credential injected at runtime ✅
- GitHub API called successfully ✅
```

---

### Phase 2 Testing (400+ Servers)

**Performance Tests**:
```python
# Load test with 400+ items
async def test_pagination_performance():
    import time
    
    # Test 1: Load page 1
    start = time.time()
    response = await client.get("/api/mcp-tools/?skip=0&limit=20")
    page1_time = time.time() - start
    
    # Test 2: Load page 10
    start = time.time()
    response = await client.get("/api/mcp-tools/?skip=180&limit=20")
    page10_time = time.time() - start
    
    # Test 3: Load page 20
    start = time.time()
    response = await client.get("/api/mcp-tools/?skip=380&limit=20")
    page20_time = time.time() - start
    
    # Assert: All pages load in <500ms
    assert page1_time < 0.5
    assert page10_time < 0.5
    assert page20_time < 0.5
    
    logger.info(f"✅ Pagination performance test passed")
    logger.info(f"  Page 1: {page1_time:.3f}s")
    logger.info(f"  Page 10: {page10_time:.3f}s")
    logger.info(f"  Page 20: {page20_time:.3f}s")
```

**Search Performance**:
```python
# Test search with 400+ items
async def test_search_performance():
    # Test 1: Generic search
    start = time.time()
    response = await client.get("/api/mcp-tools/?search=github")
    search_time = time.time() - start
    
    # Assert: Search completes in <500ms
    assert search_time < 0.5
    
    logger.info(f"✅ Search completed in {search_time:.3f}s")
```

---

## Part 5: Database Optimization for Scale

### 5.1 Indexes for Performance

```sql
-- Add indexes for pagination queries
CREATE INDEX idx_mcp_tools_category ON mcp_tools(category);
CREATE INDEX idx_mcp_tools_provider ON mcp_tools(provider);
CREATE INDEX idx_mcp_tools_status ON mcp_tools(status);
CREATE INDEX idx_mcp_tools_name ON mcp_tools(name);
CREATE INDEX idx_mcp_tools_created_at ON mcp_tools(created_at DESC);

-- Full-text search index
CREATE INDEX idx_mcp_tools_search ON mcp_tools USING GIN(
    to_tsvector('english', name || ' ' || COALESCE(description, ''))
);

-- Tag search (GIN index for JSONB)
CREATE INDEX idx_mcp_tools_tags ON mcp_tools USING GIN(tags);

-- Similarly for credentials
CREATE INDEX idx_credentials_type ON credentials(credential_type_id);
CREATE INDEX idx_credentials_environment ON credentials(environment);
CREATE INDEX idx_credentials_name ON credentials(name);
CREATE INDEX idx_credentials_search ON credentials USING GIN(
    to_tsvector('english', name || ' ' || COALESCE(description, ''))
);
```

### 5.2 Query Optimization

```python
# Optimized query with filters
@router.get("/")
async def list_mcp_tools_optimized(
    skip: int = 0,
    limit: int = 20,
    search: Optional[str] = None,
    categories: Optional[str] = None,  # Comma-separated
    providers: Optional[str] = None,   # Comma-separated
    tags: Optional[str] = None,        # Comma-separated
    status: Optional[str] = None,
    sort_by: str = 'name',
    sort_order: str = 'asc',
    db: Session = Depends(get_db)
):
    """Optimized list with multiple filters"""
    
    # Build query
    query = db.query(MCPTool)
    
    # Full-text search (uses GIN index)
    if search:
        query = query.filter(
            text("to_tsvector('english', name || ' ' || COALESCE(description, '')) @@ plainto_tsquery(:search)")
        ).params(search=search)
    
    # Category filter (uses index)
    if categories:
        cat_list = categories.split(',')
        query = query.filter(MCPTool.category.in_(cat_list))
    
    # Provider filter (uses index)
    if providers:
        prov_list = providers.split(',')
        query = query.filter(MCPTool.provider.in_(prov_list))
    
    # Tag filter (uses GIN index)
    if tags:
        tag_list = tags.split(',')
        for tag in tag_list:
            query = query.filter(MCPTool.tags.contains([tag]))
    
    # Status filter (uses index)
    if status:
        query = query.filter(MCPTool.status == status)
    
    # Total count (before pagination)
    total = query.count()
    
    # Sorting
    sort_column = getattr(MCPTool, sort_by, MCPTool.name)
    if sort_order == 'desc':
        query = query.order_by(sort_column.desc())
    else:
        query = query.order_by(sort_column.asc())
    
    # Pagination (uses LIMIT/OFFSET)
    tools = query.offset(skip).limit(limit).all()
    
    return {
        "items": [format_mcp_tool(t) for t in tools],
        "total": total,
        "skip": skip,
        "limit": limit,
        "pages": (total + limit - 1) // limit
    }
```

---

## Part 6: UI Enhancements

### 6.1 Bulk Operations

```typescript
// components/tools/bulk-operations.tsx

export function BulkOperations() {
  const [selectedTools, setSelectedTools] = useState<number[]>([])
  
  return (
    <div className="flex gap-2">
      <Button
        variant="outline"
        onClick={() => handleBulkEnable(selectedTools)}
        disabled={selectedTools.length === 0}
      >
        <Power className="h-4 w-4 mr-2" />
        Enable ({selectedTools.length})
      </Button>
      
      <Button
        variant="outline"
        onClick={() => handleBulkDisable(selectedTools)}
        disabled={selectedTools.length === 0}
      >
        <PowerOff className="h-4 w-4 mr-2" />
        Disable ({selectedTools.length})
      </Button>
      
      <Button
        variant="outline"
        onClick={() => setSelectedTools([])}
        disabled={selectedTools.length === 0}
      >
        Clear Selection
      </Button>
    </div>
  )
}
```

### 6.2 Category Stats Dashboard

```typescript
// Show distribution of 400+ servers by category

<div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-4">
  {categoryStats.map(stat => (
    <Card key={stat.category} className="glass-card cursor-pointer hover:border-primary"
      onClick={() => filterByCategory(stat.category)}
    >
      <CardContent className="p-4 text-center">
        <div className="text-2xl mb-2">{getCategoryIcon(stat.category)}</div>
        <div className="text-sm text-muted-foreground">{stat.category}</div>
        <div className="text-xl font-bold">{stat.count}</div>
      </CardContent>
    </Card>
  ))}
</div>
```

---

## Part 7: Success Metrics

### Phase 1 Success Criteria (30 Servers)
- [ ] 30 MCP servers loaded successfully
- [ ] Pagination works smoothly (page size: 20)
- [ ] Auto-activation triggers when credential added
- [ ] At least 1 end-to-end test passes (GitHub example)
- [ ] UI remains performant (<2s page load)
- [ ] Search returns results in <500ms
- [ ] Zero breaking changes to existing functionality

### Phase 2 Success Criteria (400+ Servers)
- [ ] 400+ credential types cloned from n8n
- [ ] 400+ MCP servers generated and loaded
- [ ] Pagination handles 20 pages smoothly
- [ ] Advanced filters reduce results effectively
- [ ] Database queries optimized (<500ms)
- [ ] Full-text search working
- [ ] UI loads any page instantly (<1s)
- [ ] Memory usage acceptable (<500MB)

---

## Part 8: Timeline

### Phase 1: Week 1 (8-12 hours)

**Day 1 (4-6h):**
- ✅ Create `mcp_servers_library_30.json`
- ✅ Create `load_mcp_servers.py` script
- ✅ Implement auto-activation service
- ✅ Load 30 servers into database
- ✅ Test auto-activation

**Day 2 (4-6h):**
- ✅ Add pagination to backend APIs
- ✅ Create Pagination component
- ✅ Update Tools Dashboard with pagination
- ✅ Update Credentials Tab with pagination
- ✅ Test pagination with 30 items
- ✅ Fix any bugs
- ✅ Complete end-to-end test

---

### Phase 2: Week 2-3 (20-30 hours)

**Week 2 (12-16h):**
- ⏳ Scrape n8n credentials (or manually convert)
- ⏳ Generate all 400+ credential types
- ⏳ Generate all 400+ MCP servers
- ⏳ Add database indexes
- ⏳ Optimize queries

**Week 3 (8-14h):**
- ⏳ Bulk load all 400+ items
- ⏳ Implement advanced filters
- ⏳ Add bulk operations
- ⏳ Performance testing & optimization
- ⏳ Documentation & demo

---

## Part 9: Files to Create/Modify

### Phase 1 Files

**New Files**:
1. `mcp_servers_library_30.json` - 30 MCP server definitions
2. `scripts/load_mcp_servers.py` - Bulk loader script
3. `services/mcp_auto_activation.py` - Auto-activation service
4. `components/shared/pagination.tsx` - Reusable pagination component
5. `PRDS/20-MCP-CREDENTIAL-INTEGRATION.md` - This PRD

**Modified Files**:
1. `api/mcp_tools.py` - Add pagination, search, filters
2. `api/credentials.py` - Add pagination, trigger auto-activation
3. `components/tools/tools-dashboard.tsx` - Add pagination
4. `components/settings/CredentialsTab.tsx` - Add pagination
5. `hooks/use-mcp-tools-api.ts` - Update to support pagination params

### Phase 2 Files

**New Files**:
1. `credential_types/all_credential_types_400.py` - All 400+ types
2. `mcp_servers_library_400.json` - All 400+ servers
3. `scripts/scrape_n8n_credentials.py` - n8n scraper
4. `scripts/generate_mcp_servers.py` - Generator script
5. `components/shared/advanced-filter.tsx` - Advanced filtering
6. `components/shared/multi-select.tsx` - Multi-select component
7. `components/tools/bulk-operations.tsx` - Bulk enable/disable
8. `migrations/add_mcp_performance_indexes.sql` - Performance indexes

---

## Part 10: Business Impact

### Competitive Advantage
- **400+ pre-integrated tools** vs. competitors' 10-50
- **One-click activation** vs. manual configuration
- **Credential-driven UX** (add credential → tools appear)
- **Netflix-style marketplace** for integrations

### User Experience
- **Discovery**: Browse 400+ integrations like Netflix
- **Activation**: Add credential → instant access
- **Assignment**: Drag-drop to agents
- **Execution**: Zero configuration needed

### Cost Savings
- **Engineering Time**: No custom integrations needed
- **Maintenance**: n8n community maintains 400+ integrations
- **Support**: Standardized credential system
- **Onboarding**: Users familiar with n8n patterns

### Revenue Potential
- **Premium Integrations**: Charge for enterprise tools
- **Usage-Based**: Meter tool executions
- **Enterprise Plans**: Unlimited integrations
- **Marketplace**: Community contributions

---

## Part 11: Risks & Mitigation

| Risk | Impact | Mitigation |
|------|--------|------------|
| **n8n scraping fails** | High | Manual conversion as fallback |
| **Performance degradation** | Medium | Indexes, caching, query optimization |
| **UI becomes overwhelming** | Medium | Good filters, search, categories |
| **Auto-activation false positives** | Low | Explicit credential type matching |
| **Database storage** | Low | 400 records ≈ 2MB, negligible |

---

## Part 12: Post-Implementation

### Monitoring
- Track auto-activation success rate
- Monitor pagination query performance
- Measure search latency
- Track user adoption per integration

### Maintenance
- Weekly: Review new n8n integrations
- Monthly: Update MCP server library
- Quarterly: Performance optimization review
- Yearly: Major version upgrades

### Documentation
- User guide: "How to enable integrations"
- Developer guide: "Adding new MCP servers"
- Video tutorial: "From credential to execution in 60 seconds"
- API docs: All new endpoints

---

## Conclusion

PRD-20 transforms Automatos AI into the **most comprehensive AI orchestration platform** with:

✅ **Phase 1** (THIS WEEK):
- 30 MCP servers matching 30 credential types
- Pagination system (ready for 400+)
- Auto-activation (credential → MCP servers)
- Complete testing and validation

🚀 **Phase 2** (NEXT WEEK):
- ALL 400+ n8n credential types
- ALL 400+ matching MCP servers
- Advanced search and filtering
- Bulk operations
- Performance optimization

**The Result**: 
> "Users add an AWS credential, and **15 AWS integrations** instantly appear. Add GitHub token, get **8 GitHub tools**. Add Stripe key, get **payment processing**. One credential away from ANY integration."

**This is the platform differentiator. This is what makes Automatos AI unstoppable.** 🔥

---

**Status**: Ready for Phase 1 Implementation  
**Timeline**: Phase 1: 1-2 days | Phase 2: 3-4 days  
**Priority**: P0 - CRITICAL  

**LET'S FUCKING GO!** 🚀🚀🚀

