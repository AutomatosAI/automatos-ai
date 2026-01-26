# PRD-32: Automatos AI Widget Integration System

**Version:** 1.0  
**Status:** 🟡 Design Phase  
**Priority:** HIGH - Platform Expansion  
**Author:** Automatos AI Platform Team  
**Last Updated:** 2025-01-12  
**Dependencies:** PRD-31 (Chatbot UX), PRD-20 (MCP Integration), PRD-18 (Credential Management)

---

## Executive Summary

Transform Automatos AI from a standalone platform into an **embeddable AI service** that can be integrated into any SaaS application, website, or digital product. This PRD defines a comprehensive widget system that enables clients to embed Automatos AI capabilities with minimal code—from simple script tags to advanced React components.

### The Vision: "AI as a Service, Embedded Anywhere"

> **BudStack SaaS** adds one script tag → Tenants get AI-powered chatbots, SEO assistants, and content generators  
> **E-commerce Store** embeds widget → Customers get personalized product recommendations and support  
> **Admin Dashboard** adds context-aware assistant → Editors get AI help for SEO, blog posts, email templates  
> **Developer Portal** integrates code assistant → Developers get AI-powered debugging and code generation

### Current State → Target State

| Aspect | Current | After Implementation |
|--------|---------|---------------------|
| **Integration Method** | Manual API calls only | Script tag + NPM package + React components |
| **Widget Types** | 0 | 12 specialized widgets |
| **Context Awareness** | None | Full context injection (tenant, field, user) |
| **Multi-Tenant Support** | Manual | Built-in tenant isolation |
| **Fine-Tuned Models** | Not exposed | Per-widget model selection |
| **Deployment** | Single platform | Embeddable anywhere |

---

## 1) Problem Statement

### Current Pain

- **No embeddable solution**: Clients must build custom integrations from scratch
- **High integration barrier**: Requires API knowledge, streaming handling, error management
- **No context awareness**: Widgets can't understand what field/page they're helping with
- **Limited widget variety**: Only general chatbot available, no specialized assistants
- **No multi-tenant support**: Can't easily isolate data per tenant/client

### Impact

- **Lost opportunities**: Potential clients can't easily integrate Automatos AI
- **High support burden**: Clients need extensive help building integrations
- **Limited use cases**: Can't serve specialized needs (SEO, compliance, analytics)
- **Reduced market reach**: Can't serve non-technical clients (WordPress, Shopify stores)

---

## 2) Goals & Success Metrics

### Goals

- **G1 (Ease of Integration)**: Clients can embed Automatos AI with < 5 lines of code
- **G2 (Context Awareness)**: Widgets understand their environment (field type, tenant, user)
- **G3 (Widget Variety)**: 12 specialized widgets covering all major use cases
- **G4 (Multi-Tenant)**: Built-in tenant isolation and data separation
- **G5 (Model Flexibility)**: Support for fine-tuned models per widget type
- **G6 (Zero Maintenance)**: Widgets auto-update via CDN, no client code changes needed

### Success Metrics (MVP)

- **M1**: ≥ 90% of integrations use script tag (not custom code)
- **M2**: Average integration time < 15 minutes
- **M3**: Widget load time < 500ms (CDN cached)
- **M4**: Context-aware widgets show 40% better accuracy than generic chatbot
- **M5**: Multi-tenant isolation: 0 data leakage incidents
- **M6**: Support tickets for integration issues reduced by 80%

---

## 3) Personas & Core User Journeys

### Personas

- **P1: SaaS Founder (BudStack)**: Wants to offer AI features to tenants without building AI infrastructure
- **P2: E-commerce Store Owner**: Wants customer support chatbot and product description generator
- **P3: Content Editor**: Wants AI help for SEO, blog posts, email templates
- **P4: Developer**: Wants code assistant and API integration help
- **P5: Data Analyst**: Wants natural language database queries and analytics insights
- **P6: Compliance Officer**: Wants automated compliance checking and content moderation

### Core Journeys

#### J1: SaaS Integration (BudStack → Tenants)
1. BudStack admin creates Automatos account
2. Generates API keys for each tenant
3. Adds script tag to tenant admin panel: `<script data-automatos-tenantid="123">`
4. Tenants automatically get AI chatbot with their knowledge base
5. Tenants can enable additional widgets (SEO, analytics, etc.)

#### J2: Context-Aware Content Generation
1. Editor opens product description field
2. Clicks ✨ icon next to field
3. Widget opens with context: `{"field": "product_description", "product": "Tom Ford Bois Pacifique"}`
4. AI generates SEO-optimized description
5. Editor accepts/refines via chat

#### J3: Customer Support Chatbot
1. Customer visits e-commerce store
2. Chatbot widget loads with tenant knowledge base
3. Customer asks: "Do you ship to Canada?"
4. AI answers using tenant's shipping policy from knowledge base
5. Can escalate to human if needed

#### J4: Analytics Insights
1. Admin opens analytics dashboard
2. Asks: "Why did sales drop last week?"
3. Widget queries database using NL2SQL
4. Returns chart + explanation
5. Admin can drill down via follow-up questions

---

## 4) Widget Architecture

### 4.1 Widget Categories

#### Customer-Facing Widgets
- **Customer Chatbot** (#1): General support, FAQ, product info
- **Personalization** (#8): Dynamic content per user
- **Translation** (#9): Multi-language content

#### Admin/Editor Widgets
- **Context-Aware Assistant** (#2): Field-specific help (SEO, blog, email)
- **Workflow Automation** (#3): Trigger workflows from UI
- **Analytics Insights** (#4): Natural language analytics queries
- **Data Analysis** (#6): NL2SQL database queries
- **Content Moderation** (#7): Auto-moderate user content
- **Compliance Checker** (#10): Ensure regulatory compliance
- **A/B Test Generator** (#11): Generate test variants
- **SEO Analyzer** (#12): Real-time SEO feedback

#### Developer Widgets
- **Code Assistant** (#5): Help with code, templates, integrations

### 4.2 Widget Types

#### Type 1: Customer Chatbot Widget
**Purpose**: General customer support and FAQ

**Configuration**:
```html
<script src="https://cdn.automatos.app/widget.js"
        data-automatos-type="customer-chatbot"
        data-automatos-tenantid="123"
        data-automatos-api-key="tenant-api-key">
</script>
```

**Features**:
- Connects to tenant knowledge base
- RAG-powered answers
- Conversation history
- Escalation to human support
- Multi-language support

**Model Options**:
- Default: `ragib01/Qwen3-4B-customer-support` (fine-tuned for support)
- Fallback: GPT-4 for complex queries

**Use Cases**:
- E-commerce customer support
- SaaS tenant support
- FAQ automation
- Product information

---

#### Type 2: Context-Aware Assistant Widget
**Purpose**: Field-specific content generation and assistance

**Configuration**:
```html
<button data-automatos-type="context-assistant"
        data-automatos-field="seo"
        data-automatos-context='{"product":"Tom Ford Bois Pacifique","field":"meta_description"}'
        data-automatos-agent-id="12">
  ✨ Generate
</button>
```

**Features**:
- Understands field context (SEO, blog, email, product description)
- Pre-filled prompts based on field type
- Specialized agents per use case
- Inline generation + chat refinement
- Tone/style customization

**Field Types**:
- `seo`: Meta descriptions, titles, keywords
- `blog`: Blog post content, headlines
- `email`: Email subject lines, body content
- `product-description`: Product descriptions, features
- `social-media`: Social media posts, captions

**Model Options**:
- SEO: Fine-tuned SEO models (e.g., `SEO-Data-Dive-Data-Challenge-2025`)
- Blog: GPT-4 or Claude for long-form content
- Email: Fine-tuned email copy models
- Product: GPT-4 with product knowledge base

**Use Cases**:
- Shopify product description generation
- WordPress blog post writing
- Email marketing campaign content
- Social media content creation

---

#### Type 3: Workflow Automation Widget
**Purpose**: Trigger Automatos workflows from UI

**Configuration**:
```html
<button data-automatos-type="workflow"
        data-automatos-workflow-id="45"
        data-automatos-trigger="generate-report">
  Generate Monthly Report
</button>
```

**Features**:
- One-click workflow execution
- Progress tracking
- Result delivery (email, download, display)
- Scheduled workflows
- Workflow templates

**Use Cases**:
- Automated email campaigns
- Scheduled reports
- Data processing pipelines
- Content generation workflows

---

#### Type 4: Analytics Insights Widget
**Purpose**: Natural language analytics queries

**Configuration**:
```html
<div data-automatos-type="analytics"
     data-automatos-datasource="postgresql"
     data-automatos-schema='{"tables":["orders","products","customers"]}'>
  <input type="text" placeholder="Ask about your data...">
</div>
```

**Features**:
- NL2SQL conversion
- Interactive charts and tables
- Drill-down capabilities
- Export to CSV/PDF
- Scheduled insights

**Model Options**:
- NL2SQL: Specialized SQL generation models
- Analysis: GPT-4 for insights and explanations

**Use Cases**:
- Business intelligence dashboards
- Sales analytics
- Customer behavior analysis
- Performance metrics

---

#### Type 5: Code Assistant Widget
**Purpose**: Developer help for code, templates, integrations

**Configuration**:
```html
<div data-automatos-type="code-assistant"
     data-automatos-language="javascript"
     data-automatos-context='{"framework":"nextjs","api":"shopify"}'>
  <!-- Code editor integration -->
</div>
```

**Features**:
- Code generation and completion
- Error debugging
- API integration help
- Template generation
- Code review suggestions

**Model Options**:
- Code generation: CodeLlama, StarCoder
- Debugging: GPT-4 with code context
- Templates: Fine-tuned template models

**Use Cases**:
- Email template design
- Store template customization
- API integration
- Bug fixing

---

#### Type 6: Data Analysis Widget
**Purpose**: Natural language database queries

**Configuration**:
```html
<div data-automatos-type="data-analysis"
     data-automatos-datasource="postgresql"
     data-automatos-connection-string="encrypted">
  <!-- Query interface -->
</div>
```

**Features**:
- NL2SQL with schema awareness
- Query result visualization
- Data export
- Query history
- Safe query execution (read-only)

**Model Options**:
- NL2SQL: Specialized models for SQL generation
- Visualization: GPT-4 for chart recommendations

**Use Cases**:
- Tenant admin dashboards
- Business intelligence
- Data exploration
- Report generation

---

#### Type 7: Content Moderation Widget
**Purpose**: Auto-moderate user-generated content

**Configuration**:
```html
<div data-automatos-type="content-moderation"
     data-automatos-rules='{"strictness":"high","categories":["spam","hate","medical"]}'>
  <!-- Content input -->
</div>
```

**Features**:
- Real-time content scanning
- Category detection (spam, hate, inappropriate)
- Confidence scores
- Action recommendations (approve, reject, flag)
- Custom rule sets

**Model Options**:
- Moderation: Fine-tuned moderation models
- Medical compliance: Specialized medical content models

**Use Cases**:
- Medical cannabis storefronts (BudStack)
- Social media platforms
- Comment moderation
- Review moderation

---

#### Type 8: Personalization Widget
**Purpose**: Dynamic content per user

**Configuration**:
```html
<div data-automatos-type="personalization"
     data-automatos-user-id="user-123"
     data-automatos-scope="product-recommendations">
  <!-- Personalized content -->
</div>
```

**Features**:
- User profile analysis
- Behavioral targeting
- A/B testing integration
- Real-time personalization
- Performance tracking

**Model Options**:
- Personalization: `iioos/personalization-profile-model`
- Recommendations: GPT-4 with user history

**Use Cases**:
- Product recommendations
- Personalized landing pages
- Dynamic pricing
- Customized email content

---

#### Type 9: Translation Widget
**Purpose**: Multi-language content generation

**Configuration**:
```html
<div data-automatos-type="translation"
     data-automatos-source-lang="en"
     data-automatos-target-lang="es"
     data-automatos-content="Product description">
  <!-- Translation interface -->
</div>
```

**Features**:
- Multi-language support (50+ languages)
- Context-aware translation
- SEO-optimized translations
- Bulk translation
- Quality scoring

**Model Options**:
- Translation: `Llama2-13b-Language-translate`
- Quality: GPT-4 for review

**Use Cases**:
- E-commerce product descriptions
- Blog post translation
- Email campaign localization
- Support content translation

---

#### Type 10: Compliance Checker Widget
**Purpose**: Ensure regulatory compliance

**Configuration**:
```html
<div data-automatos-type="compliance"
     data-automatos-compliance-type="medical-cannabis"
     data-automatos-region="US">
  <!-- Content checker -->
</div>
```

**Features**:
- Regulatory rule checking
- Region-specific compliance
- Real-time validation
- Compliance reports
- Audit trails

**Model Options**:
- Compliance: Fine-tuned compliance models
- Medical: Specialized medical regulations models

**Use Cases**:
- Medical cannabis storefronts
- Healthcare content
- Financial services
- Legal content review

---

#### Type 11: A/B Test Generator Widget
**Purpose**: Generate test variants

**Configuration**:
```html
<button data-automatos-type="ab-test"
        data-automatos-test-type="email"
        data-automatos-base-content="Current email content"
        data-automatos-variants="3">
  Generate A/B Test Variants
</button>
```

**Features**:
- Multiple variant generation
- Performance prediction
- Statistical significance calculation
- Integration with A/B testing platforms
- Variant preview

**Model Options**:
- Variant generation: GPT-4 with A/B testing knowledge
- Performance: Specialized conversion models

**Use Cases**:
- Email campaign variants
- Landing page variations
- Ad copy testing
- CTA optimization

---

#### Type 12: SEO Analyzer Widget
**Purpose**: Real-time SEO feedback

**Configuration**:
```html
<div data-automatos-type="seo-analyzer"
     data-automatos-seo-context='{"page":"product","keyword":"fragrance"}'>
  <!-- SEO analysis -->
</div>
```

**Features**:
- On-page SEO scoring
- Keyword optimization
- Meta tag generation
- Content suggestions
- Competitor analysis

**Model Options**:
- SEO analysis: Fine-tuned SEO models
- Keyword research: GPT-4 with SEO knowledge base

**Use Cases**:
- Content optimization
- Product page SEO
- Blog post optimization
- Landing page SEO

---

## 5) Technical Architecture

### 5.1 Widget Delivery System

#### CDN Distribution
- **Primary CDN**: Cloudflare/CloudFront
- **Versioning**: Semantic versioning (`v1.0.0`, `v1.1.0`)
- **Caching**: Aggressive caching with cache-busting for updates
- **Load Time Target**: < 500ms

#### Script Tag Widget
```html
<!-- Basic customer chatbot -->
<script src="https://cdn.automatos.app/widget.js"
        data-automatos-type="customer-chatbot"
        data-automatos-tenantid="123"
        data-automatos-api-key="key-123">
</script>

<!-- Context-aware assistant -->
<button data-automatos-type="context-assistant"
        data-automatos-field="seo"
        data-automatos-context='{"product":"..."}'
        data-automatos-agent-id="12">
  ✨ Generate
</button>
```

#### NPM Package
```bash
npm install @automatos-ai/widgets
```

```typescript
import { CustomerChatbot, ContextAssistant } from '@automatos-ai/widgets'

<CustomerChatbot
  tenantId="123"
  apiKey="key-123"
/>

<ContextAssistant
  field="seo"
  context={{ product: "..." }}
  agentId={12}
/>
```

### 5.2 Context Injection System

#### Context Hierarchy
1. **Global Context**: Tenant ID, API key, environment
2. **Widget Context**: Widget type, field type, agent ID
3. **User Context**: User ID, role, preferences
4. **Page Context**: URL, page type, metadata
5. **Field Context**: Current value, field name, constraints

#### Context Payload Example
```json
{
  "global": {
    "tenantId": "123",
    "apiKey": "key-123",
    "environment": "production"
  },
  "widget": {
    "type": "context-assistant",
    "field": "seo",
    "agentId": 12
  },
  "user": {
    "id": "user-456",
    "role": "editor",
    "preferences": {}
  },
  "page": {
    "url": "/products/tom-ford-bois-pacifique",
    "type": "product",
    "metadata": {}
  },
  "field": {
    "name": "meta_description",
    "currentValue": "Luxury fragrance...",
    "constraints": {
      "maxLength": 160,
      "required": true
    }
  }
}
```

### 5.3 Multi-Tenant Isolation

#### Tenant Data Separation
- **API Key per Tenant**: Each tenant gets unique API key
- **Knowledge Base Isolation**: RAG queries scoped to tenant
- **Agent Assignment**: Tenants can have dedicated agents
- **Workflow Isolation**: Workflows scoped to tenant
- **Audit Logging**: All actions tagged with tenant ID

#### Tenant Configuration
```typescript
interface TenantConfig {
  tenantId: string
  apiKey: string
  defaultAgentId?: number
  knowledgeBaseIds: number[]
  enabledWidgets: WidgetType[]
  customModels?: {
    [widgetType: string]: string // Model name/ID
  }
}
```

### 5.4 Model Selection System

#### Model Mapping
```typescript
interface ModelMapping {
  widgetType: WidgetType
  defaultModel: string
  fineTunedModels?: {
    [useCase: string]: string
  }
  fallbackModel: string
}

const MODEL_MAPPINGS: ModelMapping[] = [
  {
    widgetType: 'customer-chatbot',
    defaultModel: 'ragib01/Qwen3-4B-customer-support',
    fallbackModel: 'gpt-4'
  },
  {
    widgetType: 'context-assistant',
    fineTunedModels: {
      'seo': 'SEO-Data-Dive-Data-Challenge-2025',
      'blog': 'gpt-4',
      'email': 'gpt-4'
    },
    defaultModel: 'gpt-4',
    fallbackModel: 'gpt-3.5-turbo'
  },
  {
    widgetType: 'personalization',
    defaultModel: 'iioos/personalization-profile-model',
    fallbackModel: 'gpt-4'
  },
  {
    widgetType: 'translation',
    defaultModel: 'Llama2-13b-Language-translate',
    fallbackModel: 'gpt-4'
  },
  {
    widgetType: 'analytics',
    defaultModel: 'nl2sql-specialized',
    fallbackModel: 'gpt-4'
  },
  {
    widgetType: 'data-analysis',
    defaultModel: 'nl2sql-specialized',
    fallbackModel: 'gpt-4'
  }
]
```

#### Model Selection Logic
1. Check widget-specific fine-tuned model (if use case specified)
2. Use default model for widget type
3. Fallback to GPT-4 if model unavailable
4. Allow tenant override via configuration

### 5.5 API Integration

#### Widget API Endpoint
```
POST /api/widgets/chat
Headers:
  X-API-Key: tenant-api-key
  X-Tenant-ID: tenant-id
  X-Widget-Type: widget-type
  X-Context: base64-encoded-json

Body:
{
  "message": "user message",
  "context": {
    "field": "seo",
    "agentId": 12,
    "metadata": {...}
  },
  "conversationId": "optional-id"
}
```

#### Streaming Response
- **Format**: AI SDK Data Stream (text/plain)
- **Chunks**: `0:chunk-text\n`
- **Tool Calls**: `tool-start`, `tool-end`, `tool-data`
- **Artifacts**: RAG sources, SQL results, charts

### 5.6 Security & Authentication

#### API Key Management
- **Per-Tenant Keys**: Unique API key per tenant
- **Key Rotation**: Support for key rotation
- **Rate Limiting**: Per-tenant rate limits
- **Usage Tracking**: Track API usage per tenant

#### Data Isolation
- **Tenant Scoping**: All queries scoped to tenant
- **Knowledge Base Filtering**: RAG queries filtered by tenant
- **Agent Isolation**: Agents can be tenant-specific
- **Audit Logging**: All actions logged with tenant ID

---

## 6) Implementation Phases

### Phase 1: Core Widget System (Week 1-2)
**Goal**: Basic script tag widget + customer chatbot

**Tasks**:
1. Create widget CDN infrastructure
2. Build script tag loader
3. Implement customer chatbot widget
4. Add context injection system
5. Create widget API endpoint
6. Add multi-tenant support
7. Build widget configuration system

**Deliverables**:
- Script tag widget loader
- Customer chatbot widget
- Widget API endpoint
- Basic context injection
- Multi-tenant isolation

---

### Phase 2: Context-Aware Widgets (Week 3-4)
**Goal**: Context-aware assistant + SEO analyzer

**Tasks**:
1. Build context-aware assistant widget
2. Implement field detection system
3. Add SEO analyzer widget
4. Create agent selection system
5. Add fine-tuned model support
6. Build widget configuration UI

**Deliverables**:
- Context-aware assistant widget
- SEO analyzer widget
- Agent/model selection system
- Widget configuration interface

---

### Phase 3: Specialized Widgets (Week 5-6)
**Goal**: Analytics, data analysis, workflow automation

**Tasks**:
1. Build analytics insights widget
2. Implement data analysis widget (NL2SQL)
3. Create workflow automation widget
4. Add personalization widget
5. Build translation widget
6. Add compliance checker widget

**Deliverables**:
- Analytics insights widget
- Data analysis widget
- Workflow automation widget
- Personalization widget
- Translation widget
- Compliance checker widget

---

### Phase 4: Advanced Widgets (Week 7-8)
**Goal**: Code assistant, content moderation, A/B testing

**Tasks**:
1. Build code assistant widget
2. Implement content moderation widget
3. Create A/B test generator widget
4. Add widget analytics dashboard
5. Build widget documentation
6. Create integration examples

**Deliverables**:
- Code assistant widget
- Content moderation widget
- A/B test generator widget
- Widget analytics dashboard
- Complete documentation
- Integration examples

---

### Phase 5: NPM Package & React Components (Week 9-10)
**Goal**: NPM package for technical clients

**Tasks**:
1. Create NPM package structure
2. Build React components for all widgets
3. Add TypeScript types
4. Create component documentation
5. Build example Next.js integration
6. Publish to NPM

**Deliverables**:
- `@automatos-ai/widgets` NPM package
- React components for all widgets
- TypeScript definitions
- Next.js integration example
- Component documentation

---

## 7) API Specifications

### 7.1 Widget Chat Endpoint

```
POST /api/widgets/chat
```

**Headers**:
- `X-API-Key`: Tenant API key (required)
- `X-Tenant-ID`: Tenant ID (required)
- `X-Widget-Type`: Widget type (required)
- `X-Context`: Base64-encoded context JSON (optional)

**Request Body**:
```json
{
  "message": "Generate SEO meta description",
  "context": {
    "field": "seo",
    "fieldType": "meta_description",
    "agentId": 12,
    "metadata": {
      "product": "Tom Ford Bois Pacifique",
      "currentValue": "Luxury fragrance...",
      "constraints": {
        "maxLength": 160
      }
    }
  },
  "conversationId": "optional-conversation-id"
}
```

**Response**: AI SDK Data Stream (text/plain)
```
0:Generated SEO meta description
0: for Tom Ford Bois Pacifique
0:...
```

---

### 7.2 Widget Configuration Endpoint

```
GET /api/widgets/config
```

**Headers**:
- `X-API-Key`: Tenant API key (required)
- `X-Tenant-ID`: Tenant ID (required)

**Response**:
```json
{
  "tenantId": "123",
  "enabledWidgets": [
    "customer-chatbot",
    "context-assistant",
    "seo-analyzer"
  ],
  "defaultAgentId": 1,
  "knowledgeBaseIds": [5, 6, 7],
  "customModels": {
    "customer-chatbot": "ragib01/Qwen3-4B-customer-support",
    "context-assistant": {
      "seo": "SEO-Data-Dive-Data-Challenge-2025"
    }
  }
}
```

---

### 7.3 Widget Analytics Endpoint

```
GET /api/widgets/analytics
```

**Headers**:
- `X-API-Key`: Tenant API key (required)
- `X-Tenant-ID`: Tenant ID (required)

**Query Parameters**:
- `startDate`: ISO date string
- `endDate`: ISO date string
- `widgetType`: Filter by widget type

**Response**:
```json
{
  "totalRequests": 1250,
  "widgets": {
    "customer-chatbot": {
      "requests": 800,
      "avgResponseTime": 1.2,
      "successRate": 0.98
    },
    "context-assistant": {
      "requests": 450,
      "avgResponseTime": 2.1,
      "successRate": 0.95
    }
  },
  "topAgents": [
    {"agentId": 12, "usage": 450},
    {"agentId": 1, "usage": 800}
  ]
}
```

---

## 8) UI/UX Design

### 8.1 Customer Chatbot Widget

**Layout**:
- Floating button (bottom-right corner)
- Expandable chat window
- Message bubbles (user/assistant)
- Typing indicator
- Send button + input field

**Features**:
- Smooth animations
- Mobile-responsive
- Dark/light theme support
- Customizable colors (tenant branding)

---

### 8.2 Context-Aware Assistant Widget

**Layout**:
- Small sparkle icon (✨) next to field
- Click opens modal/popup
- Modal contains:
  - Context display (field type, current value)
  - Chat interface
  - Generate button (quick action)
  - Accept/Reject buttons

**Features**:
- Pre-filled prompts based on field type
- Inline generation preview
- Chat for refinement
- One-click accept/reject

---

### 8.3 Analytics Insights Widget

**Layout**:
- Natural language input field
- Results display:
  - Chart/table visualization
  - SQL query (optional, for transparency)
  - Insights/explanation
  - Export buttons (CSV, PDF)

**Features**:
- Interactive charts
- Drill-down capabilities
- Query history
- Saved insights

---

## 9) Integration Examples

### 9.1 BudStack SaaS Integration

**Step 1**: BudStack admin creates Automatos account

**Step 2**: Generate API keys for tenants
```typescript
// BudStack backend
const apiKey = await automatosClient.createTenantApiKey({
  tenantId: tenant.id,
  enabledWidgets: ['customer-chatbot', 'context-assistant']
})
```

**Step 3**: Add script tag to tenant admin panel
```html
<script src="https://cdn.automatos.app/widget.js"
        data-automatos-type="customer-chatbot"
        data-automatos-tenantid="{{ tenant.id }}"
        data-automatos-api-key="{{ tenant.automatosApiKey }}">
</script>
```

**Step 4**: Add context-aware assistant to product fields
```html
<textarea name="product_description">
  <button data-automatos-type="context-assistant"
          data-automatos-field="product-description"
          data-automatos-context='{"product":"{{ product.name }}"}'
          data-automatos-agent-id="{{ tenant.seoAgentId }}">
    ✨ Generate
  </button>
</textarea>
```

---

### 9.2 E-commerce Store Integration

**Customer Chatbot**:
```html
<!-- Footer of store -->
<script src="https://cdn.automatos.app/widget.js"
        data-automatos-type="customer-chatbot"
        data-automatos-tenantid="store-123"
        data-automatos-api-key="store-api-key">
</script>
```

**Product Description Generator**:
```html
<!-- Admin product edit page -->
<div class="product-description-field">
  <textarea name="description"></textarea>
  <button data-automatos-type="context-assistant"
          data-automatos-field="product-description"
          data-automatos-context='{"product":"{{ product.name }}","category":"{{ product.category }}"}'>
    ✨ AI Generate
  </button>
</div>
```

---

### 9.3 WordPress Plugin Integration

**Plugin Code**:
```php
function automatos_widget_enqueue() {
    wp_enqueue_script(
        'automatos-widget',
        'https://cdn.automatos.app/widget.js',
        [],
        '1.0.0',
        true
    );
    
    wp_add_inline_script('automatos-widget', "
        window.automatosConfig = {
            type: 'customer-chatbot',
            tenantId: '" . get_option('automatos_tenant_id') . "',
            apiKey: '" . get_option('automatos_api_key') . "'
        };
    ");
}
add_action('wp_enqueue_scripts', 'automatos_widget_enqueue');
```

---

## 10) Security & Compliance

### 10.1 Data Isolation
- **Tenant Scoping**: All API calls scoped to tenant ID
- **Knowledge Base Filtering**: RAG queries filtered by tenant
- **Agent Isolation**: Agents can be tenant-specific or shared
- **Audit Logging**: All actions logged with tenant ID

### 10.2 API Key Security
- **Per-Tenant Keys**: Unique API key per tenant
- **Key Rotation**: Support for key rotation without downtime
- **Rate Limiting**: Per-tenant rate limits (configurable)
- **Usage Tracking**: Track API usage per tenant for billing

### 10.3 Content Security
- **Input Validation**: All inputs validated and sanitized
- **Output Filtering**: Sensitive data filtered from responses
- **XSS Prevention**: All outputs escaped
- **CSP Headers**: Content Security Policy for widgets

---

## 11) Performance & Scalability

### 11.1 Widget Load Performance
- **CDN Caching**: Aggressive caching (24h TTL)
- **Lazy Loading**: Widgets load on demand
- **Code Splitting**: Each widget type in separate bundle
- **Target**: < 500ms load time

### 11.2 API Performance
- **Streaming**: All responses streamed (no blocking)
- **Caching**: RAG results cached per tenant
- **Rate Limiting**: Per-tenant limits to prevent abuse
- **Target**: < 2s first token, < 5s full response

### 11.3 Scalability
- **Horizontal Scaling**: API can scale horizontally
- **Database Sharding**: Tenant data can be sharded
- **CDN Distribution**: Global CDN for widget delivery
- **Target**: Support 10,000+ tenants

---

## 12) Monitoring & Analytics

### 12.1 Widget Analytics
- **Usage Metrics**: Requests per widget type
- **Performance Metrics**: Response times, error rates
- **User Metrics**: Active users, conversations
- **Model Metrics**: Model usage, costs

### 12.2 Tenant Analytics
- **Per-Tenant Usage**: API calls, tokens used
- **Widget Adoption**: Which widgets each tenant uses
- **Performance**: Response times per tenant
- **Billing**: Usage-based billing data

### 12.3 Error Tracking
- **Error Logging**: All errors logged with context
- **Alerting**: Critical errors trigger alerts
- **Debugging**: Full request/response logging (optional)

---

## 13) Dependencies

### 13.1 Existing Systems
- **PRD-31 (Chatbot UX)**: Chat streaming, tool transparency
- **PRD-20 (MCP Integration)**: Agent system, tool execution
- **PRD-18 (Credential Management)**: API key management
- **PRD-08 (RAG)**: Knowledge base queries
- **PRD-21 (Database Knowledge)**: NL2SQL capabilities

### 13.2 New Requirements
- **CDN Infrastructure**: Cloudflare/CloudFront setup
- **Widget Build System**: Webpack/Rollup for bundling
- **Model Registry**: System to manage fine-tuned models
- **Tenant Management**: API for tenant configuration

---

## 14) Success Criteria

### Phase 1 (MVP)
- [ ] Script tag widget loads in < 500ms
- [ ] Customer chatbot works with tenant knowledge base
- [ ] Multi-tenant isolation verified (0 data leakage)
- [ ] Basic context injection working
- [ ] Widget API endpoint functional

### Phase 2 (Context-Aware)
- [ ] Context-aware assistant understands field types
- [ ] SEO analyzer provides real-time feedback
- [ ] Agent selection works correctly
- [ ] Fine-tuned models can be specified
- [ ] Widget configuration UI functional

### Phase 3 (Specialized Widgets)
- [ ] Analytics insights widget queries database
- [ ] Data analysis widget generates SQL correctly
- [ ] Workflow automation triggers workflows
- [ ] Personalization widget personalizes content
- [ ] Translation widget translates accurately

### Phase 4 (Advanced Widgets)
- [ ] Code assistant helps with code/templates
- [ ] Content moderation flags inappropriate content
- [ ] A/B test generator creates variants
- [ ] All widgets have analytics tracking
- [ ] Documentation complete

### Phase 5 (NPM Package)
- [ ] NPM package published
- [ ] React components work in Next.js
- [ ] TypeScript types complete
- [ ] Integration examples working
- [ ] Component documentation complete

---

## 15) Future Enhancements

### 15.1 Advanced Features
- **Custom Widgets**: Allow tenants to create custom widgets
- **Widget Marketplace**: Share widgets between tenants
- **White-Labeling**: Full branding customization
- **Multi-Language UI**: Widget UI in multiple languages
- **Voice Input**: Voice-to-text for widgets

### 15.2 Integration Enhancements
- **Webhook Support**: Widget events trigger webhooks
- **SSO Integration**: Single sign-on for widget access
- **API Gateway**: Unified API gateway for all widgets
- **GraphQL API**: GraphQL endpoint for advanced queries

---

## 16) Timeline & Resources

### Phase 1: Core Widget System (2 weeks)
- **Backend**: Widget API, context injection, multi-tenant
- **Frontend**: Script tag loader, customer chatbot widget
- **Infrastructure**: CDN setup, widget build system

### Phase 2: Context-Aware Widgets (2 weeks)
- **Backend**: Agent selection, model mapping
- **Frontend**: Context-aware assistant, SEO analyzer
- **Integration**: Field detection, context parsing

### Phase 3: Specialized Widgets (2 weeks)
- **Backend**: NL2SQL integration, workflow triggers
- **Frontend**: Analytics, data analysis, workflow widgets
- **Models**: Fine-tuned model integration

### Phase 4: Advanced Widgets (2 weeks)
- **Backend**: Code analysis, content moderation
- **Frontend**: Code assistant, moderation, A/B testing widgets
- **Analytics**: Widget analytics dashboard

### Phase 5: NPM Package (2 weeks)
- **Package**: NPM package structure
- **Components**: React components for all widgets
- **Documentation**: Complete documentation
- **Examples**: Integration examples

**Total Timeline**: 10 weeks (2.5 months)

---

## 17) Risk Mitigation

### 17.1 Technical Risks
- **CDN Outages**: Fallback CDN, local caching
- **API Overload**: Rate limiting, auto-scaling
- **Model Failures**: Fallback models, error handling
- **Security Breaches**: Regular audits, encryption

### 17.2 Business Risks
- **Low Adoption**: Marketing, documentation, examples
- **High Support Burden**: Self-service docs, video tutorials
- **Competition**: Unique features, better UX, lower cost

---

## 18) Conclusion

The Widget Integration System transforms Automatos AI from a standalone platform into an embeddable AI service that can be integrated into any application with minimal code. By providing 12 specialized widgets covering all major use cases, supporting fine-tuned models, and ensuring multi-tenant isolation, we enable clients to offer AI-powered features to their users without building AI infrastructure.

The script tag approach makes integration accessible to non-technical clients, while the NPM package provides flexibility for technical teams. This dual approach maximizes market reach and adoption.

---

**Next Steps**:
1. Review and approve PRD
2. Set up CDN infrastructure
3. Begin Phase 1 implementation
4. Create widget build system
5. Build customer chatbot widget (MVP)
