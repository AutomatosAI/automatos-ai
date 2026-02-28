# PRD-38.5: Widget Marketplace

**Version:** 1.0
**Status:** 🟡 Future Planning
**Priority:** MEDIUM - Ecosystem Growth
**Author:** Automatos AI Platform Team
**Last Updated:** 2026-01-27
**Dependencies:** PRD-38.1-38.4
**Timeline:** Weeks 11-14

---

## Executive Summary

Create a **Widget Marketplace** where developers can build, publish, and monetize custom widgets. This transforms Automatos from a product into a **platform with an ecosystem**, enabling:

- **Developers**: Build and sell specialized widgets
- **Users**: Discover and install widgets for their needs
- **Automatos**: Platform revenue through marketplace fees

### The Vision

```
┌─────────────────────────────────────────────────────────────────────┐
│                     AUTOMATOS WIDGET MARKETPLACE                    │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  🔥 Featured                                                        │
│  ┌────────────┐ ┌────────────┐ ┌────────────┐ ┌────────────┐      │
│  │ 📊        │ │ 🎨        │ │ 📈        │ │ 🤖        │      │
│  │ Salesforce│ │ Canva     │ │ Analytics │ │ GPT-4     │      │
│  │ CRM       │ │ Designer  │ │ Pro       │ │ Assistant │      │
│  │           │ │           │ │           │ │           │      │
│  │ ⭐ 4.9    │ │ ⭐ 4.8    │ │ ⭐ 4.7    │ │ ⭐ 4.9    │      │
│  │ Free      │ │ $9/mo     │ │ $19/mo    │ │ Free      │      │
│  └────────────┘ └────────────┘ └────────────┘ └────────────┘      │
│                                                                     │
│  📁 Categories                                                      │
│  [CRM] [Analytics] [Design] [Communication] [DevTools] [AI/ML]     │
│                                                                     │
│  🆕 New Releases                    📈 Most Popular                 │
│  • Stripe Dashboard Widget         • Gmail Widget                   │
│  • Notion Integration              • Slack Notifier                 │
│  • HubSpot CRM Widget              • Chart Builder Pro              │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 1) Goals & Success Metrics

### Goals

| ID | Goal | Description |
|----|------|-------------|
| G1 | **Developer Platform** | Enable third-party widget development |
| G2 | **Discoverability** | Easy to find relevant widgets |
| G3 | **Monetization** | Revenue sharing for widget creators |
| G4 | **Quality** | Review process ensures widget quality |
| G5 | **Trust** | Security review and sandboxing |

### Success Metrics

| Metric | Target (Year 1) | Measurement |
|--------|-----------------|-------------|
| Widgets published | 50+ | Marketplace count |
| Developer signups | 200+ | Registration |
| Widget installs | 10,000+ | Install events |
| Revenue (marketplace) | $50,000+ | Payment tracking |
| Average rating | 4.0+ | Review aggregation |

---

## 2) Feature Specifications

### 2.1 Widget Builder

**Visual Widget Builder:**
```
┌─────────────────────────────────────────────────────────────────────┐
│ Widget Builder: "My Custom Widget"                    [Save] [Test] │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│ COMPONENTS                 │ CANVAS                   │ PROPERTIES  │
│ ┌────────────────────────┐ │ ┌─────────────────────┐ │ ┌─────────┐ │
│ │ 📝 Text                │ │ │                     │ │ │ Name:   │ │
│ │ 🔘 Button              │ │ │   Drop components   │ │ │ [    ]  │ │
│ │ 📊 Chart               │ │ │       here          │ │ │         │ │
│ │ 📋 Table               │ │ │                     │ │ │ Size:   │ │
│ │ 📥 Input               │ │ │                     │ │ │ [W] [H] │ │
│ │ 🖼️ Image               │ │ └─────────────────────┘ │ │         │ │
│ │ ─────────────────────  │ │                         │ │ Data:   │ │
│ │ 🔌 Data Source         │ │ DATA BINDINGS           │ │ [API ▾] │ │
│ │ 🔄 API Connector       │ │ • chart → query result  │ │         │ │
│ │ 📡 Webhook             │ │ • title → response.name │ │         │ │
│ └────────────────────────┘ │                         │ └─────────┘ │
│                            │                         │             │
└─────────────────────────────────────────────────────────────────────┘
```

**Code-Based Development:**
```typescript
// Widget Development Kit (WDK)
import { defineWidget, useWidgetData, WidgetContainer } from '@automatos-ai/wdk'

export default defineWidget({
  name: 'my-custom-widget',
  displayName: 'My Custom Widget',
  version: '1.0.0',
  description: 'A custom widget that does X',
  icon: 'chart-bar',

  // Configuration schema
  configSchema: {
    apiEndpoint: { type: 'string', required: true },
    refreshInterval: { type: 'number', default: 60 },
    theme: { type: 'enum', options: ['light', 'dark'] },
  },

  // Permissions needed
  permissions: ['network', 'storage'],

  // Widget component
  component: MyWidget,
})

function MyWidget({ config }) {
  const { data, isLoading, error, refresh } = useWidgetData({
    source: config.apiEndpoint,
    refreshInterval: config.refreshInterval,
  })

  return (
    <WidgetContainer>
      {isLoading && <Loading />}
      {error && <Error message={error.message} />}
      {data && <MyChart data={data} theme={config.theme} />}
    </WidgetContainer>
  )
}
```

### 2.2 Widget Marketplace UI

**Marketplace Pages:**
```
frontend/app/marketplace/
├── page.tsx                    # Marketplace home
├── [category]/
│   └── page.tsx               # Category listing
├── widget/[id]/
│   ├── page.tsx               # Widget detail
│   └── reviews/page.tsx       # Reviews
├── publish/
│   └── page.tsx               # Publish wizard
└── developer/
    ├── page.tsx               # Developer dashboard
    └── widgets/[id]/
        └── page.tsx           # Widget analytics
```

**Widget Detail Page:**
```
┌─────────────────────────────────────────────────────────────────────┐
│ ← Back to Marketplace                                               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│ ┌──────────┐  📊 Analytics Pro Dashboard                           │
│ │          │                                                        │
│ │   ICON   │  By DataViz Studios                                   │
│ │          │  ⭐⭐⭐⭐⭐ 4.8 (234 reviews) • 5,678 installs           │
│ └──────────┘                                                        │
│                                                                     │
│ [Install - $19/month]  [Try Demo]  [♡ Save]  [Share]               │
│                                                                     │
├─────────────────────────────────────────────────────────────────────┤
│ [Overview] [Screenshots] [Reviews] [Changelog] [Support]            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│ ## Overview                                                         │
│                                                                     │
│ Transform your data into beautiful, interactive dashboards with     │
│ Analytics Pro. Features include:                                    │
│                                                                     │
│ • 15+ chart types (bar, line, pie, scatter, heatmap, etc.)        │
│ • Real-time data updates                                           │
│ • Custom color themes                                              │
│ • Export to PDF, PNG, CSV                                          │
│ • SQL and API data sources                                         │
│                                                                     │
│ ## Screenshots                                                      │
│ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐                   │
│ │         │ │         │ │         │ │         │                   │
│ └─────────┘ └─────────┘ └─────────┘ └─────────┘                   │
│                                                                     │
│ ## Requirements                                                     │
│ • Automatos Pro plan or higher                                     │
│ • Database connection for SQL queries                              │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 2.3 Review & Publishing Process

**Publishing Workflow:**
```
Developer submits widget
        │
        ▼
┌───────────────┐
│ Automated     │ • Build verification
│ Checks        │ • Security scan
│               │ • Bundle size check
└───────┬───────┘
        │ Pass
        ▼
┌───────────────┐
│ Manual        │ • Code review (paid widgets)
│ Review        │ • UX review
│               │ • Policy compliance
└───────┬───────┘
        │ Approved
        ▼
┌───────────────┐
│ Published     │ • Listed in marketplace
│               │ • Available for install
└───────────────┘
```

**Automated Checks:**
```typescript
interface AutomatedChecks {
  // Security
  noEval: boolean           // No eval() or new Function()
  noInlineScripts: boolean  // CSP compliant
  sandboxed: boolean        // Runs in iframe sandbox
  noSensitiveData: boolean  // No hardcoded secrets

  // Performance
  bundleSize: number        // Max 500KB
  loadTime: number          // Max 2 seconds
  memoryUsage: number       // Max 50MB

  // Quality
  hasReadme: boolean
  hasChangelog: boolean
  hasScreenshots: boolean
  typesCoverage: number     // Min 80%
  testCoverage: number      // Min 60%
}
```

### 2.4 Monetization

**Revenue Model:**
```
┌─────────────────────────────────────────────────────────────────────┐
│                      WIDGET PRICING OPTIONS                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  FREE               │  PAID (ONE-TIME)      │  SUBSCRIPTION         │
│  ──────────────     │  ──────────────────   │  ────────────────     │
│  • Open source      │  • Single purchase    │  • Monthly/yearly     │
│  • Community        │  • Lifetime access    │  • Recurring revenue  │
│  • No revenue       │  • Support optional   │  • Updates included   │
│  • Attribution      │                       │  • Priority support   │
│                     │                       │                       │
│  Platform fee: 0%   │  Platform fee: 15%    │  Platform fee: 20%    │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

**Payout Schedule:**
```typescript
interface PayoutSettings {
  // Thresholds
  minimumPayout: 50  // USD
  payoutFrequency: 'monthly' | 'weekly'

  // Platform fees
  freeWidgetFee: 0
  paidWidgetFee: 0.15      // 15%
  subscriptionFee: 0.20    // 20%

  // Payment methods
  supportedMethods: ['stripe', 'paypal', 'bank_transfer']

  // Timing
  holdPeriod: 14  // Days before payout eligible
  payoutDay: 1    // Day of month
}
```

### 2.5 Developer Dashboard

```
┌─────────────────────────────────────────────────────────────────────┐
│ Developer Dashboard                                    [Docs] [API] │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│ OVERVIEW                                              Jan 2026      │
│ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐│
│ │ 3 Widgets    │ │ 5,678        │ │ $2,345       │ │ 4.7 ⭐       ││
│ │ Published    │ │ Total Users  │ │ This Month   │ │ Avg Rating   ││
│ └──────────────┘ └──────────────┘ └──────────────┘ └──────────────┘│
│                                                                     │
│ YOUR WIDGETS                                                        │
│ ┌───────────────────────────────────────────────────────────────┐  │
│ │ Widget          │ Users  │ Revenue │ Rating │ Status    │ ⋮   │  │
│ ├───────────────────────────────────────────────────────────────┤  │
│ │ Analytics Pro   │ 3,456  │ $1,890  │ 4.8 ⭐  │ Published │ ⋮   │  │
│ │ Chart Builder   │ 1,234  │ $455    │ 4.6 ⭐  │ Published │ ⋮   │  │
│ │ Data Exporter   │ 988    │ Free    │ 4.5 ⭐  │ Published │ ⋮   │  │
│ │ New Widget      │ -      │ -       │ -      │ Draft     │ ⋮   │  │
│ └───────────────────────────────────────────────────────────────┘  │
│                                                                     │
│ EARNINGS                        │ USER FEEDBACK                     │
│ ┌─────────────────────────────┐ │ ┌─────────────────────────────┐ │
│ │     [Revenue Chart]         │ │ │ "Great widget, works..."    │ │
│ │                             │ │ │ "Love the customization..."  │ │
│ │                             │ │ │ "Could use more chart..."    │ │
│ └─────────────────────────────┘ │ └─────────────────────────────┘ │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 3) Technical Architecture

### 3.1 Widget Sandbox

Widgets run in an isolated environment for security:

```typescript
// Widget isolation architecture
interface WidgetSandbox {
  // Execution context
  iframe: HTMLIFrameElement  // Sandboxed iframe
  worker?: Worker            // Optional web worker

  // Communication
  postMessage: (message: WidgetMessage) => void
  onMessage: (handler: (message: WidgetMessage) => void) => void

  // Permissions
  permissions: WidgetPermission[]

  // Resource limits
  limits: {
    memory: number     // Max heap size
    cpu: number        // Max CPU time per frame
    network: number    // Max requests per minute
    storage: number    // Max localStorage
  }
}

// Sandbox attributes
const SANDBOX_ATTRS = [
  'allow-scripts',
  'allow-forms',
  // NO: allow-same-origin (prevents parent access)
  // NO: allow-top-navigation
  // NO: allow-popups
]
```

### 3.2 Widget Communication

```typescript
// Widget ↔ Platform communication
type WidgetMessage =
  | { type: 'ready' }
  | { type: 'resize'; height: number }
  | { type: 'request'; id: string; method: string; params: any }
  | { type: 'response'; id: string; result: any; error?: string }
  | { type: 'event'; name: string; data: any }

// Platform API available to widgets
interface WidgetPlatformAPI {
  // Data access (via postMessage)
  chat: {
    send: (message: string) => Promise<Response>
    onMessage: (handler: (message: Message) => void) => void
  }

  documents: {
    search: (query: string) => Promise<Document[]>
  }

  data: {
    query: (question: string) => Promise<QueryResult>
  }

  // Storage (scoped to widget)
  storage: {
    get: (key: string) => Promise<any>
    set: (key: string, value: any) => Promise<void>
    delete: (key: string) => Promise<void>
  }

  // UI
  ui: {
    showToast: (message: string, type: 'info' | 'error' | 'success') => void
    showModal: (content: React.ReactNode) => Promise<any>
    resize: (height: number) => void
  }

  // Context
  context: {
    workspaceId: string
    userId: string
    theme: 'light' | 'dark'
    locale: string
  }
}
```

### 3.3 Database Schema

```sql
-- Widgets table
CREATE TABLE marketplace_widgets (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  name VARCHAR(100) NOT NULL UNIQUE,
  display_name VARCHAR(200) NOT NULL,
  description TEXT,
  long_description TEXT,

  -- Developer
  developer_id UUID NOT NULL REFERENCES users(id),
  developer_name VARCHAR(200),

  -- Versioning
  version VARCHAR(20) NOT NULL,
  changelog TEXT,

  -- Pricing
  pricing_type VARCHAR(20) NOT NULL,  -- 'free', 'one_time', 'subscription'
  price_cents INTEGER,
  currency VARCHAR(3) DEFAULT 'USD',

  -- Content
  icon_url TEXT,
  screenshots JSONB DEFAULT '[]',
  readme TEXT,
  keywords TEXT[],
  categories TEXT[],

  -- Technical
  bundle_url TEXT NOT NULL,
  bundle_size INTEGER,
  permissions TEXT[],
  min_plan VARCHAR(50),

  -- Stats
  install_count INTEGER DEFAULT 0,
  rating_average DECIMAL(3,2) DEFAULT 0,
  rating_count INTEGER DEFAULT 0,

  -- Status
  status VARCHAR(20) DEFAULT 'draft',  -- draft, review, published, suspended
  published_at TIMESTAMP WITH TIME ZONE,

  -- Timestamps
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Widget installations
CREATE TABLE widget_installations (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  widget_id UUID NOT NULL REFERENCES marketplace_widgets(id),
  workspace_id UUID NOT NULL,
  user_id UUID NOT NULL REFERENCES users(id),

  -- Subscription (if applicable)
  subscription_id VARCHAR(255),
  subscription_status VARCHAR(20),

  -- Usage
  last_used_at TIMESTAMP WITH TIME ZONE,
  use_count INTEGER DEFAULT 0,

  -- Timestamps
  installed_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  uninstalled_at TIMESTAMP WITH TIME ZONE
);

-- Widget reviews
CREATE TABLE widget_reviews (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  widget_id UUID NOT NULL REFERENCES marketplace_widgets(id),
  user_id UUID NOT NULL REFERENCES users(id),

  rating INTEGER NOT NULL CHECK (rating >= 1 AND rating <= 5),
  title VARCHAR(200),
  body TEXT,

  -- Moderation
  is_verified_purchase BOOLEAN DEFAULT FALSE,
  is_featured BOOLEAN DEFAULT FALSE,
  status VARCHAR(20) DEFAULT 'published',

  -- Timestamps
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

  UNIQUE(widget_id, user_id)
);

-- Developer payouts
CREATE TABLE developer_payouts (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  developer_id UUID NOT NULL REFERENCES users(id),

  amount_cents INTEGER NOT NULL,
  currency VARCHAR(3) DEFAULT 'USD',
  platform_fee_cents INTEGER NOT NULL,
  net_amount_cents INTEGER NOT NULL,

  status VARCHAR(20) DEFAULT 'pending',  -- pending, processing, completed, failed
  payout_method VARCHAR(50),
  payout_reference VARCHAR(255),

  period_start DATE NOT NULL,
  period_end DATE NOT NULL,

  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  completed_at TIMESTAMP WITH TIME ZONE
);
```

---

## 4) Implementation Plan

### Week 11: Widget Builder

| Day | Task |
|-----|------|
| 1-2 | Widget Development Kit (WDK) |
| 3-4 | Visual widget builder (drag-drop) |
| 5 | Widget preview and testing |

### Week 12: Marketplace Backend

| Day | Task |
|-----|------|
| 1-2 | Database schema and models |
| 3 | Widget submission API |
| 4 | Review workflow |
| 5 | Installation management |

### Week 13: Marketplace Frontend

| Day | Task |
|-----|------|
| 1-2 | Marketplace home and browse |
| 3 | Widget detail page |
| 4 | Reviews and ratings |
| 5 | Developer dashboard |

### Week 14: Monetization + Polish

| Day | Task |
|-----|------|
| 1-2 | Stripe integration |
| 3 | Payout system |
| 4 | Analytics and reporting |
| 5 | Testing and launch |

---

## 5) API Endpoints

```
# Marketplace
GET    /api/marketplace/widgets              # List widgets
GET    /api/marketplace/widgets/:id          # Widget details
GET    /api/marketplace/widgets/:id/reviews  # Widget reviews
GET    /api/marketplace/categories           # List categories
GET    /api/marketplace/featured             # Featured widgets

# Installation
POST   /api/marketplace/widgets/:id/install   # Install widget
DELETE /api/marketplace/widgets/:id/install   # Uninstall
GET    /api/marketplace/installed             # My installed widgets

# Publishing
POST   /api/marketplace/widgets              # Submit widget
PUT    /api/marketplace/widgets/:id          # Update widget
POST   /api/marketplace/widgets/:id/submit   # Submit for review
GET    /api/marketplace/widgets/:id/status   # Review status

# Reviews
POST   /api/marketplace/widgets/:id/reviews  # Create review
PUT    /api/marketplace/reviews/:id          # Update review
DELETE /api/marketplace/reviews/:id          # Delete review

# Developer
GET    /api/marketplace/developer/widgets    # My widgets
GET    /api/marketplace/developer/analytics  # Analytics
GET    /api/marketplace/developer/earnings   # Earnings
POST   /api/marketplace/developer/payout     # Request payout
```

---

## 6) Files to Create

```
frontend/
├── app/marketplace/
│   ├── page.tsx
│   ├── [category]/page.tsx
│   ├── widget/[id]/page.tsx
│   ├── publish/page.tsx
│   └── developer/
│       ├── page.tsx
│       └── widgets/[id]/page.tsx
│
├── components/marketplace/
│   ├── WidgetCard.tsx
│   ├── WidgetGrid.tsx
│   ├── WidgetDetail.tsx
│   ├── ReviewList.tsx
│   ├── ReviewForm.tsx
│   ├── PublishWizard.tsx
│   ├── DeveloperDashboard.tsx
│   └── EarningsChart.tsx
│
└── lib/marketplace/
    ├── api.ts
    └── types.ts

backend/
├── api/marketplace/
│   ├── router.py
│   ├── widgets.py
│   ├── reviews.py
│   ├── installations.py
│   └── developer.py
│
├── models/
│   ├── marketplace_widget.py
│   ├── widget_installation.py
│   ├── widget_review.py
│   └── developer_payout.py
│
└── services/
    ├── widget_review.py
    ├── widget_sandbox.py
    └── stripe_integration.py

packages/
└── wdk/                        # Widget Development Kit
    ├── src/
    │   ├── index.ts
    │   ├── defineWidget.ts
    │   ├── hooks/
    │   ├── components/
    │   └── types.ts
    └── package.json
```

---

## 7) Testing Checklist

### Widget Builder
- [ ] Visual builder creates valid widgets
- [ ] Code-based development works
- [ ] Preview renders correctly
- [ ] Build process succeeds

### Marketplace
- [ ] Browse and search widgets
- [ ] Filter by category
- [ ] View widget details
- [ ] Read reviews

### Installation
- [ ] Install free widget
- [ ] Purchase paid widget
- [ ] Subscribe to widget
- [ ] Uninstall widget

### Publishing
- [ ] Submit widget for review
- [ ] Automated checks pass
- [ ] Manual review workflow
- [ ] Published widget appears

### Monetization
- [ ] Stripe checkout works
- [ ] Subscription billing
- [ ] Payout calculation correct
- [ ] Payout processing works

---

## 8) Success Criteria

Phase 5 is complete when:

1. [ ] Widget Development Kit published
2. [ ] Visual builder functional
3. [ ] Marketplace UI complete
4. [ ] Review workflow working
5. [ ] Stripe integration live
6. [ ] Payout system tested
7. [ ] 10+ widgets published (internal)
8. [ ] Documentation complete
9. [ ] Security audit passed

---

## 9) Future Enhancements

### Phase 5.1: Advanced Features
- Widget analytics dashboard
- A/B testing for widgets
- Widget templates library
- Team/organization widgets

### Phase 5.2: Ecosystem Growth
- Widget certification program
- Developer rewards program
- Widget hackathons
- Partner integrations

### Phase 5.3: Enterprise
- Private marketplace (on-prem)
- Custom widget development service
- Enterprise widget licensing
- White-label marketplace

---

## 10) References

- [Shopify App Store](https://apps.shopify.com/)
- [Figma Plugins](https://www.figma.com/community/plugins)
- [VS Code Marketplace](https://marketplace.visualstudio.com/)
- [Chrome Web Store](https://chrome.google.com/webstore)
- [Stripe Connect](https://stripe.com/connect)
- PRD-32: Widget Integration System
- PRD-38.4: SDK Foundation

---

*Document Version: 1.0*
*Created: 2026-01-27*
*Estimated Implementation: 4 weeks*
