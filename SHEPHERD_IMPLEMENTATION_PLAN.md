# Shepherd.js First-Login Onboarding Implementation Plan

## Scope
Single comprehensive tour triggered **once** on first login, with easy dismissal. ChatWidget handles all ongoing help.

## User Journey
```
User logs in for first time
  → Welcome modal appears
    → [Skip Tour] → Set flag, go to /chat
    → [Start Tour] → Begin guided walkthrough
      → Step 1: Welcome to Automatos
      → Step 2: Navigation sidebar
      → Step 3: Create your first agent
      → Step 4: Connect to email (Composio)
      → Step 5: Test your agent
      → Step 6: ChatWidget for help
      → Complete → Set completion flag
```

## Implementation Steps

### 1. Installation
```bash
cd frontend
npm install shepherd.js react-shepherd
```

### 2. File Structure
```
frontend/
├── lib/
│   └── shepherd/
│       ├── first-login-tour.ts       # Tour definition
│       ├── shepherd-theme.ts         # Dark glass styling
│       └── tour-storage.ts           # LocalStorage helpers
├── components/
│   └── onboarding/
│       ├── first-login-guard.tsx     # Checks if tour needed
│       └── welcome-modal.tsx         # "Welcome! [Skip] [Start Tour]"
└── styles/
    └── shepherd-custom.css           # Automatos theme overrides
```

### 3. Core Tour Definition

**File**: `frontend/lib/shepherd/first-login-tour.ts`

```typescript
import Shepherd from 'shepherd.js'
import { shepherdTheme } from './shepherd-theme'

export function createFirstLoginTour() {
  const tour = new Shepherd.Tour({
    ...shepherdTheme,
    exitOnEsc: true,
    keyboardNavigation: true,
  })

  // Step 1: Welcome
  tour.addStep({
    id: 'welcome',
    title: 'Welcome to Automatos AI',
    text: `
      <p class="text-gray-300 mb-3">
        Let's get you started! This quick tour will show you how to:
      </p>
      <ul class="text-gray-400 text-sm space-y-1 list-disc list-inside">
        <li>Navigate the platform</li>
        <li>Create your first AI agent</li>
        <li>Connect to your email</li>
        <li>Get help when you need it</li>
      </ul>
      <p class="text-xs text-gray-500 mt-4">
        You can skip this anytime by pressing ESC
      </p>
    `,
    buttons: [
      {
        text: 'Skip Tour',
        classes: 'shepherd-button-secondary',
        action: () => {
          tour.complete()
          markTourSkipped()
        }
      },
      {
        text: 'Let\'s Go!',
        action: tour.next,
      }
    ],
  })

  // Step 2: Sidebar Navigation
  tour.addStep({
    id: 'navigation',
    title: 'Your Navigation Hub',
    text: `
      <p class="text-gray-300">
        The sidebar gives you access to all major features:
      </p>
      <ul class="text-sm text-gray-400 mt-2 space-y-1">
        <li><strong>Chat:</strong> Talk to your AI assistants</li>
        <li><strong>Agents:</strong> Create and manage AI workers</li>
        <li><strong>Workflows:</strong> Automate complex tasks</li>
        <li><strong>Tools:</strong> Connect to external services</li>
      </ul>
    `,
    attachTo: {
      element: '[data-tour="sidebar"]',
      on: 'right'
    },
    buttons: [
      { text: 'Back', action: tour.back },
      { text: 'Next', action: tour.next }
    ],
  })

  // Step 3: Navigate to Agents
  tour.addStep({
    id: 'go-to-agents',
    title: 'Create Your First Agent',
    text: `
      <p class="text-gray-300">
        Let's create an AI agent to handle your emails.
        Click on <strong>Agents</strong> in the sidebar.
      </p>
    `,
    attachTo: {
      element: '[data-tour="nav-agents"]',
      on: 'right'
    },
    advanceOn: {
      selector: '[data-tour="nav-agents"]',
      event: 'click'
    },
    buttons: [
      { text: 'Back', action: tour.back }
    ],
  })

  // Step 4: Create Agent Button (waits for page load)
  tour.addStep({
    id: 'create-agent-btn',
    title: 'Start Creating',
    text: `
      <p class="text-gray-300">
        Click <strong>Create Agent</strong> to open the agent builder.
      </p>
    `,
    beforeShowPromise: () => waitForElement('[data-tour="create-agent-btn"]'),
    attachTo: {
      element: '[data-tour="create-agent-btn"]',
      on: 'bottom'
    },
    advanceOn: {
      selector: '[data-tour="create-agent-btn"]',
      event: 'click'
    },
    buttons: [
      { text: 'Back', action: tour.back }
    ],
  })

  // Step 5: Agent Name (in modal)
  tour.addStep({
    id: 'agent-name',
    title: 'Name Your Agent',
    text: `
      <p class="text-gray-300 mb-2">
        Give your agent a descriptive name, like:
      </p>
      <code class="text-sm bg-gray-800 px-2 py-1 rounded">
        Email Assistant
      </code>
    `,
    beforeShowPromise: () => waitForElement('[data-tour="agent-name-input"]'),
    attachTo: {
      element: '[data-tour="agent-name-input"]',
      on: 'right'
    },
    buttons: [
      { text: 'Next', action: tour.next }
    ],
  })

  // Step 6: Agent Description
  tour.addStep({
    id: 'agent-description',
    title: 'Describe Its Purpose',
    text: `
      <p class="text-gray-300 mb-2">
        Tell the agent what it should do:
      </p>
      <code class="text-xs bg-gray-800 px-2 py-1 rounded block">
        "Monitor my inbox, categorize emails, and draft replies to common questions."
      </code>
    `,
    attachTo: {
      element: '[data-tour="agent-description-input"]',
      on: 'right'
    },
    buttons: [
      { text: 'Back', action: tour.back },
      { text: 'Next', action: tour.next }
    ],
  })

  // Step 7: Select Tools/Skills
  tour.addStep({
    id: 'agent-tools',
    title: 'Connect to Email',
    text: `
      <p class="text-gray-300 mb-2">
        In the <strong>Tools & Integrations</strong> section, search for and enable:
      </p>
      <ul class="text-sm text-gray-400 space-y-1">
        <li>• <strong>Gmail</strong> (for Google)</li>
        <li>• <strong>Outlook</strong> (for Microsoft)</li>
        <li>• <strong>IMAP</strong> (for other providers)</li>
      </ul>
      <p class="text-xs text-gray-500 mt-3">
        You'll authenticate with your email provider after creating the agent.
      </p>
    `,
    attachTo: {
      element: '[data-tour="agent-tools-section"]',
      on: 'left'
    },
    buttons: [
      { text: 'Back', action: tour.back },
      { text: 'Next', action: tour.next }
    ],
  })

  // Step 8: Save Agent
  tour.addStep({
    id: 'save-agent',
    title: 'Save Your Agent',
    text: `
      <p class="text-gray-300">
        Click <strong>Create Agent</strong> to finalize.
        Then you'll be able to configure the email connection.
      </p>
    `,
    attachTo: {
      element: '[data-tour="save-agent-btn"]',
      on: 'top'
    },
    advanceOn: {
      selector: '[data-tour="save-agent-btn"]',
      event: 'click'
    },
    buttons: [
      { text: 'Back', action: tour.back }
    ],
  })

  // Step 9: Agent Created - Next Steps
  tour.addStep({
    id: 'agent-created',
    title: 'Agent Created! 🎉',
    text: `
      <p class="text-gray-300 mb-3">
        Great! Your agent is ready. Next steps:
      </p>
      <ol class="text-sm text-gray-400 space-y-2 list-decimal list-inside">
        <li>Click on your agent to configure email access</li>
        <li>Authenticate with Gmail/Outlook via Composio</li>
        <li>Test it by asking "Summarize my recent emails"</li>
      </ol>
    `,
    beforeShowPromise: () => waitForElement('[data-tour="agent-roster"]'),
    attachTo: {
      element: '[data-tour="agent-roster"]',
      on: 'top'
    },
    buttons: [
      { text: 'Next', action: tour.next }
    ],
  })

  // Step 10: ChatWidget - Always Available Help
  tour.addStep({
    id: 'chat-widget',
    title: 'Need Help? Use the Chat',
    text: `
      <p class="text-gray-300 mb-3">
        This floating chat widget is always available.
        Ask questions like:
      </p>
      <ul class="text-sm text-gray-400 space-y-1">
        <li>• "How do I connect my Gmail?"</li>
        <li>• "Show me email automation examples"</li>
        <li>• "What tools can I integrate?"</li>
      </ul>
      <p class="text-xs text-gray-500 mt-3">
        It's powered by AI and knows about all Automatos features.
      </p>
    `,
    attachTo: {
      element: '[data-tour="chat-widget"]',
      on: 'left'
    },
    buttons: [
      { text: 'Back', action: tour.back },
      { text: 'Finish Tour', action: tour.complete }
    ],
  })

  return tour
}

// Helper to wait for dynamic elements
function waitForElement(selector: string, timeout = 5000): Promise<void> {
  return new Promise((resolve, reject) => {
    if (document.querySelector(selector)) {
      return resolve()
    }

    const observer = new MutationObserver(() => {
      if (document.querySelector(selector)) {
        observer.disconnect()
        resolve()
      }
    })

    observer.observe(document.body, {
      childList: true,
      subtree: true
    })

    setTimeout(() => {
      observer.disconnect()
      reject(new Error(`Element ${selector} not found within ${timeout}ms`))
    }, timeout)
  })
}

function markTourSkipped() {
  localStorage.setItem('automatos-tour-skipped', 'true')
  localStorage.setItem('automatos-tour-completed-at', new Date().toISOString())
}
```

### 4. Tour Storage Helper

**File**: `frontend/lib/shepherd/tour-storage.ts`

```typescript
const TOUR_COMPLETED_KEY = 'automatos-onboarding-completed'
const TOUR_SKIPPED_KEY = 'automatos-onboarding-skipped'
const TOUR_DISMISSED_KEY = 'automatos-tour-dismissed-at'

export function haCompletedOnboarding(): boolean {
  return !!(
    localStorage.getItem(TOUR_COMPLETED_KEY) ||
    localStorage.getItem(TOUR_SKIPPED_KEY)
  )
}

export function markOnboardingComplete() {
  localStorage.setItem(TOUR_COMPLETED_KEY, 'true')
  localStorage.setItem('automatos-tour-completed-at', new Date().toISOString())
}

export function markOnboardingSkipped() {
  localStorage.setItem(TOUR_SKIPPED_KEY, 'true')
  localStorage.setItem(TOUR_DISMISSED_KEY, new Date().toISOString())
}

export function resetOnboarding() {
  localStorage.removeItem(TOUR_COMPLETED_KEY)
  localStorage.removeItem(TOUR_SKIPPED_KEY)
  localStorage.removeItem(TOUR_DISMISSED_KEY)
}
```

### 5. Welcome Modal Component

**File**: `frontend/components/onboarding/welcome-modal.tsx`

```typescript
'use client'

import { useState } from 'react'
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription } from '@/components/ui/dialog'
import { Button } from '@/components/ui/button'
import { Sparkles, X } from 'lucide-react'
import { createFirstLoginTour } from '@/lib/shepherd/first-login-tour'
import { markOnboardingSkipped, markOnboardingComplete } from '@/lib/shepherd/tour-storage'

interface WelcomeModalProps {
  open: boolean
  onOpenChange: (open: boolean) => void
}

export function WelcomeModal({ open, onOpenChange }: WelcomeModalProps) {
  const [isStarting, setIsStarting] = useState(false)

  const handleSkip = () => {
    markOnboardingSkipped()
    onOpenChange(false)
  }

  const handleStartTour = () => {
    setIsStarting(true)
    onOpenChange(false)

    // Small delay for modal close animation
    setTimeout(() => {
      const tour = createFirstLoginTour()

      tour.on('complete', () => {
        markOnboardingComplete()
      })

      tour.on('cancel', () => {
        markOnboardingSkipped()
      })

      tour.start()
      setIsStarting(false)
    }, 300)
  }

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-2xl">
        <DialogHeader>
          <div className="flex items-center gap-3 mb-2">
            <div className="p-3 rounded-full bg-gradient-to-br from-blue-500 to-purple-600">
              <Sparkles className="w-6 h-6 text-white" />
            </div>
            <div>
              <DialogTitle className="text-2xl">Welcome to Automatos AI</DialogTitle>
              <DialogDescription className="text-gray-400 mt-1">
                Your intelligent automation platform
              </DialogDescription>
            </div>
          </div>
        </DialogHeader>

        <div className="space-y-6 py-4">
          {/* Quick Overview */}
          <div>
            <h3 className="font-semibold text-gray-200 mb-3">
              What you can do with Automatos:
            </h3>
            <div className="grid grid-cols-2 gap-3">
              <div className="p-4 rounded-lg bg-gray-800/50 border border-gray-700">
                <div className="font-medium text-sm text-gray-300 mb-1">
                  🤖 AI Agents
                </div>
                <div className="text-xs text-gray-400">
                  Create autonomous workers for email, data, research, and more
                </div>
              </div>
              <div className="p-4 rounded-lg bg-gray-800/50 border border-gray-700">
                <div className="font-medium text-sm text-gray-300 mb-1">
                  🔌 150+ Integrations
                </div>
                <div className="text-xs text-gray-400">
                  Connect to Gmail, Slack, Jira, GitHub, and all your tools
                </div>
              </div>
              <div className="p-4 rounded-lg bg-gray-800/50 border border-gray-700">
                <div className="font-medium text-sm text-gray-300 mb-1">
                  ⚡ Workflows
                </div>
                <div className="text-xs text-gray-400">
                  Build complex automations with no-code visual builder
                </div>
              </div>
              <div className="p-4 rounded-lg bg-gray-800/50 border border-gray-700">
                <div className="font-medium text-sm text-gray-300 mb-1">
                  💬 AI Chat
                </div>
                <div className="text-xs text-gray-400">
                  Get help anytime from your AI assistant
                </div>
              </div>
            </div>
          </div>

          {/* Tour CTA */}
          <div className="p-4 rounded-lg bg-blue-500/10 border border-blue-500/30">
            <div className="flex items-start gap-3">
              <div className="mt-0.5">
                <Sparkles className="w-5 h-5 text-blue-400" />
              </div>
              <div className="flex-1">
                <div className="font-medium text-gray-200 mb-1">
                  Take a 2-minute tour
                </div>
                <div className="text-sm text-gray-400">
                  We'll walk you through creating your first email assistant agent.
                  You can skip or exit anytime by pressing <kbd className="px-1.5 py-0.5 text-xs bg-gray-700 rounded">ESC</kbd>
                </div>
              </div>
            </div>
          </div>

          {/* Actions */}
          <div className="flex items-center justify-between pt-2">
            <Button
              variant="ghost"
              onClick={handleSkip}
              className="text-gray-400 hover:text-gray-200"
            >
              Skip, I'll explore on my own
            </Button>
            <Button
              onClick={handleStartTour}
              disabled={isStarting}
              className="bg-gradient-to-r from-blue-500 to-purple-600 hover:from-blue-600 hover:to-purple-700"
            >
              {isStarting ? 'Starting...' : 'Start Tour'}
              <Sparkles className="w-4 h-4 ml-2" />
            </Button>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  )
}
```

### 6. First Login Guard Component

**File**: `frontend/components/onboarding/first-login-guard.tsx`

```typescript
'use client'

import { useEffect, useState } from 'react'
import { useUser } from '@clerk/nextjs'
import { WelcomeModal } from './welcome-modal'
import { hasCompletedOnboarding } from '@/lib/shepherd/tour-storage'

export function FirstLoginGuard() {
  const { user, isLoaded } = useUser()
  const [showWelcome, setShowWelcome] = useState(false)

  useEffect(() => {
    if (!isLoaded || !user) return

    // Check if this is truly first login
    const onboardingComplete = hasCompletedOnboarding()
    const userCreatedRecently = user.createdAt &&
      (Date.now() - new Date(user.createdAt).getTime()) < 5 * 60 * 1000 // 5 mins

    if (!onboardingComplete && userCreatedRecently) {
      // Small delay to let the app render first
      setTimeout(() => setShowWelcome(true), 1000)
    }
  }, [isLoaded, user])

  return (
    <WelcomeModal
      open={showWelcome}
      onOpenChange={setShowWelcome}
    />
  )
}
```

### 7. Shepherd Theme

**File**: `frontend/lib/shepherd/shepherd-theme.ts`

```typescript
export const shepherdTheme = {
  defaultStepOptions: {
    classes: 'shepherd-theme-automatos',
    scrollTo: { behavior: 'smooth' as const, block: 'center' as const },
    cancelIcon: {
      enabled: true,
    },
    modalOverlayOpeningPadding: 8,
    modalOverlayOpeningRadius: 8,
    when: {
      show() {
        const currentStepElement = document.querySelector('.shepherd-element')
        if (currentStepElement) {
          // Smooth fade-in animation
          currentStepElement.classList.add('animate-in', 'fade-in', 'zoom-in-95')
        }
      },
    },
  },
  useModalOverlay: true,
}
```

**File**: `frontend/styles/shepherd-custom.css`

```css
/* Import base Shepherd styles */
@import 'shepherd.js/dist/css/shepherd.css';

/* Automatos Dark Glass Theme */
.shepherd-theme-automatos {
  background: rgba(17, 24, 39, 0.95) !important;
  backdrop-filter: blur(16px);
  border: 1px solid rgba(55, 65, 81, 0.6);
  border-radius: 12px;
  box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.4),
              0 10px 10px -5px rgba(0, 0, 0, 0.2);
  max-width: 400px;
}

.shepherd-theme-automatos .shepherd-header {
  padding: 1rem 1rem 0.5rem;
}

.shepherd-theme-automatos .shepherd-title {
  color: rgba(243, 244, 246, 1);
  font-size: 1.125rem;
  font-weight: 600;
  margin: 0;
}

.shepherd-theme-automatos .shepherd-text {
  color: rgba(209, 213, 219, 1);
  font-size: 0.875rem;
  line-height: 1.6;
  padding: 1rem;
}

.shepherd-theme-automatos .shepherd-text p {
  margin: 0;
}

.shepherd-theme-automatos .shepherd-text ul,
.shepherd-theme-automatos .shepherd-text ol {
  margin: 0.5rem 0;
  padding-left: 1.25rem;
}

.shepherd-theme-automatos .shepherd-text code {
  background: rgba(31, 41, 55, 0.8);
  border: 1px solid rgba(55, 65, 81, 0.5);
  border-radius: 4px;
  padding: 0.125rem 0.375rem;
  font-size: 0.8125rem;
  font-family: 'Monaco', 'Menlo', monospace;
  color: rgba(96, 165, 250, 1);
}

.shepherd-theme-automatos .shepherd-footer {
  padding: 0.75rem 1rem;
  border-top: 1px solid rgba(55, 65, 81, 0.3);
  display: flex;
  justify-content: flex-end;
  gap: 0.5rem;
}

.shepherd-theme-automatos .shepherd-button {
  background: linear-gradient(135deg, #3b82f6, #2563eb);
  border: none;
  border-radius: 6px;
  color: white;
  padding: 0.5rem 1rem;
  font-size: 0.875rem;
  font-weight: 500;
  cursor: pointer;
  transition: all 0.2s ease;
}

.shepherd-theme-automatos .shepherd-button:hover {
  transform: translateY(-1px);
  box-shadow: 0 4px 12px rgba(59, 130, 246, 0.4);
}

.shepherd-theme-automatos .shepherd-button-secondary {
  background: transparent;
  border: 1px solid rgba(75, 85, 99, 0.5);
  color: rgba(156, 163, 175, 1);
}

.shepherd-theme-automatos .shepherd-button-secondary:hover {
  background: rgba(31, 41, 55, 0.5);
  border-color: rgba(107, 114, 128, 0.7);
  color: rgba(209, 213, 219, 1);
  box-shadow: none;
}

.shepherd-theme-automatos .shepherd-cancel-icon {
  color: rgba(156, 163, 175, 1);
  width: 20px;
  height: 20px;
  transition: color 0.2s;
}

.shepherd-theme-automatos .shepherd-cancel-icon:hover {
  color: rgba(243, 244, 246, 1);
}

/* Modal Overlay */
.shepherd-modal-overlay-container {
  background: rgba(0, 0, 0, 0.6) !important;
  backdrop-filter: blur(4px);
}

/* Highlighted Element */
.shepherd-target-highlight {
  animation: pulse-ring 2s ease-out infinite;
}

@keyframes pulse-ring {
  0% {
    box-shadow: 0 0 0 0 rgba(59, 130, 246, 0.6);
  }
  50% {
    box-shadow: 0 0 0 8px rgba(59, 130, 246, 0.2);
  }
  100% {
    box-shadow: 0 0 0 0 rgba(59, 130, 246, 0);
  }
}

/* Arrow/Pointer */
.shepherd-arrow {
  display: none; /* Cleaner look without arrow */
}

/* Animations */
.animate-in {
  animation-duration: 0.2s;
  animation-fill-mode: both;
}

.fade-in {
  animation-name: fadeIn;
}

.zoom-in-95 {
  animation-name: zoomIn95;
}

@keyframes fadeIn {
  from {
    opacity: 0;
  }
  to {
    opacity: 1;
  }
}

@keyframes zoomIn95 {
  from {
    transform: scale(0.95);
  }
  to {
    transform: scale(1);
  }
}
```

### 8. Add to Root Layout

**File**: `frontend/app/layout.tsx`

```typescript
import { FirstLoginGuard } from '@/components/onboarding/first-login-guard'
import '@/styles/shepherd-custom.css'

export default function RootLayout({ children }) {
  return (
    <html>
      <body>
        <ClerkProvider>
          <QueryClientProvider>
            <ThemeProvider>
              {/* Add this */}
              <FirstLoginGuard />

              {children}
            </ThemeProvider>
          </QueryClientProvider>
        </ClerkProvider>
      </body>
    </html>
  )
}
```

### 9. Add data-tour Attributes

Update these components to add tour targets:

**Sidebar** (`frontend/components/layout/sidebar.tsx`):
```tsx
<aside data-tour="sidebar" className="...">
  {/* ... */}
  <Link href="/agents" data-tour="nav-agents">
    Agents
  </Link>
</aside>
```

**Agents Page** (`frontend/app/agents/page.tsx`):
```tsx
<Button data-tour="create-agent-btn" onClick={openCreateModal}>
  Create Agent
</Button>

<div data-tour="agent-roster">
  {/* Agent list */}
</div>
```

**Create Agent Modal** (`frontend/components/agents/create-agent-modal.tsx`):
```tsx
<Input
  data-tour="agent-name-input"
  placeholder="Agent Name"
  {...field}
/>

<Textarea
  data-tour="agent-description-input"
  placeholder="Describe what this agent does..."
  {...field}
/>

<div data-tour="agent-tools-section">
  {/* Tools selection UI */}
</div>

<Button data-tour="save-agent-btn" type="submit">
  Create Agent
</Button>
```

**Chat Widget** (`frontend/components/chatbot/chat-widget.tsx`):
```tsx
<div data-tour="chat-widget" className="...">
  {/* Chat widget content */}
</div>
```

## Testing & Rollout

### Local Testing
1. Clear localStorage
2. Create new test user account
3. Verify welcome modal appears
4. Test both "Skip" and "Start Tour" paths
5. Test ESC key dismissal during tour
6. Verify tour only appears once

### A/B Testing (Optional)
```typescript
// Only show to 50% of new users to measure impact
if (Math.random() > 0.5) {
  setShowWelcome(true)
}

// Track metrics:
// - Tour completion rate
// - Time to first agent creation
// - User retention (7-day)
```

### Settings Override
Add option for users to replay tour:

**File**: `frontend/app/settings/page.tsx`
```tsx
import { resetOnboarding } from '@/lib/shepherd/tour-storage'

<Button onClick={() => {
  resetOnboarding()
  window.location.href = '/chat'
}}>
  Replay Onboarding Tour
</Button>
```

## Key Features

✅ **One-time only** - Shows once on first login
✅ **Easy skip** - Dismiss button + ESC key
✅ **Contextual** - Guides through real agent creation
✅ **Non-intrusive** - ChatWidget takes over after
✅ **Accessible** - Keyboard navigation, ARIA support
✅ **Styled to match** - Dark glass Automatos theme
✅ **Remembers state** - Won't show again after completion

## Next Steps

1. Install dependencies
2. Create the 9 files above
3. Add `data-tour` attributes to components
4. Test with fresh user account
5. Monitor completion rates
6. Iterate based on feedback

Total implementation time: ~4-6 hours
