import Shepherd from 'shepherd.js'
import { shepherdTheme } from './shepherd-theme'
import { markOnboardingComplete, markOnboardingSkipped } from './tour-storage'

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
          markOnboardingSkipped()
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

  // Handle tour completion
  tour.on('complete', () => {
    markOnboardingComplete()
  })

  tour.on('cancel', () => {
    markOnboardingSkipped()
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
