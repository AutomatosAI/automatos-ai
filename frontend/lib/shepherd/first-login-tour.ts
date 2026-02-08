import Shepherd from 'shepherd.js'
import { shepherdTheme } from './shepherd-theme'
import { markOnboardingComplete, markOnboardingSkipped } from './tour-storage'
import { switchTabAndWaitForElement } from './tour-bridge'

export function createFirstLoginTour(userId: string) {
  const tour = new Shepherd.Tour({
    ...shepherdTheme,
    exitOnEsc: true,
    keyboardNavigation: true,
  })

  // Helper for two-colour titles: white + gradient-text (orange)
  // &nbsp; required because gradient-text's -webkit-text-fill-color:transparent collapses normal spaces
  const title = (white: string, accent: string) =>
    `<span style="color:#fff">${white}</span>&nbsp;<span class="gradient-text">${accent}</span>`

  // Step 1: Welcome
  tour.addStep({
    id: 'welcome',
    title: title('Welcome to', 'Automatos AI'),
    text: `
      <p class="text-gray-300 mb-3">
        We'll walk you through your workspace, help you design your first agent,
        pick a persona and model, and connect your tools. Takes about 2 minutes!
      </p>
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
          markOnboardingSkipped(userId)
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
    title: title('Your', 'Navigation Hub'),
    text: `
      <p class="text-gray-300 mb-2">
        The sidebar gives you access to all major features:
      </p>
      <ul class="text-sm text-gray-400 mt-2 space-y-1">
        <li><strong>Chat:</strong> Talk to your AI assistants</li>
        <li><strong>Agents:</strong> Create and manage AI workers</li>
        <li><strong>Workflows:</strong> Automate complex tasks</li>
        <li><strong>Tools:</strong> Connect to external services</li>
        <li><strong>Marketplace:</strong> Discover agents, recipes &amp; tools</li>
        <li><strong>Analytics:</strong> Monitor performance &amp; usage</li>
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
    title: title('Create Your', 'First Agent'),
    text: `
      <p class="text-gray-300">
        <span style="margin-right:6px">🤖</span>
        Let's create an AI agent! Click on <strong>Agents</strong> in the sidebar.
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
    title: title('Start', 'Creating'),
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

  // Step 5: Agent Category (modal Tab 1 - Config)
  tour.addStep({
    id: 'agent-category',
    title: title('Pick a', 'Category'),
    text: `
      <p class="text-gray-300 mb-2">
        Choose the category that best describes your agent's role.
        This filters which personas are available in the next step.
      </p>
    `,
    beforeShowPromise: () => switchTabAndWaitForElement(1, '[data-tour="agent-category-select"]'),
    attachTo: {
      element: '[data-tour="agent-category-select"]',
      on: 'right'
    },
    buttons: [
      { text: 'Next', action: tour.next }
    ],
  })

  // Step 6: Agent Name (modal Tab 1 - Config)
  tour.addStep({
    id: 'agent-name',
    title: title('Name Your', 'Agent'),
    text: `
      <p class="text-gray-300 mb-2">
        Give your agent a descriptive name, like:
      </p>
      <code class="text-sm bg-gray-800 px-2 py-1 rounded">
        Email Assistant
      </code>
    `,
    beforeShowPromise: () => switchTabAndWaitForElement(1, '[data-tour="agent-name-input"]'),
    attachTo: {
      element: '[data-tour="agent-name-input"]',
      on: 'right'
    },
    buttons: [
      { text: 'Next', action: tour.next }
    ],
  })

  // Step 6: Agent Description (modal Tab 1 - Config)
  tour.addStep({
    id: 'agent-description',
    title: title('Describe Its', 'Purpose'),
    text: `
      <p class="text-gray-300 mb-2">
        Tell the agent what it should do:
      </p>
      <code class="text-xs bg-gray-800 px-2 py-1 rounded block">
        "Monitor my inbox, categorize emails, and draft replies to common questions."
      </code>
    `,
    beforeShowPromise: () => switchTabAndWaitForElement(1, '[data-tour="agent-description-input"]'),
    attachTo: {
      element: '[data-tour="agent-description-input"]',
      on: 'right'
    },
    buttons: [
      { text: 'Back', action: tour.back },
      { text: 'Next', action: tour.next }
    ],
  })

  // Step 7: Persona Tab (modal Tab 2)
  tour.addStep({
    id: 'agent-persona',
    title: title('Choose a', 'Persona'),
    text: `
      <p class="text-gray-300 mb-2">
        Give your agent a personality! You can choose:
      </p>
      <ul class="text-sm text-gray-400 space-y-1">
        <li><strong>No Persona:</strong> Default behaviour</li>
        <li><strong>Predefined:</strong> Pick from curated personalities</li>
        <li><strong>Custom:</strong> Write your own system prompt</li>
      </ul>
      <p class="text-xs text-gray-500 mt-3">
        Try selecting a <strong>Predefined</strong> persona — the list is filtered by the category you chose on the Config tab.
      </p>
    `,
    beforeShowPromise: () => switchTabAndWaitForElement(2, '[data-tour="agent-persona-section"]'),
    attachTo: {
      element: '[data-tour="agent-persona-section"]',
      on: 'left'
    },
    buttons: [
      { text: 'Back', action: tour.back },
      { text: 'Next', action: tour.next }
    ],
  })

  // Step 8: Model Tab (modal Tab 3)
  tour.addStep({
    id: 'agent-model',
    title: title('Select an', 'LLM Model'),
    text: `
      <p class="text-gray-300 mb-2">
        Choose the AI model that best suits your agent's task.
        Different models have different strengths — some are faster,
        others are more creative or analytical.
      </p>
      <p class="text-xs text-gray-500 mt-3">
        Pick a model from the dropdown, then adjust parameters like temperature if you'd like.
      </p>
    `,
    beforeShowPromise: () => switchTabAndWaitForElement(3, '[data-tour="agent-model-section"]'),
    attachTo: {
      element: '[data-tour="agent-model-section"]',
      on: 'left'
    },
    buttons: [
      { text: 'Back', action: tour.back },
      { text: 'Next', action: tour.next }
    ],
  })

  // Step 9: Tools Tab (modal Tab 4)
  tour.addStep({
    id: 'agent-tools',
    title: title('Connect', 'Tools'),
    text: `
      <p class="text-gray-300 mb-2">
        Tools let your agent connect to external services like
        <strong>email</strong>, <strong>calendars</strong>, <strong>CRMs</strong>, and more.
      </p>
      <p class="text-gray-400 text-sm">
        You probably don't have any tools set up yet — that's OK!
        You can come back and add them later from the <strong>Tools</strong> page.
      </p>
    `,
    beforeShowPromise: () => switchTabAndWaitForElement(4, '[data-tour="agent-tools-section"]'),
    attachTo: {
      element: '[data-tour="agent-tools-section"]',
      on: 'left'
    },
    buttons: [
      { text: 'Back', action: tour.back },
      { text: 'Next', action: tour.next }
    ],
  })

  // Step 10: Plugins Tab (modal Tab 5)
  tour.addStep({
    id: 'agent-plugins',
    title: title('Add', 'Capabilities'),
    text: `
      <p class="text-gray-300 mb-2">
        Capabilities are extra skills you can add to make your agent smarter —
        like writing professional emails, summarising documents, or generating reports.
      </p>
      <p class="text-gray-400 text-sm">
        You may not have any available yet — you can always add them later.
        When you're ready, click <strong>Create Agent</strong> below!
      </p>
    `,
    beforeShowPromise: () => switchTabAndWaitForElement(5, '[data-tour="agent-plugins-section"]'),
    attachTo: {
      element: '[data-tour="agent-plugins-section"]',
      on: 'left'
    },
    buttons: [
      { text: 'Back', action: tour.back },
      {
        text: 'Next',
        action: tour.next,
      }
    ],
  })

  // Step 11: Create Agent — attached to the real button so user can click it
  tour.addStep({
    id: 'save-agent',
    title: title('Create Your', 'Agent'),
    text: `
      <p>Click the <strong>Create Agent</strong> button to finish up!</p>
    `,
    beforeShowPromise: () => switchTabAndWaitForElement(5, '[data-tour="save-agent-btn"]'),
    attachTo: {
      element: '[data-tour="save-agent-btn"]',
      on: 'top'
    },
    buttons: [
      { text: 'Back', action: tour.back },
      {
        text: 'Skip to End',
        classes: 'shepherd-button-secondary',
        action: () => tour.show('chat-widget')
      }
    ],
    when: {
      show: () => {
        // Auto-advance when agent is created (agent-roster appears AND modal disappears)
        const observer = new MutationObserver(() => {
          const rosterVisible = document.querySelector('[data-tour="agent-roster"]')
          const modalVisible = document.querySelector('[data-tour="save-agent-btn"]')
          if (rosterVisible && !modalVisible) {
            observer.disconnect()
            tour.next()
          }
        })
        observer.observe(document.body, { childList: true, subtree: true })
        ;(tour as any).__saveStepObserver = observer
      },
      hide: () => {
        const observer = (tour as any).__saveStepObserver as MutationObserver | undefined
        if (observer) {
          observer.disconnect()
          delete (tour as any).__saveStepObserver
        }
      }
    }
  })

  // Step 12: Agent Created — show kebab menu
  tour.addStep({
    id: 'agent-created',
    title: title('Agent', 'Created!'),
    text: `
      <p class="text-gray-300 mb-3">
        Well done — you've created your first agent! You can see it here in your roster.
      </p>
      <p class="text-gray-400 text-sm mb-2">
        Use this menu to <strong>view details</strong>, <strong>configure</strong>,
        <strong>start/pause</strong>, or <strong>delete</strong> your agent later.
      </p>
      <p class="text-xs text-gray-500">
        Don't worry about getting everything perfect now — you can always come back and edit.
      </p>
    `,
    beforeShowPromise: () => waitForElement('[data-tour="agent-card-menu"]'),
    attachTo: {
      element: '[data-tour="agent-card-menu"]',
      on: 'left'
    },
    buttons: [
      { text: 'Next', action: tour.next }
    ],
  })

  // Step 13: ChatWidget - Always Available Help
  tour.addStep({
    id: 'chat-widget',
    title: title('Need Help?', 'Ask the Chat'),
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
    markOnboardingComplete(userId)
  })

  tour.on('cancel', () => {
    markOnboardingSkipped(userId)
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
