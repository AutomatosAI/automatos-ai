import Shepherd from 'shepherd.js'
import { shepherdTheme } from '../shepherd-theme'
import { markTourComplete, markTourSkipped } from '../tour-storage'
import { title, waitForElement, stepProgress, tabList } from '../tour-utils'

const TOUR_ID = 'chat'
const TOTAL = 5

export function createChatTour(userId: string) {
  const tour = new Shepherd.Tour({
    ...shepherdTheme,
    exitOnEsc: true,
    keyboardNavigation: true,
  })

  // Step 1: About (centered)
  tour.addStep({
    id: 'chat-about',
    title: title('AI', 'Command Line'),
    text: `
      <p class="text-gray-300 mb-2">
        This is your main conversation interface. Chat with any agent,
        switch models on the fly, and toggle special modes like Code,
        Plan, and Mission — all from one place.
      </p>
      ${stepProgress(1, TOTAL)}
    `,
    buttons: [
      { text: 'Skip', classes: 'shepherd-button-secondary', action: () => tour.cancel() },
      { text: 'Next', action: tour.next },
    ],
  })

  // Step 2: Agent selector
  tour.addStep({
    id: 'chat-agent',
    title: title('Pick an', 'Agent'),
    text: `
      <p class="text-gray-300 mb-2">
        Every conversation runs through an agent. Each one has its own persona,
        memory, tool access and skillset — so picking the right agent changes
        what the assistant can actually do.
      </p>
      <p class="text-gray-400 text-sm">
        Start with <strong>Auto</strong> if you're not sure — it routes your message
        to the best agent for the job.
      </p>
      ${stepProgress(2, TOTAL)}
    `,
    beforeShowPromise: () => waitForElement('[data-tour="chat-agent-selector"]'),
    attachTo: { element: '[data-tour="chat-agent-selector"]', on: 'bottom' },
    buttons: [
      { text: 'Back', classes: 'shepherd-button-secondary', action: tour.back },
      { text: 'Next', action: tour.next },
    ],
  })

  // Step 3: Quick modes
  tour.addStep({
    id: 'chat-modes',
    title: title('Quick', 'Modes'),
    text: `
      <p class="text-gray-300 mb-2">
        Toggle <strong>Code</strong> for syntax-highlighted output,
        <strong>Plan</strong> to get a structured step-by-step, or
        <strong>Mission</strong> to launch a multi-agent objective — all
        without leaving the chat.
      </p>
      ${stepProgress(3, TOTAL)}
    `,
    beforeShowPromise: () => waitForElement('[data-tour="chat-mode-bar"]'),
    attachTo: { element: '[data-tour="chat-mode-bar"]', on: 'top' },
    buttons: [
      { text: 'Back', classes: 'shepherd-button-secondary', action: tour.back },
      { text: 'Next', action: tour.next },
    ],
  })

  // Step 4: Header toolbar
  tour.addStep({
    id: 'chat-header',
    title: title('Your', 'Toolbar'),
    text: `
      <p class="text-gray-300 mb-2">
        Your toolbar icons, left to right:
      </p>
      ${tabList([
        ['Docs (book icon)', 'Opens documentation and API reference in a new tab — everything you need to build with the platform.'],
        ['Theme (monitor icon)', 'Switch between light and dark mode instantly.'],
        ['Alerts (bell icon)', 'Notifications from your agents, missions, and system events — check here for anything that needs your attention.'],
        ['Profile (your avatar)', 'Account settings, workspace management, and Tour this page — click your avatar on any page to relaunch that page\'s tour.'],
      ])}
      ${stepProgress(4, TOTAL)}
    `,
    beforeShowPromise: () => waitForElement('[data-tour="header-actions"]'),
    attachTo: { element: '[data-tour="header-actions"]', on: 'bottom' },
    buttons: [
      { text: 'Back', classes: 'shepherd-button-secondary', action: tour.back },
      { text: 'Next', action: tour.next },
    ],
  })

  // Step 5: Meet Auto
  tour.addStep({
    id: 'chat-auto',
    title: title('Meet', 'Auto'),
    text: `
      <p class="text-gray-300 mb-2">
        See the chat bubble in the <strong>bottom right</strong>? That's <strong>Auto</strong> —
        your AI assistant that follows you to every page in the platform.
      </p>
      <p class="text-gray-300 mb-2">
        Ask Auto anything — how to build agents, create playbooks, set up missions,
        connect tools, or understand your analytics. If you're ever stuck, Auto is
        right there to help.
      </p>
      <p class="text-gray-400 text-sm">
        Try it now — ask Auto <em>"What can you help me with?"</em>
      </p>
      ${stepProgress(5, TOTAL)}
    `,
    buttons: [
      { text: 'Back', classes: 'shepherd-button-secondary', action: tour.back },
      { text: 'Got it!', action: tour.complete },
    ],
  })

  tour.on('complete', () => markTourComplete(TOUR_ID, userId))
  tour.on('cancel', () => markTourSkipped(TOUR_ID, userId))

  return tour
}
