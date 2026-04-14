import Shepherd from 'shepherd.js'
import { shepherdTheme } from '../shepherd-theme'
import { markTourComplete, markTourSkipped } from '../tour-storage'
import { title, waitForElement, stepProgress } from '../tour-utils'

const TOUR_ID = 'chat'
const TOTAL = 3

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
        Plan, and Mission — all from one place. Every conversation is
        saved automatically.
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
        Every conversation runs through an agent. Each one has its own
        persona, memory, tools and skills. Start with <strong>Auto</strong>
        — it routes your message to the best agent automatically.
      </p>
      ${stepProgress(2, TOTAL)}
    `,
    beforeShowPromise: () => waitForElement('[data-tour="chat-agent-selector"]'),
    attachTo: { element: '[data-tour="chat-agent-selector"]', on: 'top' },
    buttons: [
      { text: 'Back', classes: 'shepherd-button-secondary', action: tour.back },
      { text: 'Next', action: tour.next },
    ],
  })

  // Step 3: Header icons
  tour.addStep({
    id: 'chat-header',
    title: title('Your', 'Toolbar'),
    text: `
      <p class="text-gray-300 mb-2">
        Up here you'll find <strong>Docs</strong> (documentation),
        <strong>Theme</strong> toggle, <strong>Alerts</strong>, and your
        <strong>Profile</strong> menu — which is also where you'll find
        <em>Tour this page</em> on every page.
      </p>
      ${stepProgress(3, TOTAL)}
    `,
    beforeShowPromise: () => waitForElement('[data-tour="header-actions"]'),
    attachTo: { element: '[data-tour="header-actions"]', on: 'bottom' },
    buttons: [
      { text: 'Back', classes: 'shepherd-button-secondary', action: tour.back },
      { text: 'Got it!', action: tour.complete },
    ],
  })

  tour.on('complete', () => markTourComplete(TOUR_ID, userId))
  tour.on('cancel', () => markTourSkipped(TOUR_ID, userId))

  return tour
}
