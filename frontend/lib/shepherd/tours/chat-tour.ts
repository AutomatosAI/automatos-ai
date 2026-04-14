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
      { text: 'Got it!', action: tour.complete },
    ],
  })

  tour.on('complete', () => markTourComplete(TOUR_ID, userId))
  tour.on('cancel', () => markTourSkipped(TOUR_ID, userId))

  return tour
}
