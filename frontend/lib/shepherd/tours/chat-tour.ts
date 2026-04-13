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
      ${stepProgress(1, TOTAL)}
    `,
    beforeShowPromise: () => waitForElement('[data-tour="chat-agent-selector"]'),
    attachTo: { element: '[data-tour="chat-agent-selector"]', on: 'bottom' },
    buttons: [{ text: 'Next', action: tour.next }],
  })

  tour.addStep({
    id: 'chat-model',
    title: title('Choose a', 'Model'),
    text: `
      <p class="text-gray-300 mb-2">
        Pick which LLM powers this conversation. Faster, cheaper models (Haiku, GPT-mini,
        DeepSeek) are great for quick back-and-forth; bigger models (Opus, Sonnet, GPT-4-class)
        are better for planning, writing, and reasoning.
      </p>
      <p class="text-gray-400 text-sm">
        You can change models mid-conversation — the history stays.
      </p>
      ${stepProgress(2, TOTAL)}
    `,
    beforeShowPromise: () => waitForElement('[data-tour="chat-model-selector"]'),
    attachTo: { element: '[data-tour="chat-model-selector"]', on: 'bottom' },
    buttons: [
      { text: 'Back', classes: 'shepherd-button-secondary', action: tour.back },
      { text: 'Next', action: tour.next },
    ],
  })

  tour.addStep({
    id: 'chat-history',
    title: title('Chat', 'History'),
    text: `
      <p class="text-gray-300 mb-2">
        Every conversation is saved automatically. Toggle the chat sidebar to browse,
        search, rename, pin and resume previous sessions — across any agent.
      </p>
      ${stepProgress(3, TOTAL)}
    `,
    beforeShowPromise: () => waitForElement('[data-tour="nav-chat"]'),
    attachTo: { element: '[data-tour="nav-chat"]', on: 'right' },
    buttons: [
      { text: 'Back', classes: 'shepherd-button-secondary', action: tour.back },
      { text: 'Got it!', action: tour.complete },
    ],
  })

  tour.on('complete', () => markTourComplete(TOUR_ID, userId))
  tour.on('cancel', () => markTourSkipped(TOUR_ID, userId))

  return tour
}
