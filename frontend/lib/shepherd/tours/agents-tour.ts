import Shepherd from 'shepherd.js'
import { shepherdTheme } from '../shepherd-theme'
import { markTourComplete, markTourSkipped } from '../tour-storage'
import { title, waitForElement, stepProgress } from '../tour-utils'

const TOUR_ID = 'agents'
const TOTAL = 4

export function createAgentsTour(userId: string) {
  const tour = new Shepherd.Tour({
    ...shepherdTheme,
    exitOnEsc: true,
    keyboardNavigation: true,
  })

  tour.addStep({
    id: 'agents-overview',
    title: title('Agent', 'Management'),
    text: `
      <p class="text-gray-300 mb-2">
        This is your agent command center — create, configure, and monitor all your AI workers.
      </p>
      ${stepProgress(1, TOTAL)}
    `,
    beforeShowPromise: () => waitForElement('[data-tour="agents-page-header"]'),
    attachTo: { element: '[data-tour="agents-page-header"]', on: 'bottom' },
    buttons: [{ text: 'Next', action: tour.next }],
  })

  tour.addStep({
    id: 'agents-create',
    title: title('Create an', 'Agent'),
    text: `
      <p class="text-gray-300 mb-2">
        Click here to build a new agent — pick a category, persona, model, and tools.
      </p>
      ${stepProgress(2, TOTAL)}
    `,
    beforeShowPromise: () => waitForElement('[data-tour="create-agent-btn"]'),
    attachTo: { element: '[data-tour="create-agent-btn"]', on: 'bottom' },
    buttons: [
      { text: 'Back', classes: 'shepherd-button-secondary', action: tour.back },
      { text: 'Next', action: tour.next },
    ],
  })

  tour.addStep({
    id: 'agents-roster',
    title: title('Your', 'Agent Roster'),
    text: `
      <p class="text-gray-300 mb-2">
        All your agents appear here. Switch between grid and list view to find them fast.
      </p>
      ${stepProgress(3, TOTAL)}
    `,
    beforeShowPromise: () => waitForElement('[data-tour="agent-roster"]'),
    attachTo: { element: '[data-tour="agent-roster"]', on: 'top' },
    buttons: [
      { text: 'Back', classes: 'shepherd-button-secondary', action: tour.back },
      { text: 'Next', action: tour.next },
    ],
  })

  tour.addStep({
    id: 'agents-actions',
    title: title('Agent', 'Actions'),
    text: `
      <p class="text-gray-300 mb-2">
        Use the menu on each card to view details, configure, start/pause, or delete an agent.
      </p>
      ${stepProgress(4, TOTAL)}
    `,
    beforeShowPromise: () => waitForElement('[data-tour="agent-card-menu"]'),
    attachTo: { element: '[data-tour="agent-card-menu"]', on: 'left' },
    buttons: [
      { text: 'Back', classes: 'shepherd-button-secondary', action: tour.back },
      { text: 'Got it!', action: tour.complete },
    ],
  })

  tour.on('complete', () => markTourComplete(TOUR_ID, userId))
  tour.on('cancel', () => markTourSkipped(TOUR_ID, userId))

  return tour
}
