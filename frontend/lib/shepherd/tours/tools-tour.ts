import Shepherd from 'shepherd.js'
import { shepherdTheme } from '../shepherd-theme'
import { markTourComplete, markTourSkipped } from '../tour-storage'
import { title, waitForElement, stepProgress } from '../tour-utils'

const TOUR_ID = 'tools'
const TOTAL = 3

export function createToolsTour(userId: string) {
  const tour = new Shepherd.Tour({
    ...shepherdTheme,
    exitOnEsc: true,
    keyboardNavigation: true,
  })

  tour.addStep({
    id: 'tools-overview',
    title: title('Tools &', 'Integrations'),
    text: `
      <p class="text-gray-300 mb-2">
        Connect your agents to external services — Gmail, Slack, GitHub, CRMs, and 150+ more.
      </p>
      ${stepProgress(1, TOTAL)}
    `,
    beforeShowPromise: () => waitForElement('[data-tour="tools-page-header"]'),
    attachTo: { element: '[data-tour="tools-page-header"]', on: 'bottom' },
    buttons: [{ text: 'Next', action: tour.next }],
  })

  tour.addStep({
    id: 'tools-connected',
    title: title('Connected', 'Apps'),
    text: `
      <p class="text-gray-300 mb-2">
        Apps you've already connected show up here. Click any card to manage or reconfigure.
      </p>
      ${stepProgress(2, TOTAL)}
    `,
    beforeShowPromise: () => waitForElement('[data-tour="tools-connected-section"]'),
    attachTo: { element: '[data-tour="tools-connected-section"]', on: 'top' },
    buttons: [
      { text: 'Back', classes: 'shepherd-button-secondary', action: tour.back },
      { text: 'Next', action: tour.next },
    ],
  })

  tour.addStep({
    id: 'tools-search',
    title: title('Find &', 'Connect'),
    text: `
      <p class="text-gray-300 mb-2">
        Search for any app or browse categories to find new integrations.
      </p>
      ${stepProgress(3, TOTAL)}
    `,
    beforeShowPromise: () => waitForElement('[data-tour="tools-search"]'),
    attachTo: { element: '[data-tour="tools-search"]', on: 'bottom' },
    buttons: [
      { text: 'Back', classes: 'shepherd-button-secondary', action: tour.back },
      { text: 'Got it!', action: tour.complete },
    ],
  })

  tour.on('complete', () => markTourComplete(TOUR_ID, userId))
  tour.on('cancel', () => markTourSkipped(TOUR_ID, userId))

  return tour
}
