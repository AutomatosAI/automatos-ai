import Shepherd from 'shepherd.js'
import { shepherdTheme } from '../shepherd-theme'
import { markTourComplete, markTourSkipped } from '../tour-storage'
import { title, waitForElement, stepProgress } from '../tour-utils'

const TOUR_ID = 'analytics'
const TOTAL = 3

export function createAnalyticsTour(userId: string) {
  const tour = new Shepherd.Tour({
    ...shepherdTheme,
    exitOnEsc: true,
    keyboardNavigation: true,
  })

  tour.addStep({
    id: 'analytics-overview',
    title: title('', 'Analytics'),
    text: `
      <p class="text-gray-300 mb-2">
        Track agent performance, workflow success rates, document usage, and LLM costs.
      </p>
      ${stepProgress(1, TOTAL)}
    `,
    beforeShowPromise: () => waitForElement('[data-tour="analytics-page-header"]'),
    attachTo: { element: '[data-tour="analytics-page-header"]', on: 'bottom' },
    buttons: [{ text: 'Next', action: tour.next }],
  })

  tour.addStep({
    id: 'analytics-tabs',
    title: title('Drill', 'Down'),
    text: `
      <p class="text-gray-300 mb-2">
        Switch between Overview, Agents, Workflows, Documents, Costs, and Tools views.
      </p>
      ${stepProgress(2, TOTAL)}
    `,
    beforeShowPromise: () => waitForElement('[data-tour="analytics-tabs"]'),
    attachTo: { element: '[data-tour="analytics-tabs"]', on: 'bottom' },
    buttons: [
      { text: 'Back', classes: 'shepherd-button-secondary', action: tour.back },
      { text: 'Next', action: tour.next },
    ],
  })

  tour.addStep({
    id: 'analytics-timerange',
    title: title('Time', 'Range'),
    text: `
      <p class="text-gray-300 mb-2">
        Filter data by 7, 30, or 90 days to spot trends and track changes.
      </p>
      ${stepProgress(3, TOTAL)}
    `,
    beforeShowPromise: () => waitForElement('[data-tour="analytics-time-range"]'),
    attachTo: { element: '[data-tour="analytics-time-range"]', on: 'bottom' },
    buttons: [
      { text: 'Back', classes: 'shepherd-button-secondary', action: tour.back },
      { text: 'Got it!', action: tour.complete },
    ],
  })

  tour.on('complete', () => markTourComplete(TOUR_ID, userId))
  tour.on('cancel', () => markTourSkipped(TOUR_ID, userId))

  return tour
}
