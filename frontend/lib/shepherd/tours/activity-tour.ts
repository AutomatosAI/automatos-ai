import Shepherd from 'shepherd.js'
import { shepherdTheme } from '../shepherd-theme'
import { markTourComplete, markTourSkipped } from '../tour-storage'
import { title, waitForElement, stepProgress } from '../tour-utils'

const TOUR_ID = 'activity'
const TOTAL = 4

export function createActivityTour(userId: string) {
  const tour = new Shepherd.Tour({
    ...shepherdTheme,
    exitOnEsc: true,
    keyboardNavigation: true,
  })

  tour.addStep({
    id: 'act-overview',
    title: title('Command', 'Centre'),
    text: `
      <p class="text-gray-300 mb-2">
        Your AI workforce at a glance — see every chat, routine, recipe, and mission in one place.
      </p>
      ${stepProgress(1, TOTAL)}
    `,
    beforeShowPromise: () => waitForElement('[data-tour="activity-page-header"]'),
    attachTo: { element: '[data-tour="activity-page-header"]', on: 'bottom' },
    buttons: [{ text: 'Next', action: tour.next }],
  })

  tour.addStep({
    id: 'act-stats',
    title: title('Live', 'Stats'),
    text: `
      <p class="text-gray-300 mb-2">
        Real-time counters — what's working now, connected channels, completions, and anything that needs attention.
      </p>
      ${stepProgress(2, TOTAL)}
    `,
    beforeShowPromise: () => waitForElement('[data-tour="activity-stats"]'),
    attachTo: { element: '[data-tour="activity-stats"]', on: 'bottom' },
    buttons: [
      { text: 'Back', classes: 'shepherd-button-secondary', action: tour.back },
      { text: 'Next', action: tour.next },
    ],
  })

  tour.addStep({
    id: 'act-tabs',
    title: title('Navigate', 'Tabs'),
    text: `
      <p class="text-gray-300 mb-2">
        Switch between the unified Feed, Routines (recurring agent tasks), Recipes (automations), and Missions (coming soon).
      </p>
      ${stepProgress(3, TOTAL)}
    `,
    beforeShowPromise: () => waitForElement('[data-tour="activity-tabs"]'),
    attachTo: { element: '[data-tour="activity-tabs"]', on: 'bottom' },
    buttons: [
      { text: 'Back', classes: 'shepherd-button-secondary', action: tour.back },
      { text: 'Next', action: tour.next },
    ],
  })

  tour.addStep({
    id: 'act-content',
    title: title('Activity', 'Feed'),
    text: `
      <p class="text-gray-300 mb-2">
        The live feed shows all activity across your workspace. Filter by type or status, and click any item to drill in.
      </p>
      ${stepProgress(4, TOTAL)}
    `,
    beforeShowPromise: () => waitForElement('[data-tour="activity-content"]'),
    attachTo: { element: '[data-tour="activity-content"]', on: 'top' },
    buttons: [
      { text: 'Back', classes: 'shepherd-button-secondary', action: tour.back },
      { text: 'Got it!', action: tour.complete },
    ],
  })

  tour.on('complete', () => markTourComplete(TOUR_ID, userId))
  tour.on('cancel', () => markTourSkipped(TOUR_ID, userId))

  return tour
}
