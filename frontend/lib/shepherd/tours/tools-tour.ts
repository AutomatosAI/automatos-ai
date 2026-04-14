import Shepherd from 'shepherd.js'
import { shepherdTheme } from '../shepherd-theme'
import { markTourComplete, markTourSkipped } from '../tour-storage'
import { title, waitForElement, stepProgress } from '../tour-utils'

const TOUR_ID = 'tools'
const TOTAL = 2

export function createToolsTour(userId: string) {
  const tour = new Shepherd.Tour({
    ...shepherdTheme,
    exitOnEsc: true,
    keyboardNavigation: true,
  })

  // Step 1: About (centered)
  tour.addStep({
    id: 'tools-about',
    title: title('Tools &', 'Integrations'),
    text: `
      <p class="text-gray-300 mb-2">
        Connect Gmail, Slack, GitHub, Jira, HubSpot, CRMs, databases —
        1,000+ integrations. Authenticate once here and any agent you
        give access to can start calling them as tools.
      </p>
      ${stepProgress(1, TOTAL)}
    `,
    buttons: [
      { text: 'Skip', classes: 'shepherd-button-secondary', action: () => tour.cancel() },
      { text: 'Next', action: tour.next },
    ],
  })

  // Step 2: How to connect
  tour.addStep({
    id: 'tools-connect',
    title: title('Connect an', 'App'),
    text: `
      <p class="text-gray-300 mb-2">
        Search or browse by category, hit <strong>Connect</strong>, and
        walk through an OAuth flow or API key entry. Connected apps show
        a green badge. Click any connected app to reconfigure, re-auth,
        or see which agents are using it.
      </p>
      ${stepProgress(2, TOTAL)}
    `,
    beforeShowPromise: () => waitForElement('[data-tour="tools-page-header"]'),
    attachTo: { element: '[data-tour="tools-page-header"]', on: 'bottom' },
    buttons: [
      { text: 'Back', classes: 'shepherd-button-secondary', action: tour.back },
      { text: 'Got it!', action: tour.complete },
    ],
  })

  tour.on('complete', () => markTourComplete(TOUR_ID, userId))
  tour.on('cancel', () => markTourSkipped(TOUR_ID, userId))

  return tour
}
