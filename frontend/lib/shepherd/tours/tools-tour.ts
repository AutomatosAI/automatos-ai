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
        This is where your workspace plugs into the outside world. Connect Gmail,
        Slack, GitHub, Jira, HubSpot, CRMs, databases — 150+ integrations — and any
        agent you give access to can start calling them as tools.
      </p>
      <p class="text-gray-400 text-sm">
        Authentication happens here once; agents reuse the saved credentials.
      </p>
      ${stepProgress(1, TOTAL)}
    `,
    buttons: [
      { text: 'Skip', classes: 'shepherd-button-secondary', action: () => tour.cancel() },
      { text: 'Next', action: tour.next },
    ],
  })

  tour.addStep({
    id: 'tools-connected',
    title: title('Connected', 'Apps'),
    text: `
      <p class="text-gray-300 mb-2">
        Anything you've already authenticated appears here as a card. Click one to
        reconfigure, re-auth, disable, or see which agents are currently using it.
      </p>
      <p class="text-gray-400 text-sm">
        Green = healthy connection. Amber = needs re-auth.
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
        Search for any service or browse by category — Communication, CRM, Dev,
        Marketing, Productivity. Hit <strong>Connect</strong> on a card and you'll
        walk through an OAuth flow or key entry, then it's available to every agent.
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
