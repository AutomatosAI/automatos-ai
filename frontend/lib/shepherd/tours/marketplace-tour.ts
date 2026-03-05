import Shepherd from 'shepherd.js'
import { shepherdTheme } from '../shepherd-theme'
import { markTourComplete, markTourSkipped } from '../tour-storage'
import { title, waitForElement, stepProgress } from '../tour-utils'

const TOUR_ID = 'marketplace'
const TOTAL = 4

export function createMarketplaceTour(userId: string) {
  const tour = new Shepherd.Tour({
    ...shepherdTheme,
    exitOnEsc: true,
    keyboardNavigation: true,
  })

  tour.addStep({
    id: 'mp-overview',
    title: title('Community', 'Marketplace'),
    text: `
      <p class="text-gray-300 mb-2">
        Browse pre-built agents, recipes, tools, and LLMs — install anything with one click.
      </p>
      ${stepProgress(1, TOTAL)}
    `,
    beforeShowPromise: () => waitForElement('[data-tour="marketplace-stats"]'),
    attachTo: { element: '[data-tour="marketplace-stats"]', on: 'bottom' },
    buttons: [{ text: 'Next', action: tour.next }],
  })

  tour.addStep({
    id: 'mp-tabs',
    title: title('Browse by', 'Category'),
    text: `
      <p class="text-gray-300 mb-2">
        Switch between Applications, Agents, Recipes, LLMs, Capabilities, and Skills.
      </p>
      ${stepProgress(2, TOTAL)}
    `,
    beforeShowPromise: () => waitForElement('[data-tour="marketplace-tabs"]'),
    attachTo: { element: '[data-tour="marketplace-tabs"]', on: 'bottom' },
    buttons: [
      { text: 'Back', classes: 'shepherd-button-secondary', action: tour.back },
      { text: 'Next', action: tour.next },
    ],
  })

  tour.addStep({
    id: 'mp-search',
    title: title('Search', 'Anything'),
    text: `
      <p class="text-gray-300 mb-2">
        Looking for something specific? Type here to filter by name.
      </p>
      ${stepProgress(3, TOTAL)}
    `,
    beforeShowPromise: () => waitForElement('[data-tour="marketplace-search"]'),
    attachTo: { element: '[data-tour="marketplace-search"]', on: 'bottom' },
    buttons: [
      { text: 'Back', classes: 'shepherd-button-secondary', action: tour.back },
      { text: 'Next', action: tour.next },
    ],
  })

  tour.addStep({
    id: 'mp-install',
    title: title('One-Click', 'Install'),
    text: `
      <p class="text-gray-300 mb-2">
        Found something you like? Hit <strong>Install</strong> or <strong>Add</strong> to start using it immediately.
      </p>
      ${stepProgress(4, TOTAL)}
    `,
    beforeShowPromise: () => waitForElement('[data-tour="marketplace-content"]'),
    attachTo: { element: '[data-tour="marketplace-content"]', on: 'top' },
    buttons: [
      { text: 'Back', classes: 'shepherd-button-secondary', action: tour.back },
      { text: 'Got it!', action: tour.complete },
    ],
  })

  tour.on('complete', () => markTourComplete(TOUR_ID, userId))
  tour.on('cancel', () => markTourSkipped(TOUR_ID, userId))

  return tour
}
