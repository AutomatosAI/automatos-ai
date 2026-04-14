import Shepherd from 'shepherd.js'
import { shepherdTheme } from '../shepherd-theme'
import { markTourComplete, markTourSkipped } from '../tour-storage'
import { title, waitForElement, stepProgress, tabList } from '../tour-utils'

const TOUR_ID = 'marketplace'
const TOTAL = 2

export function createMarketplaceTour(userId: string) {
  const tour = new Shepherd.Tour({
    ...shepherdTheme,
    exitOnEsc: true,
    keyboardNavigation: true,
  })

  // Step 1: About (centered)
  tour.addStep({
    id: 'mp-about',
    title: title('Community', 'Marketplace'),
    text: `
      <p class="text-gray-300 mb-2">
        Pre-built agents, playbooks, integrations, LLM models, and
        capabilities — all installable with one click. The fastest way
        to get your workspace productive.
      </p>
      ${stepProgress(1, TOTAL)}
    `,
    buttons: [
      { text: 'Skip', classes: 'shepherd-button-secondary', action: () => tour.cancel() },
      { text: 'Next', action: tour.next },
    ],
  })

  // Step 2: Tabs (attached to tab bar)
  tour.addStep({
    id: 'mp-tabs',
    title: title('Browse by', 'Category'),
    text: `
      ${tabList([
        ['Applications', 'Tool integrations — Gmail, Slack, GitHub, and 1,000+ more.'],
        ['Agents', 'Ready-made personas you can install and customise.'],
        ['Playbooks', 'Multi-step automation recipes for common workflows.'],
        ['LLMs', 'Model catalogue with pricing, context windows, and capabilities.'],
        ['Capabilities', 'Plugins and skills that extend what agents can do.'],
      ])}
      ${stepProgress(2, TOTAL)}
    `,
    beforeShowPromise: () => waitForElement('[data-tour="marketplace-tabs"]'),
    attachTo: { element: '[data-tour="marketplace-tabs"]', on: 'bottom' },
    buttons: [
      { text: 'Back', classes: 'shepherd-button-secondary', action: tour.back },
      { text: 'Got it!', action: tour.complete },
    ],
  })

  tour.on('complete', () => markTourComplete(TOUR_ID, userId))
  tour.on('cancel', () => markTourSkipped(TOUR_ID, userId))

  return tour
}
