import Shepherd from 'shepherd.js'
import { shepherdTheme } from '../shepherd-theme'
import { markTourComplete, markTourSkipped } from '../tour-storage'
import { title, waitForElement, stepProgress, tabList } from '../tour-utils'

const TOUR_ID = 'marketplace'
const TOTAL = 4

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
        Pre-built agents, automation playbooks, integration tools, LLM models,
        and capabilities — all installable with one click. This is the fastest
        way to get your workspace productive.
      </p>
      ${stepProgress(1, TOTAL)}
    `,
    buttons: [
      { text: 'Skip', classes: 'shepherd-button-secondary', action: () => tour.cancel() },
      { text: 'Next', action: tour.next },
    ],
  })

  // Step 2: Tabs
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
      { text: 'Next', action: tour.next },
    ],
  })

  // Step 3: Featured
  tour.addStep({
    id: 'mp-featured',
    title: title('Featured &', 'Recommended'),
    text: `
      <p class="text-gray-300 mb-2">
        The featured section highlights curated picks. Below that,
        <strong>Recommended for You</strong> suggests items based on your
        workspace setup. Browse, search, or filter within any tab.
      </p>
      ${stepProgress(3, TOTAL)}
    `,
    beforeShowPromise: () => waitForElement('[data-tour="marketplace-tabs"]'),
    attachTo: { element: '[data-tour="marketplace-tabs"]', on: 'bottom' },
    buttons: [
      { text: 'Back', classes: 'shepherd-button-secondary', action: tour.back },
      { text: 'Next', action: tour.next },
    ],
  })

  // Step 4: Install
  tour.addStep({
    id: 'mp-install',
    title: title('One-Click', 'Install'),
    text: `
      <p class="text-gray-300 mb-2">
        Click <strong>Add to Workspace</strong> and it's yours.
      </p>
      <p class="text-gray-400 text-sm">
        <strong>Important:</strong> LLMs and Tools must be added from the
        Marketplace before agents can use them. Installed items appear in
        their respective pages — Agents, Tools, Playbooks, or Settings.
      </p>
      ${stepProgress(4, TOTAL)}
    `,
    buttons: [
      { text: 'Back', classes: 'shepherd-button-secondary', action: tour.back },
      { text: 'Got it!', action: tour.complete },
    ],
  })

  tour.on('complete', () => markTourComplete(TOUR_ID, userId))
  tour.on('cancel', () => markTourSkipped(TOUR_ID, userId))

  return tour
}
