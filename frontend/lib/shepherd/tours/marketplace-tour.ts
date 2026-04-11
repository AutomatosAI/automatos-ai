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

  tour.addStep({
    id: 'mp-overview',
    title: title('Community', 'Marketplace'),
    text: `
      <p class="text-gray-300 mb-2">
        The fastest way to get productive. Browse pre-built agents, recipes,
        integrations and models — install any of them into your workspace with one click.
      </p>
      <p class="text-gray-400 text-sm">
        Anything you install here becomes available to every agent you create.
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
        Five top tabs cover everything you can add to your workspace:
      </p>
      ${tabList([
        ['Applications', 'Tool integrations — Gmail, Slack, GitHub, Jira, HubSpot and 150+ more.'],
        ['Agents', 'Ready-made personas (marketer, researcher, SDR, analyst) you can install and use immediately.'],
        ['Recipes', 'Multi-step playbooks and workflows — automations you can drop into any agent.'],
        ['LLMs', 'Model catalogue — pick which providers (OpenAI, Anthropic, DeepSeek, Qwen…) your agents can use.'],
        ['Capabilities', 'Plugins and skills that extend an agent with new behaviours or tool packs.'],
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

  tour.addStep({
    id: 'mp-search',
    title: title('Search', 'Anything'),
    text: `
      <p class="text-gray-300 mb-2">
        Looking for a specific tool or agent? Filter by name, category, or tag.
        Search works across whichever top tab you're in.
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
        Every card has an <strong>Install</strong> or <strong>Add</strong> button. Installed items
        show up in the matching workspace page — Agents, Tools, Playbooks, or Settings → Orchestrator.
      </p>
      <p class="text-gray-400 text-sm">
        You can remove anything later from its home page — installs are never permanent.
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
