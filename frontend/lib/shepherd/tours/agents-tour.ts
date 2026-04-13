import Shepherd from 'shepherd.js'
import { shepherdTheme } from '../shepherd-theme'
import { markTourComplete, markTourSkipped } from '../tour-storage'
import { title, waitForElement, stepProgress, tabList } from '../tour-utils'

const TOUR_ID = 'agents'
const TOTAL = 5

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
        Your agent command centre. Every AI worker in the workspace lives here —
        you can create them, configure personas and tools, wire them into orgs,
        and hand them reusable recipes to run.
      </p>
      ${stepProgress(1, TOTAL)}
    `,
    beforeShowPromise: () => waitForElement('[data-tour="agents-page-header"]'),
    attachTo: { element: '[data-tour="agents-page-header"]', on: 'bottom' },
    buttons: [{ text: 'Next', action: tour.next }],
  })

  tour.addStep({
    id: 'agents-tabs',
    title: title('Five', 'Agent Views'),
    text: `
      <p class="text-gray-300 mb-2">
        The tab strip below the header gives you every angle on your workforce:
      </p>
      ${tabList([
        ['Roster', 'The full list of agents as cards or rows — create, edit, start/pause, delete.'],
        ['Org Chart', 'Visualise reporting lines and coordination hierarchy between agents.'],
        ['Configuration', 'Deep settings — model, system prompt, memory, guardrails.'],
        ['Coordination', 'Rules for how agents hand off work to each other and share context.'],
        ['Recipes', 'Reusable playbooks assigned to an agent — the jobs they know how to run.'],
      ])}
      ${stepProgress(2, TOTAL)}
    `,
    beforeShowPromise: () => waitForElement('[data-tour="agents-page-header"]'),
    attachTo: { element: '[data-tour="agents-page-header"]', on: 'bottom' },
    buttons: [
      { text: 'Back', classes: 'shepherd-button-secondary', action: tour.back },
      { text: 'Next', action: tour.next },
    ],
  })

  tour.addStep({
    id: 'agents-create',
    title: title('Create an', 'Agent'),
    text: `
      <p class="text-gray-300 mb-2">
        Build a new agent from scratch — pick a category, persona and model,
        then attach tools and skills. You can also start from a Marketplace template
        and customise from there.
      </p>
      ${stepProgress(3, TOTAL)}
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
        All your agents live here. Toggle between grid and list view, search by name,
        and filter by status to find what you need quickly.
      </p>
      ${stepProgress(4, TOTAL)}
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
        Each card has a menu for view details, configure, start/pause, clone, and delete.
        Click the card itself to open the full agent workspace with its chat, reports and memory.
      </p>
      ${stepProgress(5, TOTAL)}
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
