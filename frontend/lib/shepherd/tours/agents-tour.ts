import Shepherd from 'shepherd.js'
import { shepherdTheme } from '../shepherd-theme'
import { markTourComplete, markTourSkipped } from '../tour-storage'
import { title, waitForElement, stepProgress, tabList } from '../tour-utils'

const TOUR_ID = 'agents'
const TOTAL = 4

export function createAgentsTour(userId: string) {
  const tour = new Shepherd.Tour({
    ...shepherdTheme,
    exitOnEsc: true,
    keyboardNavigation: true,
  })

  // Step 1: Overview (centered)
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
    buttons: [
      { text: 'Skip', classes: 'shepherd-button-secondary', action: () => tour.cancel() },
      { text: 'Next', action: tour.next },
    ],
  })

  // Step 2: Tabs
  tour.addStep({
    id: 'agents-tabs',
    title: title('Five', 'Agent Views'),
    text: `
      ${tabList([
        ['Roster', 'The full list of agents as cards or rows — create, edit, start/pause, delete.'],
        ['Org Chart', 'Visualise reporting lines and coordination hierarchy between agents.'],
        ['Configuration', 'Deep settings — model, system prompt, memory, guardrails.'],
        ['Coordination', 'Rules for how agents hand off work to each other and share context.'],
        ['Recipes', 'Reusable playbooks assigned to an agent — the jobs they know how to run.'],
      ])}
      ${stepProgress(2, TOTAL)}
    `,
    beforeShowPromise: () => waitForElement('[data-tour="agents-tabs"]'),
    attachTo: { element: '[data-tour="agents-tabs"]', on: 'bottom' },
    buttons: [
      { text: 'Back', classes: 'shepherd-button-secondary', action: tour.back },
      { text: 'Next', action: tour.next },
    ],
  })

  // Step 3: Create agent button
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

  // Step 4: Agent cards (centered)
  tour.addStep({
    id: 'agents-cards',
    title: title('Your', 'Agent Roster'),
    text: `
      <p class="text-gray-300 mb-2">
        Each agent appears as a card with status, model, and a menu for configure,
        start/pause, clone, and delete. Click a card to open its full workspace
        with chat, reports, and memory.
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
