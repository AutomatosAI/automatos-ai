import Shepherd from 'shepherd.js'
import { shepherdTheme } from '../shepherd-theme'
import { markTourComplete, markTourSkipped } from '../tour-storage'
import { title, waitForElement, stepProgress, tabList } from '../tour-utils'

const TOUR_ID = 'agents'
const TOTAL = 2

export function createAgentsTour(userId: string) {
  const tour = new Shepherd.Tour({
    ...shepherdTheme,
    exitOnEsc: true,
    keyboardNavigation: true,
  })

  // Step 1: About (centered)
  tour.addStep({
    id: 'agents-about',
    title: title('Agent', 'Management'),
    text: `
      <p class="text-gray-300 mb-2">
        Your agent command centre. Every AI worker in the workspace lives
        here — create them, configure personas and tools, wire them into
        orgs, and hand them reusable playbooks to run.
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
    id: 'agents-tabs',
    title: title('Agent', 'Views'),
    text: `
      ${tabList([
        ['Roster', 'All agents as cards or rows — create, edit, start/pause, delete.'],
        ['Org Chart', 'Reporting lines and coordination hierarchy between agents.'],
        ['Configuration', 'Deep settings — model, system prompt, memory, guardrails.'],
        ['Coordination', 'Rules for how agents hand off work and share context.'],
        ['Playbooks', 'Reusable recipes assigned to an agent — the jobs they run.'],
      ])}
      ${stepProgress(2, TOTAL)}
    `,
    beforeShowPromise: () => waitForElement('[data-tour="agents-tabs"]'),
    attachTo: { element: '[data-tour="agents-tabs"]', on: 'bottom' },
    buttons: [
      { text: 'Back', classes: 'shepherd-button-secondary', action: tour.back },
      { text: 'Got it!', action: tour.complete },
    ],
  })

  tour.on('complete', () => markTourComplete(TOUR_ID, userId))
  tour.on('cancel', () => markTourSkipped(TOUR_ID, userId))

  return tour
}
