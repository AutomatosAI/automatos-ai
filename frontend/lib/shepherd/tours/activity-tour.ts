import Shepherd from 'shepherd.js'
import { shepherdTheme } from '../shepherd-theme'
import { markTourComplete, markTourSkipped } from '../tour-storage'
import { title, waitForElement, stepProgress, tabList } from '../tour-utils'

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
        A single pane of glass over your whole AI workforce. Every chat, scheduled task,
        running mission and completed report flows through this page, so you always know
        what's happening and what needs you.
      </p>
      ${stepProgress(1, TOTAL)}
    `,
    buttons: [
      { text: 'Skip', classes: 'shepherd-button-secondary', action: () => tour.cancel() },
      { text: 'Next', action: tour.next },
    ],
  })

  tour.addStep({
    id: 'act-stats',
    title: title('Live', 'Stats'),
    text: `
      <p class="text-gray-300 mb-2">
        Real-time counters at the top — what's working right now, connected channels,
        completions today, and anything that needs your attention. Click any tile
        to jump straight to a filtered view.
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
    title: title('Six', 'Activity Views'),
    text: `
      <p class="text-gray-300 mb-2">
        Switch between different ways of looking at the same workforce:
      </p>
      ${tabList([
        ['Summary', 'Dashboard of widgets — KPIs, recent reports, alerts, completions.'],
        ['Board', 'Kanban board of running tasks and executions, grouped by status.'],
        ['Calendar', 'Scheduled runs, routines and mission deadlines on a calendar.'],
        ['Memory', 'What your agents remember — long-term memory, daily logs, context.'],
        ['Missions', 'Multi-step objectives agents are working through.'],
        ['Blog', 'Chronological event log of everything that happened in the workspace.'],
      ])}
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
    title: title('Drill', 'Into Anything'),
    text: `
      <p class="text-gray-300 mb-2">
        Whichever view you pick, every item is clickable. Open a report to grade it,
        open a mission to see progress, open a memory entry to inspect what the
        agent learned.
      </p>
      ${stepProgress(4, TOTAL)}
    `,
    // Centered — no specific attach target needed for final step
    buttons: [
      { text: 'Back', classes: 'shepherd-button-secondary', action: tour.back },
      { text: 'Got it!', action: tour.complete },
    ],
  })

  tour.on('complete', () => markTourComplete(TOUR_ID, userId))
  tour.on('cancel', () => markTourSkipped(TOUR_ID, userId))

  return tour
}
