import Shepherd from 'shepherd.js'
import { shepherdTheme } from '../shepherd-theme'
import { markTourComplete, markTourSkipped } from '../tour-storage'
import { title, waitForElement, stepProgress, tabList } from '../tour-utils'

const TOUR_ID = 'activity'
const TOTAL = 2

export function createActivityTour(userId: string) {
  const tour = new Shepherd.Tour({
    ...shepherdTheme,
    exitOnEsc: true,
    keyboardNavigation: true,
  })

  // Step 1: About (centered)
  tour.addStep({
    id: 'act-about',
    title: title('Command', 'Centre'),
    text: `
      <p class="text-gray-300 mb-2">
        A single pane of glass over your whole AI workforce. Every chat,
        scheduled task, running mission and completed report flows through
        this page — so you always know what's happening.
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
    id: 'act-tabs',
    title: title('Activity', 'Views'),
    text: `
      ${tabList([
        ['Summary', 'Dashboard of widgets — KPIs, recent reports, alerts, completions.'],
        ['Board', 'Kanban board of running tasks, grouped by status.'],
        ['Calendar', 'Scheduled runs, routines and mission deadlines.'],
        ['Memory', 'What your agents remember — long-term memory and context.'],
        ['Missions', 'Multi-step objectives agents are working through.'],
        ['Blog', 'Chronological event log of everything in the workspace.'],
      ])}
      ${stepProgress(2, TOTAL)}
    `,
    beforeShowPromise: () => waitForElement('[data-tour="activity-tabs"]'),
    attachTo: { element: '[data-tour="activity-tabs"]', on: 'bottom' },
    buttons: [
      { text: 'Back', classes: 'shepherd-button-secondary', action: tour.back },
      { text: 'Got it!', action: tour.complete },
    ],
  })

  tour.on('complete', () => markTourComplete(TOUR_ID, userId))
  tour.on('cancel', () => markTourSkipped(TOUR_ID, userId))

  return tour
}
