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
      <p class="text-gray-300 mb-3">
        A single pane of glass over your whole AI workforce — this is where you
        monitor, manage, and steer everything that's happening in your workspace.
      </p>
      <p class="text-gray-300 mb-3">
        See which agents are <strong>running right now</strong>, review completed
        <strong>reports</strong>, track <strong>mission progress</strong>, check
        <strong>scheduled tasks</strong>, and catch anything that needs your attention.
      </p>
      <p class="text-gray-400 text-sm">
        Think of it as mission control — everything flows through here so
        nothing slips through the cracks.
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
      <p class="text-gray-300 mb-3">
        Every item is clickable — open a report to grade it,
        open a mission to see progress, open a memory entry to inspect what
        an agent learned.
      </p>
      <p class="text-gray-300 mb-2">
        Hit the <strong>Customize</strong> button on the Summary dashboard to
        rearrange widgets, hide ones you don't need, and make the layout yours.
        Drag and drop to reorder — your layout is saved automatically.
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
