import Shepherd from 'shepherd.js'
import { shepherdTheme } from '../shepherd-theme'
import { markTourComplete, markTourSkipped } from '../tour-storage'
import { title, waitForElement, stepProgress, tabList } from '../tour-utils'

const TOUR_ID = 'workspace'
const TOTAL = 3

export function createWorkspaceTour(userId: string) {
  const tour = new Shepherd.Tour({
    ...shepherdTheme,
    exitOnEsc: true,
    keyboardNavigation: true,
  })

  // Step 1: About (centered)
  tour.addStep({
    id: 'workspace-about',
    title: title('Your', 'Workspace Files'),
    text: `
      <p class="text-gray-300 mb-2">
        Every deliverable your agents produce — reports, code, documents,
        images — lands here. Three views give you gallery browsing, a full
        file explorer, and an activity timeline.
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
    id: 'workspace-tabs',
    title: title('Three Ways', 'to Browse'),
    text: `
      ${tabList([
        ['Outputs', 'Gallery of agent deliverables — reports, images, code, and docs. Filter by type or source (chat, tasks, missions, heartbeats, playbooks).'],
        ['Explorer', 'VS Code-style file browser with tabbed code editor, syntax highlighting, and a built-in interactive terminal.'],
        ['Activity', 'Live feed of all executions — chats, routines, playbooks, and missions with status tracking.'],
      ])}
      ${stepProgress(2, TOTAL)}
    `,
    beforeShowPromise: () => waitForElement('[data-tour="workspace-tabs"]'),
    attachTo: { element: '[data-tour="workspace-tabs"]', on: 'bottom' },
    buttons: [
      { text: 'Back', classes: 'shepherd-button-secondary', action: tour.back },
      { text: 'Next', action: tour.next },
    ],
  })

  // Step 3: Outputs preview
  tour.addStep({
    id: 'workspace-outputs',
    title: title('Agent', 'Deliverables'),
    text: `
      <p class="text-gray-300 mb-2">
        Click any output to preview it — markdown reports render inline,
        code gets syntax highlighting, images display full-size. Download,
        open in Explorer, or delete from the preview panel.
      </p>
      ${stepProgress(3, TOTAL)}
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
