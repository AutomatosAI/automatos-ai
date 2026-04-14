import Shepherd from 'shepherd.js'
import { shepherdTheme } from '../shepherd-theme'
import { markTourComplete, markTourSkipped } from '../tour-storage'
import { title, waitForElement, stepProgress, tabList } from '../tour-utils'

const TOUR_ID = 'workspace'
const TOTAL = 2

export function createWorkspaceTour(userId: string) {
  const tour = new Shepherd.Tour({
    ...shepherdTheme,
    exitOnEsc: true,
    keyboardNavigation: true,
  })

  // Step 1: About (centered)
  tour.addStep({
    id: 'workspace-about',
    title: title('Your', 'Workspace'),
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

  // Step 2: Tabs (attached to tab bar)
  tour.addStep({
    id: 'workspace-tabs',
    title: title('Three', 'Views'),
    text: `
      ${tabList([
        ['Outputs', 'Gallery of agent deliverables — reports, images, code, and docs.'],
        ['Explorer', 'VS Code-style file browser with tabbed editor and terminal.'],
        ['Activity', 'Live feed of all executions — chats, routines, playbooks, missions.'],
      ])}
      ${stepProgress(2, TOTAL)}
    `,
    beforeShowPromise: () => waitForElement('[data-tour="workspace-tabs"]'),
    attachTo: { element: '[data-tour="workspace-tabs"]', on: 'bottom' },
    buttons: [
      { text: 'Back', classes: 'shepherd-button-secondary', action: tour.back },
      { text: 'Got it!', action: tour.complete },
    ],
  })

  tour.on('complete', () => markTourComplete(TOUR_ID, userId))
  tour.on('cancel', () => markTourSkipped(TOUR_ID, userId))

  return tour
}
