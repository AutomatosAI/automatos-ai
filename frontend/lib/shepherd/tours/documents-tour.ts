import Shepherd from 'shepherd.js'
import { shepherdTheme } from '../shepherd-theme'
import { markTourComplete, markTourSkipped } from '../tour-storage'
import { title, waitForElement, stepProgress, tabList } from '../tour-utils'

const TOUR_ID = 'documents'
const TOTAL = 2

export function createDocumentsTour(userId: string) {
  const tour = new Shepherd.Tour({
    ...shepherdTheme,
    exitOnEsc: true,
    keyboardNavigation: true,
  })

  // Step 1: About (centered)
  tour.addStep({
    id: 'docs-about',
    title: title('Knowledge', 'Bases'),
    text: `
      <p class="text-gray-300 mb-2">
        This is where your agents get context. Upload files, connect
        databases, map code, and model the business — everything here is
        searchable and retrievable by any agent in the workspace.
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
    id: 'docs-tabs',
    title: title('Five', 'Knowledge Sources'),
    text: `
      ${tabList([
        ['Documents', 'Upload PDFs, Word, Markdown, images. Auto-chunked and searchable via RAG.'],
        ['Database', 'Connect Postgres, MySQL, Snowflake — agents query in natural language.'],
        ['Templates', 'Reusable document templates agents can fill in — proposals, reports, briefs.'],
        ['CodeGraph', 'Index a repo into a graph of files, symbols and calls for code reasoning.'],
        ['Business Graph', 'Your org as entities — customers, products, processes — agents can traverse.'],
      ])}
      ${stepProgress(2, TOTAL)}
    `,
    beforeShowPromise: () => waitForElement('[data-tour="documents-tabs"]'),
    attachTo: { element: '[data-tour="documents-tabs"]', on: 'bottom' },
    buttons: [
      { text: 'Back', classes: 'shepherd-button-secondary', action: tour.back },
      { text: 'Got it!', action: tour.complete },
    ],
  })

  tour.on('complete', () => markTourComplete(TOUR_ID, userId))
  tour.on('cancel', () => markTourSkipped(TOUR_ID, userId))

  return tour
}
