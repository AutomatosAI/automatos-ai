import Shepherd from 'shepherd.js'
import { shepherdTheme } from '../shepherd-theme'
import { markTourComplete, markTourSkipped } from '../tour-storage'
import { title, waitForElement, stepProgress } from '../tour-utils'

const TOUR_ID = 'documents'
const TOTAL = 3

export function createDocumentsTour(userId: string) {
  const tour = new Shepherd.Tour({
    ...shepherdTheme,
    exitOnEsc: true,
    keyboardNavigation: true,
  })

  tour.addStep({
    id: 'docs-overview',
    title: title('Knowledge', 'Bases'),
    text: `
      <p class="text-gray-300 mb-2">
        Upload documents, connect cloud storage, and build searchable knowledge for your agents.
      </p>
      ${stepProgress(1, TOTAL)}
    `,
    beforeShowPromise: () => waitForElement('[data-tour="documents-page-header"]'),
    attachTo: { element: '[data-tour="documents-page-header"]', on: 'bottom' },
    buttons: [{ text: 'Next', action: tour.next }],
  })

  tour.addStep({
    id: 'docs-upload',
    title: title('Upload', 'Documents'),
    text: `
      <p class="text-gray-300 mb-2">
        Drag and drop files or click to upload. Supports PDF, Word, text, and more.
      </p>
      ${stepProgress(2, TOTAL)}
    `,
    beforeShowPromise: () => waitForElement('[data-tour="documents-upload-btn"]'),
    attachTo: { element: '[data-tour="documents-upload-btn"]', on: 'bottom' },
    buttons: [
      { text: 'Back', classes: 'shepherd-button-secondary', action: tour.back },
      { text: 'Next', action: tour.next },
    ],
  })

  tour.addStep({
    id: 'docs-tabs',
    title: title('Browse', 'Sources'),
    text: `
      <p class="text-gray-300 mb-2">
        Switch between local uploads, cloud storage, databases, and semantic search.
      </p>
      ${stepProgress(3, TOTAL)}
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
