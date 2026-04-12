import Shepherd from 'shepherd.js'
import { shepherdTheme } from '../shepherd-theme'
import { markTourComplete, markTourSkipped } from '../tour-storage'
import { title, waitForElement, stepProgress, tabList } from '../tour-utils'

const TOUR_ID = 'documents'
const TOTAL = 4

export function createDocumentsTour(userId: string) {
  const tour = new Shepherd.Tour({
    ...shepherdTheme,
    exitOnEsc: true,
    keyboardNavigation: true,
  })

  // Step 1 — Overview
  tour.addStep({
    id: 'docs-overview',
    title: title('Knowledge', 'Bases'),
    text: `
      <p class="text-gray-300 mb-2">
        This is where your agents get their brains. Upload files, connect databases,
        map code, and model the business — everything here is searchable and retrievable
        by any agent in the workspace.
      </p>
      <p class="text-gray-400 text-sm">
        Think of it as five knowledge sources sitting behind one unified RAG layer.
      </p>
      ${stepProgress(1, TOTAL)}
    `,
    beforeShowPromise: () => waitForElement('[data-tour="documents-page-header"]'),
    attachTo: { element: '[data-tour="documents-page-header"]', on: 'bottom' },
    buttons: [{ text: 'Next', action: tour.next }],
  })

  // Step 2 — Top-level tabs walkthrough
  tour.addStep({
    id: 'docs-tabs',
    title: title('The Five', 'Knowledge Sources'),
    text: `
      <p class="text-gray-300 mb-2">
        Each top tab is a different way of giving your agents context:
      </p>
      ${tabList([
        ['Documents', 'Upload PDFs, Word, Markdown, images. Auto-chunked, embedded, and searchable via RAG.'],
        ['Database', 'Connect Postgres, MySQL, Snowflake and let agents query them in natural language with a semantic layer on top.'],
        ['Templates', 'Reusable document templates agents can fill in — proposals, reports, briefs.'],
        ['CodeGraph', 'Index a repo into a graph of files, symbols and calls so agents can reason about code structure.'],
        ['Business Graph', 'Your org modelled as entities — customers, products, processes — agents can traverse.'],
      ])}
      ${stepProgress(2, TOTAL)}
    `,
    beforeShowPromise: () => waitForElement('[data-tour="documents-tabs"]'),
    attachTo: { element: '[data-tour="documents-tabs"]', on: 'bottom' },
    buttons: [
      { text: 'Back', classes: 'shepherd-button-secondary', action: tour.back },
      { text: 'Next', action: tour.next },
    ],
  })

  // Step 3 — Upload
  tour.addStep({
    id: 'docs-upload',
    title: title('Upload', 'Documents'),
    text: `
      <p class="text-gray-300 mb-2">
        Drag-and-drop or click to upload. Files are chunked, embedded, and indexed
        automatically — you'll see them appear in <strong>Library</strong> once processing completes.
      </p>
      <p class="text-gray-400 text-sm">
        Supports PDF, DOCX, Markdown, TXT, CSV, and images (OCR runs automatically).
      </p>
      ${stepProgress(3, TOTAL)}
    `,
    beforeShowPromise: () => waitForElement('[data-tour="documents-upload-btn"]'),
    attachTo: { element: '[data-tour="documents-upload-btn"]', on: 'bottom' },
    buttons: [
      { text: 'Back', classes: 'shepherd-button-secondary', action: tour.back },
      { text: 'Next', action: tour.next },
    ],
  })

  // Step 4 — Sub-tabs inside Documents
  tour.addStep({
    id: 'docs-subtabs',
    title: title('Inside the', 'Documents Tab'),
    text: `
      <p class="text-gray-300 mb-2">
        Once you're in <strong>Documents</strong>, the inner strip gives you everything
        you need end-to-end:
      </p>
      ${tabList([
        ['Library', 'Every uploaded file — search, preview, delete.'],
        ['Processing', 'Live status of chunking, embedding, OCR jobs.'],
        ['Multimodal', 'Images and mixed-content files with OCR + vision extraction.'],
        ['Search', 'Semantic search across the whole knowledge base.'],
        ['RAG Test', 'Sandbox to test how your agents will retrieve this content.'],
        ['Upload', 'The drop zone again, for convenience.'],
      ])}
      ${stepProgress(4, TOTAL)}
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
