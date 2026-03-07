import Shepherd from 'shepherd.js'
import { shepherdTheme } from '../shepherd-theme'
import { markTourComplete, markTourSkipped } from '../tour-storage'
import { title, waitForElement, stepProgress } from '../tour-utils'

const TOUR_ID = 'workflows'
const TOTAL = 4

export function createWorkflowsTour(userId: string) {
  const tour = new Shepherd.Tour({
    ...shepherdTheme,
    exitOnEsc: true,
    keyboardNavigation: true,
  })

  tour.addStep({
    id: 'wf-overview',
    title: title('Workflow', 'Management'),
    text: `
      <p class="text-gray-300 mb-2">
        Create multi-agent workflows from recipes, monitor running jobs, and track results.
      </p>
      ${stepProgress(1, TOTAL)}
    `,
    beforeShowPromise: () => waitForElement('[data-tour="workflows-page-header"]'),
    attachTo: { element: '[data-tour="workflows-page-header"]', on: 'bottom' },
    buttons: [{ text: 'Next', action: tour.next }],
  })

  tour.addStep({
    id: 'wf-recipes',
    title: title('Workflow', 'Recipes'),
    text: `
      <p class="text-gray-300 mb-2">
        Recipes are reusable workflow templates. Pick one, customize inputs, and run.
      </p>
      ${stepProgress(2, TOTAL)}
    `,
    beforeShowPromise: () => waitForElement('[data-tour="workflows-recipes-tab"]'),
    attachTo: { element: '[data-tour="workflows-recipes-tab"]', on: 'bottom' },
    buttons: [
      { text: 'Back', classes: 'shepherd-button-secondary', action: tour.back },
      { text: 'Next', action: tour.next },
    ],
  })

  tour.addStep({
    id: 'wf-active',
    title: title('Active', 'Workflows'),
    text: `
      <p class="text-gray-300 mb-2">
        See what's cooking — monitor running workflows, view progress, and check results.
      </p>
      ${stepProgress(3, TOTAL)}
    `,
    beforeShowPromise: () => waitForElement('[data-tour="workflows-active-tab"]'),
    attachTo: { element: '[data-tour="workflows-active-tab"]', on: 'bottom' },
    buttons: [
      { text: 'Back', classes: 'shepherd-button-secondary', action: tour.back },
      { text: 'Next', action: tour.next },
    ],
  })

  tour.addStep({
    id: 'wf-create',
    title: title('Create a', 'Recipe'),
    text: `
      <p class="text-gray-300 mb-2">
        Design your own workflow recipe with custom steps, agents, and triggers.
      </p>
      ${stepProgress(4, TOTAL)}
    `,
    beforeShowPromise: () => waitForElement('[data-tour="workflows-create-btn"]'),
    attachTo: { element: '[data-tour="workflows-create-btn"]', on: 'bottom' },
    buttons: [
      { text: 'Back', classes: 'shepherd-button-secondary', action: tour.back },
      { text: 'Got it!', action: tour.complete },
    ],
  })

  tour.on('complete', () => markTourComplete(TOUR_ID, userId))
  tour.on('cancel', () => markTourSkipped(TOUR_ID, userId))

  return tour
}
