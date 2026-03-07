import Shepherd from 'shepherd.js'
import { shepherdTheme } from '../shepherd-theme'
import { markTourComplete, markTourSkipped } from '../tour-storage'
import { title, waitForElement, stepProgress } from '../tour-utils'

const TOUR_ID = 'settings'
const TOTAL = 3

export function createSettingsTour(userId: string) {
  const tour = new Shepherd.Tour({
    ...shepherdTheme,
    exitOnEsc: true,
    keyboardNavigation: true,
  })

  tour.addStep({
    id: 'settings-overview',
    title: title('System', 'Settings'),
    text: `
      <p class="text-gray-300 mb-2">
        Manage your workspace configuration, API keys, credentials, and security settings.
      </p>
      ${stepProgress(1, TOTAL)}
    `,
    beforeShowPromise: () => waitForElement('[data-tour="settings-page-header"]'),
    attachTo: { element: '[data-tour="settings-page-header"]', on: 'bottom' },
    buttons: [{ text: 'Next', action: tour.next }],
  })

  tour.addStep({
    id: 'settings-tabs',
    title: title('Navigate', 'Tabs'),
    text: `
      <p class="text-gray-300 mb-2">
        System Settings, Orchestrator, Webhooks, API Keys, Credentials, Audit Logs, Channels, and more.
      </p>
      ${stepProgress(2, TOTAL)}
    `,
    beforeShowPromise: () => waitForElement('[data-tour="settings-tabs"]'),
    attachTo: { element: '[data-tour="settings-tabs"]', on: 'bottom' },
    buttons: [
      { text: 'Back', classes: 'shepherd-button-secondary', action: tour.back },
      { text: 'Next', action: tour.next },
    ],
  })

  tour.addStep({
    id: 'settings-credentials',
    title: title('API Keys &', 'Credentials'),
    text: `
      <p class="text-gray-300 mb-2">
        Store your API keys and service credentials securely. Agents use these to connect to external services.
      </p>
      ${stepProgress(3, TOTAL)}
    `,
    beforeShowPromise: () => waitForElement('[data-tour="settings-credentials-tab"]'),
    attachTo: { element: '[data-tour="settings-credentials-tab"]', on: 'bottom' },
    buttons: [
      { text: 'Back', classes: 'shepherd-button-secondary', action: tour.back },
      { text: 'Got it!', action: tour.complete },
    ],
  })

  tour.on('complete', () => markTourComplete(TOUR_ID, userId))
  tour.on('cancel', () => markTourSkipped(TOUR_ID, userId))

  return tour
}
