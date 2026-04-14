import Shepherd from 'shepherd.js'
import { shepherdTheme } from '../shepherd-theme'
import { markTourComplete, markTourSkipped } from '../tour-storage'
import { title, waitForElement, stepProgress, tabList } from '../tour-utils'

const TOUR_ID = 'settings'
const TOTAL = 2

export function createSettingsTour(userId: string) {
  const tour = new Shepherd.Tour({
    ...shepherdTheme,
    exitOnEsc: true,
    keyboardNavigation: true,
  })

  // Step 1: About (centered)
  tour.addStep({
    id: 'settings-about',
    title: title('Workspace', 'Settings'),
    text: `
      <p class="text-gray-300 mb-2">
        Everything that configures your workspace — model routing, API keys,
        external auth, channels, webhooks and embeddable widgets — lives here.
        Changes apply workspace-wide.
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
    id: 'settings-tabs',
    title: title('Settings', 'Areas'),
    text: `
      ${tabList([
        ['Orchestrator', 'LLM routing — which models handle chat, tools, embeddings, fallbacks.'],
        ['Webhooks', 'Outbound webhooks that fire on agent events or completions.'],
        ['API Keys', 'BYOK — bring your own OpenAI, Anthropic, or OpenRouter keys.'],
        ['Credentials', 'OAuth tokens and secrets agents use for external services.'],
        ['Channels', 'Inbound channels — email, Slack, Teams, voice — that feed agents.'],
        ['Voices', 'Voice profiles for TTS output and voice-channel agents.'],
        ['Widget SDK', 'Embeddable chat widget keys and site configuration.'],
      ])}
      ${stepProgress(2, TOTAL)}
    `,
    beforeShowPromise: () => waitForElement('[data-tour="settings-tabs"]'),
    attachTo: { element: '[data-tour="settings-tabs"]', on: 'bottom' },
    buttons: [
      { text: 'Back', classes: 'shepherd-button-secondary', action: tour.back },
      { text: 'Got it!', action: tour.complete },
    ],
  })

  tour.on('complete', () => markTourComplete(TOUR_ID, userId))
  tour.on('cancel', () => markTourSkipped(TOUR_ID, userId))

  return tour
}
