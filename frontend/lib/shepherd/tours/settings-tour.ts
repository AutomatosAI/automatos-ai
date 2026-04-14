import Shepherd from 'shepherd.js'
import { shepherdTheme } from '../shepherd-theme'
import { markTourComplete, markTourSkipped } from '../tour-storage'
import { title, waitForElement, stepProgress, tabList } from '../tour-utils'

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
    title: title('Workspace', 'Settings'),
    text: `
      <p class="text-gray-300 mb-2">
        Everything that configures your workspace — system behaviour, model routing,
        external auth, webhooks, channels and embeddable widgets — lives here.
      </p>
      <p class="text-gray-400 text-sm">
        Changes apply to the whole workspace, so admin access is usually required.
      </p>
      ${stepProgress(1, TOTAL)}
    `,
    buttons: [
      { text: 'Skip', classes: 'shepherd-button-secondary', action: () => tour.cancel() },
      { text: 'Next', action: tour.next },
    ],
  })

  tour.addStep({
    id: 'settings-tabs',
    title: title('Seven', 'Settings Areas'),
    text: `
      <p class="text-gray-300 mb-2">
        Each tab handles a different class of configuration:
      </p>
      ${tabList([
        ['Orchestrator', 'LLM routing rules — which models handle chat, tools, embeddings, fallbacks.'],
        ['Webhooks', 'Outbound webhooks that fire on agent events, completions, or errors.'],
        ['API Keys', 'BYOK — bring your own OpenAI / Anthropic / OpenRouter keys.'],
        ['Credentials', 'Third-party credentials (OAuth, tokens) agents use to call external services.'],
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
      { text: 'Next', action: tour.next },
    ],
  })

  tour.addStep({
    id: 'settings-channels',
    title: title('Connect', 'Channels'),
    text: `
      <p class="text-gray-300 mb-2">
        The <strong>Channels</strong> tab is where you wire in email, Slack, Teams,
        Discord, Telegram, WhatsApp, voice, and more — so your agents can receive
        and respond on the platforms your team already uses.
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
