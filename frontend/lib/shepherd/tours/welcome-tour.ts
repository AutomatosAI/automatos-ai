import Shepherd from 'shepherd.js'
import { shepherdTheme } from '../shepherd-theme'
import { markTourComplete, markTourSkipped } from '../tour-storage'
import { title, waitForElement, stepProgress, tabList } from '../tour-utils'

const TOUR_ID = 'welcome'
const TOTAL_STEPS = 5

export function createWelcomeTour(userId: string) {
  const tour = new Shepherd.Tour({
    ...shepherdTheme,
    exitOnEsc: true,
    keyboardNavigation: true,
  })

  // Step 1: Welcome (no target — centered)
  tour.addStep({
    id: 'welcome-intro',
    title: title('Welcome to', 'Automatos AI'),
    text: `
      <p class="text-gray-300 mb-3">
        Quick 60-second orientation — we'll show you where everything lives.
      </p>
      <p class="text-xs text-gray-500">
        Press ESC anytime to skip
      </p>
      ${stepProgress(1, TOTAL_STEPS)}
    `,
    buttons: [
      {
        text: 'Skip Tour',
        classes: 'shepherd-button-secondary',
        action: () => tour.cancel(),
      },
      {
        text: "Let's Go!",
        action: tour.next,
      },
    ],
  })

  // Step 2: Sidebar navigation
  tour.addStep({
    id: 'welcome-sidebar',
    title: title('Your', 'Navigation Hub'),
    text: `
      <p class="text-gray-300 mb-2">
        The sidebar is the map of the whole platform. Each item is a major area
        of your workspace:
      </p>
      ${tabList([
        ['Chat', 'Talk to any agent — your day-to-day home base.'],
        ['Activity', 'Command centre for everything your agents are doing.'],
        ['Agents', 'Create, configure and coordinate your AI workforce.'],
        ['Marketplace', 'One-click installs for agents, recipes, tools and LLMs.'],
        ['Knowledge Bases', 'Documents, databases, templates, code & business graphs.'],
        ['Tools', '150+ integrations — Gmail, Slack, GitHub, CRMs, and more.'],
        ['Analytics', 'Performance, usage and cost dashboards.'],
        ['Settings', 'Workspace config, API keys, credentials and channels.'],
      ])}
      ${stepProgress(2, TOTAL_STEPS)}
    `,
    beforeShowPromise: () => waitForElement('[data-tour="sidebar"]'),
    attachTo: {
      element: '[data-tour="sidebar"]',
      on: 'right',
    },
    buttons: [
      { text: 'Back', classes: 'shepherd-button-secondary', action: tour.back },
      { text: 'Next', action: tour.next },
    ],
  })

  // Step 3: Chat input area
  tour.addStep({
    id: 'welcome-chat',
    title: title('Chat is', 'Home Base'),
    text: `
      <p class="text-gray-300 mb-2">
        Pick an agent and model, then ask anything. Your AI assistants live here.
      </p>
      ${stepProgress(3, TOTAL_STEPS)}
    `,
    beforeShowPromise: () => waitForElement('[data-tour="chat-input-area"]'),
    attachTo: {
      element: '[data-tour="chat-input-area"]',
      on: 'top',
    },
    buttons: [
      { text: 'Back', classes: 'shepherd-button-secondary', action: tour.back },
      { text: 'Next', action: tour.next },
    ],
  })

  // Step 4: Marketplace nav link
  tour.addStep({
    id: 'welcome-marketplace',
    title: title('Start at the', 'Marketplace'),
    text: `
      <p class="text-gray-300 mb-2">
        Pre-built agents, recipes, tools, and LLMs — ready to install with one click.
      </p>
      <p class="text-gray-400 text-sm">
        Great first stop if you want to get productive fast.
      </p>
      ${stepProgress(4, TOTAL_STEPS)}
    `,
    beforeShowPromise: () => waitForElement('[data-tour="nav-marketplace"]'),
    attachTo: {
      element: '[data-tour="nav-marketplace"]',
      on: 'right',
    },
    buttons: [
      { text: 'Back', classes: 'shepherd-button-secondary', action: tour.back },
      { text: 'Next', action: tour.next },
    ],
  })

  // Step 5: Chat widget + Guide button
  tour.addStep({
    id: 'welcome-help',
    title: title('Need Help?', 'We Got You'),
    text: `
      <p class="text-gray-300 mb-2">
        This assistant knows every feature — ask it anything.
      </p>
      <p class="text-gray-400 text-sm">
        The <strong>Guide</strong> button (bottom-left) has page-specific tours you can replay anytime.
      </p>
      ${stepProgress(5, TOTAL_STEPS)}
    `,
    beforeShowPromise: () => waitForElement('[data-tour="chat-widget"]'),
    attachTo: {
      element: '[data-tour="chat-widget"]',
      on: 'left',
    },
    buttons: [
      { text: 'Back', classes: 'shepherd-button-secondary', action: tour.back },
      { text: "Let's Explore!", action: tour.complete },
    ],
  })

  tour.on('complete', () => {
    markTourComplete(TOUR_ID, userId)
  })

  tour.on('cancel', () => {
    markTourSkipped(TOUR_ID, userId)
  })

  return tour
}
