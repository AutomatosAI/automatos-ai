import Shepherd from 'shepherd.js'
import { shepherdTheme, isMobile, mobilePosition } from '../shepherd-theme'
import { markTourComplete, markTourSkipped } from '../tour-storage'
import { title, waitForElement, stepProgress, tabList } from '../tour-utils'

const TOUR_ID = 'chat'
const TOTAL = 5

export function createChatTour(userId: string) {
  const mobile = isMobile()

  const tour = new Shepherd.Tour({
    ...shepherdTheme,
    exitOnEsc: true,
    keyboardNavigation: true,
  })

  // Step 1: About (centered — no attachTo)
  tour.addStep({
    id: 'chat-about',
    title: title('Meet', 'Auto'),
    text: mobile
      ? `
        <p class="text-gray-300 mb-2">
          <strong>Auto</strong> is your AI assistant — he knows the platform,
          remembers you, and can manage everything via plain English.
        </p>
        <p class="text-gray-400 text-sm">
          Try: <em>"Hi Auto, what can you help me with?"</em>
        </p>
        ${stepProgress(1, TOTAL)}
      `
      : `
        <p class="text-gray-300 mb-3">
          Welcome to your AI command line. This is where you talk to <strong>Auto</strong> —
          your personal AI assistant who knows the entire platform inside and out.
        </p>
        <p class="text-gray-300 mb-3">
          Auto <strong>remembers you</strong> between conversations — introduce yourself,
          tell him about your work, and he'll tailor every interaction to you from now on.
        </p>
        <p class="text-gray-300 mb-3">
          Auto can <strong>manage the platform for you</strong>: create agents, launch missions,
          connect tools, check analytics, read your emails — just ask in plain English.
        </p>
        <p class="text-gray-300 mb-3">
          Auto <strong>follows you everywhere</strong> — on other pages you'll see a chat bubble
          in the bottom-right corner so you can ask questions without leaving the page.
          You can always jump back here for a full conversation.
        </p>
        <p class="text-gray-400 text-sm">
          Try saying: <em>"Hi Auto, I'm [your name]. What can you help me with?"</em>
        </p>
        ${stepProgress(1, TOTAL)}
      `,
    buttons: [
      { text: 'Skip', classes: 'shepherd-button-secondary', action: () => tour.cancel() },
      { text: 'Next', action: tour.next },
    ],
  })

  // Step 2: Agent selector
  tour.addStep({
    id: 'chat-agent',
    title: title('Pick an', 'Agent'),
    text: `
      <p class="text-gray-300 mb-2">
        ${mobile
          ? 'Each agent has its own persona, memory and tools. Pick the right one for the job.'
          : 'Every conversation runs through an agent. Each one has its own persona, memory, tool access and skillset — so picking the right agent changes what the assistant can actually do.'
        }
      </p>
      <p class="text-gray-400 text-sm">
        Start with <strong>Auto</strong> — it routes to the best agent automatically.
      </p>
      ${stepProgress(2, TOTAL)}
    `,
    beforeShowPromise: () => waitForElement('[data-tour="chat-agent-selector"]'),
    attachTo: { element: '[data-tour="chat-agent-selector"]', on: 'bottom' },
    buttons: [
      { text: 'Back', classes: 'shepherd-button-secondary', action: tour.back },
      { text: 'Next', action: tour.next },
    ],
  })

  // Step 3: Quick modes
  tour.addStep({
    id: 'chat-modes',
    title: title('Quick', 'Modes'),
    text: `
      <p class="text-gray-300 mb-2">
        Toggle <strong>Code</strong>, <strong>Plan</strong>, or <strong>Mission</strong>
        mode without leaving the chat.
      </p>
      ${stepProgress(3, TOTAL)}
    `,
    beforeShowPromise: () => waitForElement('[data-tour="chat-mode-bar"]'),
    attachTo: { element: '[data-tour="chat-mode-bar"]', on: 'top' },
    buttons: [
      { text: 'Back', classes: 'shepherd-button-secondary', action: tour.back },
      { text: 'Next', action: tour.next },
    ],
  })

  // Step 4: Header toolbar
  tour.addStep({
    id: 'chat-header',
    title: title('Your', 'Toolbar'),
    text: mobile
      ? `
        <p class="text-gray-300 mb-2">Docs, theme, alerts, and your profile — all up here.</p>
        <p class="text-gray-400 text-sm">Tap your avatar on any page to relaunch that page's tour.</p>
        ${stepProgress(4, TOTAL)}
      `
      : `
        <p class="text-gray-300 mb-2">
          Your toolbar icons, left to right:
        </p>
        ${tabList([
          ['Docs (book icon)', 'Opens documentation and API reference in a new tab — everything you need to build with the platform.'],
          ['Theme (monitor icon)', 'Switch between light and dark mode instantly.'],
          ['Alerts (bell icon)', 'Notifications from your agents, missions, and system events — check here for anything that needs your attention.'],
          ['Profile (your avatar)', 'Account settings, workspace management, and Tour this page — click your avatar on any page to relaunch that page\'s tour.'],
        ])}
        ${stepProgress(4, TOTAL)}
      `,
    beforeShowPromise: () => waitForElement('[data-tour="header-actions"]'),
    attachTo: { element: '[data-tour="header-actions"]', on: 'bottom' },
    buttons: [
      { text: 'Back', classes: 'shepherd-button-secondary', action: tour.back },
      { text: 'Next', action: tour.next },
    ],
  })

  // Step 5: Navigation sidebar
  tour.addStep({
    id: 'chat-nav',
    title: title('Your', 'Navigation'),
    text: mobile
      ? `
        <p class="text-gray-300 mb-2">
          Use the menu to reach Activity, Agents, Tools, Knowledge Base, Marketplace, and Analytics.
        </p>
        <p class="text-gray-400 text-sm">
          Each page has its own tour — tap your avatar and select <em>Tour this page</em>.
        </p>
        ${stepProgress(5, TOTAL)}
      `
      : `
        <p class="text-gray-300 mb-2">
          This sidebar is your launchpad to every part of the platform:
        </p>
        ${tabList([
          ['Activity', 'Live dashboard of everything your agents are doing — missions, tasks, and reports.'],
          ['Workspace', 'Your shared file system — agent outputs, code, and collaboration files.'],
          ['Agents', 'Build, configure, and manage your AI workforce.'],
          ['Tools', 'Connect integrations like Gmail, Slack, GitHub, and more.'],
          ['Knowledge Base', 'Upload documents, connect data sources — give your agents context.'],
          ['Marketplace', 'Discover and install pre-built agents, playbooks, and tools.'],
          ['Analytics', 'Track performance, costs, and usage across your workspace.'],
        ])}
        <p class="text-gray-400 text-sm mt-2">
          Each page has its own guided tour — click your avatar and select
          <em>Tour this page</em> on any page to learn more.
        </p>
        ${stepProgress(5, TOTAL)}
      `,
    beforeShowPromise: () => waitForElement('[data-tour="sidebar"]'),
    attachTo: { element: '[data-tour="sidebar"]', on: mobilePosition('right') },
    buttons: [
      { text: 'Back', classes: 'shepherd-button-secondary', action: tour.back },
      { text: 'Got it!', action: tour.complete },
    ],
  })

  tour.on('complete', () => markTourComplete(TOUR_ID, userId))
  tour.on('cancel', () => markTourSkipped(TOUR_ID, userId))

  return tour
}
