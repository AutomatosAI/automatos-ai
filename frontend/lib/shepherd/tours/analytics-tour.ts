import Shepherd from 'shepherd.js'
import { shepherdTheme } from '../shepherd-theme'
import { markTourComplete, markTourSkipped } from '../tour-storage'
import { title, waitForElement, stepProgress, tabList } from '../tour-utils'

const TOUR_ID = 'analytics'
const TOTAL = 2

export function createAnalyticsTour(userId: string) {
  const tour = new Shepherd.Tour({
    ...shepherdTheme,
    exitOnEsc: true,
    keyboardNavigation: true,
  })

  // Step 1: About (centered)
  tour.addStep({
    id: 'analytics-about',
    title: title('Performance &', 'Costs'),
    text: `
      <p class="text-gray-300 mb-2">
        Measure what your AI workforce is actually doing. Every agent run,
        mission, document lookup and LLM call gets tracked here — see
        impact, spot bottlenecks, and keep costs in check.
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
    id: 'analytics-tabs',
    title: title('Analytics', 'Views'),
    text: `
      ${tabList([
        ['Overview', 'High-level dashboard — headline KPIs across everything.'],
        ['Agents', 'Per-agent performance — runs, success rate, time-to-completion.'],
        ['Missions', 'Mission and workflow execution stats, step-level timings.'],
        ['Documents', 'Knowledge base usage — top files, hit rates, RAG quality.'],
        ['LLM & Costs', 'Token spend by model and provider, cost per agent and per day.'],
        ['Tools', 'Which tools get used most, error rates, latency.'],
      ])}
      ${stepProgress(2, TOTAL)}
    `,
    beforeShowPromise: () => waitForElement('[data-tour="analytics-tabs"]'),
    attachTo: { element: '[data-tour="analytics-tabs"]', on: 'bottom' },
    buttons: [
      { text: 'Back', classes: 'shepherd-button-secondary', action: tour.back },
      { text: 'Got it!', action: tour.complete },
    ],
  })

  tour.on('complete', () => markTourComplete(TOUR_ID, userId))
  tour.on('cancel', () => markTourSkipped(TOUR_ID, userId))

  return tour
}
