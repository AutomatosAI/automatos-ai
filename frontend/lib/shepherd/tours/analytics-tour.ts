import Shepherd from 'shepherd.js'
import { shepherdTheme } from '../shepherd-theme'
import { markTourComplete, markTourSkipped } from '../tour-storage'
import { title, waitForElement, stepProgress, tabList } from '../tour-utils'

const TOUR_ID = 'analytics'
const TOTAL = 3

export function createAnalyticsTour(userId: string) {
  const tour = new Shepherd.Tour({
    ...shepherdTheme,
    exitOnEsc: true,
    keyboardNavigation: true,
  })

  tour.addStep({
    id: 'analytics-overview',
    title: title('', 'Analytics'),
    text: `
      <p class="text-gray-300 mb-2">
        Measure what your AI workforce is actually doing. Every agent run, mission,
        document lookup and LLM call gets tracked here so you can see impact,
        spot bottlenecks, and keep costs in check.
      </p>
      ${stepProgress(1, TOTAL)}
    `,
    beforeShowPromise: () => waitForElement('[data-tour="analytics-page-header"]'),
    attachTo: { element: '[data-tour="analytics-page-header"]', on: 'bottom' },
    buttons: [{ text: 'Next', action: tour.next }],
  })

  tour.addStep({
    id: 'analytics-tabs',
    title: title('Six', 'Analytics Views'),
    text: `
      <p class="text-gray-300 mb-2">
        Each top tab answers a different question about your workspace:
      </p>
      ${tabList([
        ['Overview', 'High-level dashboard — headline KPIs across everything.'],
        ['Agents', 'Per-agent performance — runs, success rate, time-to-completion.'],
        ['Missions', 'Mission and workflow execution stats, step-level timings.'],
        ['Documents', 'How your knowledge base is used — top files, hit rates, RAG quality.'],
        ['LLM & Costs', 'Token spend by model and provider, cost per agent and per day.'],
        ['Tools & Integrations', 'Which tools get used most, error rates, latency.'],
      ])}
      ${stepProgress(2, TOTAL)}
    `,
    beforeShowPromise: () => waitForElement('[data-tour="analytics-tabs"]'),
    attachTo: { element: '[data-tour="analytics-tabs"]', on: 'bottom' },
    buttons: [
      { text: 'Back', classes: 'shepherd-button-secondary', action: tour.back },
      { text: 'Next', action: tour.next },
    ],
  })

  tour.addStep({
    id: 'analytics-timerange',
    title: title('Time', 'Range'),
    text: `
      <p class="text-gray-300 mb-2">
        The range picker applies to every chart on the page — flip between 7, 30 and 90 days
        to compare trends. Pick a wider window to see week-over-week movement.
      </p>
      ${stepProgress(3, TOTAL)}
    `,
    beforeShowPromise: () => waitForElement('[data-tour="analytics-time-range"]'),
    attachTo: { element: '[data-tour="analytics-time-range"]', on: 'bottom' },
    buttons: [
      { text: 'Back', classes: 'shepherd-button-secondary', action: tour.back },
      { text: 'Got it!', action: tour.complete },
    ],
  })

  tour.on('complete', () => markTourComplete(TOUR_ID, userId))
  tour.on('cancel', () => markTourSkipped(TOUR_ID, userId))

  return tour
}
