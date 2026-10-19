// PRD-261 A-12h: the Command Center data hello theme. Its frame only wraps the page; its
// registerPage('/command-center') entry reads the page's data through the SDK alone
// (pages/command-center: useActivityStats, useNeedsYou, useActivityFeed, useActivitySchedule,
// useQuestions, useWatches, useBoardEventStream) and counts the board changes the stream pushes
// (events: useHostEvent). It imports nothing else of the host. It reaches the browser only through
// the lane's route interception (e2e/browser/specs/pages-command-center.spec.ts, through
// support/hello-theme.ts), and its bare imports resolve through the host's import map.
import { useState } from 'react'
import { jsx, jsxs } from 'react/jsx-runtime'
import {
  defineTheme,
  useActivityFeed,
  useActivitySchedule,
  useActivityStats,
  useBoardEventStream,
  useHostEvent,
  useNeedsYou,
  useQuestions,
  useWatches,
} from '@automatos/theme-sdk'

const COMMAND_CENTER_ROUTE = '/command-center'
const NOT_YET = '…'
const FEED_LIMIT = 5

/** @param {{ children?: import('react').ReactNode }} props */
function HelloFrame({ children }) {
  return jsx('div', { 'data-testid': 'hello-a12h-frame', className: 'hello-a12h-frame', children })
}

/**
 * One value the page read, or why it has none yet.
 * @param {string} testId
 * @param {{ data: unknown, error: string | null }} query
 * @param {(data: any) => unknown} pick
 */
function value(testId, query, pick) {
  const shown = query.error ? `error: ${query.error}` : query.data ? String(pick(query.data)) : NOT_YET
  return jsx('output', { 'data-testid': testId, children: shown })
}

/** @param {readonly any[]} rows @param {string} field */
function first(rows, field) {
  return rows.length > 0 ? rows[0][field] : 'none'
}

function HelloCommandCenter() {
  const stats = useActivityStats('1d')
  const needsYou = useNeedsYou()
  const feed = useActivityFeed({ limit: FEED_LIMIT })
  const schedule = useActivitySchedule('7d')
  const questions = useQuestions()
  const watches = useWatches()
  const [boardChanges, setBoardChanges] = useState(0)
  useBoardEventStream(true)
  useHostEvent('board-changed', () => setBoardChanges((count) => count + 1))

  return jsxs('section', {
    'data-testid': 'hello-a12h-page',
    className: 'hello-a12h-page',
    'aria-label': 'Hello Command Center',
    children: [
      jsx('h2', { children: 'Hello Command Center' }),
      value('hello-a12h-working-now', stats, (data) => data.working_now),
      value('hello-a12h-needs-you', needsYou, (data) => data.total),
      value('hello-a12h-feed', feed, (data) => first(data.items, 'name')),
      value('hello-a12h-schedule', schedule, (data) => first(data.scheduled, 'name')),
      value('hello-a12h-question', questions, (data) => first(data.grants, 'question_md')),
      value('hello-a12h-watch', watches, (data) => first(data.watches, 'title')),
      jsx('output', { 'data-testid': 'hello-a12h-board-changes', children: String(boardChanges) }),
    ],
  })
}

export default defineTheme({
  register(host) {
    host.registerSlot('frame@1', HelloFrame)
    host.registerPage(COMMAND_CENTER_ROUTE, HelloCommandCenter)
  },
  unregister() {},
})
