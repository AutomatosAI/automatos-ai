// PRD-268 US-C21f: the proactive hello theme. Its route `hello-proactive` (served at /x/hello-proactive)
// places every proactive default component of the SDK 1.3.0 cards area, BriefCard, SuggestionCard (the
// card and its pinned line), StatusLabel, ReadinessList, CalibrationCard and ReplyChips, from
// '@automatos/theme-sdk' alone, with the fixture data below, so the lane can see the host's own words,
// glyphs, AI marks and controls drawn in Studio light and Studio dark. Every handler only says, in
// the route's last line, what the card handed the theme: nothing here posts, navigates or plays. It
// reaches the browser only through the lane's route interception (e2e/browser/specs/
// proactive-defaults.spec.ts, through support/hello-theme.ts), and its bare imports resolve through
// the host's import map to the host's own React and theme SDK.
import { jsx, jsxs } from 'react/jsx-runtime'
import { useState } from 'react'
import {
  BriefCard,
  CalibrationCard,
  defineTheme,
  ReadinessList,
  ReplyChips,
  StatusLabel,
  SuggestionCard,
} from '@automatos/theme-sdk'

const ROUTE = 'hello-proactive'
const WAITING = 'waiting'

/** Brief@1 (§6.11.2) with a row in every section and a folded Today. Test data only. */
const BRIEF = {
  v: 1,
  date: '2026-10-30',
  greeting: 'Morning, Sam.',
  prose: 'One thing needs you before noon: the Lisbon flights hold lapses then. The rest can wait.',
  spoken: 'Morning, Sam. 1. Approve the Lisbon flights. 2. Draft the October update. 3. Weekly metrics report.',
  sections: {
    needs_you: [
      {
        id: 'e2e-row-needs-you',
        title: 'Approve the Lisbon flights',
        why: 'The hold lapses at noon.',
        number: '#0419',
        display_state: 'needs_you',
        opens: { type: 'ticket', id: '419' },
        action: { verb: 'answer', label: 'Answer →', done_label: 'Answered' },
      },
    ],
    suggestions: [
      {
        id: 'e2e-row-suggestion',
        title: 'Draft the October update',
        why: 'Three customers asked for one this week.',
        action: { verb: 'start', label: 'Start →', done_label: 'Started' },
      },
    ],
    today: [
      { id: 'e2e-row-today', title: 'Weekly metrics report', why: 'Runs this afternoon.', display_state: 'working' },
    ],
    since: [{ id: 'e2e-row-since', title: 'Invoice reminders sent', why: 'Four went out overnight.', display_state: 'done' }],
    unchecked: [{ id: 'e2e-row-unchecked', title: 'Phone alerts', why: 'No phone has registered yet.' }],
  },
  counts: { needs_you: 1, suggestions: 1, today: 3, since: 1 },
  labels: {
    sections: {
      needs_you: 'Needs you',
      suggestions: 'Auto suggests',
      today: 'Today',
      since: 'Since your last brief',
      unchecked: "Couldn't check",
    },
    more: { today: 'Show 2 more' },
    empty: 'Nothing needs you today.',
  },
  state_hash: 'e2e-proactive-brief',
  generated_at: '2026-10-30T07:00:00Z',
}

/** One open suggestion, as GET /api/proactive/suggestions writes it. Test data only. */
const SUGGESTION = {
  id: '00000000-0000-4000-8000-0000000c21f1',
  workspace_id: null,
  user_id: null,
  source_type: 'check_in',
  source_agent_id: null,
  run_id: null,
  title: 'Draft the October update',
  detail: 'Three customers asked for a product update this week; the last one went out in September.',
  evidence: [],
  proposed_action: { kind: 'board_task', payload: { title: 'Draft the October update' } },
  subject_key: 'e2e:october-update',
  severity: 'today',
  status: 'open',
  occurrences: 1,
  last_seen_at: null,
  started_ref: null,
  decided_via: null,
  hide_reason: null,
  decided_at: null,
  est_credits: 12,
  created_at: '2026-10-30T07:00:00Z',
  delivered_at: null,
  expires_at: null,
}

/** One row per display state (§6.11.1), with a next step and a retry where a row has one. */
const STATES = [
  { state: 'needs_you', title: 'Approve the Lisbon flights', nextStep: 'Waiting for your answer' },
  { state: 'working', title: 'Weekly metrics report', nextStep: "Reading last week's numbers" },
  { state: 'ready', title: 'Competitor scan' },
  { state: 'done', title: 'Invoice reminders' },
  { state: 'failed', title: 'Newsletter send', nextStep: 'The mail provider refused the login', retry: 'retry_ticket' },
]

/** Readiness@1 (§6.11.3): one item per state, and a fix that opens an app path. */
const READINESS = {
  v: 1,
  checked_at: '2026-10-30T07:00:00Z',
  items: [
    { id: 'model', label: 'AI model', detail: 'A model key is set for this workspace.', state: 'ready', fix: null, needed_by: ['brief', 'check_in'] },
    { id: 'push', label: 'Phone alerts', detail: 'No phone has registered yet.', state: 'not_checked', fix: null, needed_by: ['push'] },
    {
      id: 'voice',
      label: 'Server voice',
      detail: 'No voice provider is connected.',
      state: 'needs_setup',
      fix: { label: 'Set up voice', href: '/settings' },
      needed_by: ['voice'],
    },
  ],
}

/** A row only the browser knows, drawn after the server's. */
const CLIENT_ITEMS = [
  { id: 'microphone', label: 'Microphone', detail: 'This browser can use a microphone.', state: 'ready', fix: null, needed_by: ['voice'] },
]

/** Calibration@1 (§6.11.5): three lines with their sources, and three offers. */
const CALIBRATION = {
  v: 1,
  id: 'e2e-calibration',
  lines: [
    {
      id: 'e2e-line-voice',
      text: 'You write short, plain updates and skip the greeting.',
      sources: [
        { type: 'deliverable', id: 'e2e-d1', label: 'September update' },
        { type: 'deliverable', id: 'e2e-d2', label: 'August update' },
        { type: 'conversation', id: 'e2e-c1', label: 'Launch notes' },
      ],
    },
    {
      id: 'e2e-line-customers',
      text: 'Most of your customers are small hotels.',
      sources: [{ type: 'document', id: 'e2e-doc1', label: 'Customer list' }],
    },
    {
      id: 'e2e-line-cadence',
      text: 'You send a product update once a month.',
      sources: [{ type: 'deliverable', id: 'e2e-d1', label: 'September update' }],
    },
  ],
  offers: ['Draft the monthly update', 'Summarise new customer feedback', 'Check the booking pipeline'],
  empty: false,
}

/** The answers Auto offered with its last message, each at most 24 characters. */
const CHIPS = ['Yes, book it', 'Not this week', 'Show me options']

/** @param {{ children?: import('react').ReactNode }} props */
function ProactiveFrame({ children }) {
  return jsxs('div', {
    'data-testid': 'hello-proactive-frame',
    className: 'hello-proactive-frame',
    children: [
      jsx('header', { className: 'hello-proactive-header', children: jsx('strong', { children: 'Hello proactive defaults' }) }),
      jsx('div', { 'data-testid': 'hello-proactive-content', children }),
    ],
  })
}

/** One placed component, in a box the lane screenshots. */
function place(name, child) {
  return jsx('div', { 'data-testid': `hello-proactive-${name}`, className: 'hello-proactive-place', children: child })
}

/** @param {{ hear: (what: string) => void }} props */
function StatusRows({ hear }) {
  return jsx('ul', {
    children: STATES.map((row) =>
      jsx(
        'li',
        {
          children: jsx(StatusLabel, {
            state: row.state,
            nextStep: row.nextStep ?? null,
            retry: row.retry ?? null,
            onRetry: (retry) => hear(`retry:${retry}`),
            title: row.title,
          }),
        },
        row.state,
      ),
    ),
  })
}

function ProactiveRoute() {
  const [heard, setHeard] = useState(WAITING)
  const brief = jsx(BriefCard, {
    brief: BRIEF,
    onAction: (row) => {
      setHeard(`action:${row.id}`)
      return Promise.resolve()
    },
    onOpen: (row) => setHeard(`open:${row.id}`),
    onShowMore: (section) => setHeard(`more:${section}`),
  })
  const card = jsx(SuggestionCard, { suggestion: SUGGESTION, onStarted: (answer) => setHeard(`started:${answer.opens.type}`) })
  const pinned = jsx(SuggestionCard, { suggestion: SUGGESTION, variant: 'pinned', onOpen: () => setHeard('open:suggestion') })
  const readiness = jsx(ReadinessList, { readiness: READINESS, clientItems: CLIENT_ITEMS })
  const calibration = jsx(CalibrationCard, {
    calibration: CALIBRATION,
    onAnswer: (answer) => setHeard(`calibration:${answer.answer}`),
    onSomethingOff: () => setHeard('something-off'),
  })
  const chips = jsx(ReplyChips, { chips: CHIPS, latest: true, onReply: (text) => setHeard(`reply:${text}`) })
  return jsxs('section', {
    'data-testid': 'hello-proactive-route',
    className: 'hello-proactive-route',
    'aria-label': 'Proactive defaults',
    children: [
      place('brief', brief),
      place('suggestion', card),
      place('pinned', pinned),
      place('status', jsx(StatusRows, { hear: setHeard })),
      place('readiness', readiness),
      place('calibration', calibration),
      place('chips', chips),
      jsx('p', { 'data-testid': 'hello-proactive-heard', children: heard }),
    ],
  })
}

export default defineTheme({
  register(host) {
    host.registerSlot('frame@1', ProactiveFrame)
    host.registerRoute(ROUTE, ProactiveRoute)
  },
  unregister() {},
})
