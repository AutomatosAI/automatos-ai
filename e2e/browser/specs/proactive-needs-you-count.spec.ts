import { expect, test, type Locator, type Page, type Route } from '@playwright/test'
import { fulfillBackendJson } from '../support/backend-route'
import { CORE_STYLES, chooseCoreStyle, type CoreStyle } from '../support/hello-theme'

// PRD-268 US-W04 (@host): the brief's needs-you count equals the Needs-you badge on the same screen.
//
// The Command Center draws the day's brief under Auto's Read (AutosRead, in both shells) and the one Needs-you
// number (PRD-252 R5): Studio on its Board tab's badge, Classic in the Needs you widget ("N waiting"). The server
// makes the brief's counts.needs_you from needs_you() (US-C19c), so the two agree; the screen must draw each as it
// was sent and never recount one from the rows it lists. The backend answers are the spec's: GET
// /api/activity/needs-you says 7 and lists 7 rows; the day's brief (POST /api/proactive/brief/open, then its
// message) says 7 and lists 5, as it folds at 5. Under Studio and under Classic, the brief's needs-you section
// shows 5 rows and the count 7, and the badge on the same screen shows 7.

const APP_URL = process.env.E2E_APP_URL ?? 'http://localhost:3000'
const COMMAND_CENTER_PATH = '/command-center'
const OK = 200
const NEEDS_YOU = 7
/** The brief lists 5 rows of a section and says how many more (it folds at 5, never at 6). */
const LISTED = 5

const NEEDS_YOU_PATH = '/api/activity/needs-you'
const DIGEST_PATH = '/api/activity/digest'
const OPEN_PATH = '/api/proactive/brief/open'
const PROACTIVE_PREFIX = '/api/proactive/'
const BRIEF_REF = { chat_id: '00000000-0000-4000-8000-0000000b0401', message_id: '00000000-0000-4000-8000-0000000b0402' }
const MESSAGES_PATH = `/api/chat/${BRIEF_REF.chat_id}/messages`
const STATE_HASH = 'e2e-needs-you-count'
const REVIEW_STEP = 'Look over the work: approve it, or send it back with what to change.'

const TITLES = ['Monday dispatch', 'Price list for Tidewater', 'Autumn menu', 'Club newsletter', 'Café letters', 'Weekly numbers', 'Oat milk order']

const NEEDS_YOU_ANSWER = {
  total: NEEDS_YOU,
  counts: { review: NEEDS_YOU, question: 0, approval: 0, stuck: 0, failed: 0 },
  rows: {
    review: TITLES.map((title, index) => ({
      ticket_id: 760 + index, number: `#00${42 + index}`, title, agent_name: null, mission_id: null, at: null,
      display_state: 'needs_you', next_step: REVIEW_STEP, retry: null,
    })),
    question: [], approval: [], stuck: [], failed: [],
  },
}

const BRIEF = {
  v: 1,
  date: '2026-10-30',
  greeting: 'Morning, Sam.',
  prose: 'Seven things need you today.',
  spoken: 'Morning, Sam.',
  sections: {
    needs_you: TITLES.slice(0, LISTED).map((title, index) => ({
      id: `ticket-${760 + index}`, title, why: 'It waits for your review.', display_state: 'needs_you', opens: { type: 'ticket', id: String(760 + index) },
    })),
    suggestions: [], today: [], since: [], unchecked: [],
  },
  counts: { needs_you: NEEDS_YOU, suggestions: 0, today: 0, since: 0 },
  labels: { sections: { needs_you: 'Needs you' }, more: { needs_you: `Show ${NEEDS_YOU - LISTED} more` } },
  state_hash: STATE_HASH,
  generated_at: '2026-10-30T07:00:00Z',
}
const BRIEF_MESSAGES = [
  { id: BRIEF_REF.message_id, role: 'assistant', parts: [{ type: 'text', text: BRIEF.prose }, { type: 'brief@1', brief: BRIEF }] },
]
const DIGEST = { text: BRIEF.prose, generated_at: BRIEF.generated_at, state_hash: STATE_HASH, needs_attention_count: NEEDS_YOU }

function onPath(pathname: string): (url: URL) => boolean {
  return (url) => url.pathname === pathname
}

/** The proactive routes: the open names the day's message; anything else (the Suggested list) lists nothing. */
async function answerProactive(route: Route): Promise<void> {
  const pathname = new URL(route.request().url()).pathname
  await fulfillBackendJson(route, OK, pathname === OPEN_PATH ? { ...BRIEF_REF, posted: false } : { suggestions: [] })
}

/** Answers every read the brief and the badge make on the Command Center. */
async function answerTheScreen(page: Page): Promise<void> {
  await page.route(onPath(NEEDS_YOU_PATH), (route) => fulfillBackendJson(route, OK, NEEDS_YOU_ANSWER))
  await page.route(onPath(DIGEST_PATH), (route) => fulfillBackendJson(route, OK, DIGEST))
  await page.route((url) => url.pathname.startsWith(PROACTIVE_PREFIX), answerProactive)
  await page.route(onPath(MESSAGES_PATH), (route) => fulfillBackendJson(route, OK, BRIEF_MESSAGES))
}

/** The one Needs-you number on this screen: Studio's Board tab badge, Classic's Needs you widget. */
function needsYouBadge(page: Page, core: CoreStyle): Locator {
  if (core.studio) {
    return page.getByRole('navigation', { name: 'Command Center sections' }).getByRole('button', { name: /^Board/ }).locator('.cc-tab-ct')
  }
  return page.getByText(/^\d+ waiting$/)
}

function digits(text: string | null): number {
  return Number((text ?? '').replace(/\D/g, ''))
}

for (const core of CORE_STYLES) {
  test(`under ${core.style} the brief's needs-you count is the Needs-you badge's on the same screen`, { tag: '@host' }, async ({ page }) => {
    await chooseCoreStyle(page, core, APP_URL)
    await answerTheScreen(page)
    await page.goto(COMMAND_CENTER_PATH)

    const section = page.getByTestId('autos-read-brief').locator('[data-section="needs_you"]')
    const sectionCount = section.locator('[data-section-count]')
    const badge = needsYouBadge(page, core)
    // The brief lists 5 of its 7, so a count drawn from the rows would read 5.
    await expect(section.locator('li')).toHaveCount(LISTED)
    await expect(sectionCount).toHaveText(new RegExp(`${NEEDS_YOU}$`))
    await expect(badge).toHaveText(new RegExp(`^${NEEDS_YOU}( waiting)?$`))
    expect(digits(await sectionCount.textContent())).toBe(digits(await badge.textContent()))
  })
}
