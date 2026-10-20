import { expect, test, type Locator, type Page } from '@playwright/test'
import { fulfillBackendJson } from '../../support/backend-route'
import { ask, autoColumn, autoReplies, openPlayChat } from '../../support/play-chat'
import { describeParity, PARITY_TAG } from '../../support/play-parity'

// PRD-261 A-17b (@parity): approvals in Play's chat, under builtin Play and the installed package.
// One recorded reply (fixtures/play-parity/chat-turns.json, 'approvals') raises a tool approval
// (a grant, PRD-193) and a Mission plan (PRD-163). Both show inside Auto's reply as the core's own
// cards (the SDK's cards.ToolApproval and cards.MissionApproval, protected surfaces), Auto listens
// for the answer while one waits, and each button posts the core's own decision: the grant and deny
// routes for the tool, approve and reject for the Mission (answered here, so no Mission runs). A
// tool card shows its outcome; a Mission card leaves the reply once decided.

const ASK = 'Delete the old draft and send the weekly report'
const REPLY_TEXT = 'Two things need your answer before they run.'
const GRANT_ID = 26101
const MISSION_ID = 'e2e-play-parity-mission'
const GRANT_PATH = `/api/v1/approval-grants/${GRANT_ID}/grant`
const DENY_PATH = `/api/v1/approval-grants/${GRANT_ID}/deny`
const MISSION_APPROVE_PATH = `/api/missions/${MISSION_ID}/approve`
const MISSION_REJECT_PATH = `/api/missions/${MISSION_ID}/reject`
const DECISION_PATHS = [GRANT_PATH, DENY_PATH, MISSION_APPROVE_PATH, MISSION_REJECT_PATH]
/** lib/chat/approvals.ts: the cards' titles, as chat.tsx gives them. */
const TOOL_TITLE = 'Action approval needed'
const MISSION_TITLE = 'Mission plan — approval needed'
const APPROVALS_SECTION = 'Approvals'
const TOOL_CARD = '[data-protected-surface="tool-approval"]'
const MISSION_CARD = '[data-protected-surface="mission-approval"]'
const APPROVE = 'Approve & run'
const DENY = 'Deny'
const REJECT = 'Reject'
/** components/widgets/ToolApprovalWidget: the outcome lines. */
const APPROVED = 'Approved.'
const DENIED = 'Denied — the action will not run.'
/** components/play/chat/caption.ts AWAITING_ANSWER: Auto listens while a card waits. */
const AWAITING_ANSWER = 'Waiting for your answer on the card.'
const OK = 200

/** Answers the four decision routes and records each POST's path, in order. */
async function recordDecisions(page: Page): Promise<string[]> {
  const decided: string[] = []
  await page.route(
    (url) => DECISION_PATHS.includes(url.pathname),
    async (route) => {
      const { pathname } = new URL(route.request().url())
      if (route.request().method() === 'POST') decided.push(pathname)
      const grant = { grant: { id: GRANT_ID, status: pathname === GRANT_PATH ? 'granted' : 'denied', details: {} } }
      const mission = { id: MISSION_ID, state: pathname === MISSION_APPROVE_PATH ? 'executing' : 'rejected' }
      await fulfillBackendJson(route, OK, pathname.startsWith('/api/missions/') ? mission : grant)
    },
  )
  return decided
}

interface RaisedCards {
  readonly tool: Locator
  readonly mission: Locator
}

/** Sends the message whose reply raises both cards, and finds them inside that reply. */
async function raiseBothCards(page: Page): Promise<RaisedCards> {
  await ask(page, ASK)
  const reply = autoReplies(page).last()
  await expect(reply).toContainText(REPLY_TEXT)
  const approvals = reply.getByRole('region', { name: APPROVALS_SECTION, exact: true })
  const tool = approvals.locator(TOOL_CARD)
  const mission = approvals.locator(MISSION_CARD)
  await expect(tool).toBeVisible()
  await expect(tool).toContainText(TOOL_TITLE)
  await expect(mission).toBeVisible()
  await expect(mission).toContainText(MISSION_TITLE)
  await expect(autoColumn(page)).toContainText(AWAITING_ANSWER)
  return { tool, mission }
}

describeParity('approvals in the chat', (variant) => {
  test('a tool card and a Mission card in the reply: Approve & run, then Reject', { tag: PARITY_TAG }, async ({ page, request }) => {
    const decided = await recordDecisions(page)
    await openPlayChat(page, request, variant, ['approvals'])
    const { tool, mission } = await raiseBothCards(page)

    await tool.getByRole('button', { name: APPROVE }).click()
    await expect.poll(() => decided).toEqual([GRANT_PATH])
    await expect(tool.getByRole('status')).toHaveText(APPROVED)
    await expect(autoColumn(page)).toContainText(AWAITING_ANSWER)

    await mission.getByRole('button', { name: REJECT }).click()
    await expect.poll(() => decided).toEqual([GRANT_PATH, MISSION_REJECT_PATH])
    await expect(mission).toHaveCount(0)
    await expect(autoColumn(page)).not.toContainText(AWAITING_ANSWER)
  })

  test('a tool card and a Mission card in the reply: Deny, then Approve & run', { tag: PARITY_TAG }, async ({ page, request }) => {
    const decided = await recordDecisions(page)
    await openPlayChat(page, request, variant, ['approvals'])
    const { tool, mission } = await raiseBothCards(page)

    await tool.getByRole('button', { name: DENY }).click()
    await expect.poll(() => decided).toEqual([DENY_PATH])
    await expect(tool.getByRole('status')).toHaveText(DENIED)

    await mission.getByRole('button', { name: APPROVE }).click()
    await expect.poll(() => decided).toEqual([DENY_PATH, MISSION_APPROVE_PATH])
    await expect(mission).toHaveCount(0)
    await expect(autoColumn(page)).not.toContainText(AWAITING_ANSWER)
  })
})
