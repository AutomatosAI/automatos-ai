import { expect, test } from '@playwright/test'
import { recordCancels } from '../../support/chat-stream'
import { ask, autoReplies, openPlayChat, readPlayTurns } from '../../support/play-chat'
import { describeParity, PARITY_TAG } from '../../support/play-parity'

// PRD-261 A-17b (@parity), P260-FX-30 (decision F4): Stop keeps what Auto had written, under builtin
// Play and the installed package. The first reply ('partial') streams its words and stays open;
// Stop cancels the turn on the server (POST /api/chat/{id}/cancel, answered here), and the reply
// stays with its words and "You stopped this answer." (components/play/chat/auto-reply.tsx
// STOPPED_NOTICE). The next message streams a new reply under it, and the stopped one stays.

const ASK = 'Compare these three laptops for me'
const ASK_AGAIN = 'Just give me the shortlist'
const PARTIAL = 'I am still writing the comparison'
const ANSWER = 'Here is your laptop shortlist.'
const STOPPED_NOTICE = 'You stopped this answer.'

describeParity('Stop keeps the reply Auto was writing (FX-30)', (variant) => {
  test('the stopped reply keeps its words and says it was stopped; the next message streams under it', { tag: PARITY_TAG }, async ({ page, request }) => {
    const cancelled = await recordCancels(page)
    await openPlayChat(page, request, variant, ['partial', 'answer'])
    await ask(page, ASK)

    const replies = autoReplies(page)
    const stopped = replies.first()
    await expect(stopped).toContainText(PARTIAL)
    await expect(stopped).toHaveAttribute('aria-busy', 'true')

    await page.getByRole('button', { name: 'Stop', exact: true }).click()
    await expect(stopped).toHaveAttribute('aria-busy', 'false')
    await expect(stopped).toContainText(PARTIAL)
    await expect(stopped).toContainText(STOPPED_NOTICE)
    await expect(page.getByRole('button', { name: 'Send', exact: true })).toBeVisible()
    await expect.poll(() => cancelled).toEqual([`/api/chat/${readPlayTurns().chatId}/cancel`])

    await ask(page, ASK_AGAIN)
    await expect(replies).toHaveCount(2)
    await expect(replies.last()).toContainText(ANSWER)
    await expect(replies.last()).toHaveAttribute('aria-busy', 'false')
    await expect(replies.last()).not.toContainText(STOPPED_NOTICE)
    await expect(stopped).toContainText(PARTIAL)
    await expect(stopped).toContainText(STOPPED_NOTICE)
  })
})
