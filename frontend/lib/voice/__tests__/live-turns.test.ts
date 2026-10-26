// Theme voice — the words of the voice session in progress (lib/voice/live-turns.ts): they show in the conversation
// they belong to and nowhere else; once that conversation's saved rows land (chat_changed) the exchange hides for
// good, Auto's words still growing in Retell's transcript included; the member speaking again shows again.
import { afterEach, describe, expect, it } from 'vitest'

import {
  CHAT_CHANGED_EVENT,
  type LiveVoiceTurn,
  useLiveTurnStore,
  visibleLiveTurn,
  watchSavedVoiceTurns,
} from '@/lib/voice/live-turns'

const turn = (userText: string, agentText = '', chatId = 'chat-1'): LiveVoiceTurn => ({ chatId, userText, agentText })
const shownIn = (chatId: string | null) => {
  const { turn: current, saved } = useLiveTurnStore.getState()
  return visibleLiveTurn(current, saved, chatId)
}
const chatChanged = (chatId: unknown) => window.dispatchEvent(new CustomEvent(CHAT_CHANGED_EVENT, { detail: { chat_id: chatId } }))

afterEach(() => {
  useLiveTurnStore.getState().clear()
})

describe('the live words — where they show', () => {
  it('shows the exchange in progress in its own conversation only', () => {
    useLiveTurnStore.getState().publish(turn('where did we leave off', 'We were mid-way'))
    expect(shownIn('chat-1')).toEqual({ userText: 'where did we leave off', agentText: 'We were mid-way' })
    expect(shownIn('chat-2')).toBeNull()
    expect(shownIn(null)).toBeNull()
  })

  it('an empty exchange (a new hold-to-talk turn) shows nothing', () => {
    useLiveTurnStore.getState().publish(turn('hello', 'Hi'))
    useLiveTurnStore.getState().publish(turn('', ''))
    expect(useLiveTurnStore.getState().turn).toBeNull()
    expect(shownIn('chat-1')).toBeNull()
  })

  it('the same words again change nothing (a duration tick)', () => {
    useLiveTurnStore.getState().publish(turn('hello', 'Hi'))
    const before = useLiveTurnStore.getState().turn
    useLiveTurnStore.getState().publish(turn('hello', 'Hi'))
    expect(useLiveTurnStore.getState().turn).toBe(before)
  })
})

describe('the live words — dropped when the saved rows land', () => {
  it('hides the saved exchange, and keeps it hidden while Auto\'s words still grow', () => {
    const store = useLiveTurnStore.getState()
    store.publish(turn('what is on today', 'Two meet'))
    store.settle('chat-1')
    expect(shownIn('chat-1')).toBeNull()
    useLiveTurnStore.getState().publish(turn('what is on today', 'Two meetings, then a call.'))
    expect(shownIn('chat-1')).toBeNull()
  })

  it('the member speaking again is a new exchange, and it shows', () => {
    const store = useLiveTurnStore.getState()
    store.publish(turn('what is on today', 'Two meetings.'))
    store.settle('chat-1')
    useLiveTurnStore.getState().publish(turn('w'))
    expect(shownIn('chat-1')).toEqual({ userText: 'w', agentText: '' })
    useLiveTurnStore.getState().publish(turn('what is on today'))
    expect(shownIn('chat-1')).toEqual({ userText: 'what is on today', agentText: '' })
  })

  it("another conversation's saved rows leave the words alone", () => {
    useLiveTurnStore.getState().publish(turn('hello', 'Hi'))
    useLiveTurnStore.getState().settle('chat-2')
    expect(shownIn('chat-1')).toEqual({ userText: 'hello', agentText: 'Hi' })
  })

  it('settles on the window\'s chat_changed for the conversation it names', () => {
    const stop = watchSavedVoiceTurns(window)
    useLiveTurnStore.getState().publish(turn('hello', 'Hi'))
    chatChanged('chat-2')
    chatChanged(42)
    expect(shownIn('chat-1')).not.toBeNull()
    chatChanged('chat-1')
    expect(shownIn('chat-1')).toBeNull()
    stop()
    useLiveTurnStore.getState().publish(turn('next'))
    chatChanged('chat-1')
    expect(shownIn('chat-1')).toEqual({ userText: 'next', agentText: '' })
  })
})
