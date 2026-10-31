/**
 * Theme voice, web quick-ask — the engine over fakes (quick-ask-harness.ts): a press opens the browser's recognition
 * and shows the words as they are recognised; letting go, or the browser's own end of speech, sends them as a normal
 * chat turn with spokenMode, adopting a new chat's id; the streamed reply is spoken sentence by sentence (markdown,
 * links and code left to the chat) with the AI notice first, once per browser session.
 */
import { afterEach, describe, expect, it, vi } from 'vitest'

vi.mock('@/lib/api-client', () => ({
  apiClient: { getBaseUrl: () => '', getAuthHeaders: async () => ({}), request: async () => ({}) },
}))

import { AI_NOTICE_STORAGE_KEY, HOLD_MAX_MS, LEVEL_DECAY_MS, RELEASE_GRACE_MS } from '@/lib/voice/quick-ask/config'
import { QUICK_ASK_ERRORS } from '@/lib/voice/quick-ask/copy'

import { ask, flush, GREETING, quickAskHarness } from './quick-ask-harness'

afterEach(() => {
  vi.restoreAllMocks()
})

const ASKED = "What's on today?"

describe('start: the browser listens', () => {
  it('opens a recognition and primes the speech inside the press, and listens once the browser captures', async () => {
    const h = quickAskHarness()

    await h.engine.start({ chatId: 'c-1' })

    expect(h.recognition.factory).toHaveBeenCalledTimes(1)
    expect(h.synthesis.port.prime).toHaveBeenCalledTimes(1)
    expect(h.engine.snapshot()).toMatchObject({ mode: 'quick', phase: 'connecting', micOpen: false, chatId: 'c-1' })
    h.recognition.last().capture()
    expect(h.engine.snapshot()).toMatchObject({ phase: 'listening', micOpen: true })
  })

  it('shows the words as they are recognised, each lifting the member’s level', async () => {
    const h = quickAskHarness()
    await h.engine.start({ chatId: null })
    const hold = h.recognition.last()
    hold.capture()

    expect(h.engine.levels().user).toBe(0)
    hold.say("What's on")
    expect(h.engine.snapshot().heard).toBe("What's on")
    expect(h.engine.levels().user).toBeGreaterThan(0)
    hold.say(ASKED)
    expect(h.engine.snapshot().heard).toBe(ASKED)
    h.time.advance(LEVEL_DECAY_MS)
    expect(h.engine.levels().user).toBe(0)
    expect(h.phases).toEqual(['connecting', 'listening'])
  })

  it('counts the hold in whole seconds while the microphone is open', async () => {
    const h = quickAskHarness()
    await h.engine.start({ chatId: null })
    h.recognition.last().capture()

    h.time.advance(2_500)

    expect(h.engine.snapshot().durationSec).toBe(2)
  })

  it('ignores a second start while a hold is open, and reads an empty chat id as a new conversation', async () => {
    const h = quickAskHarness()

    await h.engine.start({ chatId: '' })
    await h.engine.start({ chatId: 'c-1' })

    expect(h.recognition.factory).toHaveBeenCalledTimes(1)
    expect(h.engine.snapshot().chatId).toBeNull()
  })
})

describe('release: the words go to Auto as a chat turn with spokenMode', () => {
  it('sends the final words once, on the conversation, with spokenMode, and Auto thinks', async () => {
    const h = quickAskHarness()

    const turn = await ask(h, ASKED)

    expect(h.recognition.last().stop).toHaveBeenCalledTimes(1)
    expect(h.chat.sendTurn).toHaveBeenCalledTimes(1)
    // The page's language travels with the turn (jsdom's navigator.language here).
    expect(turn.request).toEqual({ chatId: 'c-1', text: ASKED, spokenMode: true, locale: navigator.language })
    expect(h.engine.snapshot()).toMatchObject({ phase: 'thinking', heard: ASKED, micOpen: false, reply: '' })
  })

  it('sends the page’s language with the turn, so Auto answers in that variant', async () => {
    const h = quickAskHarness({ locale: 'pt-BR' })

    const turn = await ask(h, ASKED)

    expect(turn.request).toEqual({ chatId: 'c-1', text: ASKED, spokenMode: true, locale: 'pt-BR' })
  })

  it('adopts the id of the chat a new conversation’s turn creates, and tells the open chat the turn landed', async () => {
    const h = quickAskHarness()
    const turn = await ask(h, 'Hello', null)

    expect(turn.request.chatId).toBeNull()
    turn.handlers.onChatId('c-new')
    expect(h.engine.snapshot().chatId).toBe('c-new')
    turn.settle({ kind: 'finished' })
    await flush()

    expect(h.announceTurn).toHaveBeenCalledWith('c-new')
  })

  it('sends when the browser ends the speech by itself, before the member lets go', async () => {
    const h = quickAskHarness()
    await h.engine.start({ chatId: 'c-1' })
    const hold = h.recognition.last()
    hold.capture()
    hold.say('Book a room')

    hold.end()
    await flush()
    h.engine.release()

    expect(h.chat.sendTurn).toHaveBeenCalledTimes(1)
    expect(h.chat.last().request.text).toBe('Book a room')
  })

  it('a release before the browser captured stops the capture the moment it begins', async () => {
    const h = quickAskHarness()
    await h.engine.start({ chatId: 'c-1' })
    const hold = h.recognition.last()

    h.engine.release()
    expect(hold.stop).not.toHaveBeenCalled()
    hold.capture()

    expect(hold.stop).toHaveBeenCalledTimes(1)
  })

  it('ends the hold RELEASE_GRACE_MS after the release when the browser never says it ended', async () => {
    const h = quickAskHarness()
    await h.engine.start({ chatId: 'c-1' })
    const hold = h.recognition.last()
    hold.capture()
    hold.say('Is it raining?')

    h.engine.release()
    h.time.advance(RELEASE_GRACE_MS)
    await flush()

    expect(hold.abort).toHaveBeenCalled()
    expect(h.chat.last().request.text).toBe('Is it raining?')
  })

  it('lets go by itself after HOLD_MAX_MS', async () => {
    const h = quickAskHarness()
    await h.engine.start({ chatId: 'c-1' })
    const hold = h.recognition.last()
    hold.capture()
    hold.say('A long question')

    h.time.advance(HOLD_MAX_MS)
    expect(hold.stop).toHaveBeenCalledTimes(1)
    hold.end()
    await flush()

    expect(h.chat.last().request.text).toBe('A long question')
  })

  it('nothing said is "nothing heard", and nothing is sent', async () => {
    const h = quickAskHarness()
    await h.engine.start({ chatId: 'c-1' })
    const hold = h.recognition.last()
    hold.capture()

    h.engine.release()
    hold.end()
    await flush()

    expect(h.chat.sendTurn).not.toHaveBeenCalled()
    expect(h.engine.snapshot()).toMatchObject({ phase: 'error', error: QUICK_ASK_ERRORS.noSpeech, micOpen: false })
  })
})

describe('the reply is spoken sentence by sentence', () => {
  it('speaks each sentence as it completes, the AI notice first, markdown, links and code left to the chat', async () => {
    const h = quickAskHarness()
    const turn = await ask(h, ASKED)
    const opening = 'You have **two** meetings. One is at [ten](https://cal.example.com/1). '

    turn.handlers.onText(opening)
    expect(h.synthesis.spoken()).toEqual([GREETING, 'You have two meetings.'])
    turn.handlers.onText(`${opening}Details are at https://cal.example.com/2\n\n\`\`\`sh\nls -la\n\`\`\``)
    expect(h.synthesis.spoken()).toEqual([GREETING, 'You have two meetings.', 'One is at ten.', 'Details are at a link.'])
    turn.settle({ kind: 'finished' })
    await flush()

    expect(h.synthesis.spoken()).toEqual([
      GREETING, 'You have two meetings.', 'One is at ten.', 'Details are at a link.', "There's code in the chat.",
    ])
  })

  it('is speaking while a sentence plays, the reply growing as spoken, and idle once the last one ends', async () => {
    const h = quickAskHarness()
    const turn = await ask(h, ASKED)
    turn.handlers.onText('Two meetings. One at ten.')

    h.synthesis.start(0)
    expect(h.engine.snapshot()).toMatchObject({ phase: 'speaking', reply: GREETING })
    expect(h.engine.levels().agent).toBeGreaterThan(0)
    h.synthesis.end(0)
    h.synthesis.start(1)
    expect(h.engine.snapshot().reply).toBe(`${GREETING} Two meetings.`)
    h.synthesis.end(1)
    // The queue ran out while the turn still streams: Auto is working on the rest.
    expect(h.engine.snapshot().phase).toBe('thinking')
    expect(h.engine.levels().agent).toBe(0)
    turn.settle({ kind: 'finished' })
    await flush()
    h.synthesis.play(2)

    expect(h.engine.snapshot()).toMatchObject({ phase: 'idle', reply: `${GREETING} Two meetings. One at ten.` })
    expect(h.phases).toEqual(['connecting', 'listening', 'thinking', 'speaking', 'thinking', 'speaking', 'idle'])
    expect(h.announceTurn).toHaveBeenCalledWith('c-1')
  })

  it('never says narration that left the answer before it was spoken', async () => {
    const h = quickAskHarness()
    const turn = await ask(h, ASKED)

    turn.handlers.onText('Let me check')
    turn.handlers.onText('')
    turn.handlers.onText('You have two meetings.')
    turn.settle({ kind: 'finished' })
    await flush()

    expect(h.synthesis.spoken()).toEqual([GREETING, 'You have two meetings.'])
  })

  it('a turn that ends with nothing to say goes idle at once, and the notice waits for the next reply', async () => {
    const h = quickAskHarness()
    const turn = await ask(h, 'Do it quietly')

    turn.settle({ kind: 'finished' })
    await flush()

    expect(h.synthesis.spoken()).toEqual([])
    expect(h.engine.snapshot().phase).toBe('idle')
    expect(h.storage.values.has(AI_NOTICE_STORAGE_KEY)).toBe(false)
  })
})

describe('the AI notice (PRD-264 EUAI-04, D73)', () => {
  it('is said once per browser session, before the first reply, even to a new engine in the same tab', async () => {
    const h = quickAskHarness()
    const first = await ask(h, 'One?')
    first.handlers.onText('One.')
    first.settle({ kind: 'finished' })
    await flush()
    h.synthesis.play(0)
    h.synthesis.play(1)
    const second = await ask(h, 'Two?')
    second.handlers.onText('Two.')
    second.settle({ kind: 'finished' })
    await flush()

    expect(h.synthesis.spoken()).toEqual([GREETING, 'One.', 'Two.'])
    expect(h.storage.values.get(AI_NOTICE_STORAGE_KEY)).toBe('1')

    const again = quickAskHarness({ storage: h.storage })
    const third = await ask(again, 'Three?')
    third.handlers.onText('Three.')
    third.settle({ kind: 'finished' })
    await flush()
    expect(again.synthesis.spoken()).toEqual(['Three.'])
  })

  it('names the answering agent in place of Auto', async () => {
    const h = quickAskHarness()
    const turn = await ask(h, 'Who are you?')

    turn.handlers.onAgent('Research Agent')
    turn.handlers.onText('I look things up. ')
    turn.handlers.onText('I look things up. Ask me anything.')

    expect(h.synthesis.spoken()).toEqual(['Hi, this is Research Agent, your AI assistant.', 'I look things up.'])
  })
})
