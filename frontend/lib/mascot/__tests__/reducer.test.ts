// @vitest-environment node
/**
 * PRD-260 D6 — Auto's state machine, ported from Automatos Play's
 * tests/mascot/reducer.test.ts: the state table, the priority order, the
 * one-shots and the page's own states; and (T-005c) Auto listening while an
 * approval card waits for the owner.
 */
import { describe, expect, it } from 'vitest'

import { INITIAL_MASCOT, mascotReducer, pendingPose } from '@/lib/mascot/reducer'
import type { MascotEvent, ToolEndEvent } from '@/lib/mascot/types'
import type { Receipt } from '@/types'

import { receipt, replay, sequence } from './helpers'

const SEND: MascotEvent = { type: 'send' }
const FINISH: MascotEvent = { type: 'finish', finishReason: 'stop' }
const FAILED: MascotEvent = { type: 'error', message: 'x', code: null }
const say = (text: string): MascotEvent => ({ type: 'text', text })
const start = (toolCallId: string, toolName: string): MascotEvent => ({ type: 'tool-start', toolCallId, toolName, input: {} })
const end = (toolCallId: string, fields: Partial<ToolEndEvent> = {}): MascotEvent => ({
  type: 'tool-end',
  toolCallId,
  toolName: 'any',
  success: true,
  error: null,
  durationMs: 10,
  summary: null,
  skipped: false,
  waiting: false,
  ...fields,
})
const receipts = (...statuses: Receipt['status'][]): MascotEvent => ({
  type: 'receipts',
  model: null,
  above: [],
  receipts: statuses.map((status) => receipt(status)),
})
const ended = (state: 'hello' | 'celebrate'): MascotEvent => ({ type: 'clip-ended', state })

describe('mascot reducer: the state table', () => {
  it('maps each tool kind to its state', () => {
    expect(replay([SEND, start('1', 'platform_web_search')]).mascot).toBe('searching')
    expect(replay([SEND, start('1', 'search_knowledge')]).mascot).toBe('reading')
    expect(replay([SEND, start('1', 'generate_document')]).mascot).toBe('working')
  })

  it('maps progress, workflow updates and agent hand-offs to working, memories to reading', () => {
    expect(replay([SEND, { type: 'progress', text: 'Step 2' }]).mascot).toBe('working')
    const workflow: MascotEvent = { type: 'workflow-update', workflowId: 'w', status: 'running', currentStep: null, progress: null }
    expect(replay([SEND, workflow]).mascot).toBe('working')
    const assigned: MascotEvent = { type: 'agent-assigned', agentId: 2, agentName: 'Scout', routeMethod: null }
    expect(replay([SEND, assigned]).mascot).toBe('working')
    expect(replay([SEND, { type: 'memory-injected', memories: [], totalMatched: 0 }]).mascot).toBe('reading')
  })

  it('goes to oops on every failure the table lists', () => {
    const failures: MascotEvent[] = [
      FAILED,
      { type: 'limit_reached', limit: 'steps', value: 20, message: 'Stopped' },
      { type: 'tool-permission-denied', toolName: 'x', agentId: null, reason: null },
      { type: 'attachment_unavailable', attachments: [] },
      receipts('done', 'refused'),
      { type: 'request-failed' },
    ]
    for (const failure of failures) expect({ failure: failure.type, mascot: replay([SEND, failure]).mascot }).toEqual({ failure: failure.type, mascot: 'oops' })
    expect(replay([SEND, start('1', 'x'), end('1', { success: false })]).mascot).toBe('oops')
  })

  it('treats a skipped tool as no failure', () => {
    expect(replay([SEND, start('1', 'x'), end('1', { success: false, skipped: true })]).mascot).toBe('idle')
  })

  it("ignores events that don't move Auto", () => {
    const thinking = replay([SEND])
    const quiet: MascotEvent[] = [
      { type: 'chat-id', chatId: 'c' },
      { type: 'usage', promptTokens: 1, completionTokens: 1, totalTokens: 2 },
      { type: 'narration', text: 'n', retracted: false },
      { type: 'unknown', eventType: 'new-thing' },
    ]
    for (const event of quiet) expect(mascotReducer(thinking, event)).toBe(thinking)
  })

  it('listens while the message box has focus, and lets go on blur', () => {
    expect(sequence([{ type: 'input-focus' }, { type: 'input-blur' }])).toEqual(['listening', 'idle'])
  })

  it('returns to idle when a turn ends without words or tools', () => {
    expect(sequence([SEND, FINISH])).toEqual(['thinking', 'idle'])
  })

  it('stops thinking once the words arrive', () => {
    expect(sequence([SEND, say('Hello')])).toEqual(['thinking', 'idle'])
  })
})

describe('mascot reducer: priority', () => {
  it('keeps a tool state over lower-ranked events', () => {
    const events: MascotEvent[] = [SEND, start('1', 'platform_search_documents'), { type: 'progress', text: 'p' }, { type: 'reasoning', delta: 'r' }, say('hi')]
    expect(sequence(events)).toEqual(['thinking', 'reading'])
  })

  it('falls back to the tool still running when one of two ends', () => {
    const events = [SEND, start('w', 'platform_web_search'), start('g', 'generate_document'), end('w')]
    expect(replay(events).mascot).toBe('working')
    expect(replay([...events, end('g')]).mascot).toBe('idle')
  })

  it('lets oops beat celebrate, and waiting beat typing', () => {
    expect(replay([SEND, receipts('done'), FAILED]).mascot).toBe('oops')
    expect(replay([SEND, receipts('done', 'refused'), FINISH]).mascot).toBe('oops')
    const asked = replay([SEND, start('1', 'x'), end('1', { success: false, waiting: true }), FINISH])
    expect(asked.mascot).toBe('listening')
    expect(mascotReducer(asked, { type: 'input-focus' }).mascot).toBe('listening')
    expect(mascotReducer(asked, { type: 'input-blur' })).toBe(asked)
  })

  it('does not let typing clear oops; sending does', () => {
    const failed = replay([SEND, FAILED])
    expect(mascotReducer(failed, { type: 'input-focus' }).mascot).toBe('oops')
    expect(mascotReducer(failed, SEND).mascot).toBe('thinking')
  })
})

describe('mascot reducer: one-shots', () => {
  it('celebrates on finish after a done receipt, and only then', () => {
    expect(sequence([SEND, receipts('done'), FINISH])).toEqual(['thinking', 'celebrate'])
    expect(sequence([SEND, receipts('skipped', 'waiting'), FINISH])).toEqual(['thinking', 'idle'])
    const celebrated = replay([SEND, receipts('done'), FINISH, ended('celebrate')])
    expect(celebrated.mascot).toBe('idle')
    expect(sequence([SEND, FINISH], celebrated)).toEqual(['thinking', 'idle'])
  })

  it('plays hello on the first open and returns to idle when it ends', () => {
    expect(sequence([{ type: 'app-open' }, ended('hello')])).toEqual(['hello', 'idle'])
  })

  it('queues what happens during a one-shot and plays it next', () => {
    const state = replay([{ type: 'deliverable-added' }, { type: 'input-focus' }])
    expect(state.mascot).toBe('celebrate')
    expect(pendingPose(state).mascot).toBe('listening')
    expect(mascotReducer(state, ended('celebrate')).mascot).toBe('listening')
  })

  it('lets a new message during celebrate end in thinking, or idle once words arrive', () => {
    const celebrating = replay([{ type: 'deliverable-added' }, SEND])
    expect(mascotReducer(celebrating, ended('celebrate')).mascot).toBe('thinking')
    const answered = mascotReducer(celebrating, say('hi'))
    expect(mascotReducer(answered, ended('celebrate')).mascot).toBe('idle')
  })

  it('swaps urgent states in at once: oops interrupts a one-shot', () => {
    expect(sequence([{ type: 'deliverable-added' }, { type: 'request-failed' }])).toEqual(['celebrate', 'oops'])
    expect(sequence([SEND, start('1', 'platform_web_search'), { type: 'app-open' }])).toEqual(['thinking', 'searching', 'hello'])
  })

  it("ignores the end of a clip that isn't the one-shot playing", () => {
    const celebrating = replay([{ type: 'deliverable-added' }])
    expect(mascotReducer(celebrating, ended('hello'))).toBe(celebrating)
  })
})

describe('mascot reducer: client states', () => {
  it('sleeps after the idle timeout, but not mid-turn or over a stronger state', () => {
    expect(replay([{ type: 'idle-timeout' }]).mascot).toBe('sleeping')
    expect(replay([SEND, say('hi'), { type: 'idle-timeout' }]).mascot).toBe('idle')
    expect(replay([SEND, FAILED, { type: 'idle-timeout' }]).mascot).toBe('oops')
    expect(sequence([{ type: 'idle-timeout' }, { type: 'input-focus' }])).toEqual(['sleeping', 'listening'])
  })

  it('wakes from the idle nap on any input, but not from the backend being down', () => {
    expect(sequence([{ type: 'idle-timeout' }, { type: 'wake' }])).toEqual(['sleeping', 'idle'])
    expect(mascotReducer(INITIAL_MASCOT, { type: 'wake' })).toBe(INITIAL_MASCOT)
    const down = replay([{ type: 'backend-down' }])
    expect(mascotReducer(down, { type: 'wake' })).toBe(down)
  })

  it('sleeps while the backend is down and ignores everything until it is back', () => {
    const down = replay([SEND, start('1', 'x'), { type: 'backend-down' }])
    expect(down).toMatchObject({ mascot: 'sleeping', backendDown: true, turnActive: false, openTools: [] })
    expect(mascotReducer(down, { type: 'input-focus' })).toBe(down)
    expect(mascotReducer(down, SEND)).toBe(down)
    expect(mascotReducer(down, { type: 'backend-up' })).toMatchObject({ mascot: 'idle', backendDown: false })
  })

  it("ignores backend-up when the backend wasn't down", () => {
    expect(mascotReducer(INITIAL_MASCOT, { type: 'backend-up' })).toBe(INITIAL_MASCOT)
  })

  it('shows lost on the 404 page', () => {
    expect(replay([{ type: 'not-found' }]).mascot).toBe('lost')
  })

  it('shows loading only over a calm state, and returns to idle after', () => {
    expect(sequence([{ type: 'loading-start' }, { type: 'loading-end' }])).toEqual(['loading', 'idle'])
    expect(replay([SEND, { type: 'loading-start' }]).mascot).toBe('thinking')
    expect(replay([{ type: 'not-found' }, { type: 'loading-start' }]).mascot).toBe('loading')
  })

  it('returns the same state object when nothing changes', () => {
    expect(mascotReducer(INITIAL_MASCOT, { type: 'input-blur' })).toBe(INITIAL_MASCOT)
    expect(mascotReducer(INITIAL_MASCOT, { type: 'loading-end' })).toBe(INITIAL_MASCOT)
  })
})

describe('mascot reducer: an approval waits for the owner (T-005c)', () => {
  const WAITS: MascotEvent = { type: 'approval-waiting' }
  const SETTLED: MascotEvent = { type: 'approval-settled' }
  const waiting = () => replay([WAITS])

  it('listens while an approval waits, after the turn ends too, and idles once every card is settled', () => {
    expect(sequence([SEND, say('Here is the plan'), WAITS, FINISH, SETTLED])).toEqual(['thinking', 'idle', 'listening', 'idle'])
    expect(replay([SEND, WAITS, FINISH])).toMatchObject({ mascot: 'listening', approvalWaiting: true, turnActive: false })
  })

  it("outranks the turn's tools and thinking, never oops", () => {
    expect(replay([SEND, start('1', 'generate_document'), WAITS]).mascot).toBe('listening')
    expect(replay([SEND, WAITS]).mascot).toBe('listening')
    expect(replay([SEND, FAILED, WAITS])).toMatchObject({ mascot: 'oops', approvalWaiting: true })
  })

  it('rests on listening instead of idle: after a celebrate, and after a later send thinks, works and answers', () => {
    expect(sequence([{ type: 'deliverable-added' }, ended('celebrate')], waiting())).toEqual(['celebrate', 'listening'])
    expect(sequence([SEND, say('On it')], waiting())).toEqual(['thinking', 'listening'])
    const worked = [SEND, start('1', 'generate_document'), start('2', 'platform_web_search'), end('2')]
    expect(sequence(worked, waiting())).toEqual(['thinking', 'working', 'searching', 'working'])
    expect(sequence([...worked, end('1'), FINISH], waiting())).toEqual(['thinking', 'working', 'searching', 'working', 'listening'])
  })

  it('never naps while an approval waits', () => {
    const state = waiting()
    expect(mascotReducer(state, { type: 'idle-timeout' })).toBe(state)
  })

  it('keeps the flag while the backend is down, and wakes to listening or, once settled, to idle', () => {
    const down = replay([WAITS, { type: 'backend-down' }])
    expect(down).toMatchObject({ mascot: 'sleeping', approvalWaiting: true })
    expect(mascotReducer(down, { type: 'backend-up' }).mascot).toBe('listening')
    const settledWhileDown = mascotReducer(down, SETTLED)
    expect(settledWhileDown).toMatchObject({ mascot: 'sleeping', approvalWaiting: false })
    expect(mascotReducer(settledWhileDown, { type: 'backend-up' }).mascot).toBe('idle')
    expect(replay([{ type: 'backend-down' }, WAITS])).toMatchObject({ mascot: 'sleeping', approvalWaiting: true })
  })

  it('settling with nothing waiting changes nothing; a second wait returns the same state', () => {
    expect(mascotReducer(INITIAL_MASCOT, SETTLED)).toBe(INITIAL_MASCOT)
    const state = waiting()
    expect(mascotReducer(state, WAITS)).toBe(state)
  })

  it('settling ends listening for the card, not a stronger state', () => {
    expect(replay([SEND, FAILED, WAITS, SETTLED]).mascot).toBe('oops')
    const celebrating = replay([WAITS, { type: 'deliverable-added' }, SETTLED])
    expect(celebrating.mascot).toBe('celebrate')
    expect(mascotReducer(celebrating, ended('celebrate')).mascot).toBe('idle')
  })
})
