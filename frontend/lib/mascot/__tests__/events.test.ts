// @vitest-environment node
/**
 * PRD-260 D5, D6 — the core chat as Auto's events: each of useChat's onData
 * parts (lib/chat/hooks.ts, the receipts frame as useChatWithReceipts reads it)
 * and each status change maps as lib/mascot/events.ts says, unknown parts map to
 * nothing, the reply's words starting end thinking (T-003b), and whole turns
 * built from those parts play the states Play's spec expects.
 */
import { describe, expect, it } from 'vitest'

import { LIVE_KEY } from '@/lib/chat/receipts'
import {
  BLOCKED_TURN,
  type ChatStatus,
  FAILED_TURN,
  mascotEventsFromData,
  mascotEventsFromReply,
  mascotEventsFromStatus,
  NO_REPLY,
  type ReplyProgress,
} from '@/lib/mascot/events'
import type { MascotEvent } from '@/lib/mascot/types'
import { TRIAL_EXHAUSTED_CODE } from '@/lib/trial'
import type { Receipt } from '@/types'

import { receipt, replay, sequence } from './helpers'

const toolStart = (toolCallId: string, toolName: string) => ({ type: 'tool-start', data: { toolCallId, toolName, input: { q: 'laptops' } } })
const toolEnd = (toolCallId: string, fields: Record<string, unknown> = {}) => ({
  type: 'tool-end',
  data: { toolCallId, toolName: 'platform_web_search', success: true, error: null, durationMs: 120, summary: '3 results', ...fields },
})
const receiptsPart = (list: readonly unknown[], frame: Record<string, unknown> = {}) => ({
  type: 'tool-data',
  data: { [LIVE_KEY]: { receipts: list, ...frame } },
})

/** The events a run of status changes makes, starting from idle as useAutoMascot does. */
function fromStatuses(statuses: readonly ChatStatus[], errorCode: string | null = null): MascotEvent[] {
  return statuses.reduce<{ previous: ChatStatus; events: MascotEvent[] }>(
    (acc, next) => ({ previous: next, events: [...acc.events, ...mascotEventsFromStatus(acc.previous, next, errorCode)] }),
    { previous: 'idle', events: [] },
  ).events
}

const data = (part: unknown): MascotEvent[] => mascotEventsFromData(part)

/** The chat starts streaming, and later goes back to idle. */
const STARTED = mascotEventsFromStatus('idle', 'streaming', null)
const FINISHED = mascotEventsFromStatus('streaming', 'idle', null)

describe('mascotEventsFromData: tool lines', () => {
  it('maps a tool-start part to tool-start with its id, name and input', () => {
    expect(data(toolStart('c1', 'platform_web_search'))).toEqual([
      { type: 'tool-start', toolCallId: 'c1', toolName: 'platform_web_search', input: { q: 'laptops' } },
    ])
  })

  it("names an unnamed tool as useChat does, and drops a part with no call id", () => {
    expect(data({ type: 'tool-start', data: { toolCallId: 'c2' } })).toEqual([
      { type: 'tool-start', toolCallId: 'c2', toolName: 'tool', input: {} },
    ])
    expect(data({ type: 'tool-start', data: { toolName: 'x' } })).toEqual([])
    expect(data({ type: 'tool-end', data: { toolCallId: '' } })).toEqual([])
    expect(data({ type: 'tool-end', data: { toolCallId: 7 } })).toEqual([])
  })

  it('maps a tool-end part to tool-end, with skipped and waiting only when they are true', () => {
    expect(data(toolEnd('c1'))).toEqual([{
      type: 'tool-end', toolCallId: 'c1', toolName: 'platform_web_search', success: true, error: null,
      durationMs: 120, summary: '3 results', skipped: false, waiting: false,
    }])
    const [failed] = data(toolEnd('c1', { success: false, error: 'timeout' }))
    expect(failed).toMatchObject({ type: 'tool-end', success: false, error: 'timeout', skipped: false, waiting: false })
    expect(data(toolEnd('c1', { success: false, skipped: true }))[0]).toMatchObject({ skipped: true, waiting: false })
    expect(data(toolEnd('c1', { success: false, waiting: true }))[0]).toMatchObject({ skipped: false, waiting: true })
    expect(data(toolEnd('c1', { success: 'yes', skipped: 'true', waiting: 1 }))[0]).toMatchObject({ success: false, skipped: false, waiting: false })
  })
})

describe('mascotEventsFromData: receipts', () => {
  const ALL: readonly Receipt[] = [receipt('done'), receipt('refused'), receipt('skipped'), receipt('waiting')]

  it('maps the receipts frame on a tool-data part to receipts, every status kept', () => {
    const [event] = data(receiptsPart(ALL, { model: 'claude', above: ['Just to be clear: nothing was sent.'] }))
    expect(event).toEqual({ type: 'receipts', receipts: ALL, model: 'claude', above: ['Just to be clear: nothing was sent.'] })
  })

  it('keeps only real receipts, as the reply does, and reads a frame with none as an empty list', () => {
    const [event] = data(receiptsPart([receipt('done'), { action: 'a', status: 'exploded' }, null]))
    expect(event).toEqual({ type: 'receipts', receipts: [receipt('done')], model: null, above: [] })
    expect(data(receiptsPart([]))).toEqual([{ type: 'receipts', receipts: [], model: null, above: [] }])
  })

  it('maps a tool-data part without a receipts frame to nothing', () => {
    expect(data({ type: 'tool-data', data: { documents: [{ id: 'd1' }] } })).toEqual([])
    expect(data({ type: 'tool-data', data: { [LIVE_KEY]: { receipts: 'not a list' } } })).toEqual([])
  })
})

describe('mascotEventsFromData: the other parts', () => {
  it('maps memories to reading and a workflow update to working (Play spec)', () => {
    expect(data({ type: 'memory-injected', data: { memories: [{ id: 'm', memory: 'x', tier: 'agent' }], totalMatched: 4 } }))
      .toEqual([{ type: 'memory-injected', memories: [{ id: 'm', memory: 'x', tier: 'agent' }], totalMatched: 4 }])
    expect(data({ type: 'workflow-update', data: { workflowId: 'w1', status: 'running', currentStep: 'draft', progress: 40 } }))
      .toEqual([{ type: 'workflow-update', workflowId: 'w1', status: 'running', currentStep: 'draft', progress: 40 }])
  })

  it('maps a mission suggestion to nothing: the chat shows its card, Auto stays as he is', () => {
    expect(data({ type: 'mission-suggestion', data: { goal: 'Launch plan', complexity: 'high', agent_id: 3 } })).toEqual([])
  })

  it('maps unknown and malformed parts to nothing', () => {
    const quiet: unknown[] = [
      { type: 'data-usage', data: { promptTokens: 1, completionTokens: 2, totalTokens: 3 } },
      { type: 'agent-info', data: { name: 'Auto' } },
      { type: 'memory-stored', data: { memory: {}, reason: 'r' } },
      { type: 'something-new', data: { x: 1 } },
      { type: 'tool-start' },
      { type: 'tool-start', data: 'c1' },
      { type: 'tool-end', data: [] },
      { data: { toolCallId: 'c1' } },
      null,
      undefined,
      'tool-start',
      42,
    ]
    for (const part of quiet) expect({ part, events: data(part) }).toEqual({ part, events: [] })
  })
})

describe('mascotEventsFromStatus', () => {
  it('starts the turn when the chat starts streaming, from idle or after an error', () => {
    expect(mascotEventsFromStatus('idle', 'streaming', null)).toEqual([{ type: 'send' }])
    expect(mascotEventsFromStatus('error', 'streaming', null)).toEqual([{ type: 'send' }])
  })

  it('finishes the turn when streaming goes back to idle', () => {
    expect(mascotEventsFromStatus('streaming', 'idle', null)).toEqual([{ type: 'finish', finishReason: null }])
  })

  it('fails the turn on error, and on a blocked turn whose error landed with the end of the stream', () => {
    expect(mascotEventsFromStatus('streaming', 'error', null)).toEqual([{ type: 'error', message: FAILED_TURN, code: null }])
    expect(mascotEventsFromStatus('idle', 'error', TRIAL_EXHAUSTED_CODE))
      .toEqual([{ type: 'error', message: FAILED_TURN, code: TRIAL_EXHAUSTED_CODE }])
    expect(mascotEventsFromStatus('streaming', 'idle', TRIAL_EXHAUSTED_CODE))
      .toEqual([{ type: 'error', message: BLOCKED_TURN, code: TRIAL_EXHAUSTED_CODE }])
  })

  it('makes nothing when the status is unchanged or the error is already told', () => {
    for (const status of ['idle', 'streaming', 'error'] as const) {
      expect(mascotEventsFromStatus(status, status, TRIAL_EXHAUSTED_CODE)).toEqual([])
    }
    expect(mascotEventsFromStatus('error', 'idle', null)).toEqual([])
    expect(mascotEventsFromStatus('error', 'idle', TRIAL_EXHAUSTED_CODE)).toEqual([])
  })
})

describe("whole turns from the core chat's parts", () => {
  it('a web search that did something: thinking, searching, idle, then celebrate at the end', () => {
    const events = [
      ...STARTED,
      ...data(toolStart('c1', 'platform_web_search')),
      ...data(toolEnd('c1')),
      ...data(receiptsPart([receipt('done', { kind: 'write' })])),
      ...FINISHED,
    ]
    expect(sequence(events)).toEqual(['thinking', 'searching', 'idle', 'celebrate'])
    expect(replay([...events, { type: 'clip-ended', state: 'celebrate' }])).toMatchObject({ mascot: 'idle', turnActive: false })
  })

  it('an ask: Auto keeps listening after the turn, waiting for the owner', () => {
    const events = [
      ...STARTED,
      ...data(toolStart('c1', 'platform_update_agent')),
      ...data(toolEnd('c1', { success: false, waiting: true })),
      ...data(receiptsPart([receipt('waiting')])),
      ...FINISHED,
    ]
    expect(sequence(events)).toEqual(['thinking', 'working', 'listening'])
  })

  it('a refused write, a failed tool and a failed turn end on oops; the next send thinks again', () => {
    const turn = STARTED
    expect(replay([...turn, ...data(receiptsPart([receipt('refused')]))]).mascot).toBe('oops')
    expect(replay([...turn, ...data(toolStart('c1', 'query_database')), ...data(toolEnd('c1', { success: false }))]).mascot).toBe('oops')
    const failed = replay(fromStatuses(['streaming', 'error', 'idle']))
    expect(failed).toMatchObject({ mascot: 'oops', turnActive: false })
    expect(sequence(STARTED, failed)).toEqual(['thinking'])
  })

  it('a blocked turn ends on oops even when the stream ended in the same render', () => {
    expect(replay(fromStatuses(['streaming', 'idle'], TRIAL_EXHAUSTED_CODE)).mascot).toBe('oops')
  })

  it('reading memories, then a workflow running', () => {
    const events = [
      ...STARTED,
      ...data({ type: 'memory-injected', data: { memories: [], totalMatched: 0 } }),
      ...FINISHED,
    ]
    expect(sequence(events)).toEqual(['thinking', 'reading', 'idle'])
    expect(replay([...STARTED, ...data({ type: 'workflow-update', data: { workflowId: 'w', status: 'running' } })]).mascot).toBe('working')
  })
})

describe("mascotEventsFromReply: the reply's words starting (T-003b)", () => {
  const empty = (id: string): ReplyProgress => ({ id, hasText: false, toolRunning: false })
  const words = (id: string, toolRunning = false): ReplyProgress => ({ id, hasText: true, toolRunning })
  const ANSWER: MascotEvent[] = [{ type: 'text', text: '' }]

  it("the newest reply's first words while the chat streams are one text event", () => {
    expect(mascotEventsFromReply(empty('r2'), words('r2'), 'streaming')).toEqual(ANSWER)
    expect(mascotEventsFromReply(NO_REPLY, words('r2'), 'streaming')).toEqual(ANSWER)
    expect(mascotEventsFromReply(words('r1'), words('r2'), 'streaming')).toEqual(ANSWER)
  })

  it("the same reply's later words, an empty reply and no reply map to nothing", () => {
    expect(mascotEventsFromReply(words('r2'), words('r2'), 'streaming')).toEqual([])
    expect(mascotEventsFromReply(empty('r2'), empty('r2'), 'streaming')).toEqual([])
    expect(mascotEventsFromReply(words('r1'), NO_REPLY, 'streaming')).toEqual([])
  })

  it('words that start while a tool runs map to nothing, also once the tool ends: the tool decides', () => {
    expect(mascotEventsFromReply(empty('r2'), words('r2', true), 'streaming')).toEqual([])
    expect(mascotEventsFromReply(words('r2', true), words('r2'), 'streaming')).toEqual([])
  })

  it("maps nothing unless the chat streams (a stored reply, a finished or failed turn)", () => {
    for (const status of ['idle', 'error'] as const) {
      expect({ status, events: mascotEventsFromReply(empty('r2'), words('r2'), status) }).toEqual({ status, events: [] })
    }
  })
})

describe('a whole turn with its words (T-003b)', () => {
  const reply = (hasText: boolean, toolRunning = false): ReplyProgress => ({ id: 'r2', hasText, toolRunning })
  const words = (previous: ReplyProgress, next: ReplyProgress) => mascotEventsFromReply(previous, next, 'streaming')

  it('send → thinking → first text → idle → tool-start → working → tool-end → idle → finish', () => {
    const events = [
      ...STARTED,
      ...words(reply(false), reply(true)),
      ...data(toolStart('c1', 'query_database')),
      ...words(reply(true), reply(true, true)),
      ...data(toolEnd('c1')),
      ...words(reply(true, true), reply(true)),
      ...FINISHED,
    ]
    expect(sequence(events)).toEqual(['thinking', 'idle', 'working', 'idle'])
    expect(replay(events)).toMatchObject({ mascot: 'idle', turnActive: false, openTools: [] })
  })

  it('tool-start still wins: words that arrive while a tool runs keep its state until it ends', () => {
    const events = [
      ...STARTED,
      ...data(toolStart('c1', 'platform_web_search')),
      ...words(reply(false, true), reply(true, true)),
    ]
    expect(sequence(events)).toEqual(['thinking', 'searching'])
    expect(sequence([...events, ...data(toolEnd('c1')), ...FINISHED])).toEqual(['thinking', 'searching', 'idle'])
  })
})
