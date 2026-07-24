import { describe, it, expect, vi, afterEach } from 'vitest'

// PRD-205 S7 — the frontend half of background→chat delivery:
// * the SSE parser yields chat_changed frames intact (the hook fans them
//   out as window events);
// * getChatMessages maps the persisted messages.source.label into the
//   existing metadata badge slot, so "Auto · background" survives reload.

import { parseSSEFrames } from '@/hooks/use-board-event-stream'

describe('PRD-205 — SSE chat_changed frames', () => {
  it('parses a chat_changed frame with its payload', () => {
    const buffer =
      'event: chat_changed\n' +
      'data: {"workspace_id":"ws-1","chat_id":"c-9","user_id":3,"event":"chat_changed"}\n\n'
    const { events, rest } = parseSSEFrames(buffer)
    expect(rest).toBe('')
    expect(events).toHaveLength(1)
    expect(events[0].event).toBe('chat_changed')
    expect((events[0].data as { chat_id: string }).chat_id).toBe('c-9')
  })

  it('keeps board_changed frames working beside chat frames', () => {
    const buffer =
      'event: board_changed\ndata: {"workspace_id":"ws-1"}\n\n' +
      'event: chat_changed\ndata: {"workspace_id":"ws-1","chat_id":"c-1"}\n\n'
    const { events } = parseSSEFrames(buffer)
    expect(events.map((e) => e.event)).toEqual(['board_changed', 'chat_changed'])
  })
})

describe('PRD-205 — persisted source → badge slot', () => {
  afterEach(() => vi.restoreAllMocks())

  it('maps messages.source.label into metadata.source (survives reload)', async () => {
    const { apiClient } = await import('@/lib/api-client')
    vi.spyOn(apiClient, 'request').mockResolvedValue([
      {
        id: 'm1',
        role: 'assistant',
        parts: [{ type: 'text', text: 'Watched it. 8.4/10, passed.' }],
        source: { origin: 'watcher', label: 'Auto · background' },
      },
      {
        id: 'm2',
        role: 'user',
        parts: [{ type: 'text', text: 'run this and watch it' }],
        source: null,
      },
    ] as never)

    const { getChatMessages } = await import('@/lib/chat/api')
    const messages = await getChatMessages('c-9')

    expect(messages[0].metadata?.source).toBe('Auto · background')
    expect(messages[1].metadata?.source).toBeUndefined()
  })
})
