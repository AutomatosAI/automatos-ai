/**
 * PRD-266 US-007b — the agent-info frame reaches onData.
 *
 * The backend wrote the agent-info frame as d:{"type": {"type": "agent-info", …}}, so
 * useChat's parser (lib/chat/hooks.ts: data.type === 'agent-info' && data.agent) never
 * forwarded it. consumers/chatbot/streaming.py now writes the frame itself; AGENT_INFO_LINE
 * is the exact line it writes (orchestrator/tests/test_prd266_chat_agent_info_history.py
 * asserts the same string). The real hook reads a stubbed /api/chat stream; only the
 * network, the identity and the toasts are mocked.
 */
import { act, renderHook } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

vi.mock('sonner', () => ({ toast: { error: vi.fn(), success: vi.fn(), info: vi.fn() } }))
vi.mock('@/lib/auth-hooks', () => ({ useAuth: () => ({ isLoaded: true, getToken: async () => null }) }))

import { useChat } from '@/lib/chat/hooks'

const AGENT = { id: 7, name: 'Auto', type: 'auto', skills: ['web_search'] }
const AGENT_INFO_LINE =
  'd:{"type": "agent-info", "agent": {"id": 7, "name": "Auto", "type": "auto", "skills": ["web_search"]}}'
/** The line the backend wrote before the fix: the frame under `type`. */
const NESTED_LINE = `d:{"type": ${AGENT_INFO_LINE.slice('d:'.length)}}`

/** A /api/chat response whose body is these lines, then the end of the stream. */
function chatResponse(lines: string[]): Response {
  const body = new ReadableStream<Uint8Array>({
    start(controller) {
      controller.enqueue(new TextEncoder().encode(lines.map((line) => `${line}\n`).join('')))
      controller.close()
    },
  })
  return { ok: true, status: 200, headers: { get: () => null }, body } as unknown as Response
}

async function framesFor(lines: string[]): Promise<unknown[]> {
  vi.stubGlobal('fetch', vi.fn(async () => chatResponse(lines)))
  const onData = vi.fn()
  const { result } = renderHook(() => useChat({ id: 'c1', onData }))
  await act(async () => {
    await result.current.sendMessage('Who is answering?')
  })
  return onData.mock.calls.map(([frame]) => frame)
}

afterEach(() => {
  vi.unstubAllGlobals()
})

describe('the agent-info frame (PRD-266 US-007b)', () => {
  it("turns the backend's exact line into onData({ type: 'agent-info', data: agent })", async () => {
    const frames = await framesFor([AGENT_INFO_LINE, '0:"Hello."'])

    expect(frames).toContainEqual({ type: 'agent-info', data: AGENT })
  })

  it('never forwarded the nested line the backend wrote before', async () => {
    const frames = await framesFor([NESTED_LINE, '0:"Hello."'])

    expect(frames.some((frame) => (frame as { type?: string }).type === 'agent-info')).toBe(false)
  })
})
