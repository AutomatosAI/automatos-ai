/**
 * PRD-277 US-502, US-503 — in enterprise the chat box's POST and the document upload get the
 * session from the session fetch (lib/enterprise/session-fetch.ts), not from code of their own.
 *
 * Both call the page's `fetch` themselves: useChat posts to the web app's `/api/chat`, and
 * `apiClient.uploadDocument` to `/api/documents/upload` (relative, as the enterprise build calls the
 * API). Here the real wrapper is put over that `fetch`, as EnterpriseSessionProvider puts it over
 * `window.fetch` (enterprise-session-pages.test.tsx proves the provider does), and only the network
 * behind it is a double:
 *   - the chat POST and the upload carry the cross-site header and go with credentials;
 *   - a 401 in chat opens "Your session ended", with the API's reason, and shows no toast.
 */
import { act, fireEvent, render, renderHook, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi, type Mock } from 'vitest'

const toasts = vi.hoisted(() => ({ error: vi.fn(), success: vi.fn(), info: vi.fn() }))

vi.mock('@/lib/edition-name', () => ({ isEnterprise: true, resolveEdition: () => 'enterprise' }))
vi.mock('sonner', () => ({ toast: toasts }))
vi.mock('@/lib/auth-hooks', () => ({ useAuth: () => ({ isLoaded: true, getToken: async () => null }) }))
vi.mock('next/navigation', () => ({ usePathname: () => '/chat' }))

import { EnterpriseSessionProvider } from '@/components/enterprise/enterprise-session-provider'
import { apiClient } from '@/lib/api-client'
import { useChat } from '@/lib/chat/hooks'
import type { WhoAmI } from '@/lib/enterprise/auth-api'
import { SESSION_ENDED_BODY, SESSION_ENDED_REASONS, SESSION_ENDED_TITLE } from '@/lib/enterprise/messages'
import { installSessionFetch } from '@/lib/enterprise/session-fetch'
import { SESSION_CSRF_HEADER } from '@/lib/enterprise/session-transport'

type Send = (input: RequestInfo | URL, init?: RequestInit) => Promise<Response>
type Backend = Mock<Send>

const WHO: WhoAmI = {
  edition: 'enterprise',
  person: { id: 7, email: 'ada@example.test', name: 'Ada Example', system_role: 'user' },
  workspaces: [{ id: '00000000-0000-0000-0000-00000000c4a7', name: 'Research', role: 'editor' }],
  in_no_workspace: false,
  default_workspace_id: null,
}
const ENDED = { detail: { code: 'session_ended', reason: 'expired', message: 'Your session has ended. Sign in again.' } }

const pageFetch = window.fetch
let restore: () => void = () => {}

/** The page's `window.fetch` is the network double. */
function network(answer: () => Response): Backend {
  const backend = vi.fn<Send>(async () => answer())
  window.fetch = backend as unknown as typeof fetch
  return backend
}

/** The network double, with the real session fetch put over `window.fetch`, as the provider puts it. */
function pageWithSessionFetch(answer: () => Response): Backend {
  const backend = network(answer)
  restore = installSessionFetch(window, true)
  return backend
}

/** A finished /api/chat stream (the shape lib/chat/__tests__/prd266-agent-info-frame.test.ts uses). */
function chatStream(): Response {
  const body = new ReadableStream<Uint8Array>({
    start(controller) {
      controller.enqueue(new TextEncoder().encode('0:"Hello."\n'))
      controller.close()
    },
  })
  return { ok: true, status: 200, headers: { get: () => null }, body } as unknown as Response
}

/** What reached the network for `path`. */
function sentWith(backend: Backend, path: string): { headers: Headers; credentials?: RequestCredentials } {
  const call = backend.mock.calls.find(([input]) => String(input) === path)
  expect(call, `nothing was sent to ${path}`).toBeDefined()
  const init = call?.[1]
  return { headers: new Headers(init?.headers), credentials: init?.credentials }
}

afterEach(() => {
  restore()
  restore = () => {}
  window.fetch = pageFetch
  vi.unstubAllEnvs()
  vi.restoreAllMocks()
  toasts.error.mockClear()
})

describe('requests made outside apiClient, in enterprise', () => {
  it('the chat POST carries the cross-site header and goes with credentials', async () => {
    const backend = pageWithSessionFetch(chatStream)
    const { result } = renderHook(() => useChat({ id: 'c1' }))
    await act(async () => {
      await result.current.sendMessage('Hello?')
    })

    const sent = sentWith(backend, '/api/chat')
    expect(sent.headers.get(SESSION_CSRF_HEADER)).toBe('1')
    expect(sent.credentials).toBe('include')
  })

  it('the document upload carries the cross-site header and goes with credentials', async () => {
    vi.spyOn(console, 'log').mockImplementation(() => {})
    // The enterprise build calls the API by relative address (the session cookie is the page host's).
    vi.stubEnv('NEXT_PUBLIC_API_URL', '')
    const backend = pageWithSessionFetch(() => new Response('{"id": 1}', { status: 200 }))
    await apiClient.uploadDocument(new File(['notes'], 'notes.txt', { type: 'text/plain' }))

    const sent = sentWith(backend, '/api/documents/upload')
    expect(sent.headers.get(SESSION_CSRF_HEADER)).toBe('1')
    expect(sent.credentials).toBe('include')
  })

  it('a 401 in chat opens "Your session ended", with the reason, and shows no toast', async () => {
    vi.spyOn(apiClient, 'get').mockResolvedValue(WHO as never)
    // The provider puts the session fetch over this itself.
    const backend = network(() => new Response(JSON.stringify(ENDED), { status: 401 }))
    function ChatBox() {
      const { sendMessage } = useChat({ id: 'c1' })
      return <button onClick={() => void sendMessage('Hello?')}>send</button>
    }
    render(<EnterpriseSessionProvider><ChatBox /></EnterpriseSessionProvider>)

    fireEvent.click(await screen.findByText('send'))

    expect(await screen.findByText(SESSION_ENDED_TITLE)).toBeTruthy()
    expect(screen.getByText(`${SESSION_ENDED_REASONS.expired} ${SESSION_ENDED_BODY}`)).toBeTruthy()
    expect(toasts.error).not.toHaveBeenCalled()
    expect(sentWith(backend, '/api/chat').headers.get(SESSION_CSRF_HEADER)).toBe('1')
  })
})
