/**
 * PRD-266 US-006b — editors answer on the web too (§4.6, D16).
 *
 * The API decides who answers (`require_workspace_answerer`: owner, admin or editor); the
 * web reads its answer. With grants returned (an editor), the Questions tab renders a
 * QuestionCard that answers, and the Governance approvals inbox renders its cards with
 * Grant and Deny. With a 403 (a viewer), both say plainly who can answer; any other
 * failure keeps the surface's load-error line. No copy calls an editor a non-admin.
 *
 * The real apiClient and hooks run over a stubbed fetch, so every call travels the real path.
 */
import type { ReactNode } from 'react'
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import type { ApprovalGrant } from '@/lib/api-client'

vi.mock('next/link', () => ({
  default: ({ href, children, ...rest }: { href: unknown; children: ReactNode }) => (
    <a href={String(href)} {...rest}>
      {children}
    </a>
  ),
}))

vi.mock('sonner', () => ({
  toast: { success: vi.fn(), error: vi.fn(), info: vi.fn() },
}))

import { QuestionsTab } from '../questions-tab'
import { ApprovalsInbox } from '../governance/approvals-inbox'

const GRANTS = 'GET /api/v1/approval-grants'
const MISSIONS = 'GET /api/missions'
const ANSWER = 'POST /api/v1/approval-grants/7/answer'
const GRANT = 'POST /api/v1/approval-grants/8/grant'
const REFUSAL = 'Only owners, admins and editors can answer questions.'
const ADMIN_ONLY_COPY = /workspace admin/i

type Answer = { readonly status: number; readonly body: unknown }

const ok = (body: unknown): Answer => ({ status: 200, body })
const viewerRefused: Answer = { status: 403, body: { detail: REFUSAL } }
const serverFailed: Answer = { status: 500, body: { detail: 'Internal Server Error' } }

function grant(overrides: Partial<ApprovalGrant>): ApprovalGrant {
  return {
    id: 1,
    workspace_id: 'ws',
    subject_type: 'board_task',
    subject_id: '42',
    tool_name: null,
    risk_tier: null,
    agent_id: 9,
    status: 'pending',
    reason: null,
    estimated_cost_usd: null,
    requested_at: '2026-10-20T10:00:00Z',
    expires_at: null,
    granted_at: null,
    granted_by: null,
    revoked_at: null,
    revoked_by: null,
    ...overrides,
  }
}

const QUESTION = grant({ id: 7, kind: 'question', question_md: 'Ship **A** or **B**?', options: ['A', 'B'], asked_by_agent_id: 9 })
const APPROVAL = grant({ id: 8, kind: 'approval', tool_name: 'composio_gmail_send', question_md: 'Send the launch email?' })

/** The API over a stubbed fetch: each "METHOD path" answers as scripted (200 {} otherwise); every route is kept. */
function stubApi(answers: Readonly<Record<string, Answer>>): string[] {
  const calls: string[] = []
  vi.stubGlobal('fetch', vi.fn(async (url: string, init: RequestInit = {}) => {
    const route = `${init.method ?? 'GET'} ${new URL(String(url), 'http://app.test').pathname}`
    calls.push(route)
    const answer = answers[route] ?? ok({})
    return new Response(JSON.stringify(answer.body), { status: answer.status, headers: { 'Content-Type': 'application/json' } })
  }))
  return calls
}

/** One surface under its own React Query client, no retries (a refusal shows at once). */
function renderSurface(node: ReactNode): void {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false }, mutations: { retry: false } } })
  render(<QueryClientProvider client={client}>{node}</QueryClientProvider>)
}

afterEach(() => {
  cleanup()
  vi.unstubAllGlobals()
})

describe('the Questions tab for editors and viewers (PRD-266 US-006b)', () => {
  it('an editor sees the QuestionCard and answers it', async () => {
    const calls = stubApi({ [GRANTS]: ok({ grants: [QUESTION] }) })
    renderSurface(<QuestionsTab />)

    expect(await screen.findByLabelText('Question')).toHaveTextContent('Ship A or B?')
    fireEvent.click(screen.getByRole('button', { name: 'B' }))
    await waitFor(() => expect(calls).toContain(ANSWER))
    expect(screen.queryByText(ADMIN_ONLY_COPY)).not.toBeInTheDocument()
  })

  it("a viewer's 403 says plainly who can answer", async () => {
    stubApi({ [GRANTS]: viewerRefused })
    renderSurface(<QuestionsTab />)

    expect(await screen.findByText(REFUSAL)).toBeInTheDocument()
    expect(screen.queryByText(ADMIN_ONLY_COPY)).not.toBeInTheDocument()
    expect(screen.queryByLabelText('Question')).not.toBeInTheDocument()
  })

  it('any other failure keeps the load error', async () => {
    stubApi({ [GRANTS]: serverFailed })
    renderSurface(<QuestionsTab />)

    expect(await screen.findByText('Could not load questions.')).toBeInTheDocument()
    expect(screen.queryByText(REFUSAL)).not.toBeInTheDocument()
  })
})

describe('the Governance approvals inbox for editors and viewers (PRD-266 US-006b)', () => {
  it('an editor sees the approval cards and grants one', async () => {
    const calls = stubApi({ [GRANTS]: ok({ grants: [APPROVAL] }), [MISSIONS]: ok({ missions: [] }) })
    renderSurface(<ApprovalsInbox />)

    expect(await screen.findByText('Send the launch email?')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /Deny/ })).toBeInTheDocument()
    fireEvent.click(screen.getByRole('button', { name: /Grant/ }))
    await waitFor(() => expect(calls).toContain(GRANT))
    expect(screen.queryByText(ADMIN_ONLY_COPY)).not.toBeInTheDocument()
  })

  it("a viewer's 403 says plainly who can answer", async () => {
    stubApi({ [GRANTS]: viewerRefused, [MISSIONS]: ok({ missions: [] }) })
    renderSurface(<ApprovalsInbox />)

    expect(await screen.findByText(REFUSAL)).toBeInTheDocument()
    expect(screen.queryByText(ADMIN_ONLY_COPY)).not.toBeInTheDocument()
    expect(screen.queryByRole('button', { name: /Grant/ })).not.toBeInTheDocument()
  })

  it('any other failure keeps the load error', async () => {
    stubApi({ [GRANTS]: serverFailed, [MISSIONS]: ok({ missions: [] }) })
    renderSurface(<ApprovalsInbox />)

    expect(await screen.findByText('Could not load approvals.')).toBeInTheDocument()
    expect(screen.queryByText(REFUSAL)).not.toBeInTheDocument()
  })
})
