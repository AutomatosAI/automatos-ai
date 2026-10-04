/**
 * F242 (night 7b): the Run button gets a "Wait for me" switch.
 *
 * The owner ran playbook 102 "New Cafe Onboarding" from Run, and #0185 went
 * straight to Done (review_mode auto): Run had no way to ask for a check, though
 * the timer's card on a playbook set to wait stopped in Review (#0195). In the
 * Studio hub the library's Run (and a link to a playbook) now opens a dialog
 * whose switch starts at the playbook's own setting and is sent with the run.
 * These tests run the real hooks and the real apiClient against a fake backend,
 * and check what reached it.
 */
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { render, screen, cleanup, fireEvent, waitFor } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import React from 'react'

const WORDS = 'Playbook 103 "New Cafe Onboarding" can\'t run: steps 1 and 2 have no agent. Give each of them an agent, then run it again.'

const nav = vi.hoisted(() => ({ search: 'tab=playbooks', push: vi.fn(), replace: vi.fn() }))
const server = vi.hoisted(() => ({
  playbook: null as Record<string, unknown> | null,
  refusal: null as string | null,
  runs: [] as Array<{ address: string; body: unknown }>,
}))
const toasts = vi.hoisted(() => ({ shown: vi.fn(), error: vi.fn() }))

vi.mock('next/navigation', () => ({
  useRouter: () => ({ push: nav.push, replace: nav.replace }),
  usePathname: () => '/assignments',
  useSearchParams: () => new URLSearchParams(nav.search),
}))
vi.mock('sonner', () => ({ toast: Object.assign(toasts.shown, { error: toasts.error, success: vi.fn() }) }))
// The hub's other parts are their own suites' business (hub-compact.test.tsx);
// its create modals load through next/dynamic and stay closed here.
vi.mock('next/dynamic', () => ({ default: () => () => null }))
vi.mock('@/components/assignments/studio/playbooks-body', () => ({ PlaybooksBody: () => null }))
vi.mock('@/components/assignments/studio/missions-body', () => ({ MissionsBody: () => null }))
vi.mock('@/components/assignments/studio/entry-grid', () => ({ EntryGrid: () => null }))
vi.mock('@/hooks/use-missions-api', () => ({ useMissions: () => ({ data: { missions: [] } }) }))

import { RunPlaybookDialog } from '../studio/run-playbook-dialog'
import { StudioAssignmentsHub } from '../studio/assignments-hub'

function answer(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), { status, headers: { 'Content-Type': 'application/json' } })
}

/** The backend's playbook routes: read one, list them, run one. */
async function backend(url: string, init?: RequestInit): Promise<Response> {
  const path = new URL(url, 'http://app.test').pathname
  const ran = path.match(/^\/api\/workflow-recipes\/([^/]+)\/execute$/)
  if (ran && init?.method === 'POST') {
    if (server.refusal) return answer({ detail: server.refusal }, 400)
    server.runs.push({ address: decodeURIComponent(ran[1]), body: JSON.parse(String(init.body)) })
    return answer({ recipe_execution_id: 'exec-98c6b4377f74', status: 'started' })
  }
  if (path === '/api/workflow-recipes') return answer({ items: [], total: 0 })
  if (/^\/api\/workflow-recipes\/[^/]+$/.test(path)) {
    return server.playbook ? answer(server.playbook) : answer({ detail: 'not found' }, 404)
  }
  throw new Error(`unexpected ${init?.method ?? 'GET'} ${path}`)
}

function onboarding(executionConfig: Record<string, unknown> = {}) {
  return { id: 102, template_id: 'custom-aa07d786', name: 'New Cafe Onboarding', description: 'Welcome a new café.',
    execution_config: executionConfig, steps: [{}, {}] }
}

function withQueries(ui: React.ReactElement) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false }, mutations: { retry: false } } })
  return render(<QueryClientProvider client={client}>{ui}</QueryClientProvider>)
}

async function loadedSwitch() {
  const toggle = await screen.findByRole('switch', { name: /Wait for me/ })
  await waitFor(() => expect(toggle).not.toBeDisabled())
  return toggle
}

beforeEach(() => {
  Object.assign(server, { playbook: onboarding(), refusal: null, runs: [] })
  Object.assign(nav, { search: 'tab=playbooks' })
  nav.push.mockReset()
  nav.replace.mockReset()
  toasts.shown.mockReset()
  toasts.error.mockReset()
  Element.prototype.scrollIntoView = vi.fn()
  vi.stubGlobal('fetch', vi.fn(backend))
})
afterEach(() => {
  cleanup()
  vi.unstubAllGlobals()
})

describe('the Run dialog’s "Wait for me"', () => {
  it("starts at the playbook's own setting, the one its timer's runs follow", async () => {
    server.playbook = onboarding({ wait_for_me: true })
    withQueries(<RunPlaybookDialog address="custom-aa07d786" onClose={vi.fn()} />)

    expect(await loadedSwitch()).toHaveAttribute('aria-checked', 'true')
  })

  it('switched on, the run is sent asking to wait, and its page opens', async () => {
    withQueries(<RunPlaybookDialog address="102" onClose={vi.fn()} />)
    const toggle = await loadedSwitch()
    expect(toggle).toHaveAttribute('aria-checked', 'false')                // 102 has no setting of its own

    fireEvent.click(toggle)
    fireEvent.click(screen.getByRole('button', { name: 'Run now' }))

    await waitFor(() => expect(server.runs).toEqual([{ address: '102', body: { input_data: {}, wait_for_me: true } }]))
    expect(nav.push).toHaveBeenCalledWith('/activity/execution?id=exec-98c6b4377f74&recipeId=102')
  })

  it('switched off, the run is sent closing itself though its playbook waits', async () => {
    server.playbook = onboarding({ wait_for_me: true })
    withQueries(<RunPlaybookDialog address="custom-aa07d786" onClose={vi.fn()} />)

    fireEvent.click(await loadedSwitch())
    fireEvent.click(screen.getByRole('button', { name: 'Run now' }))

    await waitFor(() => expect(server.runs.map((run) => run.body)).toEqual([{ input_data: {}, wait_for_me: false }]))
  })

  it('a refusal is shown in the words it came with, and the dialog stays open (F270)', async () => {
    server.refusal = WORDS
    const onClose = vi.fn()
    withQueries(<RunPlaybookDialog address="103" onClose={onClose} />)
    await loadedSwitch()

    fireEvent.click(screen.getByRole('button', { name: 'Run now' }))

    await waitFor(() => expect(toasts.error).toHaveBeenCalledWith('The playbook did not start', { description: WORDS }))
    expect(server.runs).toEqual([])
    expect(onClose).not.toHaveBeenCalled()
    expect(nav.push).not.toHaveBeenCalled()
  })
})

describe('the hub opens it for ?tab=playbooks&id=', () => {
  it('opens the dialog for the playbook at the address, by its number too (F277)', async () => {
    nav.search = 'tab=playbooks&id=102'
    withQueries(<StudioAssignmentsHub />)

    expect(await screen.findByRole('dialog', { name: 'Run New Cafe Onboarding' })).toBeInTheDocument()
  })

  it('Cancel closes it and drops the id from the address', async () => {
    nav.search = 'tab=playbooks&id=custom-aa07d786'
    withQueries(<StudioAssignmentsHub />)
    await screen.findByRole('dialog', { name: 'Run New Cafe Onboarding' })

    fireEvent.click(screen.getByRole('button', { name: 'Cancel' }))

    expect(nav.replace).toHaveBeenCalledWith('/assignments?tab=playbooks', { scroll: false })
  })

  it("a mission's id never opens it", () => {
    nav.search = 'tab=missions&id=m-1111-2222'
    withQueries(<StudioAssignmentsHub />)

    expect(screen.queryByRole('dialog')).toBeNull()
  })
})
