/**
 * F372 (night 10c) — a Brand kit page left open never saves over a change made elsewhere, against
 * a mocked apiClient whose kit carries its stamp (`updated_at`) and whose PUT answers 409 when the
 * page's `if_updated_at` is no longer the stored one (as api/document_brand_kit.py does):
 *
 * * no edits on the page: a change elsewhere reloads the fields quietly;
 * * a save from a stale page is refused and shows the notice; "Keep my edits and save over it"
 *   saves without the stamp;
 * * edits held when a change elsewhere is seen: the notice, and Reload drops the edits.
 */
import { describe, it, expect, vi, afterEach, beforeEach } from 'vitest'
import { render, screen, cleanup, fireEvent, waitFor } from '@testing-library/react'
import { QueryClient, QueryClientProvider, focusManager } from '@tanstack/react-query'
import { useState } from 'react'

import { designKit } from './brand-kit-design-kit'

const LOADED = '2026-10-06T18:00:00+00:00'
const ELSEWHERE = '2026-10-06T18:05:00+00:00'
const server = vi.hoisted(() => ({ kit: null as any, saves: 0 }))

vi.mock('sonner', () => ({ toast: { success: vi.fn(), error: vi.fn() } }))
vi.mock('@/lib/api-client', () => {
  const apiClient = {
    get: vi.fn(async (path: string) => {
      if (path === '/api/documents/brand-kit') return server.kit
      if (path === '/api/documents/brand-kit/suggestions') return { suggestions: {} }
      throw new Error(`unexpected GET ${path}`)
    }),
    put: vi.fn(async (path: string, body: any) => {
      if (path !== '/api/documents/brand-kit') throw new Error(`unexpected PUT ${path}`)
      const { if_updated_at: expected, ...changes } = body
      if (expected !== undefined && expected !== server.kit.updated_at) {
        throw Object.assign(new Error('The brand kit changed since this page loaded; reload to see it'), { status: 409 })
      }
      server.saves += 1
      server.kit = { ...server.kit, ...changes, updated_at: `2026-10-06T19:00:0${server.saves}+00:00` }
      return server.kit
    }),
    post: vi.fn(),
    delete: vi.fn(),
    getAuthHeaders: vi.fn(async () => ({})),
    getBaseUrl: vi.fn(() => ''),
  }
  return { apiClient, default: apiClient }
})

import { apiClient } from '@/lib/api-client'
import { CHANGED_ELSEWHERE, KitChangedElsewhere } from '../brand-kit-changed-elsewhere'
import { useBrandKitForm } from '../use-brand-kit-form'

const api = apiClient as unknown as Record<string, ReturnType<typeof vi.fn>>

function Page() {
  const form = useBrandKitForm()
  return (
    <>
      <KitChangedElsewhere form={form} canEdit />
      {form.kit && (
        <input aria-label="Brand name" value={form.kit.name} onChange={(e) => form.patch({ name: e.target.value })} />
      )}
      <button type="button" onClick={() => void form.save()}>Save</button>
    </>
  )
}

function Harness() {
  const [client] = useState(() => new QueryClient({ defaultOptions: { queries: { retry: false } } }))
  return <QueryClientProvider client={client}><Page /></QueryClientProvider>
}

/** Auto, the designer or the API changed the kit: the server stamped it. */
function changeElsewhere() {
  server.kit = { ...server.kit, name: 'Changed by Auto', updated_at: ELSEWHERE }
}

/** The window regains focus: the page reads the kit's stamp again. */
function refocus() {
  focusManager.setFocused(false)
  focusManager.setFocused(true)
}

async function nameField() {
  const field = await screen.findByLabelText('Brand name')
  await waitFor(() => expect(field).toHaveValue('Harbourline'))
  return field
}

beforeEach(() => {
  server.kit = designKit({ updated_at: LOADED })
  server.saves = 0
  Object.values(api).forEach((fn) => fn.mockClear())
})
afterEach(() => { cleanup(); focusManager.setFocused(undefined) })

describe('a Brand kit page left open (F372)', () => {
  it('reloads the fields quietly when the kit changes elsewhere and the page holds no edits', async () => {
    render(<Harness />)
    const field = await nameField()
    changeElsewhere()
    refocus()
    await waitFor(() => expect(field).toHaveValue('Changed by Auto'))
    expect(screen.queryByRole('alert')).toBeNull()
  })

  it('refuses a stale save, shows the notice, and saves over the change only when asked', async () => {
    render(<Harness />)
    const field = await nameField()
    fireEvent.change(field, { target: { value: 'My edit' } })
    changeElsewhere()
    fireEvent.click(screen.getByRole('button', { name: 'Save' }))

    expect(await screen.findByRole('alert')).toHaveTextContent(CHANGED_ELSEWHERE)
    expect(api.put.mock.calls[0][1].if_updated_at).toBe(LOADED)
    expect(server.kit.name).toBe('Changed by Auto')  // nothing saved over it

    fireEvent.click(screen.getByRole('button', { name: 'Keep my edits and save over it' }))
    await waitFor(() => expect(server.kit.name).toBe('My edit'))
    expect(api.put.mock.calls[1][1]).not.toHaveProperty('if_updated_at')
    await waitFor(() => expect(screen.queryByRole('alert')).toBeNull())
  })

  it('shows the notice when the kit changes elsewhere while the page holds edits, and Reload drops them', async () => {
    render(<Harness />)
    const field = await nameField()
    fireEvent.change(field, { target: { value: 'My edit' } })
    changeElsewhere()
    refocus()
    expect(await screen.findByRole('alert')).toHaveTextContent(CHANGED_ELSEWHERE)
    expect(field).toHaveValue('My edit')

    fireEvent.click(screen.getByRole('button', { name: 'Reload' }))
    await waitFor(() => expect(screen.getByLabelText('Brand name')).toHaveValue('Changed by Auto'))
    expect(screen.queryByRole('alert')).toBeNull()
  })
})
