/**
 * PRD-251C US-C408 — Posted: what went out, newest first, with its receipts, numbers and topic,
 * filtered by plan (the URL's &plan=), channel and format.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'

const go = vi.fn()

vi.mock('@/components/workspace-provider', () => ({
  useWorkspace: () => ({ workspace: { id: 'w1', role: 'owner', socials: { available: true, enabled: true } } }),
}))
vi.mock('@/lib/api-client', () => {
  const kind = (k: string) => ({ kind: k, available: true, reason: null, needs_public_storage: false })
  const apiClient = {
    listSocialPosted: vi.fn(async () => ({
      total: 2,
      posts: [
        {
          id: 'p1', title: 'Three weeks to Lisbon', format: 'image', plan_id: 'plan-1', plan_name: 'Countdown', topic: 'The stand',
          went_out_at: '2026-10-23T08:00:00Z',
          receipts: [
            { toolkit: 'twitter', post_kind: 'image', permalink: 'https://x.com/i/web/status/1850', remote_id: '1850', published_at: '2026-10-23T08:00:00Z' },
            { toolkit: 'instagram', post_kind: 'story', permalink: null, remote_id: '1790', published_at: '2026-10-23T08:01:00Z' },
          ],
          numbers: { numbers: { views: 940, likes: 12, reposts: 1 }, by_channel: {}, engagement: 13, reading: 7, read_at: '2026-10-30T09:00:00Z' },
        },
        { id: 'p2', title: 'Stand map', format: 'video', plan_id: null, plan_name: null, topic: 'Where we are', went_out_at: '2026-10-24T08:00:00Z',
          receipts: [], numbers: null },
      ],
    })),
    listSocialPlans: vi.fn(async () => ({ plans: [{ id: 'plan-1', name: 'Countdown' }], total: 1 })),
    listSocialChannels: vi.fn(async () => [{ toolkit: 'twitter', label: 'X', post_kinds: [kind('image')], verified: true, setup_note: null }]),
  }
  return { apiClient, default: apiClient }
})

import { apiClient } from '@/lib/api-client'
import { SocialsPosted } from '@/components/deliverables/socials/studio/socials-posted'
import { NO_NUMBERS, numbersLine } from '@/components/deliverables/socials/studio/posted-model'

const api = apiClient as unknown as Record<string, ReturnType<typeof vi.fn>>

function renderPosted(planId: string | null = null) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  return render(<QueryClientProvider client={client}><SocialsPosted planId={planId} go={go} /></QueryClientProvider>)
}

beforeEach(() => { go.mockReset(); Object.values(api).forEach((fn) => fn.mockClear()) })
afterEach(cleanup)

describe('Posted', () => {
  it('lists what went out with its receipts, numbers and topic', async () => {
    renderPosted()
    const first = await screen.findByRole('listitem', { name: 'Three weeks to Lisbon' })
    expect(first).toHaveTextContent('Countdown · The stand · image')
    expect(first).toHaveTextContent('940 views · 12 likes · 1 repost (after a week)')
    expect(within(first).getByRole('link', { name: 'X' })).toHaveAttribute('href', 'https://x.com/i/web/status/1850')
    expect(first).toHaveTextContent('Instagram story')  // no link given: named only
    expect(screen.getByRole('listitem', { name: 'Stand map' })).toHaveTextContent(NO_NUMBERS)
    fireEvent.click(within(first).getByRole('button', { name: 'Three weeks to Lisbon' }))
    expect(go).toHaveBeenCalledWith({ view: 'calendar', post: 'p1', plan: null })
  })

  it('filters by plan through the URL, and by channel and format', async () => {
    renderPosted('plan-1')
    await screen.findByRole('listitem', { name: 'Three weeks to Lisbon' })
    expect(api.listSocialPosted).toHaveBeenLastCalledWith({ planId: 'plan-1', channel: null, format: null })
    fireEvent.change(screen.getByLabelText('Channel'), { target: { value: 'twitter' } })
    await waitFor(() => expect(api.listSocialPosted).toHaveBeenLastCalledWith({ planId: 'plan-1', channel: 'twitter', format: null }))
    fireEvent.change(screen.getByLabelText('Format'), { target: { value: 'video' } })
    await waitFor(() => expect(api.listSocialPosted).toHaveBeenLastCalledWith({ planId: 'plan-1', channel: 'twitter', format: 'video' }))
    fireEvent.change(await screen.findByLabelText('Plan'), { target: { value: '' } })
    expect(go).toHaveBeenCalledWith({ plan: null })
  })

  it('says when the channels gave no numbers, and after which reading', () => {
    expect(numbersLine({ numbers: {}, by_channel: {}, engagement: 0, reading: 1, read_at: null })).toBe('The channels gave no numbers (after a day).')
  })
})
