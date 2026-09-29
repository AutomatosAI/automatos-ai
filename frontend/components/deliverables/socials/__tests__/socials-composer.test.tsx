/**
 * PRD-251 S2.2a (US-207) — the composer's first step, against a mocked apiClient:
 * "New post" opens it (the bare form stays as "Blank draft"); a brief and the
 * channel chips → "Draft it" → the proposal fills the composer → "Save draft"
 * creates the post and sets its targets.
 */
import { describe, it, expect, vi, afterEach, beforeEach } from 'vitest'
import { render, screen, cleanup, fireEvent, waitFor, within } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'

const api = vi.hoisted(() => ({
  listSocialPosts: vi.fn(),
  getSocialsUsage: vi.fn(),
  listSocialChannels: vi.fn(),
  composeSocialPost: vi.fn(),
  createSocialPost: vi.fn(),
  setSocialPostTargets: vi.fn(),
}))

vi.mock('sonner', () => ({ toast: { success: vi.fn(), error: vi.fn() } }))
vi.mock('@/lib/api-client', () => ({ apiClient: api, default: api }))
vi.mock('@/components/workspace-provider', () => ({
  useWorkspace: () => ({ workspace: { id: 'w1', role: 'owner', socials: { available: true, enabled: true } } }),
}))
vi.mock('@/components/documents/blocks/BrandKitDialog', () => ({ BrandKitDialog: () => null }))
vi.mock('@/hooks/use-mobile', () => ({ useIsMobile: () => false, useIsTabletOrBelow: () => false }))

import { SocialsPostList } from '@/components/deliverables/socials/socials-post-list'
import type { SocialChannel, SocialComposeProposal } from '@/lib/api-client'

const kind = (k: string, available = true) => ({ kind: k, available, reason: available ? null : 'no action', needs_public_storage: false })
const CHANNELS: SocialChannel[] = [
  { toolkit: 'twitter', label: 'X', post_kinds: [kind('text'), kind('image')] as any, verified: true, setup_note: null, copy_limits: { text: 280 } },
  { toolkit: 'linkedin', label: 'LinkedIn', post_kinds: [kind('text'), kind('image', false), kind('video')] as any, verified: true, setup_note: null, copy_limits: { text: 3000 } },
]
const PROPOSAL: SocialComposeProposal = {
  title: 'Harvest Club opens Friday',
  copy: { base: 'Harvest Club opens Friday.', per_channel: { twitter: 'Friday. Harvest Club.', linkedin: 'We open Harvest Club on Friday.' } },
  format: 'image',
  template_id: 'tpl-1',
  variables: { headline: { value: 'Opens Friday', claim: false }, members: { value: 1200, claim: true } },
  sources: { members: { kind: 'metric', ref: 'members', as_of: '2026-09-28T00:00:00+00:00' } },
  channels: ['twitter', 'linkedin'],
  warnings: ['linkedin: the copy was trimmed to 3000 characters at a word boundary'],
}

function renderList() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false }, mutations: { retry: false } } })
  return render(
    <QueryClientProvider client={client}>
      <SocialsPostList role="owner" />
    </QueryClientProvider>,
  )
}

beforeEach(() => {
  Object.values(api).forEach((fn) => fn.mockReset())
  api.listSocialPosts.mockResolvedValue({ posts: [], total: 0 })
  api.getSocialsUsage.mockResolvedValue({
    render_minutes: {
      used_minutes: 0, used_seconds: 0, quota_minutes: null, remaining_minutes: null, exhausted: false,
      period_start: '2026-09-01T00:00:00Z', period_end: '2026-10-01T00:00:00Z',
    },
  })
  api.listSocialChannels.mockResolvedValue(CHANNELS)
  api.composeSocialPost.mockResolvedValue(PROPOSAL)
  api.createSocialPost.mockImplementation(async (input: any) => ({ id: 'post-9', ...input, status: 'draft' }))
  api.setSocialPostTargets.mockImplementation(async (id: string, targets: any) => ({ id, targets, status: 'draft' }))
})
afterEach(() => cleanup())

async function openComposer() {
  renderList()
  fireEvent.click(await screen.findByRole('button', { name: /New post/ }))
  return screen.getByRole('region', { name: 'Composer' })
}

describe('the composer: a brief becomes a draft', () => {
  it('"New post" opens the composer; "Blank draft" is still the bare form', async () => {
    renderList()
    fireEvent.click(await screen.findByRole('button', { name: /Blank draft/ }))
    expect(screen.getByRole('form', { name: 'New draft' })).toBeInTheDocument()
    expect(screen.queryByRole('region', { name: 'Composer' })).toBeNull()
  })

  it('brief → Draft it → the proposal fills the composer → Save draft creates the post and sets its targets', async () => {
    const composer = await openComposer()
    const chips = await within(composer).findByRole('group', { name: 'Channels' })
    expect(within(chips).getByRole('button', { name: 'X' })).toHaveAttribute('aria-pressed', 'true')
    expect(within(chips).getByRole('button', { name: 'LinkedIn' })).toHaveAttribute('aria-pressed', 'true')

    const draftIt = within(composer).getByRole('button', { name: /Draft it/ })
    expect(draftIt).toBeDisabled()
    fireEvent.change(within(composer).getByLabelText('Brief'), { target: { value: 'Announce Harvest Club' } })
    fireEvent.click(draftIt)
    await waitFor(() =>
      expect(api.composeSocialPost).toHaveBeenCalledWith({ brief: 'Announce Harvest Club', channels: ['twitter', 'linkedin'] }),
    )

    expect(await within(composer).findByDisplayValue('Harvest Club opens Friday')).toBeInTheDocument()
    expect(within(composer).getByLabelText('X copy')).toHaveValue('Friday. Harvest Club.')
    expect(within(composer).getByLabelText('LinkedIn copy')).toHaveValue('We open Harvest Club on Friday.')
    expect(within(composer).getByRole('list', { name: 'Warnings' })).toHaveTextContent('trimmed to 3000 characters')
    expect(api.createSocialPost).not.toHaveBeenCalled() // a proposal is not saved

    fireEvent.change(within(composer).getByLabelText('X copy'), { target: { value: 'Friday!' } })
    fireEvent.click(within(composer).getByRole('button', { name: 'Save draft' }))

    await waitFor(() => expect(api.setSocialPostTargets).toHaveBeenCalled())
    expect(api.createSocialPost).toHaveBeenCalledWith({
      title: 'Harvest Club opens Friday',
      brief: 'Announce Harvest Club',
      copy: { base: 'Harvest Club opens Friday.', channels: { twitter: 'Friday!', linkedin: 'We open Harvest Club on Friday.' } },
      format: 'image',
      template_id: 'tpl-1',
      variables: PROPOSAL.variables,
      sources: PROPOSAL.sources,
    })
    // Each channel posts the image as an image where it can, else its first available kind.
    expect(api.setSocialPostTargets).toHaveBeenCalledWith('post-9', [
      { toolkit: 'twitter', post_kind: 'image' },
      { toolkit: 'linkedin', post_kind: 'text' },
    ])
    await waitFor(() => expect(screen.queryByRole('region', { name: 'Composer' })).toBeNull())
  })

  it('a channel chip left off is not written for', async () => {
    const composer = await openComposer()
    fireEvent.click(await within(composer).findByRole('button', { name: 'LinkedIn' }))
    fireEvent.change(within(composer).getByLabelText('Brief'), { target: { value: 'Just X' } })
    fireEvent.click(within(composer).getByRole('button', { name: /Draft it/ }))
    await waitFor(() => expect(api.composeSocialPost).toHaveBeenCalledWith({ brief: 'Just X', channels: ['twitter'] }))
  })

  it('back to the brief keeps it', async () => {
    const composer = await openComposer()
    await within(composer).findByRole('group', { name: 'Channels' })
    fireEvent.change(within(composer).getByLabelText('Brief'), { target: { value: 'Announce Harvest Club' } })
    fireEvent.click(within(composer).getByRole('button', { name: /Draft it/ }))
    fireEvent.click(await within(composer).findByRole('button', { name: 'Back to the brief' }))
    expect(within(composer).getByLabelText('Brief')).toHaveValue('Announce Harvest Club')
  })
})
