/**
 * PRD-251 S2.4 (US-210, D6) — the Campaigns view and the "Approve series" confirmation.
 *
 * * The Studio's Plans view (PRD-251B US-B107; B6, a plan is a campaign): the
 *   workspace's campaigns, and the selected one with its posts and their statuses.
 * * "Approve series" is offered only when the workspace's series approval is on and
 *   the campaign approves as a series; otherwise it says why.
 * * The confirmation lists the posts it will approve (those waiting for approval)
 *   and those it will not; confirming sends each post with the content_hash on
 *   screen, then shows what the server approved and what it left, with why. A post
 *   left for unsourced claims needs a second confirmation naming them, which sends
 *   only that post with override_unsourced.
 * * An owner turns series approval on through apiClient; a viewer sees no switch
 *   and no approve button. The api client is mocked: nothing is fetched raw.
 */
import { describe, it, expect, vi, afterEach, beforeEach } from 'vitest'
import { render, screen, cleanup, fireEvent, within, waitFor } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'

const state = vi.hoisted(() => ({
  posts: [] as any[],
  campaign: null as any,
  workspace: {
    id: 'w1', role: 'editor', socials: { available: true, enabled: true, series_approval: true },
  } as any,
}))

vi.mock('next/navigation', () => ({
  usePathname: () => '/deliverables',
  // PRD-251B US-B107: the Studio holds its view in the URL (router.push).
  useRouter: () => ({ push: vi.fn(), replace: vi.fn() }),
  useSearchParams: () => new URLSearchParams(''),
}))
vi.mock('@/hooks/use-mobile', () => ({ useIsMobile: () => false, useIsTabletOrBelow: () => false }))
vi.mock('@/components/workspace-provider', () => ({
  useWorkspace: () => ({ workspace: state.workspace, isLoading: false, refreshWorkspace: vi.fn(async () => {}) }),
}))
vi.mock('sonner', () => ({ toast: { success: vi.fn(), error: vi.fn() } }))
vi.mock('@/lib/api-client', () => {
  const apiClient = {
    listSocialPosts: vi.fn(async () => ({ posts: state.posts, total: state.posts.length })),
    getSocialsUsage: vi.fn(async () => ({ render_minutes: null })),
    listSocialCampaigns: vi.fn(async () => ({
      campaigns: state.campaign ? [{ ...state.campaign, post_count: state.campaign.posts.length }] : [],
      total: state.campaign ? 1 : 0,
    })),
    getSocialCampaign: vi.fn(async () => state.campaign),
    approveSocialCampaignSeries: vi.fn(),
    addSocialCampaignPost: vi.fn(async () => ({})),
    removeSocialCampaignPost: vi.fn(async () => ({})),
    updateSocialCampaign: vi.fn(async () => state.campaign),
    createSocialCampaign: vi.fn(),
    setWorkspaceSeriesApproval: vi.fn(async () => ({ status: 'saved' })),
  }
  return { apiClient, default: apiClient }
})

import { apiClient } from '@/lib/api-client'
import { SocialsTab } from '@/components/deliverables/socials/socials-tab'

function seed(title: string, status: string, hash: string, extra: Record<string, unknown> = {}) {
  const at = new Date(Date.UTC(2026, 9, 1, 9, 0)).toISOString()
  return {
    id: `post-${title.toLowerCase().replace(/\W+/g, '-')}`, workspace_id: 'w1', created_by: 'user-1', title,
    brief: null, copy: { base: `${title} copy` }, format: null, template_id: null, variables: {}, sources: {},
    media: {}, status, content_hash: hash, approved_hash: null, approved_by: null, approved_at: null,
    override_unsourced: false, review_log: [], scheduled_for: null, timezone: null, created_at: at, updated_at: at,
    ...extra,
  }
}

const THREE_WEEKS = seed('Three weeks to go', 'needs_approval', 'a'.repeat(64))
const TWO_WEEKS = seed('Two weeks to go', 'needs_approval', 'b'.repeat(64), {
  variables: { users: { value: 1200, claim: true } },
})
const LAUNCH = seed('Launch day', 'draft', 'c'.repeat(64))
const LOOSE = seed('Not in a campaign', 'draft', 'd'.repeat(64))

function campaign(mode = 'series') {
  return {
    id: 'camp-1', workspace_id: 'w1', name: 'Web Summit countdown', approval_mode: mode, approved_hash_set: [],
    approved_by: null, approved_at: null, created_by: 'user-1', created_at: THREE_WEEKS.created_at,
    updated_at: THREE_WEEKS.created_at, posts: [THREE_WEEKS, TWO_WEEKS, LAUNCH],
  }
}

function renderTab() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false }, mutations: { retry: false } } })
  return render(
    <QueryClientProvider client={client}>
      <SocialsTab />
    </QueryClientProvider>,
  )
}

async function openCampaigns() {
  renderTab()
  fireEvent.click(await screen.findByRole('button', { name: 'Plans' }))
  return screen.findByRole('region', { name: 'Campaign Web Summit countdown' })
}

beforeEach(() => {
  state.posts = [THREE_WEEKS, TWO_WEEKS, LAUNCH, LOOSE]
  state.campaign = campaign()
  state.workspace = { id: 'w1', role: 'editor', socials: { available: true, enabled: true, series_approval: true } }
  vi.mocked(apiClient.approveSocialCampaignSeries).mockReset()
  vi.mocked(apiClient.setWorkspaceSeriesApproval).mockClear()
  vi.mocked(apiClient.addSocialCampaignPost).mockClear()
})
afterEach(cleanup)

describe('the Campaigns view', () => {
  it('is the Plans view: the campaigns, and the selected one with its posts and their statuses', async () => {
    const detail = await openCampaigns()
    expect(screen.getByRole('button', { name: 'Plans' })).toHaveAttribute('aria-current', 'page')
    const list = screen.getByRole('list', { name: 'Campaigns' })
    expect(within(list).getByRole('button', { name: /Web Summit countdown/ })).toHaveTextContent('As a series · 3 posts')

    const posts = within(detail).getByRole('list', { name: 'Campaign posts' })
    const rows = within(posts).getAllByRole('listitem').map((row) => row.textContent)
    expect(rows).toEqual([
      expect.stringContaining('Three weeks to goNeeds approval'),
      expect.stringContaining('Two weeks to goNeeds approval'),
      expect.stringContaining('Launch dayDraft'),
    ])
    expect(within(detail).getByRole('combobox', { name: 'Approval mode' })).toHaveValue('series')
  })

  it('adds a post of the workspace to the campaign through apiClient', async () => {
    const detail = await openCampaigns()
    const picker = within(detail).getByRole('combobox', { name: 'Post to add' })
    expect(within(picker).getAllByRole('option').map((o) => o.textContent)).toEqual(['Choose a post', 'Not in a campaign'])
    fireEvent.change(picker, { target: { value: LOOSE.id } })
    fireEvent.click(within(detail).getByRole('button', { name: 'Add to campaign' }))
    await waitFor(() => expect(apiClient.addSocialCampaignPost).toHaveBeenCalledWith('camp-1', LOOSE.id))
  })

  it('an owner turns series approval on; a viewer sees no switch and no Approve series', async () => {
    state.workspace = { id: 'w1', role: 'owner', socials: { available: true, enabled: true, series_approval: false } }
    await openCampaigns()
    fireEvent.click(screen.getByRole('switch'))
    await waitFor(() => expect(apiClient.setWorkspaceSeriesApproval).toHaveBeenCalledWith(true))
    cleanup()

    state.workspace = { id: 'w1', role: 'viewer', socials: { available: true, enabled: true, series_approval: true } }
    await openCampaigns()
    expect(screen.queryByRole('switch')).toBeNull()
    expect(screen.getByText('Series approval is on for this workspace.')).toBeInTheDocument()
    expect(screen.queryByRole('button', { name: 'Approve series' })).toBeNull()
  })
})

describe('Approve series', () => {
  it('is not offered with the workspace switch off, or for a per-post campaign, and says why', async () => {
    state.workspace = { id: 'w1', role: 'editor', socials: { available: true, enabled: true, series_approval: false } }
    let detail = await openCampaigns()
    expect(within(detail).getByRole('button', { name: 'Approve series' })).toBeDisabled()
    expect(within(detail).getByText(/Series approval is off for this workspace/)).toBeInTheDocument()
    cleanup()

    state.workspace = { id: 'w1', role: 'editor', socials: { available: true, enabled: true, series_approval: true } }
    state.campaign = campaign('per_post')
    detail = await openCampaigns()
    expect(within(detail).getByRole('button', { name: 'Approve series' })).toBeDisabled()
    expect(within(detail).getByText(/approves post by post/)).toBeInTheDocument()
  })

  it('confirms the posts it will approve and those it will not, then sends each hash on screen', async () => {
    vi.mocked(apiClient.approveSocialCampaignSeries).mockResolvedValue({
      campaign: { ...campaign(), approved_hash_set: [THREE_WEEKS.content_hash] },
      approved: [{ ...THREE_WEEKS, status: 'approved' }],
      left: [
        {
          post_id: TWO_WEEKS.id, title: TWO_WEEKS.title, status: 'needs_approval', reason: 'unsourced',
          message: 'these claims have no source: users', claims: ['users'],
        },
      ],
    } as any)
    const detail = await openCampaigns()
    fireEvent.click(within(detail).getByRole('button', { name: 'Approve series' }))

    const confirm = screen.getByRole('alertdialog', { name: 'Approve series' })
    const will = within(confirm).getByRole('list', { name: 'Will be approved' })
    expect(within(will).getAllByRole('listitem').map((li) => li.textContent)).toEqual(['Three weeks to go', 'Two weeks to go'])
    const notCovered = within(confirm).getByRole('list', { name: 'Not covered' })
    expect(within(notCovered).getAllByRole('listitem').map((li) => li.textContent)).toEqual(['Launch day · Draft'])
    expect(apiClient.approveSocialCampaignSeries).not.toHaveBeenCalled()

    fireEvent.click(within(confirm).getByRole('button', { name: 'Approve 2 posts' }))

    await waitFor(() => expect(apiClient.approveSocialCampaignSeries).toHaveBeenCalledTimes(1))
    expect(apiClient.approveSocialCampaignSeries).toHaveBeenCalledWith('camp-1', [
      { post_id: THREE_WEEKS.id, content_hash: THREE_WEEKS.content_hash, override_unsourced: false },
      { post_id: TWO_WEEKS.id, content_hash: TWO_WEEKS.content_hash, override_unsourced: false },
    ])
    const result = await screen.findByRole('status', { name: 'Series approval result' })
    expect(result).toHaveTextContent('1 post approved.')
    const left = within(result).getByRole('list', { name: 'Left unapproved' })
    expect(left).toHaveTextContent('Two weeks to go — Unsourced claims to confirm')

    // The second confirmation names the claims, and sends only that post, overridden.
    const second = within(result).getByRole('alertdialog', { name: 'Approve with unsourced claims' })
    expect(second).toHaveTextContent('Two weeks to go (users)')
    vi.mocked(apiClient.approveSocialCampaignSeries).mockResolvedValue({
      campaign: campaign(), approved: [{ ...TWO_WEEKS, status: 'approved' }], left: [],
    } as any)
    fireEvent.click(within(second).getByRole('button', { name: 'Approve with unsourced claims' }))
    await waitFor(() => expect(apiClient.approveSocialCampaignSeries).toHaveBeenCalledTimes(2))
    expect(vi.mocked(apiClient.approveSocialCampaignSeries).mock.calls[1]).toEqual([
      'camp-1', [{ post_id: TWO_WEEKS.id, content_hash: TWO_WEEKS.content_hash, override_unsourced: true }],
    ])
  })

  it('reports a post that changed between display and approval, and cancel sends nothing', async () => {
    vi.mocked(apiClient.approveSocialCampaignSeries).mockResolvedValue({
      campaign: campaign(), approved: [],
      left: [
        {
          post_id: THREE_WEEKS.id, title: THREE_WEEKS.title, status: 'needs_approval', reason: 'changed',
          message: 'the post changed after you were shown it: review the current version', content_hash: 'e'.repeat(64),
        },
      ],
    } as any)
    const detail = await openCampaigns()
    fireEvent.click(within(detail).getByRole('button', { name: 'Approve series' }))
    fireEvent.click(screen.getByRole('button', { name: 'Cancel' }))
    expect(screen.queryByRole('alertdialog')).toBeNull()
    expect(apiClient.approveSocialCampaignSeries).not.toHaveBeenCalled()

    fireEvent.click(within(detail).getByRole('button', { name: 'Approve series' }))
    fireEvent.click(screen.getByRole('button', { name: 'Approve 2 posts' }))
    const result = await screen.findByRole('status', { name: 'Series approval result' })
    expect(result).toHaveTextContent('0 posts approved.')
    expect(result).toHaveTextContent('Three weeks to go — Changed since you saw it')
    expect(within(result).queryByRole('alertdialog')).toBeNull()
  })
})
