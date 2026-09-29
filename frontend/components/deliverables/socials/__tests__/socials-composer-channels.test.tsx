/**
 * PRD-251 S2.2c (US-209) — formats and channels, against a mocked apiClient that
 * keeps the post: only available post kinds can be chosen and unavailable ones say
 * why; each channel's count turns red over its limit and blocks submit; saving
 * writes the targets; and the whole flow — brief → preview → channels → submit —
 * ends in needs_approval.
 */
import { describe, it, expect, vi, afterEach, beforeEach } from 'vitest'
import { render, screen, cleanup, fireEvent, waitFor, within } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'

const server = vi.hoisted(() => ({ post: null as any }))
const api = vi.hoisted(() => ({
  listSocialChannels: vi.fn(),
  composeSocialPost: vi.fn(),
  createSocialPost: vi.fn(),
  updateSocialPost: vi.fn(),
  setSocialPostTargets: vi.fn(),
  renderSocialPost: vi.fn(),
  getSocialPost: vi.fn(),
  submitSocialPost: vi.fn(),
  searchSocialSources: vi.fn(),
  getSocialPostMedia: vi.fn(),
}))

vi.mock('sonner', () => ({ toast: { success: vi.fn(), error: vi.fn() } }))
vi.mock('@/lib/api-client', () => ({ apiClient: api, default: api }))
vi.mock('@/components/workspace-provider', () => ({
  useWorkspace: () => ({ workspace: { id: 'w1', role: 'owner', socials: { available: true, enabled: true } } }),
}))
vi.mock('@/components/widgets/FileWidget/FilePreview', async (importOriginal) => {
  const actual = await importOriginal<typeof import('@/components/widgets/FileWidget/FilePreview')>()
  return { ...actual, FilePreview: (props: { url?: string }) => <div data-testid="file-preview">{props.url}</div> }
})

import { SocialsComposer } from '@/components/deliverables/socials/socials-composer'
import type { SocialChannel, SocialComposeProposal } from '@/lib/api-client'

const kind = (k: string, available = true, reason: string | null = null) => ({ kind: k, available, reason, needs_public_storage: false })
const CHANNELS: SocialChannel[] = [
  { toolkit: 'twitter', label: 'X', verified: true, copy_limits: { text: 280 },
    setup_note: 'X needs your own X app: connect it with your API keys.',
    post_kinds: [kind('text'), kind('image'), kind('video', false, 'needs public storage for media')] as any },
  { toolkit: 'youtube', label: 'YouTube', verified: true, setup_note: null, copy_limits: { text: 5000, title: 100 },
    post_kinds: [kind('video')] as any },
  { toolkit: 'instagram', label: 'Instagram', verified: true, setup_note: null, copy_limits: { text: 2200, hashtags: 30 },
    post_kinds: [kind('reel', false, 'the action INSTAGRAM_CREATE_REEL is missing')] as any },
]
const PROPOSAL: SocialComposeProposal = {
  title: 'Harvest Club', copy: { base: 'Opens Friday.', per_channel: { twitter: 'Friday!' } }, format: 'video',
  template_id: 'tpl-v',
  template: { id: 'tpl-v', name: 'Countdown', format: 'social_video', sizes: ['1080x1920', '1080x1350'], variables_schema: {} },
  variables: {}, sources: {}, channels: ['twitter'], warnings: [],
}

function renderComposer(onDone = vi.fn()) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false }, mutations: { retry: false } } })
  render(
    <QueryClientProvider client={client}>
      <SocialsComposer onDone={onDone} />
    </QueryClientProvider>,
  )
  return onDone
}

async function drafted(onDone = vi.fn()) {
  renderComposer(onDone)
  await waitFor(() => expect(api.listSocialChannels).toHaveBeenCalled())
  fireEvent.change(await screen.findByLabelText('Brief'), { target: { value: 'Harvest Club countdown' } })
  fireEvent.click(screen.getByRole('button', { name: /Draft it/ }))
  await screen.findByRole('button', { name: /^Next: variables/ })
  return onDone
}

function toChannels() {
  fireEvent.click(screen.getByRole('button', { name: /Formats and channels/ }))
  return screen.getByRole('region', { name: 'Channels' })
}

beforeEach(() => {
  Object.values(api).forEach((fn) => fn.mockReset())
  server.post = null
  api.listSocialChannels.mockResolvedValue(CHANNELS)
  api.composeSocialPost.mockResolvedValue(PROPOSAL)
  api.createSocialPost.mockImplementation(async (input: any) => {
    server.post = { id: 'post-1', ...input, status: 'draft', content_hash: 'h1', media: {}, preview: null, targets: [] }
    return server.post
  })
  api.updateSocialPost.mockImplementation(async (_id: string, input: any) => (server.post = { ...server.post, ...input }))
  api.setSocialPostTargets.mockImplementation(async (_id: string, targets: any) => (server.post = { ...server.post, targets }))
  api.renderSocialPost.mockImplementation(async () => (server.post = {
    ...server.post, preview: { status: 'done', content_hash: 'h1', error: null, at: '',
      files: [{ name: 'preview-video-9x16.mp4', url: '/api/socials/posts/post-1/media/preview-video-9x16.mp4', content_type: 'video/mp4' }] },
  }))
  api.getSocialPost.mockImplementation(async () => server.post)
  api.submitSocialPost.mockImplementation(async () => {
    if (server.post.status !== 'draft') throw Object.assign(new Error('cannot submit'), { status: 409 })
    return (server.post = { ...server.post, status: 'needs_approval' })
  })
})
afterEach(() => cleanup())

describe('formats and channels (US-209)', () => {
  it('shows the formats, and only available kinds can be chosen; unavailable ones say why', async () => {
    await drafted()
    const channels = toChannels()
    expect(screen.getByRole('region', { name: 'Formats' })).toHaveTextContent('1080×1920 (9:16), 1080×1350 (4:5)')
    const x = within(channels).getByTestId('socials-channel-twitter')
    expect(within(x).getByRole('checkbox')).toBeChecked()
    expect(within(x).getByRole('radio', { name: 'text' })).toBeEnabled()
    expect(within(x).getByRole('radio', { name: /video/ })).toBeDisabled()
    expect(x).toHaveTextContent('needs public storage for media')
    expect(x).toHaveTextContent('X needs your own X app')
    const insta = within(channels).getByTestId('socials-channel-instagram')
    expect(within(insta).getByRole('checkbox')).toBeDisabled()
    expect(insta).toHaveTextContent('the action INSTAGRAM_CREATE_REEL is missing')
  })

  it('a count turns red over the limit and blocks submit until the copy is shortened', async () => {
    await drafted()
    fireEvent.change(screen.getByLabelText('X copy'), { target: { value: 'x'.repeat(281) } })
    const count = screen.getByTestId('socials-copy-count-twitter')
    expect(count).toHaveTextContent('281 / 280')
    expect(count).toHaveClass('text-destructive')
    toChannels()
    expect(screen.getByRole('button', { name: 'Submit for approval' })).toBeDisabled()
    expect(screen.getByRole('alert')).toHaveTextContent('Over the limit: X')

    fireEvent.click(screen.getByRole('button', { name: /1\. Copy/ }))
    fireEvent.change(screen.getByLabelText('X copy'), { target: { value: 'x'.repeat(280) } })
    expect(screen.getByTestId('socials-copy-count-twitter')).not.toHaveClass('text-destructive')
    toChannels()
    expect(screen.getByRole('button', { name: 'Submit for approval' })).toBeEnabled()
  })

  it('saving writes the targets: each chosen channel, its kind and its options', async () => {
    await drafted()
    const channels = toChannels()
    const x = within(channels).getByTestId('socials-channel-twitter')
    fireEvent.click(within(x).getByRole('radio', { name: 'image' }))
    fireEvent.click(within(within(channels).getByTestId('socials-channel-youtube')).getByRole('checkbox'))
    fireEvent.change(screen.getByLabelText('YouTube privacy'), { target: { value: 'unlisted' } })
    fireEvent.click(screen.getByRole('button', { name: 'Save draft' }))
    await waitFor(() => expect(api.setSocialPostTargets).toHaveBeenCalled())
    expect(api.setSocialPostTargets).toHaveBeenCalledWith('post-1', [
      { toolkit: 'twitter', post_kind: 'image' },
      { toolkit: 'youtube', post_kind: 'video', options: { privacy_status: 'unlisted', category_id: '22' } },
    ])
  })

  it('the full flow: brief → preview → channels → submit ends in needs_approval', async () => {
    const onDone = await drafted()
    fireEvent.click(screen.getByRole('button', { name: /^Next: variables/ }))
    fireEvent.click(within(screen.getByRole('region', { name: 'Preview' })).getByRole('button', { name: /Render a preview/ }))
    await waitFor(() => expect(api.renderSocialPost).toHaveBeenCalledWith('post-1', { preview: true }))
    await screen.findByTestId('file-preview')

    fireEvent.click(screen.getByRole('button', { name: /^Next: formats and channels/ }))
    fireEvent.click(screen.getByRole('button', { name: 'Submit for approval' }))

    await waitFor(() => expect(onDone).toHaveBeenCalled())
    expect(api.createSocialPost).toHaveBeenCalledTimes(1)
    expect(api.submitSocialPost).toHaveBeenCalledWith('post-1')
    expect(onDone.mock.calls[0][0]).toMatchObject({ id: 'post-1', status: 'needs_approval' })
    expect(server.post.status).toBe('needs_approval')
    // X cannot take the video yet (no public storage), so it posts its first available kind.
    expect(server.post.targets).toEqual([{ toolkit: 'twitter', post_kind: 'text' }])
  })
})
