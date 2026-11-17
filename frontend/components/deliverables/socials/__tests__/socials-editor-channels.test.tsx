/**
 * PRD-251D US-D104 (E4) and US-D105 (E5) — the editor's Channels card, against the editor's
 * apiClient double.
 *
 * * A new post starts with its channels ticked: the one connected channel; the most recent
 *   post's channels that can take the format, each at the format's own kind; none for two
 *   channels and no earlier post. A format change keeps the existing rule; the person's own
 *   tick is never overridden; a saved post opens with its own channels.
 * * Each supported channel that is not connected has a Connect button, which starts the
 *   Composio connect flow in its own window; with none connected, the buttons stand in for the
 *   old sentence; the channels are read again when the connect window reports back.
 */
import { describe, it, expect, vi, afterEach, beforeEach } from 'vitest'
import { act, screen, cleanup, fireEvent, waitFor, within } from '@testing-library/react'

vi.mock('@/lib/api-client', async () => (await import('./socials-editor-harness')).apiModule)
vi.mock('sonner', () => ({ toast: { success: vi.fn(), error: vi.fn() } }))
vi.mock('@/components/workspace-provider', () => ({
  useWorkspace: () => ({ workspace: { id: 'w1', role: 'owner', socials: { available: true, enabled: true } } }),
}))
vi.mock('@/components/deliverables/socials/socials-voice-picker', () => ({ SocialsVoicePicker: () => null }))
vi.mock('@/components/deliverables/socials/studio/socials-editor-activity', () => ({ SocialsEditorActivity: () => null }))
vi.mock('@/components/widgets/FileWidget/FilePreview', () => ({
  FilePreview: ({ url }: { url: string }) => <div data-testid="file-preview" data-url={url} />,
  inferPreviewType: () => 'image',
}))

import type { SocialChannel, SocialPost, SocialPostsResponse } from '@/lib/api-client'
import { CONNECTED_MESSAGE } from '@/components/deliverables/socials/connect-toolkit'
import { SocialsEditor } from '@/components/deliverables/socials/studio/socials-editor'
import { CHANNELS, CONNECT_URL, api, post, renderWith, resetApi } from './socials-editor-harness'

const go = vi.fn()
const card = () => screen.getByRole('region', { name: 'Channels and sizes' })
const row = (label: string) => within(card()).getByRole('listitem', { name: label })
const ticked = (label: string) => within(row(label)).getAllByRole('checkbox')[0] as HTMLInputElement
const storySwitch = (label: string) => within(row(label)).queryByRole('checkbox', { name: 'As a story' })
const connect = (label: string) => within(card()).queryByRole('button', { name: `Connect ${label}` })
const [X, LINKEDIN] = CHANNELS
const YOUTUBE = {
  toolkit: 'youtube', label: 'YouTube', verified: true, setup_note: null, copy_limits: { text: 5000 },
  post_kinds: [{ kind: 'video', available: true, reason: null, needs_public_storage: false }],
} as unknown as SocialChannel
const target = (toolkit: string, postKind: string) => ({ id: `t-${toolkit}`, toolkit, post_kind: postKind, options: {}, status: 'pending' }) as any

function posts(...list: SocialPost[]): SocialPostsResponse {
  return { posts: list, total: list.length } as SocialPostsResponse
}

/** Lets the posts and the channels the editor read arrive and their effects run: a check that
 * nothing was ticked means something only once they have. */
async function settle() {
  await waitFor(() => expect(api.listSocialPosts).toHaveBeenCalled())
  await act(() => new Promise<void>((resolve) => setTimeout(resolve, 30)))
}

function renderEditor(p: SocialPost | null = null) {
  return renderWith(<SocialsEditor role="owner" post={p} go={go} />)
}

beforeEach(() => { resetApi(); go.mockReset() })
afterEach(() => { cleanup(); vi.unstubAllGlobals() })

describe('a new post starts with its channels ticked (US-D104)', () => {
  it('ticks the one connected channel', async () => {
    api.listSocialChannels.mockResolvedValue([X])
    renderEditor()
    await waitFor(() => expect(ticked('X')).toBeChecked())
  })

  it("ticks the most recent post's channels that can take the format, each at the format's own kind", async () => {
    api.listSocialPosts.mockResolvedValue(posts(
      post({ id: 'latest', targets: [target('linkedin', 'video'), target('instagram', 'story'), target('tiktok', 'video')] }),
      post({ id: 'older', targets: [target('twitter', 'image')] }),
    ))
    renderEditor()
    await waitFor(() => expect(ticked('LinkedIn')).toBeChecked())
    expect(ticked('Instagram')).toBeChecked()
    expect(storySwitch('Instagram')).not.toBeChecked()  // the image kind, not the old story
    expect(ticked('TikTok')).not.toBeChecked()  // it takes no image
    expect(ticked('X')).not.toBeChecked()  // only on an older post

    fireEvent.click(screen.getByRole('button', { name: 'Save draft' }))
    await waitFor(() => expect(api.setSocialPostTargets).toHaveBeenCalledWith('post-new', [
      { toolkit: 'linkedin', post_kind: 'image' }, { toolkit: 'instagram', post_kind: 'image' },
    ]))
  })

  it('ticks none with two channels and no earlier post', async () => {
    api.listSocialChannels.mockResolvedValue([X, LINKEDIN])
    renderEditor()
    await screen.findByText('LinkedIn')
    await settle()
    expect(ticked('X')).not.toBeChecked()
    expect(ticked('LinkedIn')).not.toBeChecked()
  })

  it('a format change keeps the existing rule: a channel that cannot take it is left out', async () => {
    api.listSocialPosts.mockResolvedValue(posts(post({ targets: [target('twitter', 'image'), target('instagram', 'image')] })))
    renderEditor()
    await waitFor(() => expect(ticked('Instagram')).toBeChecked())
    expect(ticked('X')).toBeChecked()

    fireEvent.click(within(screen.getByRole('region', { name: 'Format' })).getByRole('button', { name: 'Text only' }))

    expect(ticked('X')).toBeChecked()
    expect(ticked('Instagram')).not.toBeChecked()
    expect(ticked('Instagram')).toBeDisabled()
  })

  it("never overrides the person's own tick", async () => {
    let answer: (value: SocialPostsResponse) => void = () => undefined
    api.listSocialPosts.mockImplementation(() => new Promise<SocialPostsResponse>((resolve) => { answer = resolve }))
    renderEditor()
    await screen.findByText('TikTok')
    fireEvent.click(ticked('X'))  // the person ticks X before the posts arrive

    await act(async () => answer(posts(post({ targets: [target('linkedin', 'image')] }))))
    await settle()

    expect(ticked('X')).toBeChecked()
    expect(ticked('LinkedIn')).not.toBeChecked()
  })

  it('a saved post opens with its own channels', async () => {
    api.listSocialPosts.mockResolvedValue(posts(post({ id: 'latest', targets: [target('linkedin', 'image')] })))
    renderEditor(post({ targets: [target('twitter', 'image')] }))
    await screen.findByText('TikTok')
    await settle()
    expect(ticked('X')).toBeChecked()
    expect(ticked('LinkedIn')).not.toBeChecked()
  })
})

describe('connect a channel from the editor (US-D105)', () => {
  it('shows a Connect button for each supported channel that is not connected, and none for a connected one', async () => {
    renderEditor()
    await screen.findByText('TikTok')
    expect(connect('YouTube')).toBeInTheDocument()
    for (const label of ['X', 'LinkedIn', 'Instagram', 'TikTok']) expect(connect(label)).toBeNull()
    expect(connect('Reddit (unverified channel)')).toBeNull()  // not a supported channel: never offered
  })

  it('with no channel connected, the buttons stand in for the sentence', async () => {
    api.listSocialChannels.mockResolvedValue([])
    renderEditor()
    const buttons = await within(card()).findAllByRole('button', { name: /^Connect / })
    expect(buttons.map((button) => button.textContent)).toEqual([
      'Connect LinkedIn', 'Connect X', 'Connect Instagram', 'Connect TikTok', 'Connect YouTube',
    ])
    expect(screen.queryByText(/No social channel is connected/)).toBeNull()
    expect(within(card()).queryByRole('list', { name: 'Channels' })).toBeNull()
  })

  it('Connect starts the Composio connect flow in a window of its own', async () => {
    vi.stubGlobal('open', vi.fn())
    renderEditor()
    await screen.findByText('TikTok')
    fireEvent.click(connect('YouTube') as HTMLElement)
    await waitFor(() => expect(api.post).toHaveBeenCalledWith(
      '/api/composio/connect/YOUTUBE', { callback_url: `${window.location.origin}/tools/callback?connected=YOUTUBE` },
    ))
    await waitFor(() => expect(window.open).toHaveBeenCalledWith(`${CONNECT_URL}/YOUTUBE`, 'Connect YouTube', expect.any(String)))
  })

  it('reads the channels again when the connect window reports back', async () => {
    renderEditor()
    await screen.findByText('TikTok')
    expect(connect('YouTube')).toBeInTheDocument()

    api.listSocialChannels.mockResolvedValue([...CHANNELS, YOUTUBE])
    window.dispatchEvent(new MessageEvent('message', { origin: window.location.origin, data: { type: CONNECTED_MESSAGE, app: 'YOUTUBE' } }))

    expect(await within(card()).findByRole('listitem', { name: 'YouTube' })).toBeInTheDocument()
    expect(connect('YouTube')).toBeNull()
  })
})
