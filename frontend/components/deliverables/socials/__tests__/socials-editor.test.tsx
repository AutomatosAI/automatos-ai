/**
 * PRD-251B US-B109 — the post editor (Editor.dc.html), against a stateful apiClient double.
 *
 * * The cards in the mockup's order, with Save draft, Render preview and Submit for approval.
 * * Format: a video's Length chips are exactly its template's lengths, with the words that
 *   fit; a carousel steps 2 to 10 slides; a text post disables the channels that take no
 *   text, saying why; a channel the format excludes says why too.
 * * Channels and sizes: each ticked channel's size, and "Renders" the distinct ratios.
 * * Save draft writes the post (POST, or PATCH once it exists), its channels (PUT /targets)
 *   and its slot (PUT /slot, only when it moved); Redraft with Auto sends the editor's
 *   choices and replaces the copy, the variables and the sources only; Submit submits.
 */
import { describe, it, expect, vi, afterEach, beforeEach } from 'vitest'
import { screen, cleanup, fireEvent, waitFor, within } from '@testing-library/react'

vi.mock('@/lib/api-client', async () => (await import('./socials-editor-harness')).apiModule)
vi.mock('sonner', () => ({ toast: { success: vi.fn(), error: vi.fn() } }))
vi.mock('@/components/workspace-provider', () => ({
  useWorkspace: () => ({ workspace: { id: 'w1', role: 'owner', socials: { available: true, enabled: true } } }),
}))
vi.mock('@/components/deliverables/socials/socials-voice-picker', () => ({ SocialsVoicePicker: () => <div data-testid="voice-picker" /> }))
vi.mock('@/components/deliverables/socials/studio/socials-editor-activity', () => ({ SocialsEditorActivity: () => null }))
vi.mock('@/components/deliverables/socials/socials-composer-preview', () => ({
  SocialsComposerPreview: ({ stale }: { stale: boolean }) => <div data-testid="preview" data-stale={String(stale)} />,
}))

import { SocialsEditor } from '@/components/deliverables/socials/studio/socials-editor'
import { TEXT_ONLY_NOTE } from '@/components/deliverables/socials/studio/editor-format-card'
import { api, post, renderWith, resetApi } from './socials-editor-harness'

const go = vi.fn()
const card = (name: string) => screen.getByRole('region', { name })
const channelRow = (label: string) => within(card('Channels and sizes')).getByRole('listitem', { name: label })
const tick = (label: string) => fireEvent.click(within(channelRow(label)).getByRole('checkbox'))

function renderEditor(p = null as ReturnType<typeof post> | null) {
  return renderWith(<SocialsEditor role="owner" post={p} go={go} />)
}

async function chooseFormat(label: string) {
  fireEvent.click(within(card('Format')).getByRole('button', { name: label }))
}

beforeEach(() => { resetApi(); go.mockReset() })
afterEach(cleanup)

describe('the post editor', () => {
  it('shows the cards in the mockup order with the three header actions', async () => {
    renderEditor()
    await screen.findByText('Fact card')
    const names = screen.getAllByRole('region').map((r) => r.getAttribute('aria-label'))
    const cards = names.filter((n) => ['Brief', 'Format', 'Channels and sizes', 'Look', 'Claims and sources', 'When'].includes(n ?? ''))
    expect(cards).toEqual(['Brief', 'Format', 'Channels and sizes', 'Look', 'Claims and sources', 'When'])
    for (const name of ['Save draft', 'Render preview', 'Submit for approval']) {
      expect(screen.getByRole('button', { name })).toBeInTheDocument()
    }
    expect(screen.getByRole('button', { name: 'Back to calendar' })).toBeInTheDocument()
    expect(screen.getByTestId('socials-editor-status')).toHaveTextContent('New')
  })

  it('a video offers exactly its template lengths, with the words that fit', async () => {
    renderEditor()
    await chooseFormat('Video')
    fireEvent.click(await within(card('Look')).findByRole('button', { name: /Story promo/ }))
    const lengths = within(card('Format')).getByRole('group', { name: 'Length' })
    expect(within(lengths).getAllByRole('button').map((b) => b.textContent)).toEqual(['0:15', '0:30', '0:40'])
    expect(within(lengths).getByRole('button', { name: '0:15' })).toHaveAttribute('aria-pressed', 'true')
    expect(within(card('Format')).getByText('About 38 spoken words fit in 0:15.')).toBeInTheDocument()
    fireEvent.click(within(lengths).getByRole('button', { name: '0:30' }))
    expect(within(card('Format')).getByText('About 75 spoken words fit in 0:30.')).toBeInTheDocument()
    expect(within(card('Format')).getByRole('group', { name: 'AI footage' })).toBeInTheDocument()
  })

  it('a carousel steps from 2 to 10 slides', async () => {
    renderEditor()
    await chooseFormat('Carousel')
    const more = screen.getByRole('button', { name: 'One slide more' })
    const fewer = screen.getByRole('button', { name: 'One slide fewer' })
    for (let i = 0; i < 6; i += 1) fireEvent.click(more)
    expect(within(card('Format')).getByText('10')).toBeInTheDocument()
    expect(more).toBeDisabled()
    for (let i = 0; i < 9; i += 1) fireEvent.click(fewer)
    expect(within(card('Format')).getByText('2')).toBeInTheDocument()
    expect(fewer).toBeDisabled()
  })

  it('a text post disables the channels that take no text, and says why', async () => {
    renderEditor()
    await screen.findByText('TikTok')
    await chooseFormat('Text only')
    expect(screen.getByText(TEXT_ONLY_NOTE)).toBeInTheDocument()
    expect(within(channelRow('TikTok')).getByRole('checkbox')).toBeDisabled()
    expect(channelRow('TikTok')).toHaveTextContent('Needs an image or video')
    expect(within(channelRow('X')).getByRole('checkbox')).not.toBeDisabled()
    expect(screen.queryByRole('region', { name: 'Look' })).toBeNull()
  })

  it('a channel the format excludes says why, and an unverified channel is named once', async () => {
    renderEditor()
    await screen.findByText('TikTok')
    expect(channelRow('TikTok')).toHaveTextContent('Takes video only')
    expect(screen.getAllByText(/unverified channel/)).toHaveLength(1)
    await chooseFormat('Video')
    expect(channelRow('Reddit (unverified channel)')).toHaveTextContent('Takes images only')
  })

  it('shows each ticked channel its size and Renders the distinct ratios', async () => {
    renderEditor()
    fireEvent.click(await within(card('Look')).findByRole('button', { name: /Fact card/ }))
    tick('X')
    tick('LinkedIn')
    tick('Instagram')
    expect(channelRow('X')).toHaveTextContent('1920 × 1080')
    expect(channelRow('LinkedIn')).toHaveTextContent('1080 × 1080')
    expect(card('Channels and sizes')).toHaveTextContent('Renders 16:9 · 1:1')
  })

  it('Save draft creates the post, then sets its channels and its slot', async () => {
    renderEditor()
    await screen.findByText('TikTok')
    fireEvent.change(screen.getByRole('textbox', { name: 'Title' }), { target: { value: 'Launch week teaser' } })
    fireEvent.change(screen.getByRole('textbox', { name: 'Brief' }), { target: { value: 'Tease the launch' } })
    fireEvent.change(screen.getByRole('textbox', { name: 'Copy' }), { target: { value: 'Something lands Monday.' } })
    tick('X')
    fireEvent.change(screen.getByLabelText('Date'), { target: { value: '2026-10-20' } })
    fireEvent.change(screen.getByLabelText('Time'), { target: { value: '12:00' } })
    fireEvent.change(screen.getByLabelText('Timezone'), { target: { value: 'UTC' } })
    fireEvent.click(screen.getByRole('button', { name: 'Save draft' }))

    await waitFor(() => expect(go).toHaveBeenCalledWith({ post: 'post-new' }))
    expect(api.createSocialPost).toHaveBeenCalledWith(expect.objectContaining({
      title: 'Launch week teaser', brief: 'Tease the launch', copy: { base: 'Something lands Monday.', channels: {} },
      format: 'image', template_id: null, length_seconds: null,
    }))
    expect(api.setSocialPostTargets).toHaveBeenCalledWith('post-new', [{ toolkit: 'twitter', post_kind: 'image' }])
    expect(api.setSocialPostSlot).toHaveBeenCalledWith('post-new', '2026-10-20T12:00:00.000Z', 'UTC')
  })

  it('Save draft edits a saved post (PATCH) and leaves an unmoved slot alone', async () => {
    renderEditor(post({ planned_for: '2026-10-20T12:00:00Z', timezone: 'UTC', targets: [{ toolkit: 'twitter', post_kind: 'image', options: {} } as any] }))
    await screen.findByText('TikTok')
    fireEvent.change(screen.getByRole('textbox', { name: 'Copy' }), { target: { value: 'Edited.' } })
    fireEvent.click(screen.getByRole('button', { name: 'Save draft' }))
    await waitFor(() => expect(api.updateSocialPost).toHaveBeenCalled())
    expect(api.updateSocialPost.mock.calls[0][0]).toBe('post-1')
    expect(api.updateSocialPost.mock.calls[0][1].copy).toEqual({ base: 'Edited.', channels: {} })
    await waitFor(() => expect(api.setSocialPostTargets).toHaveBeenCalledWith('post-1', [{ toolkit: 'twitter', post_kind: 'image' }]))
    expect(api.createSocialPost).not.toHaveBeenCalled()
    expect(api.setSocialPostSlot).not.toHaveBeenCalled()
    expect(go).not.toHaveBeenCalled()
  })

  it('Redraft with Auto sends the editor choices and replaces the copy, variables and sources only', async () => {
    api.composeSocialPost.mockResolvedValue({
      title: 'Another title', copy: { base: 'Fresh copy.', per_channel: { tiktok: 'Fresh for TikTok.' } },
      format: 'video', template_id: 'tpl-vid', template: null, variables: { hook: { value: 'Look', claim: false } },
      sources: {}, channels: ['tiktok'], warnings: [],
    })
    renderEditor(post({
      format: 'video', template_id: 'tpl-vid', length_seconds: 30,
      targets: [{ toolkit: 'tiktok', post_kind: 'video', options: {} } as any],
    }))
    await within(card('Format')).findByRole('group', { name: 'Length' })
    fireEvent.click(screen.getByRole('button', { name: /Redraft with Auto/ }))
    await waitFor(() => expect(api.composeSocialPost).toHaveBeenCalledWith({
      brief: 'The approval rule', channels: ['tiktok'], format: 'video', template_id: 'tpl-vid', length_seconds: 30,
    }))
    await waitFor(() => expect(screen.getByRole('textbox', { name: 'Copy' })).toHaveValue('Fresh copy.'))
    expect(screen.getByRole('textbox', { name: 'Title' })).toHaveValue('Nothing posts without you')
    expect(within(card('Format')).getByRole('button', { name: '0:30' })).toHaveAttribute('aria-pressed', 'true')
  })

  it('Submit for approval saves, then submits', async () => {
    renderEditor(post())
    await screen.findByText('TikTok')
    fireEvent.click(screen.getByRole('button', { name: 'Submit for approval' }))
    await waitFor(() => expect(api.submitSocialPost).toHaveBeenCalledWith('post-1'))
    expect(api.updateSocialPost).toHaveBeenCalled()
  })

  it('Render preview: a video asks for the preview, an image renders for real', async () => {
    renderEditor(post())
    await screen.findByText('TikTok')
    fireEvent.click(screen.getByRole('button', { name: 'Render preview' }))
    await waitFor(() => expect(api.renderSocialPost).toHaveBeenCalledWith('post-1', {}))
    cleanup()
    api.renderSocialPost.mockClear()
    renderEditor(post({ format: 'video', template_id: 'tpl-vid', length_seconds: 15 }))
    await screen.findByText('TikTok')
    fireEvent.click(screen.getByRole('button', { name: 'Render preview' }))
    await waitFor(() => expect(api.renderSocialPost).toHaveBeenCalledWith('post-1', { preview: true }))
  })

  it('a claim shows Unsourced until it has a source, and Add a source opens its search', async () => {
    renderEditor()
    fireEvent.click(await within(card('Look')).findByRole('button', { name: /Fact card/ }))
    const claims = card('Claims and sources')
    expect(within(claims).getByText(/Unsourced/)).toBeInTheDocument()
    expect(within(claims).getByText('A claim without a source needs a second confirmation from whoever approves.')).toBeInTheDocument()
    fireEvent.click(within(claims).getByRole('button', { name: 'Add a source' }))
    expect(await within(claims).findByRole('textbox', { name: 'Search sources for members' })).toBeInTheDocument()
  })

  it('submit waits while a ticked channel is over its limit', async () => {
    renderEditor()
    await screen.findByText('TikTok')
    tick('X')
    fireEvent.change(screen.getByRole('textbox', { name: 'Copy' }), { target: { value: 'x'.repeat(281) } })
    expect(screen.getByRole('button', { name: 'Submit for approval' })).toBeDisabled()
  })
})
