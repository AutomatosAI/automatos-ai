/**
 * PRD-251B US-B110 — the editor's preview: one tab per ticked channel; each shows the post
 * as it will look there (the brand, the channel, the slot, the channel's copy, the media at
 * its aspect through FilePreview), its copy with the count against the channel's limit (red
 * over it, and Submit waits), "Preview out of date" after an edit, and the note on what was
 * rendered or what the full render costs. Render preview asks for the half-resolution preview.
 */
import { describe, it, expect, vi, afterEach, beforeEach } from 'vitest'
import { screen, cleanup, fireEvent, waitFor, within } from '@testing-library/react'

vi.mock('@/lib/api-client', async () => (await import('./socials-editor-harness')).apiModule)
vi.mock('sonner', () => ({ toast: { success: vi.fn(), error: vi.fn() } }))
vi.mock('@/components/workspace-provider', () => ({
  useWorkspace: () => ({ workspace: { id: 'w1', name: 'Acme', role: 'owner', socials: { available: true, enabled: true } } }),
}))
vi.mock('@/components/deliverables/socials/socials-voice-picker', () => ({ SocialsVoicePicker: () => null }))
vi.mock('@/components/deliverables/socials/studio/socials-editor-activity', () => ({ SocialsEditorActivity: () => null }))
vi.mock('@/components/widgets/FileWidget/FilePreview', () => ({
  FilePreview: ({ url }: { url: string }) => <div data-testid="file-preview" data-url={url} />,
  inferPreviewType: () => 'image',
}))

import { SocialsEditor } from '@/components/deliverables/socials/studio/socials-editor'
import { PREVIEW_STALE, TEXT_NOTE, renderNote } from '@/components/deliverables/socials/studio/preview-model'
import { api, post, renderWith, resetApi } from './socials-editor-harness'

const go = vi.fn()
const preview = () => screen.getByRole('region', { name: 'Preview' })
const target = (toolkit: string, kind: string) => ({ id: `t-${toolkit}`, toolkit, post_kind: kind, options: {} }) as any

const TWO_CHANNELS = post({
  copy: { base: 'Base copy.', channels: { twitter: 'x'.repeat(281) } },
  targets: [target('twitter', 'image'), target('linkedin', 'image')],
  media: { '16:9': ['d-wide'], '1:1': ['d-square'] },
  planned_for: '2026-10-14T11:00:00Z', timezone: 'Europe/London',
  review_log: [{ at: '2026-10-14T07:02:00Z', by: 'agent', action: 'render_done', comment: null }] as any,
})

function renderEditor(p: ReturnType<typeof post>) {
  return renderWith(<SocialsEditor role="owner" post={p} go={go} />)
}

beforeEach(() => {
  resetApi()
  go.mockReset()
  api.getSocialPostMedia.mockResolvedValue([
    { aspect: '16:9', deliverable_id: 'd-wide', name: 'image-16x9.png', url: 'https://cdn.test/wide.png', content_type: 'image/png', bytes: 10, error: null },
    { aspect: '1:1', deliverable_id: 'd-square', name: 'image-1x1.png', url: 'https://cdn.test/square.png', content_type: 'image/png', bytes: 10, error: null },
  ])
})
afterEach(cleanup)

describe('the preview', () => {
  it('has one tab per ticked channel, each with its own copy, count and limit', async () => {
    renderEditor(TWO_CHANNELS)
    const tabs = await within(preview()).findByRole('group', { name: 'Preview channel' })
    await waitFor(() => expect(within(tabs).getAllByRole('button').map((b) => b.textContent)).toEqual(['X', 'LinkedIn']))
    expect(within(preview()).getByRole('textbox', { name: 'Copy for X' })).toHaveValue('x'.repeat(281))
    const count = within(preview()).getByTestId('socials-copy-count-twitter')
    expect(count).toHaveTextContent('281 / 280')
    expect(count.className).toContain('text-destructive')
    expect(screen.getByRole('button', { name: 'Submit for approval' })).toBeDisabled()

    fireEvent.click(within(tabs).getByRole('button', { name: 'LinkedIn' }))
    expect(within(preview()).getByRole('textbox', { name: 'Copy for LinkedIn' })).toHaveValue('Base copy.')
    expect(within(preview()).getByTestId('socials-copy-count-linkedin')).toHaveTextContent('10 / 3000')
    expect(within(preview()).getByText('Each channel starts from the same base copy. An edit here changes this channel only.')).toBeInTheDocument()
  })

  it('an edit to one channel changes that channel only, and shortening it lets Submit go', async () => {
    renderEditor(TWO_CHANNELS)
    const copy = await within(preview()).findByRole('textbox', { name: 'Copy for X' })
    fireEvent.change(copy, { target: { value: 'Short for X.' } })
    expect(screen.getByRole('button', { name: 'Submit for approval' })).not.toBeDisabled()
    expect(screen.getByRole('textbox', { name: 'Copy' })).toHaveValue('Base copy.')
  })

  it('shows the media at the channel aspect through FilePreview, with the brand, channel and slot', async () => {
    renderEditor(TWO_CHANNELS)
    const frame = await within(preview()).findByTestId('socials-preview-frame')
    await waitFor(() => expect(within(frame).getByTestId('file-preview')).toHaveAttribute('data-url', 'https://cdn.test/wide.png'))
    expect(frame.style.aspectRatio).toBe('16 / 9')
    expect(await within(preview()).findByText('Automatos')).toBeInTheDocument()
    expect(within(preview()).getByText(/X · Wed 14 Oct 2026, 12:00/)).toBeInTheDocument()
    fireEvent.click(within(preview()).getByRole('button', { name: 'LinkedIn' }))
    expect(within(preview()).getByTestId('file-preview')).toHaveAttribute('data-url', 'https://cdn.test/square.png')
    expect(within(preview()).getByTestId('socials-preview-frame').style.aspectRatio).toBe('1 / 1')
  })

  it('a text post shows the copy and no media frame', async () => {
    renderEditor(post({ format: 'text', copy: { base: 'Just words.' }, targets: [target('twitter', 'text')] }))
    await within(preview()).findByRole('textbox', { name: 'Copy for X' })
    expect(within(preview()).queryByTestId('socials-preview-frame')).toBeNull()
    expect(within(preview()).getByText(TEXT_NOTE)).toBeInTheDocument()
  })

  it('an edit to what a render reads marks the preview out of date', async () => {
    renderEditor(TWO_CHANNELS)
    const frame = await within(preview()).findByTestId('socials-preview-frame')
    await within(frame).findByTestId('file-preview')
    expect(within(frame).queryByText(PREVIEW_STALE)).toBeNull()
    fireEvent.change(screen.getByRole('textbox', { name: 'Brief' }), { target: { value: 'A new angle' } })
    expect(within(frame).getByText(PREVIEW_STALE)).toBeInTheDocument()
  })

  it('the note says what was rendered, or what the full render will cost', async () => {
    renderEditor(TWO_CHANNELS)
    expect(await within(preview()).findByText(/Rendered \d\d:\d\d\. One PNG per size \(16:9, 1:1\)\. Images use no render minutes\./)).toBeInTheDocument()
    expect(renderNote({ format: 'video', lengthSeconds: 30, ratios: ['9:16'], sizes: 3, renderedAt: null })).toBe(
      'Half-resolution preview, 0:30. The full render is made when you submit: about 1.5 render minutes (3 sizes x 0:30).',
    )
    expect(renderNote({ format: 'text', lengthSeconds: null, ratios: [], sizes: 0, renderedAt: null })).toBe(TEXT_NOTE)
  })

  it('Render preview asks for the half-resolution preview of a video', async () => {
    renderEditor(post({ format: 'video', template_id: 'tpl-vid', length_seconds: 30, targets: [target('tiktok', 'video')] }))
    expect(await within(preview()).findByText(/about 1 render minutes \(2 sizes x 0:30\)/)).toBeInTheDocument()
    fireEvent.click(screen.getByRole('button', { name: 'Render preview' }))
    await waitFor(() => expect(api.renderSocialPost).toHaveBeenCalledWith('post-1', { preview: true }))
  })
})
