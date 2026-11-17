/**
 * PRD-251D US-D102 (E1) — Submit for approval makes the file, against the editor's apiClient double.
 *
 * * A post with a visual (a template, Let Auto pick) and no rendered file yet is saved, then given
 *   the full render (POST /render with no preview), which ends it in needs_approval by itself:
 *   never the submit route.
 * * US-D108: a post sent back with its rendered file renders again only when the file does not
 *   show it: its own `file_current` is false, the editor holds unsaved changes to what the file
 *   shows (a field, not the brief alone), or the save itself answers that the file is older than
 *   the post. Otherwise its file already shows it, and it is submitted through the submit route as before.
 * * A video makes the full render, never the half-resolution preview, and the line under Submit
 *   says its render minutes; a still of the person's own is cropped; a text post submits.
 * * A refusal (422 naming what Auto could not write, 429 no render minutes) shows as Render
 *   preview's does, and the post stays as it was. Render preview stays.
 */
import { describe, it, expect, vi, afterEach, beforeEach } from 'vitest'
import { screen, cleanup, fireEvent, waitFor, within } from '@testing-library/react'

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

import { toast } from 'sonner'
import type { SocialPost } from '@/lib/api-client'
import { SocialsEditor } from '@/components/deliverables/socials/studio/socials-editor'
import {
  RENDERING_FOR_APPROVAL, SENT_FOR_APPROVAL, SUBMIT_CROPS, SUBMIT_RENDERS,
} from '@/components/deliverables/socials/studio/editor-submit'
import { api, post, renderWith, resetApi } from './socials-editor-harness'

const go = vi.fn()
const submitButton = () => screen.getByRole('button', { name: 'Submit for approval' })
const note = () => screen.queryByTestId('socials-submit-note')

function renderEditor(p: SocialPost | null) {
  return renderWith(<SocialsEditor role="owner" post={p} go={go} />)
}

async function submit(p: SocialPost | null) {
  renderEditor(p)
  await screen.findByText('TikTok')
  fireEvent.click(submitButton())
}

/** A post rendered, then sent back by its approver: its file is from before any change since. */
const SENT_BACK = post({
  status: 'changes_requested', template_id: 'tpl-img', variables: { headline: { value: 'OPEN LATE', claim: false } },
  media: { '1:1': ['d-square'] },
  review_log: [
    { at: '2026-10-14T07:02:00Z', by: 'u1', action: 'render_done', comment: 'Rendered 1080 x 1080.' },
    { at: '2026-10-14T08:00:00Z', by: 'u2', action: 'request_changes', comment: 'Say Friday, not Thursday.' },
  ] as any,
})

beforeEach(() => { resetApi(); go.mockReset(); vi.mocked(toast.success).mockReset(); vi.mocked(toast.error).mockReset() })
afterEach(cleanup)

describe('Submit for approval', () => {
  it('saves an image post, then gives it the full render, never the submit route, and says so', async () => {
    renderEditor(post({ template_id: 'tpl-img' }))
    await screen.findByText('TikTok')
    expect(note()).toHaveTextContent(SUBMIT_RENDERS)

    fireEvent.click(submitButton())

    await waitFor(() => expect(api.renderSocialPost).toHaveBeenCalledWith('post-1', {}))
    expect(api.submitSocialPost).not.toHaveBeenCalled()
    expect(api.updateSocialPost.mock.invocationCallOrder[0]).toBeLessThan(api.renderSocialPost.mock.invocationCallOrder[0])
    await waitFor(() => expect(toast.success).toHaveBeenCalledWith(RENDERING_FOR_APPROVAL))
  })

  it('renders a new post left to Let Auto pick, on the one post it creates', async () => {
    await submit(null)
    await waitFor(() => expect(api.renderSocialPost).toHaveBeenCalledWith('post-new', {}))
    expect(api.createSocialPost).toHaveBeenCalledTimes(1)
    expect(api.submitSocialPost).not.toHaveBeenCalled()
    await waitFor(() => expect(go).toHaveBeenCalledWith({ post: 'post-new' }))
  })

  it('renders again a post sent back whose file is older than its words (US-D108)', async () => {
    renderEditor({ ...SENT_BACK, file_current: false })
    await screen.findByText('TikTok')
    expect(note()).toHaveTextContent(SUBMIT_RENDERS)
    fireEvent.click(submitButton())
    await waitFor(() => expect(api.renderSocialPost).toHaveBeenCalledWith('post-1', {}))
    expect(api.submitSocialPost).not.toHaveBeenCalled()
  })

  it('submits a post sent back whose file still shows it, through the submit route, with no line about a render (US-D108)', async () => {
    renderEditor({ ...SENT_BACK, file_current: true })
    await screen.findByText('TikTok')
    expect(note()).toBeNull()
    fireEvent.click(submitButton())
    await waitFor(() => expect(api.submitSocialPost).toHaveBeenCalledWith('post-1'))
    expect(api.renderSocialPost).not.toHaveBeenCalled()
    await waitFor(() => expect(toast.success).toHaveBeenCalledWith(SENT_FOR_APPROVAL))
  })

  it('renders such a post once the editor holds an unsaved change to what its file shows (US-D108)', async () => {
    renderEditor({ ...SENT_BACK, file_current: true })
    await screen.findByText('TikTok')
    const fields = await screen.findByRole('region', { name: 'Text on the image' })
    fireEvent.change(await within(fields).findByLabelText('Headline'), { target: { value: 'OPEN LATE FRIDAY' } })
    expect(note()).toHaveTextContent(SUBMIT_RENDERS)
    fireEvent.click(submitButton())
    await waitFor(() => expect(api.renderSocialPost).toHaveBeenCalledWith('post-1', {}))
    expect(api.submitSocialPost).not.toHaveBeenCalled()
  })

  it('submits it, with no second render, when the only unsaved edit is to the brief (US-D108)', async () => {
    // The brief is what Auto writes from: it is not on the file, so the file still shows the post.
    renderEditor({ ...SENT_BACK, file_current: true })
    await screen.findByText('TikTok')
    fireEvent.change(screen.getByRole('textbox', { name: 'Brief' }), { target: { value: 'Say Friday, not Thursday.' } })
    expect(note()).toBeNull()
    fireEvent.click(submitButton())
    await waitFor(() => expect(api.submitSocialPost).toHaveBeenCalledWith('post-1'))
    expect(api.renderSocialPost).not.toHaveBeenCalled()
  })

  it("renders when the save's own answer says the file is now older than the post (US-D108)", async () => {
    // A change the editor does not compare (a footage prompt, say): the server's flag after the save decides.
    api.setSocialPostTargets.mockImplementationOnce(async (id: string) => post({ ...SENT_BACK, id, file_current: false }))
    renderEditor({ ...SENT_BACK, file_current: true })
    await screen.findByText('TikTok')
    fireEvent.click(submitButton())
    await waitFor(() => expect(api.renderSocialPost).toHaveBeenCalledWith('post-1', {}))
    expect(api.submitSocialPost).not.toHaveBeenCalled()
    await waitFor(() => expect(toast.success).toHaveBeenCalledWith(RENDERING_FOR_APPROVAL))
  })

  it('gives a video the full render, not the preview, and says its render minutes', async () => {
    renderEditor(post({ format: 'video', template_id: 'tpl-vid', length_seconds: 30 }))
    await screen.findByText('TikTok')
    // The video template makes 2 sizes (once the gallery has loaded): 0:30 each is a minute of rendering.
    await waitFor(() => expect(note()).toHaveTextContent(
      'Submit for approval makes the full render first, then it waits for approval: about 1 render minutes (2 sizes x 0:30).',
    ))

    fireEvent.click(submitButton())

    await waitFor(() => expect(api.renderSocialPost).toHaveBeenCalledWith('post-1', {}))
    expect(api.renderSocialPost).not.toHaveBeenCalledWith('post-1', { preview: true })
    expect(api.submitSocialPost).not.toHaveBeenCalled()
  })

  it("crops a still of the person's own to each channel before it goes for approval", async () => {
    renderEditor(post({ media: { original: ['d-1'] } }))
    await screen.findByText('TikTok')
    expect(note()).toHaveTextContent(SUBMIT_CROPS)
    fireEvent.click(submitButton())
    await waitFor(() => expect(api.renderSocialPost).toHaveBeenCalledWith('post-1', {}))
    expect(api.submitSocialPost).not.toHaveBeenCalled()
  })

  it("submits a text post and a video of the person's own as today, with no line about a render", async () => {
    renderEditor(post({ format: 'text' }))
    await screen.findByText('TikTok')
    expect(note()).toBeNull()
    fireEvent.click(submitButton())
    await waitFor(() => expect(api.submitSocialPost).toHaveBeenCalledWith('post-1'))
    await waitFor(() => expect(toast.success).toHaveBeenCalledWith(SENT_FOR_APPROVAL))
    cleanup()

    renderEditor(post({ format: 'video', media: { original: ['d-1'] } }))
    await screen.findByText('TikTok')
    expect(note()).toBeNull()
    fireEvent.click(submitButton())
    await waitFor(() => expect(api.submitSocialPost).toHaveBeenCalledTimes(2))
    expect(api.renderSocialPost).not.toHaveBeenCalled()
  })

  it.each([
    [422, 'Auto could not write these from the brief: Their words. Say them in the brief, or fill them in yourself.'],
    [429, 'This workspace has used its 10 render minutes this month.'],
  ])('a refusal (%i) shows as it does for Render preview, and the post stays as it was', async (status, message) => {
    api.renderSocialPost.mockRejectedValueOnce(Object.assign(new Error(message), { status }))

    await submit(post({ template_id: 'tpl-img' }))

    await waitFor(() => expect(toast.error).toHaveBeenCalledWith(message))
    expect(api.submitSocialPost).not.toHaveBeenCalled()
    expect(screen.getByTestId('socials-editor-status')).toHaveTextContent('Draft')
    expect(go).not.toHaveBeenCalled()
  })

  it('keeps Render preview for a look before submitting', async () => {
    renderEditor(post({ template_id: 'tpl-img' }))
    await screen.findByText('TikTok')
    fireEvent.click(screen.getByRole('button', { name: 'Render preview' }))
    await waitFor(() => expect(api.renderSocialPost).toHaveBeenCalledWith('post-1', {}))
    expect(api.submitSocialPost).not.toHaveBeenCalled()
  })
})
