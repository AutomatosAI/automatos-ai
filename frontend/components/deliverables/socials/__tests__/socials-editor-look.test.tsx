/**
 * PRD-251B US-B109 — the editor's Look: where the visual comes from.
 *
 * * Template: the format's templates, "Let Auto pick" first, each with its thumbnail and name.
 * * Upload: the dropped file goes to POST /posts/{id}/media (a new post is saved first) and
 *   the post's media follows.
 * * Library: the workspace's image and video Deliverables; picking one sets the post's media.
 * * PRD-251B: with a template marked Photo, an upload or a Library picture fills its photo spot
 *   (the template's words over it), or the whole post when asked; the spot shows its picture.
 * * AI-made (PRD-251B US-B305) has its own tests: socials-editor-ai.test.tsx.
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
vi.mock('@/hooks/use-authed-image', () => ({ useAuthedImage: (path: string | null) => (path ? `blob:${path}` : null) }))
vi.mock('@/components/widgets/FileWidget/FilePreview', () => ({
  FilePreview: ({ url }: { url: string }) => <div data-testid="file-preview" data-url={url} />,
  inferPreviewType: () => 'image',
  useAuthenticatedBlobUrl: (url?: string) => ({ src: url ? `blob:${url}` : null, error: null }),
}))

import { SocialsEditor } from '@/components/deliverables/socials/studio/socials-editor'
import { DROP_HINT, DROP_TITLE } from '@/components/deliverables/socials/studio/editor-look-sources'
import { AUTO_PICK_NOTE } from '@/components/deliverables/socials/studio/editor-look-card'
import { DROP_PHOTO_TITLE } from '@/components/deliverables/socials/studio/editor-look-sources'
import { PHOTO_SPOT_HINT, PHOTO_TARGET, WHOLE_POST_HINT } from '@/components/deliverables/socials/studio/editor-look-photo'
import { OWN_PICTURE } from '@/components/deliverables/socials/studio/editor-look-ai'
import { IMAGE_TEMPLATE, PHOTO_TEMPLATE, api, post, renderWith, resetApi } from './socials-editor-harness'

const go = vi.fn()
const look = () => screen.getByRole('region', { name: 'Look' })
const source = (name: string) => fireEvent.click(within(within(look()).getByRole('group', { name: 'Where the visual comes from' })).getByRole('button', { name }))

beforeEach(() => { resetApi(); go.mockReset() })
afterEach(cleanup)

describe('the Look card', () => {
  it('offers Template, Upload, Library and AI-made', () => {
    renderWith(<SocialsEditor role="owner" post={null} go={go} />)
    const group = within(look()).getByRole('group', { name: 'Where the visual comes from' })
    expect(within(group).getAllByRole('button').map((b) => b.textContent)).toEqual(['Template', 'Upload', 'Library', 'AI-made'])
  })

  it('Template lists the format templates, Let Auto pick first, with their thumbnails', async () => {
    renderWith(<SocialsEditor role="owner" post={null} go={go} />)
    const gallery = await within(look()).findByRole('list', { name: 'Templates' })
    const cards = within(gallery).getAllByRole('button')
    // The Auto card says what Auto does with it (AUTO_PICK_NOTE).
    expect(cards.map((b) => b.textContent)).toEqual([`AutoLet Auto pick${AUTO_PICK_NOTE}`, 'Fact card'])
    expect(cards[0]).toHaveAttribute('aria-pressed', 'true')
    expect(cards[1].querySelector('img')).toHaveAttribute('src', 'https://cdn.test/fact-card.png')
    fireEvent.click(cards[1])
    expect(cards[1]).toHaveAttribute('aria-pressed', 'true')
    expect(api.listSocialTemplates).toHaveBeenCalledWith('image')
  })

  it('Upload saves a new post, then sends the file, and opens the saved post', async () => {
    renderWith(<SocialsEditor role="owner" post={null} go={go} />)
    source('Upload')
    expect(within(look()).getByText(DROP_TITLE)).toBeInTheDocument()
    expect(within(look()).getByText(DROP_HINT)).toBeInTheDocument()
    const file = new File([new Uint8Array([0x89, 0x50, 0x4e, 0x47])], 'visual.png', { type: 'image/png' })
    fireEvent.change(within(look()).getByLabelText('File to upload'), { target: { files: [file] } })
    await waitFor(() => expect(api.uploadSocialPostMedia).toHaveBeenCalledWith('post-new', file))
    expect(api.createSocialPost).toHaveBeenCalled()
    await waitFor(() => expect(go).toHaveBeenCalledWith({ post: 'post-new' }))
  })

  it('Library lists image and video Deliverables, and picking one sets the post media', async () => {
    renderWith(<SocialsEditor role="owner" post={post()} go={go} />)
    source('Library')
    const library = await within(look()).findByRole('list', { name: 'Library' })
    expect(api.listDeliverables).toHaveBeenCalledWith({ artifact_type: 'image', limit: 24 })
    expect(api.listDeliverables).toHaveBeenCalledWith({ artifact_type: 'video', limit: 24 })
    const items = within(library).getAllByRole('button')
    expect(items.map((b) => b.textContent)).toEqual(['videoA videoVideo', 'A imageImage']) // newest first
    fireEvent.click(items[0])
    await waitFor(() =>
      expect(api.updateSocialPost).toHaveBeenLastCalledWith('post-1', {
        media: { original: ['d-video'] }, template_id: null, length_seconds: null, format: 'video',
      }),
    )
  })

  it('a Library picture loads with the auth headers, as Outputs loads it: its preview_url is an API path', async () => {
    api.listDeliverables.mockImplementation(async ({ artifact_type }: { artifact_type: string }) => ({
      total: artifact_type === 'image' ? 1 : 0,
      deliverables: artifact_type === 'image'
        ? [{ id: 'd-up', title: 'Untitled post (upload)', artifact_type: 'image', file_name: null,
            preview_url: '/api/socials/posts/p9/media/upload-1.png', created_at: '2026-10-10T09:00:00Z' }]
        : [],
    }))
    renderWith(<SocialsEditor role="owner" post={post()} go={go} />)
    source('Library')
    const library = await within(look()).findByRole('list', { name: 'Library' })
    // 3 Oct 2026: a plain <img src="/api/..."> asked the web app, not the API, without the auth.
    expect(library.querySelector('img')).toHaveAttribute('src', 'blob:/api/socials/posts/p9/media/upload-1.png')
  })

  it('with a Photo template, Upload fills its photo spot, or the whole post when asked', async () => {
    api.listSocialTemplates.mockResolvedValue([IMAGE_TEMPLATE, PHOTO_TEMPLATE])
    renderWith(<SocialsEditor role="owner" post={post({ template_id: 'tpl-photo' })} go={go} />)
    source('Upload')
    const where = await within(look()).findByRole('group', { name: PHOTO_TARGET })
    expect(within(where).getAllByRole('button').map((b) => b.textContent)).toEqual(['Photo', 'The whole post'])
    expect(within(where).getByRole('button', { name: 'Photo' })).toHaveAttribute('aria-pressed', 'true')
    expect(within(look()).getByText(PHOTO_SPOT_HINT)).toBeInTheDocument()
    expect(within(look()).getByText(DROP_PHOTO_TITLE)).toBeInTheDocument()
    const file = new File([new Uint8Array([0x89, 0x50, 0x4e, 0x47])], 'salon.png', { type: 'image/png' })
    fireEvent.change(within(look()).getByLabelText('File to upload'), { target: { files: [file] } })
    await waitFor(() => expect(api.uploadSocialPostMedia).toHaveBeenCalledWith('post-1', file, 'photo'))

    fireEvent.click(within(where).getByRole('button', { name: 'The whole post' }))
    expect(within(look()).getByText(WHOLE_POST_HINT)).toBeInTheDocument()
    fireEvent.change(within(look()).getByLabelText('File to upload'), { target: { files: [file] } })
    await waitFor(() => expect(api.uploadSocialPostMedia).toHaveBeenLastCalledWith('post-1', file))
  })

  it('with a Photo template, the Library lists pictures only and the pick fills the photo spot', async () => {
    api.listSocialTemplates.mockResolvedValue([IMAGE_TEMPLATE, PHOTO_TEMPLATE])
    renderWith(<SocialsEditor role="owner" post={post({ template_id: 'tpl-photo' })} go={go} />)
    source('Library')
    await within(look()).findByRole('group', { name: PHOTO_TARGET })
    const library = await within(look()).findByRole('list', { name: 'Library' })
    const items = within(library).getAllByRole('button')
    expect(items.map((b) => b.textContent)).toEqual(['A imageImage']) // the video is no photo
    fireEvent.click(items[0])
    await waitFor(() => expect(api.setSocialPostPhoto).toHaveBeenCalledWith('post-1', 'photo', 'd-image'))
    expect(api.updateSocialPost).not.toHaveBeenCalledWith('post-1', expect.objectContaining({ media: expect.anything() }))
  })

  it('a photo spot that holds your picture shows it', async () => {
    api.listSocialTemplates.mockResolvedValue([IMAGE_TEMPLATE, PHOTO_TEMPLATE])
    const own = { prompt: 'Photo: your own picture (upload-1.png)', status: 'done' as const, toolkit: 'upload', name: 'upload-1.png' }
    renderWith(<SocialsEditor role="owner" post={post({ template_id: 'tpl-photo', footage: { photo: own } })} go={go} />)
    source('Upload')
    expect(await within(look()).findByText(OWN_PICTURE)).toBeInTheDocument()
    expect(within(look()).getByRole('img', { name: OWN_PICTURE })).toHaveAttribute('src', 'blob:/api/socials/posts/post-1/media/upload-1.png')
  })
})
