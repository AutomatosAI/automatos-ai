/**
 * PRD-251B US-B109 — the editor's Look: where the visual comes from.
 *
 * * Template: the format's templates, "Let Auto pick" first, each with its thumbnail and name.
 * * Upload: the dropped file goes to POST /posts/{id}/media (a new post is saved first) and
 *   the post's media follows.
 * * Library: the workspace's image and video Deliverables; picking one sets the post's media.
 * * AI-made visuals are Wave 3: not offered.
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

import { SocialsEditor } from '@/components/deliverables/socials/studio/socials-editor'
import { DROP_HINT, DROP_TITLE } from '@/components/deliverables/socials/studio/editor-look-sources'
import { api, post, renderWith, resetApi } from './socials-editor-harness'

const go = vi.fn()
const look = () => screen.getByRole('region', { name: 'Look' })
const source = (name: string) => fireEvent.click(within(within(look()).getByRole('group', { name: 'Where the visual comes from' })).getByRole('button', { name }))

beforeEach(() => { resetApi(); go.mockReset() })
afterEach(cleanup)

describe('the Look card', () => {
  it('offers Template, Upload and Library, and no AI-made visuals', () => {
    renderWith(<SocialsEditor role="owner" post={null} go={go} />)
    const group = within(look()).getByRole('group', { name: 'Where the visual comes from' })
    expect(within(group).getAllByRole('button').map((b) => b.textContent)).toEqual(['Template', 'Upload', 'Library'])
    expect(within(look()).queryByText(/AI/)).toBeNull()
  })

  it('Template lists the format templates, Let Auto pick first, with their thumbnails', async () => {
    renderWith(<SocialsEditor role="owner" post={null} go={go} />)
    const gallery = await within(look()).findByRole('list', { name: 'Templates' })
    const cards = within(gallery).getAllByRole('button')
    expect(cards.map((b) => b.textContent)).toEqual(['AutoLet Auto pick', 'Fact card'])
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
})
