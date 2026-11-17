/**
 * PRD-251D US-D201 (E8) — the instant preview of an image post, in the editor: an image post with
 * a template whose file is not current shows the template's own page for the words on screen, in a
 * frame whose sandbox is exactly "allow-scripts", labelled a preview; the page is shown only when
 * it opens with its own content security policy; the kit's font faces and the post's photo go in
 * at the page's markers; it is asked for a moment after the typing stops, once for a burst of
 * typing; a refusal keeps the last page and says why, any other failure keeps today's frame; a
 * video and a post whose render is current keep today's preview, and an edit brings the page back.
 *
 * The shared editor harness's apiClient double does not carry the preview's two methods or
 * requestBlob, so this file adds them to it (each test file has its own copy of the harness).
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
vi.mock('@/hooks/use-authed-image', () => ({ useAuthedImage: () => null }))
vi.mock('@/components/widgets/FileWidget/FilePreview', () => ({
  FilePreview: ({ url }: { url: string }) => <div data-testid="file-preview" data-url={url} />,
  inferPreviewType: () => 'image',
}))

import { SocialsEditor } from '@/components/deliverables/socials/studio/socials-editor'
import { INSTANT_PREVIEW_SANDBOX } from '@/components/deliverables/socials/studio/instant-preview-frame'
import {
  FONTS_MARKER, INSTANT_PREVIEW_DELAY_MS, INSTANT_PREVIEW_LABEL, INSTANT_PREVIEW_NOTE, PAGE_POLICY_META, PREVIEW_PAUSED, photoMarker,
} from '@/components/deliverables/socials/studio/instant-preview-model'
import { PHOTO_TEMPLATE, api, post, renderWith, resetApi } from './socials-editor-harness'

const instantApi = Object.assign(api, {
  getSocialTemplatePreviewPage: vi.fn(),
  getSocialPreviewFonts: vi.fn(),
  requestBlob: vi.fn(),
})

const go = vi.fn()
const WAIT = { timeout: 3000 }
const FONTS_CSS = '@font-face{font-family:"Brand Sans";src:url("data:font/woff2;base64,d09GMg==") format("woff2")}'
const PHOTO_URI = 'data:image/png;base64,cG5nLWJ5dGVz' // the bytes "png-bytes"
const REFUSAL = 'headline is longer than 60 characters (61 given)'
const pageOf = (words: string) =>
  `<!doctype html>${PAGE_POLICY_META}<script>window.gsap = {}</script><html><head><style>:root{--brand-primary:#e96235;}${FONTS_MARKER}</style>` +
  `</head><body><img data-slot="photo" src="${photoMarker('photo')}" alt=""><h1>${words}</h1></body></html>`
const target = (toolkit: string, kind: string) => ({ id: `t-${toolkit}`, toolkit, post_kind: kind, options: {} }) as any
const preview = () => screen.getByRole('region', { name: 'Preview' })
const card = (name: string) => screen.getByRole('region', { name })
const instantFrame = () => within(preview()).queryByTestId('socials-instant-preview')
const asked = () => instantApi.getSocialTemplatePreviewPage.mock.calls.map(([, input]) => input)
const settle = () => new Promise((resolve) => setTimeout(resolve, INSTANT_PREVIEW_DELAY_MS + 250))

const IMAGE_POST = post({
  template_id: 'tpl-img', targets: [target('linkedin', 'image')], variables: { headline: { value: 'Open late', claim: false } },
})

function renderEditor(p: ReturnType<typeof post>) {
  return renderWith(<SocialsEditor role="owner" post={p} go={go} />)
}

beforeEach(() => {
  resetApi()
  go.mockReset()
  instantApi.getSocialTemplatePreviewPage.mockImplementation(async (_id: string, input: any) => ({
    html: pageOf(String(input.values.headline ?? '')), fonts_version: 'kit-1',
  }))
  instantApi.getSocialPreviewFonts.mockResolvedValue({ css: FONTS_CSS })
  instantApi.requestBlob.mockResolvedValue(new Blob(['png-bytes'], { type: 'image/png' }))
})
afterEach(cleanup)

describe('the instant preview of an image post', () => {
  it('shows the page in a frame that allows its own scripts and nothing else, labelled a preview', async () => {
    renderEditor(IMAGE_POST)
    const frame = await within(preview()).findByTestId('socials-instant-preview', {}, WAIT)
    expect(frame.tagName).toBe('IFRAME')
    expect(frame.getAttribute('sandbox')).toBe('allow-scripts')
    expect(INSTANT_PREVIEW_SANDBOX).toBe('allow-scripts')
    const page = frame.getAttribute('srcdoc') ?? ''
    expect(page.startsWith(`<!doctype html>${PAGE_POLICY_META}`)).toBe(true)
    expect(page).toContain(`${FONTS_CSS}</style>`)
    expect(page).not.toContain(FONTS_MARKER)
    expect(page).toContain('<h1>Open late</h1>')
    expect(asked()).toEqual([{ values: { headline: 'Open late' }, size: '1080x1080', story: false, photos: [] }])
    expect(instantApi.getSocialPreviewFonts).toHaveBeenCalledTimes(1)
    expect(within(frame.parentElement as HTMLElement).getByText(INSTANT_PREVIEW_LABEL)).toBeInTheDocument()
    expect(within(preview()).getByText(INSTANT_PREVIEW_NOTE)).toBeInTheDocument()
    expect(within(preview()).getByTestId('socials-preview-frame').style.aspectRatio).toBe('1080 / 1080')
    expect(within(preview()).queryByText('Not rendered yet')).toBeNull()
  })

  it('never shows a page that does not open with its own policy', async () => {
    instantApi.getSocialTemplatePreviewPage.mockResolvedValue({
      html: `<!doctype html><script>window.top</script>${PAGE_POLICY_META}<h1>Open late</h1>`, fonts_version: 'kit-1',
    })
    renderEditor(IMAGE_POST)
    await waitFor(() => expect(instantApi.getSocialPreviewFonts).toHaveBeenCalled(), WAIT)
    await settle()
    expect(instantFrame()).toBeNull()
    expect(within(preview()).getByText('Not rendered yet')).toBeInTheDocument()
  })

  it('is asked for a moment after the typing stops, once for a burst of typing', async () => {
    renderEditor(IMAGE_POST)
    await waitFor(() => expect(instantApi.getSocialTemplatePreviewPage).toHaveBeenCalledTimes(1), WAIT)
    const headline = within(card('Text on the image')).getByLabelText('Headline')
    fireEvent.change(headline, { target: { value: 'Op' } })
    fireEvent.change(headline, { target: { value: 'Open ear' } })
    fireEvent.change(headline, { target: { value: 'Open early' } })
    expect(instantApi.getSocialTemplatePreviewPage).toHaveBeenCalledTimes(1)
    await waitFor(() => expect(instantApi.getSocialTemplatePreviewPage).toHaveBeenCalledTimes(2), WAIT)
    await settle()
    expect(asked().map((input: any) => input.values.headline)).toEqual(['Open late', 'Open early'])
    await waitFor(() => expect(instantFrame()?.getAttribute('srcdoc')).toContain('<h1>Open early</h1>'), WAIT)
  })

  it('puts the photo the post holds in its spot, read through the post media route as a data: URI', async () => {
    api.listSocialTemplates.mockResolvedValue([PHOTO_TEMPLATE])
    renderEditor(post({
      ...IMAGE_POST, template_id: 'tpl-photo',
      footage: { photo: { prompt: 'Photo: your own picture', status: 'done', toolkit: 'library', name: 'library-1.png' } },
    }))
    const frame = await within(preview()).findByTestId('socials-instant-preview', {}, WAIT)
    expect(instantApi.requestBlob).toHaveBeenCalledWith('/api/socials/posts/post-1/media/library-1.png')
    expect(asked()).toEqual([{ values: { headline: 'Open late' }, size: '1080x1080', story: false, photos: ['photo'] }])
    const page = frame.getAttribute('srcdoc') ?? ''
    expect(page).toContain(`<img data-slot="photo" src="${PHOTO_URI}" alt="">`)
    expect(page).not.toContain(photoMarker('photo'))
  })

  it('a refusal keeps the last page and says why; any other failure keeps today\'s frame', async () => {
    renderEditor(IMAGE_POST)
    await within(preview()).findByTestId('socials-instant-preview', {}, WAIT)
    instantApi.getSocialTemplatePreviewPage.mockRejectedValue(
      Object.assign(new Error(REFUSAL), { status: 422, body: { detail: REFUSAL } }),
    )
    fireEvent.change(within(card('Text on the image')).getByLabelText('Headline'), { target: { value: 'x'.repeat(61) } })
    expect(await within(preview()).findByText(`${PREVIEW_PAUSED} ${REFUSAL}`, {}, WAIT)).toBeInTheDocument()
    expect(instantFrame()?.getAttribute('srcdoc')).toContain('<h1>Open late</h1>')

    instantApi.getSocialTemplatePreviewPage.mockRejectedValue(Object.assign(new Error('Server error'), { status: 500 }))
    fireEvent.change(within(card('Text on the image')).getByLabelText('Headline'), { target: { value: 'Open at six' } })
    await waitFor(() => expect(instantFrame()).toBeNull(), WAIT)
    expect(within(preview()).getByText('Not rendered yet')).toBeInTheDocument()
  })

  it('a story channel asks for the story and the size of the channel on screen', async () => {
    renderEditor(post({ ...IMAGE_POST, targets: [target('twitter', 'image'), target('instagram', 'story')] }))
    await within(preview()).findByTestId('socials-instant-preview', {}, WAIT)
    expect(asked()).toEqual([{ values: { headline: 'Open late' }, size: '1920x1080', story: true, photos: [] }])
  })
})

describe('the posts that keep today\'s preview', () => {
  it('a video asks for no page', async () => {
    renderEditor(post({ format: 'video', template_id: 'tpl-vid', length_seconds: 30, targets: [target('tiktok', 'video')] }))
    await within(preview()).findByText(/render minutes/)
    await settle()
    expect(instantApi.getSocialTemplatePreviewPage).not.toHaveBeenCalled()
    expect(instantFrame()).toBeNull()
  })

  it('a current render shows the rendered file; an edit to its words brings the instant page back', async () => {
    api.getSocialPostMedia.mockResolvedValue([
      { aspect: '1:1', deliverable_id: 'd-square', name: 'image-1x1.png', url: 'https://cdn.test/square.png', content_type: 'image/png', bytes: 10, error: null },
    ])
    renderEditor(post({ ...IMAGE_POST, media: { '1:1': ['d-square'] } }))
    await within(preview()).findByTestId('file-preview')
    await settle()
    expect(instantApi.getSocialTemplatePreviewPage).not.toHaveBeenCalled()
    expect(instantFrame()).toBeNull()

    fireEvent.change(within(card('Text on the image')).getByLabelText('Headline'), { target: { value: 'Open early' } })
    const frame = await within(preview()).findByTestId('socials-instant-preview', {}, WAIT)
    expect(frame.getAttribute('srcdoc')).toContain('<h1>Open early</h1>')
    expect(within(preview()).queryByTestId('file-preview')).toBeNull()
  })
})
