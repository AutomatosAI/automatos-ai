/**
 * PRD-251B US-B305 — AI-made in the editor's Look, and the footage a save asks for.
 *
 * * AI-made lists the template's image slots; a prompt and "Make 4 options" save the draft
 *   first, then ask POST /posts/{id}/ai-options; the four options arrive on the post and one
 *   is picked (PUT /posts/{id}/ai-options/{slot});
 * * a template without an image slot, or a workspace without an AI images tool, says so;
 * * a save keeps every AI image slot the post asks for, with its own prompt, so it never
 *   drops the options made for it; the AI footage switch asks only the hook and b-roll.
 */
import { describe, it, expect, vi, afterEach, beforeEach } from 'vitest'
import { screen, cleanup, fireEvent, waitFor, within } from '@testing-library/react'

vi.mock('@/lib/api-client', async () => (await import('./socials-editor-harness')).apiModule)
vi.mock('sonner', () => ({ toast: { success: vi.fn(), error: vi.fn() } }))
vi.mock('@/components/workspace-provider', () => ({
  useWorkspace: () => ({ workspace: { id: 'w1', role: 'owner', socials: { available: true, enabled: true } } }),
}))
vi.mock('@/hooks/use-authed-image', () => ({ useAuthedImage: () => null }))
vi.mock('@/components/deliverables/socials/socials-voice-picker', () => ({ SocialsVoicePicker: () => null }))
vi.mock('@/components/deliverables/socials/studio/socials-editor-activity', () => ({ SocialsEditorActivity: () => null }))
vi.mock('@/components/widgets/FileWidget/FilePreview', () => ({
  FilePreview: ({ url }: { url: string }) => <div data-testid="file-preview" data-url={url} />,
  inferPreviewType: () => 'image',
}))

import type { SocialTemplateSummary } from '@/lib/api-client'
import { SocialsEditor } from '@/components/deliverables/socials/studio/socials-editor'
import { NO_IMAGE_SLOT } from '@/components/deliverables/socials/studio/editor-look-ai'
import { footageAsked, newDraft } from '@/components/deliverables/socials/studio/editor-model'
import { anyInFlight } from '@/components/deliverables/socials/socials-status'
import { IMAGE_TEMPLATE, VIDEO_TEMPLATE, api, post, renderWith, resetApi } from './socials-editor-harness'

const STILL_TEMPLATE: SocialTemplateSummary = { ...IMAGE_TEMPLATE, id: 'tpl-still', name: 'Photo card', footage_slots: ['hero'], image_slots: ['hero'] }
const STORY_TEMPLATE: SocialTemplateSummary = { ...VIDEO_TEMPLATE, footage_slots: ['broll', 'hero', 'hook'], image_slots: ['hero'] }
const OPTIONS = [1, 2, 3, 4].map((n) => ({
  name: `footage-hero_option_${n}-0000000${n}.png`, prompt: 'A harbour at dawn', toolkit: 'fal_ai', model: 'flux', content_type: 'image/png',
  bytes: 1000, estimate_usd: 0.06, generated_at: '2026-10-14T07:05:00Z',
}))

const go = vi.fn()
const look = () => screen.getByRole('region', { name: 'Look' })
const source = (name: string) => fireEvent.click(within(within(look()).getByRole('group', { name: 'Where the visual comes from' })).getByRole('button', { name }))

beforeEach(() => {
  resetApi()
  go.mockReset()
  api.listSocialTemplates.mockImplementation(async (format?: string) => (format === 'video' ? [STORY_TEMPLATE] : [STILL_TEMPLATE, IMAGE_TEMPLATE]))
  api.getSocialFootageSources.mockResolvedValue({
    kinds: { video: { available: true, toolkit: 'fal_ai', label: 'fal.ai' }, image: { available: true, toolkit: 'fal_ai', label: 'fal.ai' } },
  })
})
afterEach(cleanup)

describe('AI-made in the Look', () => {
  it('asks for four options of an image slot, saving the draft first', async () => {
    renderWith(<SocialsEditor role="owner" post={post({ template_id: 'tpl-still' })} go={go} />)
    source('AI-made')
    const slot = await within(look()).findByRole('group', { name: 'AI image for hero' })
    fireEvent.change(within(slot).getByLabelText('Prompt for hero'), { target: { value: 'A harbour at dawn' } })
    fireEvent.click(within(slot).getByRole('button', { name: /Make 4 options/ }))
    await waitFor(() => expect(api.makeSocialAiOptions).toHaveBeenCalledWith('post-1', 'hero', 'A harbour at dawn'))
    expect(api.updateSocialPost).toHaveBeenCalled()
  })

  it('shows the options made and picks one', async () => {
    const made = post({ template_id: 'tpl-still', footage: { hero: { prompt: 'A harbour at dawn', options: OPTIONS, options_state: 'ready' } } as any })
    renderWith(<SocialsEditor role="owner" post={made} go={go} />)
    source('AI-made')
    const options = await within(look()).findByRole('list', { name: 'Options for hero' })
    expect(within(options).getAllByRole('button')).toHaveLength(4)
    fireEvent.click(within(options).getByRole('button', { name: 'Use option 2' }))
    await waitFor(() => expect(api.pickSocialAiOption).toHaveBeenCalledWith('post-1', 'hero', OPTIONS[1].name))
  })

  it('says why when the options could not be made', async () => {
    const failed = post({ template_id: 'tpl-still', footage: { hero: { prompt: 'x', options: [], options_state: 'failed', options_error: 'The media cap is spent.' } } as any })
    renderWith(<SocialsEditor role="owner" post={failed} go={go} />)
    source('AI-made')
    expect(await within(look()).findByRole('alert')).toHaveTextContent('The media cap is spent.')
  })

  it('a template without an image slot, or no AI images tool, says so', async () => {
    renderWith(<SocialsEditor role="owner" post={post({ template_id: 'tpl-img' })} go={go} />)
    source('AI-made')
    expect(await within(look()).findByText(NO_IMAGE_SLOT)).toBeInTheDocument()
    cleanup()

    api.getSocialFootageSources.mockResolvedValue({ kinds: { image: { available: false, reason: 'No generation toolkit that makes a still is connected.' } } })
    renderWith(<SocialsEditor role="owner" post={post({ template_id: 'tpl-still' })} go={go} />)
    source('AI-made')
    expect(await within(look()).findByText(/No generation toolkit that makes a still is connected/)).toBeInTheDocument()
  })
})

describe('the footage a save asks for', () => {
  it('keeps an AI image slot with its own prompt', async () => {
    const made = post({ template_id: 'tpl-still', footage: { hero: { prompt: 'A harbour at dawn', options: OPTIONS, options_state: 'ready' } } as any })
    renderWith(<SocialsEditor role="owner" post={made} go={go} />)
    await within(look()).findByRole('list', { name: 'Templates' })
    fireEvent.click(screen.getByRole('button', { name: 'Save draft' }))
    await waitFor(() => expect(api.updateSocialPost).toHaveBeenCalled())
    const [, fields] = api.updateSocialPost.mock.calls[0]
    expect(fields.footage).toEqual({ hero: { prompt: 'A harbour at dawn' } })
  })

  it('asks the hook and b-roll from the brief while the switch is on, and keeps the stills', () => {
    const draft = { ...newDraft(), format: 'video', brief: 'Three weeks to Lisbon', footage: { hero: 'A harbour' } }
    expect(footageAsked({ ...draft, footageOn: true }, STORY_TEMPLATE.footage_slots, ['hero'])).toEqual({
      broll: { prompt: 'Three weeks to Lisbon' }, hook: { prompt: 'Three weeks to Lisbon' }, hero: { prompt: 'A harbour' },
    })
    expect(footageAsked({ ...draft, footageOn: false }, STORY_TEMPLATE.footage_slots, ['hero'])).toEqual({ hero: { prompt: 'A harbour' } })
    expect(footageAsked({ ...draft, footageOn: null, footage: { hook: 'x' } }, STORY_TEMPLATE.footage_slots, ['hero'])).toEqual({
      broll: { prompt: 'Three weeks to Lisbon' }, hook: { prompt: 'Three weeks to Lisbon' },
    })
    expect(footageAsked({ ...draft, footageOn: false, footage: {} }, STORY_TEMPLATE.footage_slots, ['hero'])).toBeNull()
  })

  it('the post list polls while options are being made', () => {
    expect(anyInFlight([post({ footage: { hero: { prompt: 'x', options: [], options_state: 'making' } } as any })])).toBe(true)
    expect(anyInFlight([post({ footage: { hero: { prompt: 'x', options: [], options_state: 'ready' } } as any })])).toBe(false)
  })
})
