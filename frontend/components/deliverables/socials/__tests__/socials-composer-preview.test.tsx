/**
 * PRD-251 S2.2b (US-208) — the composer's variables and live preview, against a
 * mocked apiClient: fields come from the template's variables_schema; a claim
 * shows Unsourced until a source is picked; a video's render asks for the preview
 * (preview: true) and shows it through FilePreview; an edit marks it stale.
 */
import { describe, it, expect, vi, afterEach, beforeEach } from 'vitest'
import { render, screen, cleanup, fireEvent, waitFor, within } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'

const api = vi.hoisted(() => ({
  listSocialChannels: vi.fn(),
  composeSocialPost: vi.fn(),
  createSocialPost: vi.fn(),
  updateSocialPost: vi.fn(),
  setSocialPostTargets: vi.fn(),
  renderSocialPost: vi.fn(),
  getSocialPost: vi.fn(),
  searchSocialSources: vi.fn(),
  getSocialPostMedia: vi.fn(),
}))
const previews = vi.hoisted(() => [] as Array<{ url?: string; previewType?: string }>)

vi.mock('sonner', () => ({ toast: { success: vi.fn(), error: vi.fn() } }))
vi.mock('@/lib/api-client', () => ({ apiClient: api, default: api }))
vi.mock('@/components/workspace-provider', () => ({
  useWorkspace: () => ({ workspace: { id: 'w1', role: 'owner', socials: { available: true, enabled: true } } }),
}))
vi.mock('@/components/widgets/FileWidget/FilePreview', async (importOriginal) => {
  const actual = await importOriginal<typeof import('@/components/widgets/FileWidget/FilePreview')>()
  return {
    ...actual,
    FilePreview: (props: { url?: string; previewType?: string }) => {
      previews.push(props)
      return <div data-testid="file-preview">{props.url}</div>
    },
  }
})

import { SocialsComposer } from '@/components/deliverables/socials/socials-composer'
import type { SocialComposeProposal, SocialPost } from '@/lib/api-client'

const SCHEMA = {
  headline: { type: 'text' as const, label: 'Headline', max_chars: 60 },
  members: { type: 'number' as const, label: 'Members', claim: true },
  show_logo: { type: 'boolean' as const, label: 'Show the logo', default: true },
}
const PROPOSAL: SocialComposeProposal = {
  title: 'Harvest Club', copy: { base: 'Opens Friday.', per_channel: {} }, format: 'video', template_id: 'tpl-v',
  template: { id: 'tpl-v', name: 'Countdown', format: 'social_video', sizes: ['1080x1920'], variables_schema: SCHEMA },
  variables: { headline: { value: 'Opens Friday', claim: false } }, sources: {}, channels: [], warnings: [],
}
const HASH = 'h'.repeat(64)
let saved: SocialPost

function renderComposer() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false }, mutations: { retry: false } } })
  return render(
    <QueryClientProvider client={client}>
      <SocialsComposer onDone={vi.fn()} />
    </QueryClientProvider>,
  )
}

async function toVariables() {
  renderComposer()
  await waitFor(() => expect(api.listSocialChannels).toHaveBeenCalled())
  fireEvent.change(screen.getByLabelText('Brief'), { target: { value: 'A countdown for Harvest Club' } })
  fireEvent.click(screen.getByRole('button', { name: /Draft it/ }))
  fireEvent.click(await screen.findByRole('button', { name: /^Next: variables/ }))
  return screen.getByRole('region', { name: 'Variables' })
}

beforeEach(() => {
  Object.values(api).forEach((fn) => fn.mockReset())
  previews.length = 0
  saved = { id: 'post-1', status: 'draft', content_hash: HASH, media: {}, preview: null } as unknown as SocialPost
  api.listSocialChannels.mockResolvedValue([])
  api.composeSocialPost.mockResolvedValue(PROPOSAL)
  api.createSocialPost.mockImplementation(async () => saved)
  api.updateSocialPost.mockImplementation(async () => saved)
  api.setSocialPostTargets.mockImplementation(async () => saved)
  api.searchSocialSources.mockResolvedValue({
    candidates: [{ kind: 'metric', ref: 'members', title: 'Members', as_of: '2026-09-28T00:00:00Z' }], total: 1,
  })
  api.renderSocialPost.mockImplementation(async () => {
    saved = { ...saved, preview: { status: 'rendering', content_hash: HASH, files: [], error: null, at: '' } } as SocialPost
    return saved
  })
  api.getSocialPost.mockImplementation(async () => ({
    ...saved,
    preview: {
      status: 'done', content_hash: HASH, error: null, at: '',
      files: [{ name: 'preview-video-9x16.mp4', url: '/api/socials/posts/post-1/media/preview-video-9x16.mp4', content_type: 'video/mp4', width: 540, height: 960 }],
    },
  }))
})
afterEach(() => cleanup())

describe('variables (US-208)', () => {
  it('a field for each variable of the schema, of its type, holding the proposal values', async () => {
    const form = await toVariables()
    expect(within(form).getByLabelText('Headline')).toHaveValue('Opens Friday')
    expect(within(form).getByLabelText('Members')).toHaveAttribute('type', 'number')
    expect(within(form).getByRole('checkbox')).toBeInTheDocument()
    fireEvent.click(within(form).getByRole('button', { name: 'JSON view' }))
    expect(JSON.parse((within(form).getByLabelText('Variables as JSON') as HTMLTextAreaElement).value)).toEqual(PROPOSAL.variables)
  })

  it('a claim shows Unsourced until a source is picked', async () => {
    const form = await toVariables()
    const claim = within(form).getByTestId('socials-claim-source-members')
    expect(within(claim).getByText('Unsourced')).toBeInTheDocument()
    fireEvent.click(within(claim).getByRole('button', { name: 'Pick a source' }))
    fireEvent.click(await within(claim).findByRole('button', { name: /Members/ }))
    expect(within(claim).queryByText('Unsourced')).toBeNull()
    expect(claim).toHaveTextContent('Source: metric members')
  })
})

describe('the live preview (US-208)', () => {
  it('saves the draft, asks for the preview (preview: true) and shows it through FilePreview', async () => {
    await toVariables()
    fireEvent.change(screen.getByLabelText('Members'), { target: { value: '1200' } })
    const pane = screen.getByRole('region', { name: 'Preview' })
    fireEvent.click(within(pane).getByRole('button', { name: /Render a preview/ }))
    await waitFor(() => expect(api.renderSocialPost).toHaveBeenCalledWith('post-1', { preview: true }))
    expect(api.createSocialPost).toHaveBeenCalledWith(expect.objectContaining({
      variables: { headline: { value: 'Opens Friday', claim: false }, members: { value: 1200, claim: true } },
    }))
    await within(pane).findByTestId('file-preview')
    expect(previews.at(-1)).toMatchObject({ url: '/api/socials/posts/post-1/media/preview-video-9x16.mp4', previewType: 'video' })
    expect(within(pane).queryByText('Preview out of date — render again')).toBeNull()
  })

  it('an edit after the render marks it stale; a second render updates the saved post, never creates another', async () => {
    await toVariables()
    const pane = screen.getByRole('region', { name: 'Preview' })
    fireEvent.click(within(pane).getByRole('button', { name: /Render a preview/ }))
    await within(pane).findByTestId('file-preview')

    fireEvent.change(screen.getByLabelText('Headline'), { target: { value: 'Opens this Friday' } })
    expect(within(pane).getByText('Preview out of date — render again')).toBeInTheDocument()

    fireEvent.click(within(pane).getByRole('button', { name: /Render again/ }))
    await waitFor(() => expect(api.renderSocialPost).toHaveBeenCalledTimes(2))
    expect(api.createSocialPost).toHaveBeenCalledTimes(1)
    expect(api.updateSocialPost).toHaveBeenCalledWith('post-1', expect.objectContaining({
      variables: expect.objectContaining({ headline: { value: 'Opens this Friday', claim: false } }),
    }))
    await waitFor(() => expect(within(pane).queryByText('Preview out of date — render again')).toBeNull())
  })

  it('an image renders for real, never as a preview', async () => {
    api.composeSocialPost.mockResolvedValue({
      ...PROPOSAL, format: 'image', template: { ...PROPOSAL.template!, format: 'social_image' },
    })
    await toVariables()
    fireEvent.click(within(screen.getByRole('region', { name: 'Preview' })).getByRole('button', { name: /^Render$/ }))
    await waitFor(() => expect(api.renderSocialPost).toHaveBeenCalledWith('post-1', {}))
  })
})
