/**
 * PRD-251B US-B207 — the Plans view and the Plan page (Plan.dc.html).
 *
 * * Plans lists each plan with its state, dates, cadence and bank; Open plan goes to its page.
 * * A new plan: five steps; Save waits for a name and a channel on every row; it creates the
 *   plan with what the form says, then opens the plan as itself.
 * * The content bank says to save first on a new plan; on a saved one it lists the topics and
 *   adds one through the bank (the server refuses with its reason), and says when research
 *   cannot run (PRD-251C US-C101).
 * * The editor's Music picker saves the post's music (a render setting).
 * * An owner or admin deletes a plan from its page after one question; an editor sees no Delete.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import type { ReactElement } from 'react'

const state = vi.hoisted(() => ({ go: vi.fn(), plans: [] as any[], topics: [] as any[], researchNote: null as string | null }))

vi.mock('sonner', () => ({ toast: { success: vi.fn(), error: vi.fn() } }))
vi.mock('@/components/workspace-provider', () => ({
  useWorkspace: () => ({ workspace: { id: 'w1', name: 'Acme', role: 'owner', socials: { available: true, enabled: true } } }),
}))
vi.mock('@/components/deliverables/socials/socials-campaigns-view', () => ({ SocialsCampaigns: () => <div data-testid="campaigns" /> }))
vi.mock('@/lib/api-client', () => {
  const kind = (k: string) => ({ kind: k, available: true, reason: null, needs_public_storage: false })
  const apiClient = {
    listSocialPlans: vi.fn(async () => ({ plans: state.plans, total: state.plans.length })),
    getSocialPlan: vi.fn(async (id: string) => state.plans.find((p) => p.id === id)),
    createSocialPlan: vi.fn(async (input: any) => ({ ...input, id: 'new-plan-id' })),
    updateSocialPlan: vi.fn(async (id: string, input: any) => ({ ...input, id })),
    listSocialChannels: vi.fn(async () => [
      { toolkit: 'twitter', label: 'X', post_kinds: [kind('image')], verified: true, setup_note: null, copy_limits: { text: 280 } },
      { toolkit: 'linkedin', label: 'LinkedIn', post_kinds: [kind('image')], verified: true, setup_note: null, copy_limits: { text: 3000 } },
    ]),
    listSocialTemplates: vi.fn(async () => []),
    listSocialPlanTopics: vi.fn(async () => ({ topics: state.topics, total: state.topics.length, unused: state.topics.length, research_note: state.researchNote })),
    addSocialPlanTopic: vi.fn(async (_id: string, input: any) => ({ ...input, id: 't-new' })),
    researchSocialPlan: vi.fn(async () => ({ execution_id: 'research-1' })),
    deleteSocialPlan: vi.fn(async () => undefined),
    listSocialMusic: vi.fn(async () => ({ available: true, tracks: [{ id: 'spring-of-2026', title: 'Spring of 2026', artist: 'S', style: 'tropical house', duration: 120, licence: 'CC BY 4.0', credit_required: true }] })),
    updateSocialPost: vi.fn(async (id: string, changes: any) => ({ id, ...changes })),
  }
  return { apiClient, default: apiClient }
})

import { apiClient } from '@/lib/api-client'
import { SocialsPlansView } from '@/components/deliverables/socials/plans/socials-plans-view'
import { SAVE_FIRST } from '@/components/deliverables/socials/plans/plan-step-bank'
import { PLAN_DELETE_CONFIRM } from '@/components/deliverables/socials/plans/plan-page'
import { SocialsMusicPicker } from '@/components/deliverables/socials/socials-music-picker'

const api = apiClient as unknown as Record<string, ReturnType<typeof vi.fn>>
const PLAN = {
  id: 'p1', name: 'Countdown', kind: 'plan', status: 'active', goal: 'Fill the stand', audience: null,
  starts_on: '2026-10-05', ends_on: '2026-11-08', timezone: 'Europe/London',
  cadence: [{ id: 'r1', channels: ['twitter'], format: 'image', length_seconds: null, template_id: null, days: ['mon', 'wed', 'fri'], time: '09:00' }],
  sources: { knowledge: true, deliverables: true, website: true, github: false, notes: '', never_say: [] },
  make: { time: '07:00', video_days_early: 1, image_days_early: 0, max_per_day: null, visual_mix: { templates: 100 } },
  research: { enabled: true, day: 'mon', time: '06:00' }, late_policy: 'skip', approval_mode: 'per_post', slot_overrides: {},
  bank: { topics: 2, unused: 1 },
}

function renderWithClient(ui: ReactElement) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  return render(<QueryClientProvider client={client}>{ui}</QueryClientProvider>)
}

beforeEach(() => {
  vi.useFakeTimers({ toFake: ['Date'] })
  vi.setSystemTime(new Date('2026-10-14T07:20:00Z'))
  state.go.mockReset()
  state.plans = [PLAN]
  state.topics = []
  state.researchNote = null
  Object.values(api).forEach((fn) => fn.mockClear())
})
afterEach(() => { cleanup(); vi.useRealTimers() })

describe('the Plans view', () => {
  it('lists each plan with its state, cadence and bank, and opens it', async () => {
    renderWithClient(<SocialsPlansView role="owner" posts={[]} planId={null} go={state.go} />)
    const card = await screen.findByRole('article', { name: 'Countdown' })
    expect(card).toHaveTextContent('Running · day 10 of 35')
    expect(card).toHaveTextContent('X image · mon, wed, fri at 09:00')
    expect(card).toHaveTextContent('Content bank: 2 topics, 1 unused')
    fireEvent.click(within(card).getByRole('button', { name: 'Open plan' }))
    expect(state.go).toHaveBeenCalledWith({ view: 'plans', plan: 'p1', post: null })
    expect(screen.getByTestId('campaigns')).toBeInTheDocument()
  })
})

describe('a new plan', () => {
  it('saves once it has a name and a channel, with what the form says, then opens as itself', async () => {
    renderWithClient(<SocialsPlansView role="owner" posts={[]} planId="new" go={state.go} />)
    const save = () => screen.getAllByRole('button', { name: 'Save plan' })[0]
    expect(screen.getByRole('navigation', { name: 'Plan steps' })).toHaveTextContent('Content bank')
    expect(save()).toBeDisabled()
    fireEvent.change(screen.getByLabelText('Name'), { target: { value: 'Launch week' } })
    fireEvent.click(screen.getByRole('button', { name: 'Cadence' }))
    fireEvent.click(await screen.findByRole('button', { name: 'LinkedIn' }))
    expect(screen.getByRole('region', { name: 'Cadence summary' })).toHaveTextContent('posts (')
    expect(save()).toBeEnabled()
    fireEvent.click(save())
    await waitFor(() => expect(api.createSocialPlan).toHaveBeenCalled())
    const input = api.createSocialPlan.mock.calls[0][0]
    expect(input).toMatchObject({ name: 'Launch week', timezone: expect.any(String), late_policy: 'skip', starts_on: '2026-10-14', ends_on: '2026-11-17' })
    expect(input.cadence).toEqual([{ channels: ['linkedin'], format: 'image', length_seconds: null, template_id: null, days: ['mon', 'tue', 'wed', 'thu', 'fri'], time: '09:00' }])
    await waitFor(() => expect(state.go).toHaveBeenCalledWith({ view: 'plans', plan: 'new-plan-id', post: null }))
  })

  it('the content bank waits for the plan to exist', () => {
    renderWithClient(<SocialsPlansView role="owner" posts={[]} planId="new" go={state.go} />)
    fireEvent.click(screen.getByRole('button', { name: 'Content bank' }))
    expect(screen.getByText(SAVE_FIRST)).toBeInTheDocument()
  })
})

describe('a saved plan', () => {
  it('opens on its cadence, and its bank adds a topic through the server', async () => {
    state.topics = [{ id: 't1', plan_id: 'p1', title: 'Three weeks to Lisbon', angle: 'Why visit', facts: [{ text: 'Stand B12.', source: { kind: 'web', ref: 'https://x.test', label: 'Stand page' } }], formats: ['image'], pinned_on: null, used_at: null, used_post_id: null, origin: 'research' }]
    renderWithClient(<SocialsPlansView role="owner" posts={[]} planId="p1" go={state.go} />)
    expect(await screen.findByRole('region', { name: 'Cadence' })).toBeInTheDocument()
    expect(await screen.findByRole('button', { name: 'Pause plan' })).toBeInTheDocument()
    fireEvent.click(screen.getByRole('button', { name: 'Content bank' }))
    const topic = await screen.findByRole('article', { name: 'Three weeks to Lisbon' })
    expect(topic).toHaveTextContent('Stand page · web')
    fireEvent.click(screen.getByRole('button', { name: 'Add a topic' }))
    const form = screen.getByRole('form', { name: 'Add a topic' })
    fireEvent.change(within(form).getByLabelText('Title'), { target: { value: 'The roadmap' } })
    fireEvent.click(within(form).getByRole('button', { name: 'Add to the bank' }))
    await waitFor(() => expect(api.addSocialPlanTopic).toHaveBeenCalledWith('p1', { title: 'The roadmap', angle: null, formats: [], facts: [] }))
    fireEvent.click(screen.getByRole('button', { name: 'Research again' }))
    await waitFor(() => expect(api.researchSocialPlan).toHaveBeenCalledWith('p1'))
    expect(screen.queryByRole('status', { name: 'Research' })).not.toBeInTheDocument()  // research can run: nothing to say
  })

  it('its bank says when research cannot run, in the server\'s words', async () => {
    state.researchNote = 'Research is not set up in this workspace yet. Research again sets it up and runs it.'
    renderWithClient(<SocialsPlansView role="owner" posts={[]} planId="p1" go={state.go} />)
    fireEvent.click(await screen.findByRole('button', { name: 'Content bank' }))
    expect(await screen.findByRole('status', { name: 'Research' })).toHaveTextContent('Research is not set up in this workspace yet. Research again sets it up and runs it.')
  })
})

describe('deleting a plan', () => {
  it('an owner deletes it from its page after one question, then the plans list opens', async () => {
    renderWithClient(<SocialsPlansView role="owner" posts={[]} planId="p1" go={state.go} />)
    fireEvent.click(await screen.findByRole('button', { name: 'Delete' }))
    const ask = screen.getByRole('alertdialog', { name: 'Delete plan' })
    expect(ask).toHaveTextContent(PLAN_DELETE_CONFIRM)
    expect(api.deleteSocialPlan).not.toHaveBeenCalled()
    fireEvent.click(within(ask).getByRole('button', { name: 'Delete plan' }))
    await waitFor(() => expect(api.deleteSocialPlan).toHaveBeenCalledWith('p1'))
    await waitFor(() => expect(state.go).toHaveBeenCalledWith({ view: 'plans', plan: null, post: null }))
  })

  it('an editor sees no Delete', async () => {
    renderWithClient(<SocialsPlansView role="editor" posts={[]} planId="p1" go={state.go} />)
    await screen.findByRole('button', { name: 'Pause plan' })
    expect(screen.queryByRole('button', { name: 'Delete' })).toBeNull()
  })
})

describe('the Music picker', () => {
  it('saves the post music: another library track, or none', async () => {
    renderWithClient(<SocialsMusicPicker post={{ id: 'post-1', music: null } as any} editable />)
    const select = await screen.findByLabelText('Music')
    await screen.findByRole('option', { name: 'Library · Tropical house' })
    fireEvent.change(select, { target: { value: 'spring-of-2026' } })
    await waitFor(() => expect(api.updateSocialPost).toHaveBeenCalledWith('post-1', { music: { track: 'spring-of-2026' } }))
    fireEvent.change(select, { target: { value: '__none__' } })
    await waitFor(() => expect(api.updateSocialPost).toHaveBeenLastCalledWith('post-1', { music: { track: null } }))
  })
})
