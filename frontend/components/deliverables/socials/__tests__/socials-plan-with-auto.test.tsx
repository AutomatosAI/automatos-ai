/**
 * PRD-251B (3 Oct 2026 pass) — Plan with Auto.
 *
 * * Plans offers Plan with Auto to whoever can author posts; it opens a new plan.
 * * A new plan starts with the card: the person's words (a greyed example shows the kind of
 *   thing), what Auto may read, and Draft my plan, which waits for words. Auto's draft fills
 *   the steps, its notes say what it could not use, and its ideas wait on the Content bank
 *   step, where any can be left out.
 * * Saving creates the plan with what the form says, adds the kept ideas to its bank in
 *   Auto's order, starts research, says so, and opens the plan. An idea the bank refuses,
 *   or research that cannot start, is said with the server's reason; the plan still opens.
 * * When Auto cannot draft, the server's reason is said and the form is left as it was.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import type { ReactElement } from 'react'

const state = vi.hoisted(() => ({ go: vi.fn() }))

vi.mock('sonner', () => ({ toast: { success: vi.fn(), error: vi.fn() } }))
vi.mock('@/components/workspace-provider', () => ({
  useWorkspace: () => ({ workspace: { id: 'w1', name: 'Salon', role: 'owner', socials: { available: true, enabled: true } } }),
}))
vi.mock('@/components/deliverables/socials/socials-campaigns-view', () => ({ SocialsCampaigns: () => <div data-testid="campaigns" /> }))
vi.mock('@/lib/api-client', () => {
  const kind = (k: string) => ({ kind: k, available: true, reason: null, needs_public_storage: false })
  const apiClient = {
    listSocialPlans: vi.fn(async () => ({ plans: [], total: 0 })),
    getSocialPlan: vi.fn(async () => undefined),
    createSocialPlan: vi.fn(async (input: any) => ({ ...input, id: 'new-plan-id' })),
    updateSocialPlan: vi.fn(async (id: string, input: any) => ({ ...input, id })),
    draftSocialPlan: vi.fn(),
    listSocialChannels: vi.fn(async () => [
      { toolkit: 'instagram', label: 'Instagram', post_kinds: [kind('image')], verified: true, setup_note: null, copy_limits: { text: 2200 } },
    ]),
    listSocialTemplates: vi.fn(async () => []),
    listSocialPlanTopics: vi.fn(async () => ({ topics: [], total: 0, unused: 0 })),
    addSocialPlanTopic: vi.fn(async (_id: string, input: any) => ({ ...input, id: 't-new' })),
    researchSocialPlan: vi.fn(async () => ({ plan_id: 'new-plan-id', execution_id: 'research-1' })),
  }
  return { apiClient, default: apiClient }
})

import { toast } from 'sonner'
import { apiClient } from '@/lib/api-client'
import { PLAN_WITH_AUTO_LEAD, SocialsPlansView } from '@/components/deliverables/socials/plans/socials-plans-view'
import { AUTO_EXAMPLE, adoptedNotes } from '@/components/deliverables/socials/plans/plan-auto-model'

const api = apiClient as unknown as Record<string, ReturnType<typeof vi.fn>>
const WORDS = 'Three posts this week about our autumn colour offer, and a customer review.'
const OFFER = { title: '20% off colour, Monday to Thursday', angle: 'The offer, plainly', formats: ['image'] }
const REVIEW = { title: "Sarah's review", angle: null, formats: [] }
const DRAFT = {
  plan: {
    name: 'Autumn colour week', goal: 'Fill the colour chairs on quiet weekdays.', audience: 'Local regulars',
    starts_on: '2026-10-14', ends_on: '2026-10-20', timezone: 'Europe/London',
    cadence: [{ channels: ['instagram'], format: 'image', length_seconds: null, template_id: null, days: ['mon', 'wed', 'fri'], time: '10:00' }],
    sources: { knowledge: true, website: false, deliverables: true, github: false, notes: `What the person asked for: ${WORDS}`, never_say: [] },
  },
  topics: [OFFER, REVIEW],
  warnings: ['Row 1: tiktok is not connected here, so it was left out.'],
}

function renderWithClient(ui: ReactElement) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  return render(<QueryClientProvider client={client}>{ui}</QueryClientProvider>)
}

async function draftNewPlan() {
  renderWithClient(<SocialsPlansView role="owner" posts={[]} planId="new" go={state.go} />)
  const card = screen.getByRole('region', { name: 'Plan with Auto' })
  fireEvent.change(within(card).getByLabelText('What should your socials do?'), { target: { value: WORDS } })
  fireEvent.click(within(card).getByRole('button', { name: 'Draft my plan' }))
  await screen.findByDisplayValue('Autumn colour week')
  return card
}

const save = () => fireEvent.click(screen.getAllByRole('button', { name: 'Save plan' })[0])

beforeEach(() => {
  vi.useFakeTimers({ toFake: ['Date'] })
  vi.setSystemTime(new Date('2026-10-14T07:20:00Z'))
  state.go.mockReset()
  Object.values(api).forEach((fn) => fn.mockClear())
  api.draftSocialPlan.mockImplementation(async () => DRAFT)
  vi.mocked(toast.success).mockClear()
  vi.mocked(toast.error).mockClear()
})
afterEach(() => { cleanup(); vi.useRealTimers() })

describe('Plans offers Plan with Auto', () => {
  it('to whoever can author posts, and it opens a new plan', async () => {
    renderWithClient(<SocialsPlansView role="owner" posts={[]} planId={null} go={state.go} />)
    const start = screen.getByRole('region', { name: 'Plan with Auto' })
    expect(start).toHaveTextContent(PLAN_WITH_AUTO_LEAD)
    fireEvent.click(within(start).getByRole('button', { name: 'Plan with Auto' }))
    expect(state.go).toHaveBeenCalledWith({ view: 'plans', plan: 'new', post: null })
    cleanup()
    renderWithClient(<SocialsPlansView role="viewer" posts={[]} planId={null} go={state.go} />)
    expect(screen.queryByRole('region', { name: 'Plan with Auto' })).toBeNull()
  })
})

describe('a new plan drafted by Auto', () => {
  it('drafts from the words and the sources picked, and fills the steps', async () => {
    renderWithClient(<SocialsPlansView role="owner" posts={[]} planId="new" go={state.go} />)
    const card = screen.getByRole('region', { name: 'Plan with Auto' })
    const words = within(card).getByLabelText('What should your socials do?')
    expect(words).toHaveAttribute('placeholder', AUTO_EXAMPLE)
    expect(within(card).getByRole('button', { name: 'Draft my plan' })).toBeDisabled()
    fireEvent.change(words, { target: { value: WORDS } })
    fireEvent.click(within(card).getByRole('checkbox', { name: /Your website/ }))
    fireEvent.click(within(card).getByRole('button', { name: 'Draft my plan' }))
    await waitFor(() => expect(api.draftSocialPlan).toHaveBeenCalledWith({
      request: WORDS, timezone: expect.any(String), sources: { knowledge: true, website: false, deliverables: true },
    }))
    expect(await screen.findByDisplayValue('Autumn colour week')).toBeInTheDocument()
    expect(screen.getByDisplayValue('Fill the colour chairs on quiet weekdays.')).toBeInTheDocument()
    expect(screen.getByRole('region', { name: "Auto's notes" })).toHaveTextContent('tiktok is not connected here')
    expect(within(card).getByRole('button', { name: 'Draft again' })).toBeEnabled()
    fireEvent.click(screen.getByRole('button', { name: 'Content bank' }))
    const ideas = screen.getByRole('region', { name: "Auto's ideas" })
    expect(within(ideas).getAllByRole('article').map((a) => a.getAttribute('aria-label'))).toEqual([OFFER.title, REVIEW.title])
  })

  it('saves the plan, adds the kept ideas, starts research, says so and opens the plan', async () => {
    await draftNewPlan()
    fireEvent.click(screen.getByRole('button', { name: 'Content bank' }))
    fireEvent.click(within(screen.getByRole('article', { name: REVIEW.title })).getByRole('button', { name: 'Leave out' }))
    expect(screen.queryByRole('article', { name: REVIEW.title })).toBeNull()
    save()
    await waitFor(() => expect(state.go).toHaveBeenCalledWith({ view: 'plans', plan: 'new-plan-id', post: null }))
    const input = api.createSocialPlan.mock.calls[0][0]
    expect(input).toMatchObject({ name: 'Autumn colour week', starts_on: '2026-10-14', ends_on: '2026-10-20', timezone: 'Europe/London' })
    expect(input.sources).toMatchObject({ knowledge: true, website: false, deliverables: true, notes: `What the person asked for: ${WORDS}` })
    expect(input.cadence).toEqual([DRAFT.plan.cadence[0]])
    expect(input.make).toMatchObject({ rhythm: 'weekly', batch_day: 'sun' })  // PRD-251C: Auto drafts a weekly plan
    expect(api.addSocialPlanTopic.mock.calls).toEqual([['new-plan-id', { ...OFFER, facts: [] }]])
    expect(api.researchSocialPlan).toHaveBeenCalledWith('new-plan-id')
    expect(toast.success).toHaveBeenCalledWith("Plan saved. 1 of Auto's ideas is in its content bank. Research has started on the sources you picked.")
  })

  it('says which idea the bank refused and that research did not start, and still opens the plan', async () => {
    api.addSocialPlanTopic.mockRejectedValueOnce(new Error('The bank already holds that title'))
    api.researchSocialPlan.mockRejectedValueOnce(new Error('Research is already running.'))
    await draftNewPlan()
    save()
    await waitFor(() => expect(state.go).toHaveBeenCalledWith({ view: 'plans', plan: 'new-plan-id', post: null }))
    expect(api.addSocialPlanTopic).toHaveBeenCalledTimes(2)
    expect(toast.error).toHaveBeenCalledWith(`The content bank refused "${OFFER.title}": The bank already holds that title`)
    expect(toast.error).toHaveBeenCalledWith('Research did not start: Research is already running. Start it from the Content bank with Research again.')
    expect(toast.success).toHaveBeenCalledWith("Plan saved. 1 of Auto's ideas is in its content bank.")
  })

  it('says why when Auto cannot draft, and leaves the form as it was', async () => {
    api.draftSocialPlan.mockImplementationOnce(async () => { throw new Error('Auto did not answer within 60 seconds. Try again.') })
    renderWithClient(<SocialsPlansView role="owner" posts={[]} planId="new" go={state.go} />)
    const card = screen.getByRole('region', { name: 'Plan with Auto' })
    fireEvent.change(within(card).getByLabelText('What should your socials do?'), { target: { value: WORDS } })
    fireEvent.click(within(card).getByRole('button', { name: 'Draft my plan' }))
    await waitFor(() => expect(toast.error).toHaveBeenCalledWith('Auto did not answer within 60 seconds. Try again.'))
    expect(screen.getByLabelText('Name')).toHaveValue('')
    expect(screen.queryByRole('region', { name: "Auto's notes" })).toBeNull()
  })

  it('a plan with no ideas and no sources is only saved', () => {
    expect(adoptedNotes({ added: 0, refused: [], researching: false, researchError: null })).toEqual({ said: 'Plan saved.', warned: [] })
    expect(adoptedNotes({ added: 3, refused: [], researching: false, researchError: null }).said).toBe("Plan saved. 3 of Auto's ideas are in its content bank.")
  })
})
