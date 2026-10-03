/**
 * PRD-251B (3 Oct 2026 pass) — Plan with Auto's model (pure): the words on the card (a greyed
 * example shows the kind of thing to say), the sources Auto may read, Auto's draft as the Plan
 * page's form, the ideas it heard as content bank topics, and what the person is told once
 * the plan is saved.
 */
import type { SocialPlanAutoDraft, SocialPlanAutoSources, SocialPlanAutoTopic, SocialTopicInput } from '@/lib/socials-plan-types'
import type { PlanDraft } from './plan-model'

export const AUTO_TITLE = 'Plan with Auto'
export const AUTO_LEAD =
  'Say what your socials should do, in your own words. Auto drafts the plan: its dates, which channel posts what and when, ' +
  'and the post ideas it heard. You check each step, change anything, and save.'
export const AUTO_LABEL = 'What should your socials do?'
export const AUTO_EXAMPLE =
  'e.g. Three posts this week on Instagram for the salon: our autumn colour offer (20% off, Monday to Thursday), ' +
  'a review from a happy customer, and a photo of the new treatment room. Friendly, not salesy.'
export const AUTO_SOURCES_LABEL = 'Auto may read'
export const AUTO_SOURCES: ReadonlyArray<{ key: keyof SocialPlanAutoSources; name: string; what: string }> = [
  { key: 'knowledge', name: 'Your knowledge', what: 'the documents in this workspace' },
  { key: 'website', name: 'Your website', what: 'the site in your brand kit' },
  { key: 'deliverables', name: 'Your Deliverables', what: 'blog posts and reports' },
]
export const ALL_AUTO_SOURCES: SocialPlanAutoSources = { knowledge: true, website: true, deliverables: true }
export const DRAFTING = 'Auto is drafting your plan…'
export const DRAFTING_NOTE = 'This can take up to a minute.'
export const REDRAFT_NOTE = 'Drafting again replaces the name, goal, dates, cadence, sources and ideas.'
export const DRAFTED_NOTE = 'Auto drafted this plan from what you said. Check each step, change anything, then save.'
export const SUGGESTIONS_TITLE = "Auto's ideas"
export const SUGGESTIONS_LEAD = 'These join the content bank when you save; research adds facts from the sources you picked.'
export const RESEARCH_STARTED_NOTE = 'Research has started on the sources you picked.'
export const RESEARCH_AGAIN_NOTE = 'Start it from the Content bank with Research again.'

/** The card's button: what it does now. */
export function draftButtonLabel(drafting: boolean, drafted: boolean): string {
  if (drafting) return DRAFTING
  return drafted ? 'Draft again' : 'Draft my plan'
}

/** The line beside the button: the wait while Auto drafts, then what drafting again replaces. */
export function draftStatusNote(drafting: boolean, drafted: boolean): string {
  if (drafting) return DRAFTING_NOTE
  return drafted ? REDRAFT_NOTE : ''
}

/** The fields Auto drafts, as the form holds them; the person's other choices stay as they are. */
export function autoChanges(auto: SocialPlanAutoDraft): Partial<PlanDraft> {
  const { plan } = auto
  return {
    name: plan.name, goal: plan.goal, audience: plan.audience,
    startsOn: plan.starts_on, endsOn: plan.ends_on, timezone: plan.timezone,
    cadence: plan.cadence.map((row) => ({
      channels: [...row.channels], format: row.format, lengthSeconds: row.length_seconds,
      templateId: row.template_id, days: [...row.days], time: row.time,
    })),
    sources: { knowledge: plan.sources.knowledge, deliverables: plan.sources.deliverables, website: plan.sources.website, github: plan.sources.github, notes: plan.sources.notes },
  }
}

/** Auto's ideas as the content bank takes them: no facts yet, research brings those. */
export function topicInputs(topics: ReadonlyArray<SocialPlanAutoTopic>): SocialTopicInput[] {
  return topics.map((topic) => ({ title: topic.title, angle: topic.angle, formats: [...topic.formats], facts: [] }))
}

/** Research starts on save when the plan reads any source. */
export function researchAsked(sources: PlanDraft['sources']): boolean {
  return sources.knowledge || sources.website || sources.deliverables || sources.github
}

/** What saving Auto's draft did: the plan always; each idea the bank refused, and why; research. */
export interface AdoptedDraft {
  added: number
  refused: string[]
  researching: boolean
  researchError: string | null
}

/** What the person is told: the saved line, then each refusal (an idea, research) on its own. */
export function adoptedNotes(adopted: AdoptedDraft): { said: string; warned: string[] } {
  const ideas = adopted.added === 1 ? "1 of Auto's ideas is" : `${adopted.added} of Auto's ideas are`
  const said = ['Plan saved.', adopted.added ? `${ideas} in its content bank.` : '', adopted.researching ? RESEARCH_STARTED_NOTE : '']
  const warned = adopted.refused.map((reason) => `The content bank refused ${reason}`)
  const research = adopted.researchError ? [`Research did not start: ${adopted.researchError.replace(/\.$/, '')}. ${RESEARCH_AGAIN_NOTE}`] : []
  return { said: said.filter(Boolean).join(' '), warned: [...warned, ...research] }
}
