/**
 * PRD-251B US-B207 and US-B208 — the Plan page's model and a plan's slots on the calendar (pure).
 *
 * * "How often" reads a row's days back as its preset; a weekly row keeps its day.
 * * A plan becomes the form's draft and the draft the plan's input, unchanged.
 * * The cadence summary counts each channel's posts over the plan's days and the render
 *   minutes its videos need; the status line says where the plan is.
 * * A planned slot's id names its plan and key; only slots not made become chips.
 * * The music picker's three choices map to the post's music and back.
 * * PRD-251C: the repeat window round-trips, and the form keeps it within what the server takes.
 */
import { describe, expect, it } from 'vitest'

import type { SocialPlan } from '@/lib/socials-plan-types'
import {
  cadenceSummary, daysFor, draftFromPlan, emptyDraft, inputFromDraft, missingFields, mixOf, oftenOf, repeatDays, statusLine,
} from '@/components/deliverables/socials/plans/plan-model'
import {
  parsePlannedId, plannedEvents, plannedId, plannedSlots, planRailLine,
} from '@/components/deliverables/socials/studio/plan-calendar-model'
import { musicFrom, musicValue, trackLabel } from '@/components/deliverables/socials/socials-music-picker'

const PLAN_ID = '0b9d8f3e-6c1a-4c55-9f1e-2a7b3c4d5e6f'

function plan(over: Partial<SocialPlan> = {}): SocialPlan {
  return {
    id: PLAN_ID, workspace_id: 'w1', name: 'Countdown', kind: 'plan', status: 'active', goal: 'Fill the stand', audience: 'Founders',
    starts_on: '2026-10-05', ends_on: '2026-11-08', timezone: 'Europe/London',
    cadence: [
      { id: 'r1', channels: ['twitter'], format: 'image', length_seconds: null, template_id: null, days: ['mon', 'tue', 'wed', 'thu', 'fri', 'sat', 'sun'], time: '09:00' },
      { id: 'r2', channels: ['instagram', 'tiktok'], format: 'video', length_seconds: 30, template_id: null, days: ['fri'], time: '17:00' },
    ],
    sources: { knowledge: true, deliverables: true, website: false, github: true, notes: 'Lead with the local edition.', never_say: ['prices'] },
    make: { time: '07:00', video_days_early: 1, image_days_early: 0, max_per_day: null, visual_mix: { templates: 50, ai_images: 50 } },
    research: { enabled: true, day: 'mon', time: '06:00' },
    late_policy: 'next_slot', approval_mode: 'per_post', slot_overrides: {}, created_by: 'u', created_at: null, updated_at: null,
    bank: { topics: 24, unused: 18 },
    ...over,
  }
}

describe('how often', () => {
  it('reads presets back from days, and keeps a weekly row on its day', () => {
    expect(oftenOf(['mon', 'tue', 'wed', 'thu', 'fri', 'sat', 'sun'])).toBe('daily')
    expect(oftenOf(['fri', 'mon', 'wed'])).toBe('mwf')
    expect(oftenOf(['thu'])).toBe('weekly')
    expect(oftenOf(['mon', 'thu'])).toBe('custom')
    expect(daysFor('weekdays', [])).toEqual(['mon', 'tue', 'wed', 'thu', 'fri'])
    expect(daysFor('weekly', ['thu', 'fri'])).toEqual(['thu'])
  })
})

describe('the draft', () => {
  it('maps a plan to the form and back without loss', () => {
    const input = inputFromDraft(draftFromPlan(plan()))
    expect(input).toMatchObject({
      name: 'Countdown', goal: 'Fill the stand', starts_on: '2026-10-05', ends_on: '2026-11-08', timezone: 'Europe/London',
      late_policy: 'next_slot', research: { enabled: true, day: 'mon', time: '06:00' },
      make: { time: '07:00', image_days_early: 0, video_days_early: 1, visual_mix: { templates: 50, ai_images: 50 } },
      sources: { knowledge: true, website: false, github: true, never_say: ['prices'] },
    })
    expect(input.cadence?.[1]).toEqual({ id: 'r2', channels: ['instagram', 'tiktok'], format: 'video', length_seconds: 30, template_id: null, days: ['fri'], time: '17:00' })
    expect(mixOf({ templates: 50, ai_images: 50 })).toBe('mixed')
  })

  it("carries the repeat window: the plan's, else the server's default of 60 days (PRD-251C)", () => {
    expect(inputFromDraft(draftFromPlan(plan())).research?.repeat_after_days).toBe(60)
    const kept = plan({ research: { enabled: true, day: 'mon', time: '06:00', repeat_after_days: 45 } })
    expect(inputFromDraft(draftFromPlan(kept)).research?.repeat_after_days).toBe(45)
    expect([repeatDays('0'), repeatDays('90'), repeatDays('9999'), repeatDays('x')]).toEqual([1, 90, 365, 1])
  })

  it('a new plan runs 35 days from today and needs a name and a channel', () => {
    const draft = emptyDraft('UTC', new Date('2026-10-14T07:00:00Z'))
    expect([draft.startsOn, draft.endsOn]).toEqual(['2026-10-14', '2026-11-17'])
    expect(missingFields(draft)).toEqual(['a name', 'a channel for every cadence row'])
    expect(inputFromDraft({ ...draft, cadence: [{ ...draft.cadence[0], format: 'image', lengthSeconds: 30 }] }).cadence?.[0].length_seconds).toBeNull()
  })
})

describe('the cadence summary and the status line', () => {
  it('counts posts per channel over the plan days and the videos render minutes', () => {
    const summary = cadenceSummary(draftFromPlan(plan()))
    expect(summary.days).toBe(35)
    expect(summary.perChannel).toEqual([
      { channel: 'twitter', posts: 35 }, { channel: 'instagram', posts: 5 }, { channel: 'tiktok', posts: 5 },
    ])
    expect([summary.posts, summary.videos, summary.renderMinutes]).toEqual([40, 5, 2.5])
  })

  it('says where the plan is', () => {
    expect(statusLine(plan(), new Date('2026-10-14T09:00:00Z'))).toBe('Running · day 10 of 35')
    expect(statusLine(plan({ status: 'paused' }), new Date('2026-10-14T09:00:00Z'))).toBe('Paused · day 10 of 35')
    expect(statusLine(plan(), new Date('2026-10-02T09:00:00Z'))).toBe('Starts in 3 days')
    expect(statusLine(plan({ status: 'ended' }))).toBe('Ended')
  })
})

describe('a plan on the calendar', () => {
  const slot = (key: string, state: 'planned' | 'made', at: string, channels = ['twitter']) => ({
    key, row_id: 'r1', channels, format: 'image', length_seconds: null, template_id: null, local_date: at.slice(0, 10),
    local_time: '09:00', at, moved: false, state, post: null, topic: null,
  })

  it('names its plan and key in the item id, and only slots not made become chips', () => {
    const id = plannedId(PLAN_ID, 'r1|2026-10-15|09:00')
    expect(parsePlannedId(id)).toEqual({ planId: PLAN_ID, key: 'r1|2026-10-15|09:00' })
    expect(parsePlannedId('a-post-id')).toBeNull()
    const answers = [{ plan_id: PLAN_ID, slots: [slot('r1|2026-10-15|09:00', 'planned', '2026-10-15T08:00:00Z'), slot('r1|2026-10-14|09:00', 'made', '2026-10-14T08:00:00Z')] }]
    const planned = plannedSlots(answers, [plan()])
    expect(planned.map((p) => p.slot.key)).toEqual(['r1|2026-10-15|09:00'])
    const span = { start: new Date('2026-10-12T00:00:00Z'), end: new Date('2026-10-19T00:00:00Z') }
    expect(plannedEvents(planned, span, 'all')).toHaveLength(1)
    expect(plannedEvents(planned, span, 'linkedin')).toHaveLength(0)
    expect(planRailLine(plan())).toBe('1.1 posts a day · 18 of 24 topics unused')
  })
})

describe('the music picker', () => {
  it('maps its choices to the post music and back', () => {
    expect([musicValue(null), musicValue({ track: null }), musicValue({ track: 'spring-of-2026' })]).toEqual(['', '__none__', 'spring-of-2026'])
    expect([musicFrom(''), musicFrom('__none__'), musicFrom('spring-of-2026')]).toEqual([null, { track: null }, { track: 'spring-of-2026' }])
    expect(trackLabel({ style: 'deep house', title: 'Deep House 003' })).toBe('Library · Deep house')
  })
})
