/**
 * PRD-268 §6.11 (US-C21b) — the proactive answers checked off the wire: Brief@1 read leniently
 * (unknown fields ignored, a bad row left out), ids checked before a path, a refusal's reason.
 */
import { describe, expect, it } from 'vitest'

import { briefOf, briefOfParts, isProactiveId, openedOf, readinessOf, refusalOf, suggestionsOf } from '@/lib/proactive/wire'

const ID = '7d1c0a52-3e0b-4c55-9a1e-2f6f0c1d9b10'

const BRIEF = {
  v: 1,
  date: '2026-10-30',
  greeting: 'Bom dia, Ana.',
  prose: 'Nada novo precisa de você.',
  spoken: 'Bom dia, Ana.',
  sections: { needs_you: [], suggestions: [], today: [], since: [], unchecked: [] },
  counts: { needs_you: 0, suggestions: 0, today: 0, since: 0 },
  labels: { sections: { needs_you: 'Precisa de você' }, more: {}, empty: 'Nada novo precisa de você.' },
  state_hash: 'h',
  generated_at: '2026-10-30T07:00:00Z',
}

describe('briefOf', () => {
  it('reads Brief@1 as the server writes it, ignoring a field it does not know', () => {
    const brief = briefOf({ ...BRIEF, later_field: { anything: true } })
    expect(brief?.greeting).toBe('Bom dia, Ana.')
    expect(brief?.labels).toEqual({ sections: { needs_you: 'Precisa de você' }, more: {}, empty: 'Nada novo precisa de você.' })
  })

  it('is null for anything that is not Brief@1', () => {
    expect(briefOf(null)).toBeNull()
    expect(briefOf({ ...BRIEF, v: 2 })).toBeNull()
    expect(briefOf({ ...BRIEF, greeting: 3 })).toBeNull()
    expect(briefOf([BRIEF])).toBeNull()
  })

  it('fills a missing section with no rows and leaves out a row without its id, title and why', () => {
    const brief = briefOf({
      ...BRIEF,
      sections: { today: [{ id: 'cal:1', title: 'Standup', why: '09:30' }, { id: 'cal:2', title: 'No why' }, 'row'] },
      counts: { today: '1', since: 2 },
    })
    expect(brief?.sections.today.map((row) => row.id)).toEqual(['cal:1'])
    expect(brief?.sections.needs_you).toEqual([])
    expect(brief?.counts).toEqual({ needs_you: 0, suggestions: 0, today: 0, since: 2 })
  })

  it("keeps a row's opens, action and state only when each is what it says", () => {
    const good = { id: 'sugg:1', title: 'Lisbon flights', why: 'Prices dropped', display_state: 'working',
      opens: { type: 'suggestion', id: ID }, action: { verb: 'start', label: 'Start', done_label: 'Started' } }
    const bad = { id: 'sugg:2', title: 'Odd', why: 'Odd', display_state: 'paused',
      opens: { type: 'javascript', id: 'x' }, action: 'x' }
    const brief = briefOf({ ...BRIEF, sections: { suggestions: [good, bad] } })
    expect(brief?.sections.suggestions[0]).toMatchObject({
      display_state: 'working', opens: { type: 'suggestion', id: ID }, action: { verb: 'start', label: 'Start', done_label: 'Started' },
    })
    expect(brief?.sections.suggestions[1]).toMatchObject({ display_state: null, opens: null, action: null })
  })

  it('keeps only known sections text in the labels', () => {
    const brief = briefOf({ ...BRIEF, labels: { sections: { needs_you: 'Needs you', bogus: 'x', today: 7 }, more: null } })
    expect(brief?.labels).toEqual({ sections: { needs_you: 'Needs you' }, more: {}, empty: undefined })
  })

  it("finds a message's brief@1 part and nothing else", () => {
    expect(briefOfParts([{ type: 'text', text: 'prose' }, { type: 'brief@1', brief: BRIEF }])?.date).toBe('2026-10-30')
    expect(briefOfParts([{ type: 'text', text: 'prose' }, { type: 'suggestion@1', suggestion_id: ID }])).toBeNull()
    expect(briefOfParts('not parts')).toBeNull()
  })
})

describe('suggestionsOf', () => {
  it('leaves out a suggestion without an id, a title or what Start does, and keeps evidence as {type, id}', () => {
    const row = { id: ID, title: 'Retry the export', proposed_action: { kind: 'board_task', payload: { agent_id: 4 } },
      evidence: [{ type: 'ticket', id: '0419', note: 'x' }, { type: 'report' }, 'ticket'] }
    const rows = suggestionsOf({ suggestions: [row, { ...row, proposed_action: null }, { ...row, id: 'nope' }, null] })
    expect(rows).toHaveLength(1)
    expect(rows[0].evidence).toEqual([{ type: 'ticket', id: '0419' }])
    expect(rows[0].proposed_action).toEqual({ kind: 'board_task', payload: { agent_id: 4 } })
    expect(rows[0].detail).toBe('')
  })
})

describe('readinessOf', () => {
  it.each(['/settings', '/settings?tab=integrations#voice'])('keeps the app path %s', (href) => {
    const items = readinessOf({ v: 1, checked_at: 't', items: [{ id: 'push', state: 'needs_setup', fix: { label: 'Fix', href }, needed_by: ['push', 'x'] }] }).items
    expect(items[0].fix).toEqual({ label: 'Fix', href })
    expect(items[0].needed_by).toEqual(['push'])
  })

  it.each(['//evil.example', '/\\evil.example', '/\t/evil.example', '/\n/evil.example', 'https://evil.example', 'javascript:alert(1)', 'settings'])(
    'drops the fix that opens %j',
    (href) => {
      const items = readinessOf({ v: 1, checked_at: 't', items: [{ id: 'push', state: 'needs_setup', fix: { label: 'Fix', href }, needed_by: [] }] }).items
      expect(items[0].fix).toBeNull()
    },
  )
})

describe('ids and the open answer', () => {
  it('takes only the server UUIDs as ids', () => {
    expect(isProactiveId(ID)).toBe(true)
    expect(isProactiveId('../settings')).toBe(false)
    expect(isProactiveId(`${ID}/hide`)).toBe(false)
    expect(isProactiveId(42)).toBe(false)
  })

  it("reads the open's answer, and anything without both ids as pending", () => {
    expect(openedOf({ chat_id: ID, message_id: ID, posted: true })).toEqual({ chat_id: ID, message_id: ID, posted: true })
    expect(openedOf({ chat_id: ID, message_id: ID })).toEqual({ chat_id: ID, message_id: ID, posted: false })
    expect(openedOf({ posted: false, pending: true })).toEqual({ posted: false, pending: true })
    expect(openedOf({ chat_id: ID })).toEqual({ posted: false, pending: true })
  })
})

describe('refusalOf', () => {
  it("gives the server's code and sentence, and the status", () => {
    const error = Object.assign(new Error('Já foi iniciada.'), { status: 409, body: { code: 'already_started', detail: 'Já foi iniciada.' } })
    expect(refusalOf(error)).toEqual({ code: 'already_started', detail: 'Já foi iniciada.', status: 409 })
  })

  it("falls back to the error's own message when there is no body", () => {
    expect(refusalOf(new Error('Failed to fetch'))).toEqual({ code: null, detail: 'Failed to fetch', status: null })
  })

  it('is null when nothing failed', () => {
    expect(refusalOf(null)).toBeNull()
    expect(refusalOf(undefined)).toBeNull()
  })
})
