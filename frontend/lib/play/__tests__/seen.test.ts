/**
 * PRD-260 T-007a — which Deliverables this browser has shown on the shelf
 * (lib/play/seen.ts), ported from Automatos Play's tests/deliverables/seen.test.ts
 * onto the core's created_at: the first visit only remembers, a new one is
 * newer than the newest seen, the record is capped and survives a broken or
 * blocked localStorage. It is kept per user and workspace (F5): another account
 * in this browser starts its own, and a record kept before moves once.
 */
import { afterEach, describe, expect, it, vi } from 'vitest'

import { lookScope } from '@/lib/looks/scope'

import {
  freshDeliverables,
  markSeen,
  parseSeen,
  readSeen,
  SEEN_CAP,
  SEEN_KEY,
  seenKey,
  UNSCOPED_SEEN_MOVED_KEY,
  writeSeen,
} from '../seen'

const OLD = { id: 'd-old', created_at: '2026-10-10T09:00:00+00:00' }
const NEW = { id: 'd-new', created_at: '2026-10-14T09:00:00+00:00' }
const A = lookScope('ws-1', 'user_a')
const B = lookScope('ws-1', 'user_b')

afterEach(() => {
  vi.restoreAllMocks()
  window.localStorage.clear()
})

describe('freshDeliverables', () => {
  it('counts nothing as new on the first visit', () => {
    expect(freshDeliverables(null, [NEW, OLD])).toEqual([])
  })

  it('counts an unseen Deliverable newer than the newest seen', () => {
    expect(freshDeliverables({ ids: ['d-old'], newest: OLD.created_at }, [NEW, OLD])).toEqual([NEW])
  })

  it('does not count an older one met for the first time (a chip, Load more)', () => {
    expect(freshDeliverables({ ids: ['d-new'], newest: NEW.created_at }, [OLD])).toEqual([])
  })
})

describe('markSeen', () => {
  it('remembers the ids and the newest time, never mutating the record', () => {
    const before = { ids: ['d-old'], newest: OLD.created_at }
    const after = markSeen(before, [NEW, OLD])
    expect(after).toEqual({ ids: ['d-old', 'd-new'], newest: NEW.created_at })
    expect(before.ids).toEqual(['d-old'])
  })

  it('returns the same record when nothing changes', () => {
    const record = { ids: ['d-new', 'd-old'], newest: NEW.created_at }
    expect(markSeen(record, [NEW, OLD])).toBe(record)
  })

  it(`keeps the newest ${SEEN_CAP} ids`, () => {
    const items = Array.from({ length: SEEN_CAP + 5 }, (_, index) => ({ id: `d-${index}`, created_at: OLD.created_at }))
    const record = markSeen(null, items)
    expect(record.ids).toHaveLength(SEEN_CAP)
    expect(record.ids[0]).toBe('d-5')
  })
})

describe('the stored record', () => {
  it('round-trips through localStorage', () => {
    writeSeen(A, { ids: ['d-old'], newest: OLD.created_at })
    expect(window.localStorage.getItem(seenKey(A))).not.toBeNull()
    expect(readSeen(A)).toEqual({ ids: ['d-old'], newest: OLD.created_at })
  })

  it("is kept per user and workspace: another account in this browser doesn't read it (F5)", () => {
    writeSeen(A, { ids: ['d-old'], newest: OLD.created_at })
    expect(seenKey(A)).toBe('automatos-play:seen-deliverables:ws-1:user_a')
    expect(readSeen(B)).toBeNull()
    expect(readSeen(lookScope('ws-2', 'user_a'))).toBeNull()
  })

  it('moves a record kept before F5 to the first account that reads one, once, never deleting it', () => {
    const before = JSON.stringify({ ids: ['d-old'], newest: OLD.created_at })
    window.localStorage.setItem(SEEN_KEY, before)
    expect(readSeen(A)).toEqual({ ids: ['d-old'], newest: OLD.created_at })
    expect(window.localStorage.getItem(UNSCOPED_SEEN_MOVED_KEY)).toBe(A)
    expect(readSeen(B)).toBeNull()
    expect(window.localStorage.getItem(SEEN_KEY)).toBe(before)
  })

  it('reads a broken record as none, keeping only string ids', () => {
    vi.spyOn(console, 'warn').mockImplementation(() => undefined)
    expect(parseSeen('{not json')).toBeNull()
    expect(parseSeen(JSON.stringify({ ids: 'd-old' }))).toBeNull()
    expect(parseSeen(JSON.stringify({ ids: ['d-old', 7], newest: 3 }))).toEqual({ ids: ['d-old'], newest: null })
  })

  it('survives a blocked localStorage', () => {
    vi.spyOn(console, 'warn').mockImplementation(() => undefined)
    vi.spyOn(Storage.prototype, 'getItem').mockImplementation(() => {
      throw new Error('blocked')
    })
    vi.spyOn(Storage.prototype, 'setItem').mockImplementation(() => {
      throw new Error('blocked')
    })
    expect(readSeen(A)).toBeNull()
    expect(() => writeSeen(A, { ids: [], newest: null })).not.toThrow()
    expect(window.localStorage.length).toBe(0)
  })
})
