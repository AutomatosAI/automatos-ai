import { describe, it, expect } from 'vitest'
import { KIND_META, KIND_ORDER, kindTone, layoutLanes } from '../calendar-kinds'

describe('calendar kinds', () => {
  it('every feed kind has a legend entry and a distinct colour', () => {
    expect([...KIND_ORDER]).toEqual(['routine', 'recipe', 'task', 'mission', 'task_due'])
    const tones = KIND_ORDER.map((k) => KIND_META[k].tone)
    expect(new Set(tones).size).toBe(tones.length)
    expect(kindTone('recipe')).toBe(KIND_META.recipe.tone)
  })

  it('falls back to the heartbeat tone for a kind the table does not know', () => {
    expect(kindTone('mystery' as never)).toBe(KIND_META.routine.tone)
  })
})

describe('layoutLanes', () => {
  const span = (e: { s: number; e: number }): [number, number] => [e.s, e.e]

  it('non-overlapping events each take the full column', () => {
    const out = layoutLanes([{ s: 0, e: 10 }, { s: 10, e: 20 }], span)
    expect(out.map((l) => [l.lane, l.lanes])).toEqual([[0, 1], [0, 1]])
  })

  it('overlapping events share the column', () => {
    const out = layoutLanes([{ s: 0, e: 10 }, { s: 5, e: 15 }], span)
    expect(out.map((l) => [l.lane, l.lanes])).toEqual([[0, 2], [1, 2]])
  })

  it('a chain reuses a freed lane and the whole cluster shares the lane count', () => {
    const out = layoutLanes([{ s: 25, e: 40 }, { s: 0, e: 20 }, { s: 10, e: 30 }], span)
    expect(out.map((l) => [l.evt.s, l.lane, l.lanes])).toEqual([[0, 0, 2], [10, 1, 2], [25, 0, 2]])
  })

  it('a later separate cluster starts fresh', () => {
    const out = layoutLanes([{ s: 0, e: 10 }, { s: 5, e: 15 }, { s: 30, e: 40 }], span)
    expect(out[2]).toMatchObject({ lane: 0, lanes: 1 })
  })

  it('does not mutate its input', () => {
    const input = [{ s: 5, e: 15 }, { s: 0, e: 10 }]
    layoutLanes(input, span)
    expect(input).toEqual([{ s: 5, e: 15 }, { s: 0, e: 10 }])
  })
})
