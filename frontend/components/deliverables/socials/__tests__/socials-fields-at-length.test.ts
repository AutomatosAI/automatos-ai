/**
 * F377 (night 11, 7 Oct) — a video's shorter cut asks only for the fields it shows. The
 * gallery entry carries `fields_cut_out` (core/social_cuts.py): the fields each cut never shows;
 * the editor's Text card leaves them out at that length, so a 15 s cut never shows a field of
 * the dropped stretches. (No field is marked as needed any more: Auto writes what is left empty.)
 */
import { describe, it, expect } from 'vitest'

import { schemaAtLength, withoutFields } from '@/components/deliverables/socials/socials-variables-form'
import type { SocialTemplateVariable } from '@/lib/api-client'

const SCHEMA: Record<string, SocialTemplateVariable> = {
  headline: { type: 'text', label: 'Headline' },
  stat_value: { type: 'text', label: 'Big number' },
  closing_line: { type: 'text', label: 'Closing line' },
  kicker: { type: 'text', label: 'Label', default: '' },
}
const CUT_OUT = { '15': ['stat_value', 'closing_line'], '30': [] }

describe('the fields a length asks for', () => {
  it('keeps only the fields a cut shows, so the dropped stretches ask for nothing', () => {
    const at15 = schemaAtLength(SCHEMA, CUT_OUT, 15)
    expect(Object.keys(at15)).toEqual(['headline', 'kicker'])
    expect(Object.keys(SCHEMA)).toEqual(['headline', 'stat_value', 'closing_line', 'kicker'])
  })

  it("Clear all empties the template's own fields and leaves the rest (a carousel's slide count)", () => {
    const variables = { headline: { value: 'OPEN LATE', claim: false }, kicker: { value: 'NEW', claim: false }, slides: { value: 5, claim: false } }
    expect(withoutFields(variables, SCHEMA)).toEqual({ slides: { value: 5, claim: false } })
    // A cleared claim's source goes with its value: it must not ride on to what Auto writes next.
    const sources = { stat_value: { kind: 'report', ref: 'r-1', as_of: null } }
    expect(withoutFields(sources, SCHEMA)).toEqual({})
  })

  it('keeps every field at a length without a cut, with no length, or with no cuts at all', () => {
    expect(schemaAtLength(SCHEMA, CUT_OUT, 30)).toBe(SCHEMA)
    expect(schemaAtLength(SCHEMA, CUT_OUT, 40)).toBe(SCHEMA)
    expect(schemaAtLength(SCHEMA, CUT_OUT, null)).toBe(SCHEMA)
    expect(schemaAtLength(SCHEMA, undefined, 15)).toBe(SCHEMA)
  })
})
