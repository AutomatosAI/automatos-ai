/**
 * F377 (night 11, 7 Oct) — a video's shorter cut asks only for the fields it shows. The
 * gallery entry carries `fields_cut_out` (core/social_cuts.py): the fields each cut never shows;
 * the editor's Text card leaves them out at that length, so a 15 s cut never marks a field of
 * the dropped stretches as needed.
 */
import { describe, it, expect } from 'vitest'

import { isRequired, schemaAtLength } from '@/components/deliverables/socials/socials-variables-form'
import type { SocialTemplateVariable } from '@/lib/api-client'

const SCHEMA: Record<string, SocialTemplateVariable> = {
  headline: { type: 'text', label: 'Headline' },
  stat_value: { type: 'text', label: 'Big number' },
  closing_line: { type: 'text', label: 'Closing line' },
  kicker: { type: 'text', label: 'Label', default: '' },
}
const CUT_OUT = { '15': ['stat_value', 'closing_line'], '30': [] }

const needed = (schema: Record<string, SocialTemplateVariable>) =>
  Object.entries(schema).filter(([, spec]) => isRequired(spec)).map(([name]) => name)

describe('the fields a length asks for', () => {
  it('keeps only the fields a cut shows, so the dropped stretches mark nothing needed', () => {
    const at15 = schemaAtLength(SCHEMA, CUT_OUT, 15)
    expect(Object.keys(at15)).toEqual(['headline', 'kicker'])
    expect(needed(at15)).toEqual(['headline'])
    expect(needed(SCHEMA)).toEqual(['headline', 'stat_value', 'closing_line'])
  })

  it('keeps every field at a length without a cut, with no length, or with no cuts at all', () => {
    expect(schemaAtLength(SCHEMA, CUT_OUT, 30)).toBe(SCHEMA)
    expect(schemaAtLength(SCHEMA, CUT_OUT, 40)).toBe(SCHEMA)
    expect(schemaAtLength(SCHEMA, CUT_OUT, null)).toBe(SCHEMA)
    expect(schemaAtLength(SCHEMA, undefined, 15)).toBe(SCHEMA)
  })
})
