import { describe, it, expect } from 'vitest'
import { applyPresetLayout, blankDraft, draftFromPreset, isBlankDraft, sampleDataOf } from '../presetDraft'
import type { TemplatePreset } from '../types'

const preset: TemplatePreset = {
  category: 'invoice',
  name: 'Branded Invoice',
  description: 'Invoice layout',
  format: 'pdf',
  includes: ['Line items'],
  variable_paths: ['data.client_name', 'data.line_items'],
  data_fields: ['client_name', 'line_items'],
  list_fields: [{ field: 'line_items', columns: ['description', 'total'] }],
  blocks: {
    version: 1,
    blocks: [
      { type: 'heading', id: 'h', level: 1, content: [{ type: 'text', text: 'Invoice' }] },
      { type: 'data_table', id: 'dt', path: 'data.line_items', columns: [{ key: 'description', label: 'Description' }, { key: 'total', label: 'Total' }] },
    ],
  },
  sample_data: { data: { client_name: 'Acme', line_items: [{ description: 'x', total: '1' }] } },
}

describe('presetDraft', () => {
  it('a new template from a preset takes its layout, category, format and sample values, not its name', () => {
    const d = draftFromPreset(preset)
    expect(d.id).toBeNull()
    expect(d.name).toBe('')
    expect(d.category).toBe('invoice')
    expect(d.format).toBe('pdf')
    expect(d.blocks).toBe(preset.blocks.blocks)
    expect(d.previewData).toEqual(preset.sample_data.data)
    expect(draftFromPreset(preset, { name: 'Old (copy)' }).name).toBe('Old (copy)')
  })
  it('changing layout keeps identity and swaps the blocks', () => {
    const current = { ...blankDraft(), id: 'abc', name: 'Weekly', description: 'mine', category: 'report' }
    const next = applyPresetLayout(current, preset)
    expect(next.id).toBe('abc')
    expect(next.name).toBe('Weekly')
    expect(next.description).toBe('mine')
    expect(next.category).toBe('invoice')
    expect(next.blocks).toBe(preset.blocks.blocks)
    expect(current.blocks).not.toBe(next.blocks) // immutable
  })
  it('unwraps {data: …} sample envelopes and tolerates junk', () => {
    expect(sampleDataOf({ data: { a: 1 } })).toEqual({ a: 1 })
    expect(sampleDataOf({ a: 1 })).toEqual({ a: 1 })
    expect(sampleDataOf(null)).toEqual({})
    expect(sampleDataOf({ data: [1] })).toEqual({ data: [1] })
  })
  it('a category picked on a blank draft yields that layout (what the editor does on select)', () => {
    const blank = { ...blankDraft(), category: 'general' }
    expect(isBlankDraft(blank)).toBe(true)
    const next = applyPresetLayout({ ...blank, category: 'invoice' }, preset)
    expect(next.category).toBe('invoice')
    expect(next.blocks).toBe(preset.blocks.blocks)
    expect(isBlankDraft(next)).toBe(false)
  })
  it('knows a blank draft when it sees one', () => {
    expect(isBlankDraft(blankDraft())).toBe(true)
    expect(isBlankDraft({ ...blankDraft(), blocks: [] })).toBe(true)
    expect(isBlankDraft(draftFromPreset(preset))).toBe(false)
  })
})
