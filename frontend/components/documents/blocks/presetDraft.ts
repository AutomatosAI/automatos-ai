// Pure: turn a category preset into editor state (PRD-243).
import type { EditorDraft } from './TemplateEditor'
import type { TemplatePreset } from './types'
import { newBlockId } from './inline'

export function sampleDataOf(sample: Record<string, any> | null | undefined): Record<string, any> {
  if (!sample || typeof sample !== 'object') return {}
  const inner = (sample as Record<string, any>).data
  return inner && typeof inner === 'object' && !Array.isArray(inner) ? inner : sample
}

export function blankDraft(): EditorDraft {
  return {
    id: null,
    name: '',
    description: '',
    category: 'general',
    format: 'pdf',
    blocks: [{ type: 'heading', id: newBlockId(), level: 1, content: [] }],
    previewData: {},
  }
}

// A NEW template from a preset: the layout, format, category and sample values come
// from the preset; the name is left for the author (starters already own the preset
// names, and agents look templates up by name).
export function draftFromPreset(preset: TemplatePreset, overrides: Partial<EditorDraft> = {}): EditorDraft {
  return {
    id: null,
    name: '',
    description: preset.description,
    category: preset.category,
    format: String(preset.format || 'pdf'),
    blocks: preset.blocks.blocks,
    previewData: sampleDataOf(preset.sample_data),
    ...overrides,
  }
}

// Swap an existing draft's layout for a preset's, keeping its identity (id, name,
// description) — what "Change layout" does.
export function applyPresetLayout(draft: EditorDraft, preset: TemplatePreset): EditorDraft {
  return {
    ...draft,
    category: preset.category,
    format: String(preset.format || draft.format),
    blocks: preset.blocks.blocks,
    previewData: sampleDataOf(preset.sample_data),
  }
}

// True when the draft still holds nothing worth confirming over (the blank start).
export function isBlankDraft(draft: EditorDraft): boolean {
  if (draft.blocks.length === 0) return true
  if (draft.blocks.length > 1) return false
  const only = draft.blocks[0]
  return only.type === 'heading' && (only.content?.length ?? 0) === 0
}
