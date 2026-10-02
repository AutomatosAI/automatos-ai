'use client'

/**
 * Template Studio's data and its pure steps (PRD-167 S5 → PRD-242 S5 → PRD-243): the
 * gallery (templates, the variables on file, the layouts), the draft a template opens as,
 * the body a save sends and the message a refused save shows. TemplateStudio.tsx holds the
 * screen; this file keeps it small enough to read.
 */
import { useCallback, useEffect, useState } from 'react'

import { templateBlocksApi } from './api'
import type { EditorDraft } from './TemplateEditor'
import { blankDraft, draftFromPreset, sampleDataOf } from './presetDraft'
import { SCHEMA_VERSION } from './types'
import type { TemplatePreset, TemplateSummary, VariableEntry } from './types'

/** The templates and the variables on file; the layouts load beside them and never hide the gallery. */
export function useTemplateGallery() {
  const [templates, setTemplates] = useState<TemplateSummary[]>([])
  const [variables, setVariables] = useState<VariableEntry[]>([])
  const [presets, setPresets] = useState<TemplatePreset[]>([])
  const [presetsLoading, setPresetsLoading] = useState(true)
  const [loading, setLoading] = useState(true)
  const [loadError, setLoadError] = useState<unknown>(null)

  const loadGallery = useCallback(async () => {
    setLoading(true)
    setLoadError(null)
    try {
      const [tpls, vars] = await Promise.all([templateBlocksApi.listTemplates(), templateBlocksApi.getVariables()])
      setTemplates(tpls)
      setVariables(vars.variables)
    } catch (e) {
      setLoadError(e)
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    loadGallery()
  }, [loadGallery])

  // Layouts are static; a failure here degrades to "Blank" only.
  useEffect(() => {
    templateBlocksApi
      .listPresets()
      .then(setPresets)
      .catch(() => setPresets([]))
      .finally(() => setPresetsLoading(false))
  }, [])

  return { templates, variables, presets, setPresets, presetsLoading, loading, loadError, loadGallery }
}

export interface OpenedDraft {
  draft: EditorDraft
  /** The layouts, when opening had to fetch them. */
  presets?: TemplatePreset[]
  /** What the person is told when the copy could not start from the template itself. */
  info?: string
}

/**
 * The draft a template opens as. A legacy (Jinja / uploaded) template cannot be
 * block-edited: its copy starts from its category's layout instead (PRD-243), fetching
 * the layouts first when they are not here yet rather than silently starting blank.
 */
export async function draftForTemplate(t: TemplateSummary, asCopy: boolean, presets: TemplatePreset[]): Promise<OpenedDraft> {
  const full = await templateBlocksApi.getTemplate(t.id)
  if (asCopy && !full.has_blocks) {
    let available = presets
    let fetched: TemplatePreset[] | undefined
    if (available.length === 0) {
      available = await templateBlocksApi.listPresets().catch(() => [] as TemplatePreset[])
      fetched = available
    }
    const preset = available.find((p) => p.category === (full.category || 'general')) ?? available.find((p) => p.category === 'general')
    return {
      draft: preset
        ? draftFromPreset(preset, { name: `${full.name} (copy)`, description: full.description || preset.description })
        : { ...blankDraft(), name: `${full.name} (copy)`, description: full.description || '' },
      presets: fetched,
      info: preset ? `Started from the ${preset.name} layout — the original is a built-in design that cannot be block-edited.` : 'Started blank.',
    }
  }
  return {
    draft: {
      id: asCopy ? null : full.id,
      name: asCopy ? `${full.name} (copy)` : full.name,
      description: full.description || '',
      category: full.category || 'general',
      format: full.format || 'pdf',
      blocks: full.blocks?.blocks ?? blankDraft().blocks,
      previewData: sampleDataOf(full.sample_data),
    },
  }
}

/** The body POST and PUT /api/documents/templates take for a draft. */
export function templateBody(draft: EditorDraft) {
  return {
    name: draft.name.trim(),
    description: draft.description,
    category: draft.category,
    format: draft.format,
    blocks: { version: SCHEMA_VERSION, blocks: draft.blocks },
    sample_data: { data: draft.previewData },
  }
}

/** Field-level block errors from the 422 arrive as a stringified detail (PRD-167 S2). */
export function templateSaveError(e: any): string {
  const message = String(e?.message || '')
  try {
    const detail = JSON.parse(message)
    if (detail?.errors) return `Invalid blocks: ${detail.errors.map((x: any) => `${x.loc} ${x.msg}`).join('; ')}`
  } catch {
    /* not JSON — the message as it came */
  }
  return message || 'Failed to save template'
}

/** The gallery's entry for the draft, or one made from the draft when the gallery has none yet. */
export function summaryForDraft(draft: EditorDraft & { id: string }, templates: TemplateSummary[]): TemplateSummary {
  return templates.find((t) => t.id === draft.id) ?? {
    id: draft.id, name: draft.name, description: draft.description, format: draft.format, category: draft.category,
    tags: [], version: 1, has_blocks: true, is_starter: false, variable_paths: [], data_fields: [], list_fields: [],
  }
}
