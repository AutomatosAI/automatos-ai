'use client'

import React, { useCallback, useEffect, useState } from 'react'
import { Palette, Plus } from 'lucide-react'
import { toast } from 'sonner'
import { Button } from '@/components/ui/button'
import { HelpTooltip } from '@/components/ui/help-tooltip'
import { DeleteConfirmation, ErrorState, LoadingState } from '@/components/shared'
import { templateBlocksApi } from './api'
import { BrandKitDialog } from './BrandKitDialog'
import { GenerateDocumentDialog } from './GenerateDocumentDialog'
import { PresetPicker } from './PresetPicker'
import { TemplateCards } from './TemplateCards'
import { TemplateEditor, type EditorDraft } from './TemplateEditor'
import { TemplateGuide } from './TemplateGuide'
import { applyPresetLayout, blankDraft, draftFromPreset, sampleDataOf } from './presetDraft'
import { collectMissingOnFile } from './templateFields'
import { SCHEMA_VERSION } from './types'
import type { BlockDocument, TemplatePreset, TemplateSummary, VariableEntry } from './types'

// PRD-167 S5 → PRD-242 S5: the non-technical Template Studio — a guided gallery
// (copy-on-customise), the block editor with live preview, brand kit, and a
// generate-now path that lands in Deliverables. Errors show their cause and retry.
export function TemplateStudio() {
  const [mode, setMode] = useState<'gallery' | 'editor'>('gallery')
  const [templates, setTemplates] = useState<TemplateSummary[]>([])
  const [variables, setVariables] = useState<VariableEntry[]>([])
  const [presets, setPresets] = useState<TemplatePreset[]>([])
  const [presetsLoading, setPresetsLoading] = useState(true)
  const [picker, setPicker] = useState<'new' | 'replace' | null>(null)
  const [loading, setLoading] = useState(true)
  const [loadError, setLoadError] = useState<unknown>(null)
  const [brandOpen, setBrandOpen] = useState(false)
  const [draft, setDraft] = useState<EditorDraft>(blankDraft)
  const [saving, setSaving] = useState(false)
  const [generateFor, setGenerateFor] = useState<TemplateSummary | null>(null)
  const [generateData, setGenerateData] = useState<Record<string, any> | undefined>(undefined)
  const [deleteFor, setDeleteFor] = useState<TemplateSummary | null>(null)

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

  // Layouts are static; a failure here degrades to "Blank" only, it never hides the gallery.
  useEffect(() => {
    templateBlocksApi
      .listPresets()
      .then(setPresets)
      .catch(() => setPresets([]))
      .finally(() => setPresetsLoading(false))
  }, [])

  const startNew = () => setPicker('new')

  const pickPreset = (preset: TemplatePreset | null) => {
    if (picker === 'replace') {
      setDraft((d) => (preset ? applyPresetLayout(d, preset) : { ...blankDraft(), id: d.id, name: d.name, description: d.description }))
      toast.success(preset ? `Layout replaced with ${preset.name}` : 'Layout cleared')
    } else {
      setDraft(preset ? draftFromPreset(preset) : blankDraft())
      setMode('editor')
    }
    setPicker(null)
  }

  const openTemplate = async (t: TemplateSummary, asCopy: boolean) => {
    try {
      const full = await templateBlocksApi.getTemplate(t.id)
      const doc: BlockDocument | null = full.blocks
      if (asCopy && !full.has_blocks) {
        // A legacy (Jinja / uploaded) template cannot be block-edited: the copy
        // starts from its category's layout instead of an empty heading (PRD-243).
        const preset = presets.find((p) => p.category === (full.category || 'general')) ?? presets.find((p) => p.category === 'general')
        setDraft(
          preset
            ? draftFromPreset(preset, { name: `${full.name} (copy)`, description: full.description || preset.description })
            : { ...blankDraft(), name: `${full.name} (copy)`, description: full.description || '' },
        )
        setMode('editor')
        toast.info(preset ? `Started from the ${preset.name} layout — the original is a built-in design that cannot be block-edited.` : 'Started blank.')
        return
      }
      setDraft({
        id: asCopy ? null : full.id,
        name: asCopy ? `${full.name} (copy)` : full.name,
        description: full.description || '',
        category: full.category || 'general',
        format: full.format || 'pdf',
        blocks: doc?.blocks ?? blankDraft().blocks,
        previewData: sampleDataOf(full.sample_data),
      })
      setMode('editor')
    } catch (e: any) {
      toast.error(`Could not open template: ${e?.message || 'unknown error'}`)
    }
  }

  const save = async () => {
    if (!draft.name.trim()) {
      toast.error('Template needs a name')
      return
    }
    setSaving(true)
    const body = {
      name: draft.name.trim(),
      description: draft.description,
      category: draft.category,
      format: draft.format,
      blocks: { version: SCHEMA_VERSION, blocks: draft.blocks },
      sample_data: { data: draft.previewData },
    }
    try {
      if (draft.id) {
        await templateBlocksApi.updateTemplate(draft.id, body)
        toast.success('Template saved')
      } else {
        const created = await templateBlocksApi.createTemplate(body)
        setDraft({ ...draft, id: created.id })
        toast.success('Template created — you can now generate from it or hand it to Auto')
      }
      await loadGallery()
    } catch (e: any) {
      // Field-level block errors from the 422 arrive as a stringified detail (PRD-167 S2).
      const message = String(e?.message || '')
      try {
        const detail = JSON.parse(message)
        if (detail?.errors) {
          toast.error(`Invalid blocks: ${detail.errors.map((x: any) => `${x.loc} ${x.msg}`).join('; ')}`)
          return
        }
      } catch {
        /* not JSON — fall through */
      }
      toast.error(message || 'Failed to save template')
    } finally {
      setSaving(false)
    }
  }

  const confirmDelete = async () => {
    if (!deleteFor) return
    try {
      await templateBlocksApi.deleteTemplate(deleteFor.id)
      toast.success(`Deleted “${deleteFor.name}”`)
      setDeleteFor(null)
      await loadGallery()
    } catch (e: any) {
      // DeleteConfirmation stays open on throw; say why so the user is not left guessing.
      toast.error(`Could not delete “${deleteFor.name}”: ${e?.message || 'unknown error'}`)
    }
  }

  const openGenerateForDraft = () => {
    if (!draft.id) return
    const current = templates.find((t) => t.id === draft.id)
    setGenerateData(draft.previewData)
    setGenerateFor(
      current ?? {
        id: draft.id,
        name: draft.name,
        description: draft.description,
        format: draft.format,
        category: draft.category,
        tags: [],
        version: 1,
        has_blocks: true,
        is_starter: false,
        variable_paths: [],
        data_fields: [],
        list_fields: [],
      },
    )
  }

  const missingForGenerate = generateFor ? collectMissingOnFile(generateFor.variable_paths, variables) : []

  if (mode === 'editor') {
    return (
      <>
        <TemplateEditor
          draft={draft}
          variables={variables}
          saving={saving}
          onChange={setDraft}
          onBack={() => setMode('gallery')}
          onSave={save}
          onGenerate={openGenerateForDraft}
          onOpenBrandKit={() => setBrandOpen(true)}
          onChangeLayout={() => setPicker('replace')}
        />
        <PresetPicker open={picker !== null} onOpenChange={(open) => !open && setPicker(null)} presets={presets} loading={presetsLoading} mode={picker ?? 'replace'} onPick={pickPreset} />
        <BrandKitDialog open={brandOpen} onOpenChange={setBrandOpen} onSaved={() => loadGallery()} />
        <GenerateDocumentDialog
          open={!!generateFor}
          onOpenChange={(open) => !open && setGenerateFor(null)}
          template={generateFor}
          initialData={generateData}
          missingOnFile={missingForGenerate}
          onOpenBrandKit={() => setBrandOpen(true)}
        />
      </>
    )
  }

  return (
    <div className="space-y-6">
      <div className="flex flex-col items-start justify-between gap-4 sm:flex-row sm:items-center">
        <div>
          <h2 className="flex items-center text-xl font-bold">
            Template Studio <HelpTooltip id="deliverables.templates.gallery.title" inline />
          </h2>
          <p className="text-sm text-muted-foreground">
            Branded document templates your agents fill — reports, letters, invoices — no code.
          </p>
        </div>
        <div className="flex gap-2">
          <Button variant="outline" onClick={() => setBrandOpen(true)}>
            <Palette className="mr-2 h-4 w-4" /> Brand Kit
          </Button>
          <Button onClick={startNew}>
            <Plus className="mr-2 h-4 w-4" /> New template
          </Button>
        </div>
      </div>

      <TemplateGuide onOpenBrandKit={() => setBrandOpen(true)} />

      {loading ? (
        <LoadingState variant="cards" count={3} label="Loading templates" />
      ) : loadError ? (
        <ErrorState
          title="Templates could not be loaded"
          error={loadError}
          onRetry={loadGallery}
          retryLabel="Try again"
        />
      ) : (
        <TemplateCards
          templates={templates}
          onEdit={(t) => openTemplate(t, false)}
          onCopy={(t) => openTemplate(t, true)}
          onGenerate={(t) => {
            setGenerateData(undefined)
            setGenerateFor(t)
          }}
          onDelete={setDeleteFor}
          onCreate={startNew}
        />
      )}

      <PresetPicker open={picker !== null} onOpenChange={(open) => !open && setPicker(null)} presets={presets} loading={presetsLoading} mode={picker ?? 'new'} onPick={pickPreset} />
      <BrandKitDialog open={brandOpen} onOpenChange={setBrandOpen} onSaved={() => loadGallery()} />
      <GenerateDocumentDialog
        open={!!generateFor}
        onOpenChange={(open) => !open && setGenerateFor(null)}
        template={generateFor}
        initialData={generateData}
        missingOnFile={missingForGenerate}
        onOpenBrandKit={() => setBrandOpen(true)}
      />
      <DeleteConfirmation
        open={!!deleteFor}
        onOpenChange={(open) => !open && setDeleteFor(null)}
        title="Delete template?"
        itemName={deleteFor?.name}
        onConfirm={confirmDelete}
      />
    </div>
  )
}
