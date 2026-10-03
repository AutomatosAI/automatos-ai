'use client'

import React, { useState } from 'react'
import { useRouter } from 'next/navigation'
import { Palette, Plus } from 'lucide-react'
import { toast } from 'sonner'
import { Button } from '@/components/ui/button'
import { HelpTooltip } from '@/components/ui/help-tooltip'
import { DeleteConfirmation, ErrorState, LoadingState } from '@/components/shared'
import { BRAND_KIT_HREF } from '@/lib/deliverables/tabs'
import { templateBlocksApi } from './api'
import { GenerateDocumentDialog } from './GenerateDocumentDialog'
import { PresetPicker } from './PresetPicker'
import { TemplateCards } from './TemplateCards'
import { TemplateEditor, type EditorDraft } from './TemplateEditor'
import { TemplateGuide } from './TemplateGuide'
import { applyPresetLayout, blankDraft, draftFromPreset } from './presetDraft'
import { collectMissingOnFile } from './templateFields'
import type { TemplatePreset, TemplateSummary } from './types'
import { draftForTemplate, summaryForDraft, templateBody, templateSaveError, useTemplateGallery } from './useTemplateStudio'

type Gallery = ReturnType<typeof useTemplateGallery>

function GalleryHeader({ onBrandKit, onNew }: { onBrandKit: () => void; onNew: () => void }) {
  return (
    <div className="flex flex-col items-start justify-between gap-4 sm:flex-row sm:items-center">
      <div>
        <h2 className="flex items-center text-xl font-bold">
          Template Studio <HelpTooltip id="deliverables.templates.gallery.title" inline />
        </h2>
        <p className="text-sm text-muted-foreground">Branded document templates your agents fill — reports, letters, invoices — no code.</p>
      </div>
      <div className="flex gap-2">
        <Button variant="outline" onClick={onBrandKit}>
          <Palette className="mr-2 h-4 w-4" /> Brand Kit
        </Button>
        <Button onClick={onNew}>
          <Plus className="mr-2 h-4 w-4" /> New template
        </Button>
      </div>
    </div>
  )
}

interface GalleryBodyProps {
  gallery: Gallery
  onOpen: (t: TemplateSummary, asCopy: boolean) => void
  onGenerate: (t: TemplateSummary) => void
  onDelete: (t: TemplateSummary) => void
  onNew: () => void
}

function GalleryBody({ gallery, onOpen, onGenerate, onDelete, onNew }: GalleryBodyProps) {
  if (gallery.loading) return <LoadingState variant="cards" count={3} label="Loading templates" />
  if (gallery.loadError) {
    return <ErrorState title="Templates could not be loaded" error={gallery.loadError} onRetry={gallery.loadGallery} retryLabel="Try again" />
  }
  return (
    <TemplateCards
      templates={gallery.templates}
      onEdit={(t) => onOpen(t, false)}
      onCopy={(t) => onOpen(t, true)}
      onGenerate={onGenerate}
      onDelete={onDelete}
      onCreate={onNew}
    />
  )
}

// PRD-167 S5 → PRD-242 S5: the non-technical Template Studio — a guided gallery
// (copy-on-customise), the block editor with live preview, and a generate-now path
// that lands in Deliverables. Errors show their cause and retry. PRD-251B US-B301: the
// brand kit is the Brand kit tab; the gallery goes there, and the editor (with its
// unsaved draft) opens it in a browser tab of its own.
export function TemplateStudio() {
  const router = useRouter()
  const gallery = useTemplateGallery()
  const [mode, setMode] = useState<'gallery' | 'editor'>('gallery')
  const [picker, setPicker] = useState<'new' | 'replace' | null>(null)
  const [draft, setDraft] = useState<EditorDraft>(blankDraft)
  const [saving, setSaving] = useState(false)
  const [generateFor, setGenerateFor] = useState<TemplateSummary | null>(null)
  const [generateData, setGenerateData] = useState<Record<string, any> | undefined>(undefined)
  const [deleteFor, setDeleteFor] = useState<TemplateSummary | null>(null)
  const openBrandKit = () => router.push(BRAND_KIT_HREF as any)
  const openBrandKitBeside = () => window.open(BRAND_KIT_HREF, '_blank', 'noopener')

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
      const opened = await draftForTemplate(t, asCopy, gallery.presets)
      if (opened.presets) gallery.setPresets(opened.presets)
      setDraft(opened.draft)
      setMode('editor')
      if (opened.info) toast.info(opened.info)
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
    try {
      if (draft.id) {
        await templateBlocksApi.updateTemplate(draft.id, templateBody(draft))
        toast.success('Template saved')
      } else {
        const created = await templateBlocksApi.createTemplate(templateBody(draft))
        setDraft({ ...draft, id: created.id })
        toast.success('Template created — you can now generate from it or hand it to Auto')
      }
      await gallery.loadGallery()
    } catch (e: any) {
      toast.error(templateSaveError(e))
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
      await gallery.loadGallery()
    } catch (e: any) {
      // DeleteConfirmation stays open on throw; say why so the user is not left guessing.
      toast.error(`Could not delete “${deleteFor.name}”: ${e?.message || 'unknown error'}`)
    }
  }

  const generate = (template: TemplateSummary, data?: Record<string, any>) => {
    setGenerateData(data)
    setGenerateFor(template)
  }
  const missingForGenerate = generateFor ? collectMissingOnFile(generateFor.variable_paths, gallery.variables) : []
  const dialogs = (
    <>
      <PresetPicker open={picker !== null} onOpenChange={(open) => !open && setPicker(null)} presets={gallery.presets} loading={gallery.presetsLoading} mode={picker ?? 'new'} onPick={pickPreset} />
      <GenerateDocumentDialog
        open={!!generateFor} onOpenChange={(open) => !open && setGenerateFor(null)} template={generateFor}
        initialData={generateData} missingOnFile={missingForGenerate} onOpenBrandKit={openBrandKitBeside}
      />
    </>
  )

  if (mode === 'editor') {
    return (
      <>
        <TemplateEditor
          draft={draft} variables={gallery.variables} presets={gallery.presets} saving={saving} onChange={setDraft}
          onBack={() => setMode('gallery')} onSave={save} onOpenBrandKit={openBrandKitBeside} onChangeLayout={() => setPicker('replace')}
          onGenerate={() => draft.id && generate(summaryForDraft({ ...draft, id: draft.id }, gallery.templates), draft.previewData)}
        />
        {dialogs}
      </>
    )
  }

  return (
    <div className="space-y-6">
      <GalleryHeader onBrandKit={openBrandKit} onNew={() => setPicker('new')} />
      <TemplateGuide onOpenBrandKit={openBrandKit} />
      <GalleryBody gallery={gallery} onOpen={openTemplate} onGenerate={(t) => generate(t)} onDelete={setDeleteFor} onNew={() => setPicker('new')} />
      {dialogs}
      <DeleteConfirmation open={!!deleteFor} onOpenChange={(open) => !open && setDeleteFor(null)} title="Delete template?" itemName={deleteFor?.name} onConfirm={confirmDelete} />
    </div>
  )
}
