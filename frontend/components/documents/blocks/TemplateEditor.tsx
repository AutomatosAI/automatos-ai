'use client'

import React, { useMemo } from 'react'
import { ArrowLeft, FileText, Palette, Save } from 'lucide-react'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { FieldHelp } from '@/components/ui/help-tooltip'
import { BlockEditor } from './BlockEditor'
import { PreviewDataForm } from './PreviewDataForm'
import { PreviewPane } from './PreviewPane'
import { UseWithAutoPopover } from './UseWithAutoPopover'
import { collectDataFields, collectMissingOnFile, collectVariablePaths } from './templateFields'
import { SCHEMA_VERSION } from './types'
import type { Block, VariableEntry } from './types'

export const CATEGORIES = ['general', 'report', 'invoice', 'contract', 'letter', 'proposal', 'data']
export const FORMATS = ['pdf', 'docx']

export interface EditorDraft {
  id: string | null // null = new (or a copy)
  name: string
  description: string
  category: string
  format: string
  blocks: Block[]
  previewData: Record<string, any>
}

interface TemplateEditorProps {
  draft: EditorDraft
  variables: VariableEntry[]
  saving: boolean
  onChange: (draft: EditorDraft) => void
  onBack: () => void
  onSave: () => void
  onGenerate: () => void
  onOpenBrandKit: () => void
}

// The authoring surface (PRD-167 S5 → PRD-242 S5): template metadata, the block
// editor, the fill-in fields it implies, and the live preview — with the chip
// contract summarised so the author sees what an agent will have to supply.
export function TemplateEditor({
  draft,
  variables,
  saving,
  onChange,
  onBack,
  onSave,
  onGenerate,
  onOpenBrandKit,
}: TemplateEditorProps) {
  const paths = useMemo(() => collectVariablePaths(draft.blocks), [draft.blocks])
  const dataFields = useMemo(() => collectDataFields(draft.blocks), [draft.blocks])
  const missingOnFile = useMemo(() => collectMissingOnFile(paths, variables), [paths, variables])
  const autoFilled = paths.filter((p) => !p.startsWith('data.'))
  const patch = (p: Partial<EditorDraft>) => onChange({ ...draft, ...p })

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <Button variant="ghost" size="sm" onClick={onBack}>
          <ArrowLeft className="mr-2 h-4 w-4" /> Back to templates
        </Button>
        <div className="flex flex-wrap items-center gap-2">
          <Button variant="outline" size="sm" onClick={onOpenBrandKit}>
            <Palette className="mr-2 h-4 w-4" /> Brand Kit
          </Button>
          {draft.id && (
            <UseWithAutoPopover template={{ id: draft.id, name: draft.name, format: draft.format, data_fields: dataFields }} />
          )}
          <Button variant="outline" size="sm" onClick={onGenerate} disabled={!draft.id}>
            <FileText className="mr-2 h-4 w-4" /> Generate a document
          </Button>
          <Button size="sm" onClick={onSave} disabled={saving}>
            <Save className="mr-2 h-4 w-4" /> {saving ? 'Saving…' : draft.id ? 'Save changes' : 'Save template'}
          </Button>
        </div>
      </div>
      {!draft.id && (
        <p className="text-xs text-muted-foreground">Save the template first to generate from it or hand it to Auto — it needs an id.</p>
      )}

      <div className="grid grid-cols-1 gap-3 sm:grid-cols-4">
        <div className="sm:col-span-2">
          <Label className="flex items-center text-xs">
            Name <FieldHelp id="deliverables.templates.editor.name" />
          </Label>
          <Input value={draft.name} onChange={(e) => patch({ name: e.target.value })} placeholder="Weekly Report" />
        </div>
        <div>
          <Label className="flex items-center text-xs">
            Category <FieldHelp id="deliverables.templates.editor.category" />
          </Label>
          <Select value={draft.category} onValueChange={(category) => patch({ category })}>
            <SelectTrigger><SelectValue /></SelectTrigger>
            <SelectContent>
              {CATEGORIES.map((c) => (
                <SelectItem key={c} value={c}>{c[0].toUpperCase() + c.slice(1)}</SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
        <div>
          <Label className="flex items-center text-xs">
            Default format <FieldHelp id="deliverables.templates.editor.format" />
          </Label>
          <Select value={draft.format} onValueChange={(format) => patch({ format })}>
            <SelectTrigger><SelectValue /></SelectTrigger>
            <SelectContent>
              {FORMATS.map((f) => (
                <SelectItem key={f} value={f}>{f.toUpperCase()}</SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
        <div className="sm:col-span-4">
          <Label className="text-xs">Description</Label>
          <Input
            value={draft.description}
            onChange={(e) => patch({ description: e.target.value })}
            placeholder="What this template is for — agents read this when choosing a template"
          />
        </div>
      </div>

      <div className="flex flex-wrap items-center gap-1.5 rounded-md border bg-muted/30 px-3 py-2 text-xs">
        <span className="font-medium">Chips in this template:</span>
        {paths.length === 0 && <span className="text-muted-foreground">none yet — use “Insert variable” inside a block</span>}
        {dataFields.map((f) => (
          <Badge key={f} variant="outline" className="font-mono text-[10px]">data.{f}</Badge>
        ))}
        {autoFilled.map((p) => (
          <Badge key={p} variant="secondary" className="font-mono text-[10px]">{p}</Badge>
        ))}
        {paths.length > 0 && (
          <span className="ml-1 text-muted-foreground">
            — outlined chips are filled per document (by an agent or you); grey ones fill themselves from your profile, brand kit and the date.
          </span>
        )}
      </div>

      <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
        <div className="space-y-4">
          <BlockEditor blocks={draft.blocks} variables={variables} onChange={(blocks) => patch({ blocks })} />
          <PreviewDataForm
            fields={dataFields}
            data={draft.previewData}
            onChange={(previewData) => patch({ previewData })}
            missingOnFile={missingOnFile}
            onOpenBrandKit={onOpenBrandKit}
            title="Preview values (also saved as the template's sample data)"
          />
        </div>
        <div className="lg:sticky lg:top-4 lg:h-[80vh]">
          <PreviewPane doc={{ version: SCHEMA_VERSION, blocks: draft.blocks }} data={draft.previewData} />
        </div>
      </div>
    </div>
  )
}
