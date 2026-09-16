'use client'

import React from 'react'
import { FileText, Layers } from 'lucide-react'
import { Badge } from '@/components/ui/badge'
import { Card, CardContent } from '@/components/ui/card'
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle } from '@/components/ui/dialog'
import { LoadingState } from '@/components/shared'
import type { TemplatePreset } from './types'

interface PresetPickerProps {
  open: boolean
  onOpenChange: (open: boolean) => void
  presets: TemplatePreset[]
  loading?: boolean
  // 'new' starts a template from the layout; 'replace' swaps the current draft's layout.
  mode: 'new' | 'replace'
  onPick: (preset: TemplatePreset | null) => void
}

const CATEGORY_LABEL: Record<string, string> = {
  letter: 'Letter',
  invoice: 'Invoice',
  report: 'Report',
  proposal: 'Proposal',
  contract: 'Agreement',
  data: 'Data sheet',
  general: 'Page',
}

function PresetCard({ preset, onPick }: { preset: TemplatePreset; onPick: () => void }) {
  return (
    <Card
      role="button"
      tabIndex={0}
      onClick={onPick}
      onKeyDown={(e) => {
        if (e.key === 'Enter' || e.key === ' ') {
          e.preventDefault()
          onPick()
        }
      }}
      className="cursor-pointer transition-colors hover:border-primary/40 focus:outline-none focus:ring-2 focus:ring-primary/40"
    >
      <CardContent className="p-4">
        <div className="mb-1 flex items-center justify-between gap-2">
          <h3 className="font-semibold">{CATEGORY_LABEL[preset.category] || preset.category}</h3>
          <Badge variant="outline" className="uppercase">{String(preset.format)}</Badge>
        </div>
        <p className="text-xs text-muted-foreground">{preset.description}</p>
        <ul className="mt-2 space-y-0.5 text-xs">
          {preset.includes.map((line) => (
            <li key={line} className="flex gap-1.5">
              <span className="text-primary">•</span>
              <span>{line}</span>
            </li>
          ))}
        </ul>
        {preset.data_fields.length > 0 && (
          <div className="mt-2 flex flex-wrap items-center gap-1">
            <span className="text-[11px] text-muted-foreground">Agent supplies:</span>
            {preset.data_fields.map((f) => {
              const isList = preset.list_fields.some((lf) => lf.field === f)
              return (
                <code key={f} className="rounded bg-muted px-1 py-0.5 font-mono text-[10px]">
                  {f}
                  {isList ? '[]' : ''}
                </code>
              )
            })}
          </div>
        )}
      </CardContent>
    </Card>
  )
}

// "Pick a layout" (PRD-243): a category is the structure a template starts from —
// letterhead for a letter, line items for an invoice — not a tag on a blank page.
export function PresetPicker({ open, onOpenChange, presets, loading, mode, onPick }: PresetPickerProps) {
  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-h-[90vh] max-w-4xl overflow-y-auto">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <Layers className="h-4 w-4" /> {mode === 'new' ? 'Start from a layout' : 'Change layout'}
          </DialogTitle>
          <DialogDescription>
            {mode === 'new'
              ? 'Each layout is a complete, branded document you then edit — nothing starts from scratch.'
              : 'Replaces the current blocks and preview values with the chosen layout. The name and description stay.'}
          </DialogDescription>
        </DialogHeader>

        {loading ? (
          <LoadingState variant="cards" count={3} label="Loading layouts" />
        ) : (
          <div className="grid grid-cols-1 gap-3 md:grid-cols-2 lg:grid-cols-3">
            {presets.map((preset) => (
              <PresetCard key={preset.category} preset={preset} onPick={() => onPick(preset)} />
            ))}
            <Card
              role="button"
              tabIndex={0}
              onClick={() => onPick(null)}
              onKeyDown={(e) => {
                if (e.key === 'Enter' || e.key === ' ') {
                  e.preventDefault()
                  onPick(null)
                }
              }}
              className="cursor-pointer border-dashed transition-colors hover:border-primary/40"
            >
              <CardContent className="flex h-full flex-col items-center justify-center p-4 text-center">
                <FileText className="mb-2 h-6 w-6 text-muted-foreground" />
                <h3 className="font-semibold">Blank</h3>
                <p className="text-xs text-muted-foreground">Just a heading. Build it block by block.</p>
              </CardContent>
            </Card>
          </div>
        )}
      </DialogContent>
    </Dialog>
  )
}
