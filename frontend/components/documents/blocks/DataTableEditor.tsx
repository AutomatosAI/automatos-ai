'use client'

import React from 'react'
import { Minus, Plus } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Switch } from '@/components/ui/switch'
import type { Block, DataTableBlock, DataTableColumn } from './types'

const ALIGN: DataTableColumn['align'][] = ['left', 'right', 'center']

interface DataTableEditorProps {
  block: DataTableBlock
  onChange: (b: Block) => void
}

// A table that fills from a data.* list at generation time (PRD-243) — the block the
// invoice and report presets are built on. The author names the field and its columns;
// an agent supplies the rows.
export function DataTableEditor({ block, onChange }: DataTableEditorProps) {
  const field = block.path.startsWith('data.') ? block.path.slice(5) : block.path
  const setColumn = (i: number, patch: Partial<DataTableColumn>) =>
    onChange({ ...block, columns: block.columns.map((c, idx) => (idx === i ? { ...c, ...patch } : c)) })
  const addColumn = () => onChange({ ...block, columns: [...block.columns, { key: `column_${block.columns.length + 1}`, label: '' }] })
  const removeColumn = (i: number) => {
    if (block.columns.length <= 1) return
    onChange({ ...block, columns: block.columns.filter((_, idx) => idx !== i) })
  }

  return (
    <div className="space-y-3">
      <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
        <div>
          <Label className="text-xs">Rows come from</Label>
          <div className="flex items-center gap-1">
            <span className="font-mono text-xs text-muted-foreground">data.</span>
            <Input
              value={field}
              onChange={(e) => onChange({ ...block, path: `data.${e.target.value.replace(/^data\./, '').replace(/[^A-Za-z0-9_.]/g, '')}` })}
              placeholder="line_items"
              className="font-mono text-sm"
            />
          </div>
          <p className="mt-1 text-[11px] text-muted-foreground">
            An agent passes a list here — one object per row, keyed by the column keys below.
          </p>
        </div>
        <div>
          <Label className="text-xs">When the list is empty</Label>
          <div className="flex items-center gap-2 pt-2">
            <Switch
              checked={block.empty_text !== null && block.empty_text !== undefined}
              onCheckedChange={(on) => onChange({ ...block, empty_text: on ? 'Nothing to report.' : null })}
              aria-label="Allow an empty list"
            />
            <span className="text-xs text-muted-foreground">
              {block.empty_text !== null && block.empty_text !== undefined ? 'Show a sentence instead' : 'Block the document (default)'}
            </span>
          </div>
          {block.empty_text !== null && block.empty_text !== undefined && (
            <Input value={block.empty_text} onChange={(e) => onChange({ ...block, empty_text: e.target.value })} className="mt-2 text-sm" />
          )}
        </div>
      </div>

      <div>
        <div className="mb-1 flex items-center justify-between">
          <Label className="text-xs">Columns</Label>
          <Button type="button" size="sm" variant="outline" className="h-7" onClick={addColumn}>
            <Plus className="mr-1 h-3 w-3" /> Column
          </Button>
        </div>
        <div className="space-y-1.5">
          {block.columns.map((c, i) => (
            <div key={i} className="grid grid-cols-[1fr_1fr_6.5rem_2rem] items-center gap-1.5">
              <Input
                value={c.key}
                onChange={(e) => setColumn(i, { key: e.target.value.replace(/[^A-Za-z0-9_]/g, '') })}
                placeholder="key (in the data)"
                className="h-8 font-mono text-xs"
                aria-label={`Column ${i + 1} key`}
              />
              <Input
                value={c.label}
                onChange={(e) => setColumn(i, { label: e.target.value })}
                placeholder="Header label"
                className="h-8 text-sm"
                aria-label={`Column ${i + 1} label`}
              />
              <Select value={c.align || 'left'} onValueChange={(align) => setColumn(i, { align: align as DataTableColumn['align'] })}>
                <SelectTrigger className="h-8 text-xs"><SelectValue /></SelectTrigger>
                <SelectContent>
                  {ALIGN.map((a) => (
                    <SelectItem key={a} value={a as string}>{a}</SelectItem>
                  ))}
                </SelectContent>
              </Select>
              <Button
                type="button"
                size="icon"
                variant="ghost"
                className="h-7 w-7 text-destructive"
                disabled={block.columns.length <= 1}
                onClick={() => removeColumn(i)}
                aria-label={`Remove column ${i + 1}`}
              >
                <Minus className="h-3.5 w-3.5" />
              </Button>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}
