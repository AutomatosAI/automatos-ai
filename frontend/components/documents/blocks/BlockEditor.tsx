'use client'

import React, { useRef } from 'react'
import {
  ChevronDown,
  ChevronUp,
  GripVertical,
  Heading,
  Image as ImageIcon,
  Minus,
  Plus,
  Table as TableIcon,
  Trash2,
  Type,
  Variable as VariableIcon,
  FoldVertical,
} from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Textarea } from '@/components/ui/textarea'
import { Checkbox } from '@/components/ui/checkbox'
import { Label } from '@/components/ui/label'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu'
import { VariablePicker } from './VariablePicker'
import { insertToken, newBlockId, parseInline, serializeInline } from './inline'
import type {
  Block,
  BlockType,
  HeadingBlock,
  ImageBlock,
  Inline,
  SectionBlock,
  TableBlock,
  TextBlock,
  VariableBlock,
  VariableEntry,
} from './types'

interface BlockEditorProps {
  blocks: Block[]
  variables: VariableEntry[]
  onChange: (blocks: Block[]) => void
}

// ---- a single inline-content field (string <-> Inline[] with {{token}} chips) ----

function InlineField({
  content,
  onChange,
  variables,
  multiline,
  placeholder,
}: {
  content: Inline[]
  onChange: (content: Inline[]) => void
  variables: VariableEntry[]
  multiline?: boolean
  placeholder?: string
}) {
  const ref = useRef<HTMLTextAreaElement | HTMLInputElement | null>(null)
  const caret = useRef<number>(0)
  const text = serializeInline(content)

  const captureCaret = () => {
    const el = ref.current
    if (el && typeof el.selectionStart === 'number') caret.current = el.selectionStart
  }
  const handleInsert = (path: string) => {
    const next = insertToken(text, caret.current, path)
    onChange(parseInline(next))
  }

  return (
    <div className="space-y-1.5">
      {multiline ? (
        <Textarea
          ref={ref as React.RefObject<HTMLTextAreaElement>}
          value={text}
          placeholder={placeholder}
          onChange={(e) => onChange(parseInline(e.target.value))}
          onSelect={captureCaret}
          onKeyUp={captureCaret}
          onClick={captureCaret}
          className="min-h-[72px]"
        />
      ) : (
        <Input
          ref={ref as React.RefObject<HTMLInputElement>}
          value={text}
          placeholder={placeholder}
          onChange={(e) => onChange(parseInline(e.target.value))}
          onSelect={captureCaret}
          onKeyUp={captureCaret}
          onClick={captureCaret}
        />
      )}
      <VariablePicker variables={variables} onInsert={handleInsert} label="Insert variable" />
    </div>
  )
}

// ---- per-type block editors ----

function HeadingEditor({ block, onChange, variables }: { block: HeadingBlock; onChange: (b: Block) => void; variables: VariableEntry[] }) {
  return (
    <div className="space-y-2">
      <Select value={String(block.level)} onValueChange={(v) => onChange({ ...block, level: Number(v) })}>
        <SelectTrigger className="w-28"><SelectValue /></SelectTrigger>
        <SelectContent>
          {[1, 2, 3, 4, 5, 6].map((l) => (
            <SelectItem key={l} value={String(l)}>{`Heading ${l}`}</SelectItem>
          ))}
        </SelectContent>
      </Select>
      <InlineField content={block.content} onChange={(content) => onChange({ ...block, content })} variables={variables} placeholder="Heading text…" />
    </div>
  )
}

function TextEditor({ block, onChange, variables }: { block: TextBlock; onChange: (b: Block) => void; variables: VariableEntry[] }) {
  return <InlineField content={block.content} onChange={(content) => onChange({ ...block, content })} variables={variables} multiline placeholder="Paragraph text… use Insert variable for {{chips}}" />
}

function ImageEditor({ block, onChange }: { block: ImageBlock; onChange: (b: Block) => void }) {
  return (
    <div className="grid grid-cols-2 gap-3">
      <div>
        <Label className="text-xs">Source</Label>
        <Select value={block.source} onValueChange={(v) => onChange({ ...block, source: v as ImageBlock['source'] })}>
          <SelectTrigger><SelectValue /></SelectTrigger>
          <SelectContent>
            <SelectItem value="brand_logo">Brand logo</SelectItem>
            <SelectItem value="url">URL</SelectItem>
            <SelectItem value="upload">Uploaded path</SelectItem>
          </SelectContent>
        </Select>
      </div>
      <div>
        <Label className="text-xs">Width (mm)</Label>
        <Input
          type="number"
          value={block.width_mm ?? ''}
          placeholder="auto"
          onChange={(e) => onChange({ ...block, width_mm: e.target.value ? Number(e.target.value) : null })}
        />
      </div>
      {block.source !== 'brand_logo' && (
        <div className="col-span-2">
          <Label className="text-xs">Source URL / path</Label>
          <Input value={block.src ?? ''} onChange={(e) => onChange({ ...block, src: e.target.value })} placeholder="https://… or workspace path" />
        </div>
      )}
      <div className="col-span-2">
        <Label className="text-xs">Alt text</Label>
        <Input value={block.alt} onChange={(e) => onChange({ ...block, alt: e.target.value })} placeholder="Describe the image" />
      </div>
    </div>
  )
}

function VariableBlockEditor({ block, onChange, variables }: { block: VariableBlock; onChange: (b: Block) => void; variables: VariableEntry[] }) {
  return (
    <div className="flex items-end gap-2">
      <div className="flex-1">
        <Label className="text-xs">Variable path</Label>
        <Input value={block.path} onChange={(e) => onChange({ ...block, path: e.target.value })} placeholder="data.summary" className="font-mono text-sm" />
      </div>
      <VariablePicker variables={variables} onInsert={(path) => onChange({ ...block, path })} label="Pick" />
    </div>
  )
}

function TableEditor({ block, onChange, variables }: { block: TableBlock; onChange: (b: Block) => void; variables: VariableEntry[] }) {
  const rows = block.rows
  const nCols = Math.max(1, ...rows.map((r) => r.length))

  const setCell = (r: number, c: number, content: Inline[]) => {
    const next = rows.map((row, ri) =>
      ri === r ? row.map((cell, ci) => (ci === c ? content : cell)) : row,
    )
    onChange({ ...block, rows: next })
  }
  const addRow = () => onChange({ ...block, rows: [...rows, Array.from({ length: nCols }, () => [] as Inline[])] })
  const addCol = () => onChange({ ...block, rows: rows.map((r) => [...r, [] as Inline[]]) })
  const removeRow = (r: number) => onChange({ ...block, rows: rows.filter((_, ri) => ri !== r) })
  const removeCol = (c: number) => onChange({ ...block, rows: rows.map((r) => r.filter((_, ci) => ci !== c)) })

  return (
    <div className="space-y-2">
      <div className="flex items-center gap-3">
        <label className="flex items-center gap-2 text-xs">
          <Checkbox checked={block.header} onCheckedChange={(v) => onChange({ ...block, header: !!v })} />
          First row is header
        </label>
        <Button type="button" size="sm" variant="outline" className="h-7" onClick={addRow}><Plus className="h-3 w-3 mr-1" />Row</Button>
        <Button type="button" size="sm" variant="outline" className="h-7" onClick={addCol}><Plus className="h-3 w-3 mr-1" />Column</Button>
      </div>
      <div className="overflow-x-auto">
        <table className="border-collapse">
          <tbody>
            {rows.map((row, r) => (
              <tr key={r}>
                {row.map((cell, c) => (
                  <td key={c} className="border p-1 align-top">
                    <Input
                      value={serializeInline(cell)}
                      onChange={(e) => setCell(r, c, parseInline(e.target.value))}
                      className="h-8 min-w-[120px] text-sm"
                    />
                  </td>
                ))}
                <td className="pl-1">
                  <Button type="button" size="icon" variant="ghost" className="h-7 w-7 text-destructive" onClick={() => removeRow(r)}>
                    <Minus className="h-3.5 w-3.5" />
                  </Button>
                </td>
              </tr>
            ))}
            <tr>
              {Array.from({ length: nCols }).map((_, c) => (
                <td key={c} className="text-center">
                  <Button type="button" size="icon" variant="ghost" className="h-6 w-6 text-destructive" onClick={() => removeCol(c)}>
                    <Minus className="h-3 w-3" />
                  </Button>
                </td>
              ))}
            </tr>
          </tbody>
        </table>
      </div>
      <VariablePicker
        variables={variables}
        label="Copy a {{chip}} to paste into a cell"
        onInsert={() => { /* chips are typed directly into cells; picker here is a reference */ }}
      />
    </div>
  )
}

// ---- block factory ----

function makeBlock(type: BlockType): Block {
  const id = newBlockId()
  switch (type) {
    case 'heading':
      return { type, id, level: 2, content: [] }
    case 'text':
      return { type, id, content: [] }
    case 'table':
      return { type, id, header: true, rows: [[[], []], [[], []]] }
    case 'image':
      return { type, id, source: 'brand_logo', alt: '', width_mm: 40 }
    case 'variable':
      return { type, id, path: 'data.value' }
    case 'page_break':
      return { type, id }
    case 'section':
      return { type, id, title: 'Section', children: [] }
  }
}

const BLOCK_MENU: { type: BlockType; label: string; icon: React.ComponentType<any> }[] = [
  { type: 'heading', label: 'Heading', icon: Heading },
  { type: 'text', label: 'Text', icon: Type },
  { type: 'table', label: 'Table', icon: TableIcon },
  { type: 'image', label: 'Image / Logo', icon: ImageIcon },
  { type: 'variable', label: 'Variable', icon: VariableIcon },
  { type: 'section', label: 'Section', icon: FoldVertical },
  { type: 'page_break', label: 'Page break', icon: Minus },
]

function AddBlockMenu({ onAdd }: { onAdd: (type: BlockType) => void }) {
  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <Button type="button" variant="outline" size="sm" className="gap-1.5">
          <Plus className="h-4 w-4" /> Add block
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="start">
        {BLOCK_MENU.map(({ type, label, icon: Icon }) => (
          <DropdownMenuItem key={type} onClick={() => onAdd(type)}>
            <Icon className="h-4 w-4 mr-2" /> {label}
          </DropdownMenuItem>
        ))}
      </DropdownMenuContent>
    </DropdownMenu>
  )
}

const BLOCK_LABELS: Record<BlockType, string> = {
  heading: 'Heading',
  text: 'Text',
  table: 'Table',
  image: 'Image',
  variable: 'Variable',
  page_break: 'Page break',
  section: 'Section',
}

// ---- recursive list ----

export function BlockEditor({ blocks, variables, onChange }: BlockEditorProps) {
  const updateAt = (i: number, b: Block) => onChange(blocks.map((blk, idx) => (idx === i ? b : blk)))
  const removeAt = (i: number) => onChange(blocks.filter((_, idx) => idx !== i))
  const move = (i: number, dir: -1 | 1) => {
    const j = i + dir
    if (j < 0 || j >= blocks.length) return
    const next = [...blocks]
    ;[next[i], next[j]] = [next[j], next[i]]
    onChange(next)
  }
  const add = (type: BlockType) => onChange([...blocks, makeBlock(type)])

  return (
    <div className="space-y-3">
      {blocks.map((block, i) => (
        <div key={block.id} className="rounded-md border bg-card p-3">
          <div className="mb-2 flex items-center justify-between">
            <div className="flex items-center gap-2 text-xs font-medium text-muted-foreground">
              <GripVertical className="h-3.5 w-3.5" />
              {BLOCK_LABELS[block.type]}
            </div>
            <div className="flex gap-0.5">
              <Button type="button" size="icon" variant="ghost" className="h-7 w-7" onClick={() => move(i, -1)} disabled={i === 0}>
                <ChevronUp className="h-3.5 w-3.5" />
              </Button>
              <Button type="button" size="icon" variant="ghost" className="h-7 w-7" onClick={() => move(i, 1)} disabled={i === blocks.length - 1}>
                <ChevronDown className="h-3.5 w-3.5" />
              </Button>
              <Button type="button" size="icon" variant="ghost" className="h-7 w-7 text-destructive" onClick={() => removeAt(i)}>
                <Trash2 className="h-3.5 w-3.5" />
              </Button>
            </div>
          </div>

          {block.type === 'heading' && <HeadingEditor block={block} onChange={(b) => updateAt(i, b)} variables={variables} />}
          {block.type === 'text' && <TextEditor block={block} onChange={(b) => updateAt(i, b)} variables={variables} />}
          {block.type === 'table' && <TableEditor block={block} onChange={(b) => updateAt(i, b)} variables={variables} />}
          {block.type === 'image' && <ImageEditor block={block} onChange={(b) => updateAt(i, b)} />}
          {block.type === 'variable' && <VariableBlockEditor block={block} onChange={(b) => updateAt(i, b)} variables={variables} />}
          {block.type === 'page_break' && <p className="text-xs italic text-muted-foreground">Forces a new page when rendered.</p>}
          {block.type === 'section' && (
            <div className="space-y-2">
              <Input
                value={(block as SectionBlock).title ?? ''}
                onChange={(e) => updateAt(i, { ...(block as SectionBlock), title: e.target.value })}
                placeholder="Section title"
              />
              <div className="border-l-2 border-muted pl-3">
                <BlockEditor
                  blocks={(block as SectionBlock).children}
                  variables={variables}
                  onChange={(children) => updateAt(i, { ...(block as SectionBlock), children })}
                />
              </div>
            </div>
          )}
        </div>
      ))}
      <AddBlockMenu onAdd={add} />
    </div>
  )
}
