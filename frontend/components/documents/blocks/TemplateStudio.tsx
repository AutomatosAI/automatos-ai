'use client'

import React, { useCallback, useEffect, useState } from 'react'
import { ArrowLeft, Copy, FileText, Palette, Pencil, Plus, Save } from 'lucide-react'
import { toast } from 'sonner'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Badge } from '@/components/ui/badge'
import { Card, CardContent } from '@/components/ui/card'
import { Textarea } from '@/components/ui/textarea'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'
import { apiClient } from '@/lib/api-client'
import { BlockEditor } from './BlockEditor'
import { PreviewPane } from './PreviewPane'
import { BrandKitDialog } from './BrandKitDialog'
import { templateBlocksApi } from './api'
import { newBlockId } from './inline'
import { SCHEMA_VERSION } from './types'
import type { Block, BlockDocument, VariableEntry } from './types'

interface TemplateSummary {
  id: string
  name: string
  description?: string
  format: string
  category: string
  has_blocks?: boolean
}

const CATEGORIES = ['general', 'report', 'invoice', 'contract', 'letter', 'proposal', 'data']

// PRD-167 S5: the non-technical block-template studio — gallery (copy-on-customise),
// block editor, live preview, and brand-kit access in one surface.
export function TemplateStudio() {
  const [mode, setMode] = useState<'gallery' | 'editor'>('gallery')
  const [templates, setTemplates] = useState<TemplateSummary[]>([])
  const [variables, setVariables] = useState<VariableEntry[]>([])
  const [brandOpen, setBrandOpen] = useState(false)
  const [loading, setLoading] = useState(true)

  // Editor state
  const [editingId, setEditingId] = useState<string | null>(null)
  const [name, setName] = useState('')
  const [description, setDescription] = useState('')
  const [category, setCategory] = useState('report')
  const [format, setFormat] = useState('pdf')
  const [blocks, setBlocks] = useState<Block[]>([])
  const [previewData, setPreviewData] = useState('{\n  "data": {}\n}')
  const [saving, setSaving] = useState(false)

  const loadGallery = useCallback(async () => {
    setLoading(true)
    try {
      const [tpls, vars] = await Promise.all([
        apiClient.get<TemplateSummary[]>('/api/documents/templates'),
        templateBlocksApi.getVariables(),
      ])
      setTemplates(tpls)
      setVariables(vars.variables)
    } catch (e) {
      toast.error('Failed to load templates')
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    loadGallery()
  }, [loadGallery])

  const startBlank = () => {
    setEditingId(null)
    setName('')
    setDescription('')
    setCategory('report')
    setFormat('pdf')
    setBlocks([{ type: 'heading', id: newBlockId(), level: 1, content: [] }])
    setPreviewData('{\n  "data": {}\n}')
    setMode('editor')
  }

  const openTemplate = async (id: string, asCopy: boolean) => {
    try {
      const full = await apiClient.get<any>(`/api/documents/templates/${id}`)
      setEditingId(asCopy ? null : id)
      setName(asCopy ? `${full.name} (copy)` : full.name)
      setDescription(full.description || '')
      setCategory(full.category || 'report')
      setFormat(full.format || 'pdf')
      const doc: BlockDocument | null = full.blocks
      setBlocks(doc?.blocks ?? [])
      setPreviewData(JSON.stringify(full.sample_data || { data: {} }, null, 2))
      setMode('editor')
    } catch {
      toast.error('Failed to open template')
    }
  }

  const parsedPreviewData = (() => {
    try {
      return JSON.parse(previewData)
    } catch {
      return {}
    }
  })()

  const save = async () => {
    if (!name.trim()) {
      toast.error('Template needs a name')
      return
    }
    setSaving(true)
    const doc: BlockDocument = { version: SCHEMA_VERSION, blocks }
    const body = { name, description, category, format, blocks: doc }
    try {
      if (editingId) {
        await apiClient.put(`/api/documents/templates/${editingId}`, body)
      } else {
        await apiClient.post('/api/documents/templates', body)
      }
      toast.success('Template saved')
      setMode('gallery')
      loadGallery()
    } catch (e: any) {
      // Surface field-level block errors from the 422 (PRD-167 S2).
      const detail = e?.response?.data?.detail
      if (detail?.errors) {
        toast.error(`Invalid blocks: ${detail.errors.map((x: any) => `${x.loc} ${x.msg}`).join('; ')}`)
      } else {
        toast.error(e?.message || 'Failed to save template')
      }
    } finally {
      setSaving(false)
    }
  }

  if (mode === 'editor') {
    return (
      <div className="space-y-4">
        <div className="flex items-center justify-between gap-3">
          <Button variant="ghost" size="sm" onClick={() => setMode('gallery')}>
            <ArrowLeft className="mr-2 h-4 w-4" /> Back
          </Button>
          <div className="flex items-center gap-2">
            <Button variant="outline" size="sm" onClick={() => setBrandOpen(true)}>
              <Palette className="mr-2 h-4 w-4" /> Brand Kit
            </Button>
            <Button size="sm" onClick={save} disabled={saving}>
              <Save className="mr-2 h-4 w-4" /> {saving ? 'Saving…' : 'Save template'}
            </Button>
          </div>
        </div>

        <div className="grid grid-cols-1 gap-3 sm:grid-cols-3">
          <div className="sm:col-span-1">
            <Label className="text-xs">Name</Label>
            <Input value={name} onChange={(e) => setName(e.target.value)} placeholder="Branded Letter" />
          </div>
          <div>
            <Label className="text-xs">Category</Label>
            <Select value={category} onValueChange={setCategory}>
              <SelectTrigger><SelectValue /></SelectTrigger>
              <SelectContent>
                {CATEGORIES.map((c) => (
                  <SelectItem key={c} value={c}>{c[0].toUpperCase() + c.slice(1)}</SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
          <div>
            <Label className="text-xs">Description</Label>
            <Input value={description} onChange={(e) => setDescription(e.target.value)} placeholder="What this template is for" />
          </div>
        </div>

        <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
          <div className="space-y-3">
            <BlockEditor blocks={blocks} variables={variables} onChange={setBlocks} />
            <div>
              <Label className="text-xs text-muted-foreground">Preview data (fills {'{{data.*}}'} chips)</Label>
              <Textarea
                value={previewData}
                onChange={(e) => setPreviewData(e.target.value)}
                className="font-mono text-xs min-h-[80px]"
              />
            </div>
          </div>
          <div className="lg:sticky lg:top-4 lg:h-[80vh]">
            <PreviewPane doc={{ version: SCHEMA_VERSION, blocks }} data={parsedPreviewData.data || parsedPreviewData} />
          </div>
        </div>

        <BrandKitDialog open={brandOpen} onOpenChange={setBrandOpen} />
      </div>
    )
  }

  // Gallery
  return (
    <div className="space-y-6">
      <div className="flex flex-col items-start justify-between gap-4 sm:flex-row sm:items-center">
        <div>
          <h2 className="text-xl font-bold">Template Studio</h2>
          <p className="text-sm text-muted-foreground">
            Build branded document templates with blocks and variable chips — no code.
          </p>
        </div>
        <div className="flex gap-2">
          <Button variant="outline" onClick={() => setBrandOpen(true)}>
            <Palette className="mr-2 h-4 w-4" /> Brand Kit
          </Button>
          <Button onClick={startBlank}>
            <Plus className="mr-2 h-4 w-4" /> New template
          </Button>
        </div>
      </div>

      {loading ? (
        <div className="grid grid-cols-1 gap-4 md:grid-cols-2 lg:grid-cols-3">
          {[1, 2, 3].map((i) => (
            <Card key={i} className="animate-pulse"><CardContent className="h-32 p-6" /></Card>
          ))}
        </div>
      ) : (
        <div className="grid grid-cols-1 gap-4 md:grid-cols-2 lg:grid-cols-3">
          {templates.map((t) => (
            <Card key={t.id} className="group transition-colors hover:border-primary/30">
              <CardContent className="p-5">
                <div className="mb-3 flex items-start justify-between">
                  <div className="min-w-0 flex-1">
                    <h3 className="truncate font-semibold">{t.name}</h3>
                    <p className="mt-1 line-clamp-2 text-sm text-muted-foreground">{t.description || 'No description'}</p>
                  </div>
                  <Badge variant="outline" className="uppercase">{t.format}</Badge>
                </div>
                <div className="mt-4 flex items-center justify-between">
                  <Badge variant="secondary" className="text-xs">{t.category}</Badge>
                  <div className="flex gap-1">
                    {t.has_blocks && (
                      <Button variant="ghost" size="sm" className="h-7" onClick={() => openTemplate(t.id, false)}>
                        <Pencil className="mr-1.5 h-3.5 w-3.5" /> Edit
                      </Button>
                    )}
                    <Button variant="ghost" size="sm" className="h-7" onClick={() => openTemplate(t.id, true)}>
                      <Copy className="mr-1.5 h-3.5 w-3.5" /> Copy
                    </Button>
                  </div>
                </div>
              </CardContent>
            </Card>
          ))}
          {templates.length === 0 && (
            <Card className="md:col-span-2 lg:col-span-3">
              <CardContent className="p-12 text-center text-muted-foreground">
                <FileText className="mx-auto mb-4 h-12 w-12 opacity-50" />
                <p>No templates yet. Create one to get started.</p>
              </CardContent>
            </Card>
          )}
        </div>
      )}

      <BrandKitDialog open={brandOpen} onOpenChange={setBrandOpen} />
    </div>
  )
}
