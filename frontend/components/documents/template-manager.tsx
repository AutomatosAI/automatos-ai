'use client'

import React, { useState, useEffect, useCallback } from 'react'
import {
  FileText,
  Plus,
  Pencil,
  Trash2,
  Eye,
  Upload,
} from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Card, CardContent } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogFooter,
} from '@/components/ui/dialog'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'
import { Textarea } from '@/components/ui/textarea'
import { Label } from '@/components/ui/label'
import { toast } from 'sonner'
import { apiClient } from '@/lib/api-client'

interface DocumentTemplate {
  id: string
  name: string
  description?: string
  format: string
  category: string
  tags: string[]
  version: number
  data_schema: Record<string, any>
  sample_data: Record<string, any>
  template_content?: string
  created_at?: string
}

const FORMAT_COLORS: Record<string, string> = {
  pdf: 'bg-destructive/10 text-destructive border-destructive/20',
  docx: 'bg-info/10 text-info border-info/20',
  xlsx: 'bg-success/10 text-success border-success/20',
}

const CATEGORIES = ['general', 'report', 'invoice', 'contract', 'letter', 'proposal', 'data']

export function TemplateManager() {
  const [templates, setTemplates] = useState<DocumentTemplate[]>([])
  const [loading, setLoading] = useState(true)
  const [filterFormat, setFilterFormat] = useState<string>('all')
  const [filterCategory, setFilterCategory] = useState<string>('all')
  const [showCreateModal, setShowCreateModal] = useState(false)
  const [editTemplate, setEditTemplate] = useState<DocumentTemplate | null>(null)
  const [deleteConfirm, setDeleteConfirm] = useState<string | null>(null)

  // Form state
  const [formName, setFormName] = useState('')
  const [formFormat, setFormFormat] = useState('pdf')
  const [formCategory, setFormCategory] = useState('general')
  const [formDescription, setFormDescription] = useState('')
  const [formContent, setFormContent] = useState('')
  const [formSchema, setFormSchema] = useState('{}')
  const [formSampleData, setFormSampleData] = useState('{}')

  const fetchTemplates = useCallback(async () => {
    setLoading(true)
    try {
      const params = new URLSearchParams()
      if (filterFormat !== 'all') params.set('format', filterFormat)
      if (filterCategory !== 'all') params.set('category', filterCategory)
      const data = await apiClient.get<DocumentTemplate[]>(`/api/documents/templates?${params}`)
      setTemplates(data)
    } catch (err: any) {
      console.error('[TemplateManager] Failed to fetch templates:', err)
      toast.error('Failed to load templates')
    } finally {
      setLoading(false)
    }
  }, [filterFormat, filterCategory])

  useEffect(() => { fetchTemplates() }, [fetchTemplates])

  const resetForm = () => {
    setFormName('')
    setFormFormat('pdf')
    setFormCategory('general')
    setFormDescription('')
    setFormContent('')
    setFormSchema('{}')
    setFormSampleData('{}')
  }

  const openCreate = () => {
    resetForm()
    setEditTemplate(null)
    setShowCreateModal(true)
  }

  const openEdit = (t: DocumentTemplate) => {
    setEditTemplate(t)
    setFormName(t.name)
    setFormFormat(t.format)
    setFormCategory(t.category)
    setFormDescription(t.description || '')
    setFormContent(t.template_content || '')
    setFormSchema(JSON.stringify(t.data_schema, null, 2))
    setFormSampleData(JSON.stringify(t.sample_data, null, 2))
    setShowCreateModal(true)
  }

  const handleSave = async () => {
    try {
      const body = {
        name: formName,
        format: formFormat,
        category: formCategory,
        description: formDescription,
        template_content: formFormat === 'pdf' ? formContent : undefined,
        data_schema: JSON.parse(formSchema),
        sample_data: JSON.parse(formSampleData),
      }

      if (editTemplate) {
        await apiClient.put(`/api/documents/templates/${editTemplate.id}`, body)
      } else {
        await apiClient.post('/api/documents/templates', body)
      }

      toast.success(editTemplate ? 'Template updated' : 'Template created')
      setShowCreateModal(false)
      fetchTemplates()
    } catch (e: any) {
      toast.error(e.message || 'Failed to save template')
    }
  }

  const handleDelete = async (id: string) => {
    try {
      await apiClient.delete(`/api/documents/templates/${id}`)
      toast.success('Template deleted')
      setDeleteConfirm(null)
      fetchTemplates()
    } catch {
      toast.error('Failed to delete template')
    }
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
        <div>
          <h2 className="text-xl font-bold">Document Templates</h2>
          <p className="text-sm text-muted-foreground">Manage templates for PDF, DOCX, and XLSX generation</p>
        </div>
        <Button onClick={openCreate} className="bg-orange-500 hover:bg-orange-600 text-white">
          <Plus className="w-4 h-4 mr-2" /> New Template
        </Button>
      </div>

      {/* Filters */}
      <div className="flex gap-3">
        <Select value={filterFormat} onValueChange={setFilterFormat}>
          <SelectTrigger className="w-[140px]"><SelectValue placeholder="Format" /></SelectTrigger>
          <SelectContent>
            <SelectItem value="all">All Formats</SelectItem>
            <SelectItem value="pdf">PDF</SelectItem>
            <SelectItem value="docx">DOCX</SelectItem>
            <SelectItem value="xlsx">XLSX</SelectItem>
          </SelectContent>
        </Select>
        <Select value={filterCategory} onValueChange={setFilterCategory}>
          <SelectTrigger className="w-[160px]"><SelectValue placeholder="Category" /></SelectTrigger>
          <SelectContent>
            <SelectItem value="all">All Categories</SelectItem>
            {CATEGORIES.map(c => (
              <SelectItem key={c} value={c}>{c.charAt(0).toUpperCase() + c.slice(1)}</SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>

      {/* Template Grid */}
      {loading ? (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {[1, 2, 3].map(i => (
            <Card key={i} className="animate-pulse">
              <CardContent className="p-6 h-40" />
            </Card>
          ))}
        </div>
      ) : templates.length === 0 ? (
        <Card>
          <CardContent className="p-12 text-center text-muted-foreground">
            <FileText className="h-12 w-12 mx-auto mb-4 opacity-50" />
            <p>No templates yet. Create one to get started.</p>
          </CardContent>
        </Card>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {templates.map(t => (
            <Card key={t.id} className="group hover:border-orange-500/30 transition-colors">
              <CardContent className="p-5">
                <div className="flex items-start justify-between mb-3">
                  <div className="flex-1 min-w-0">
                    <h3 className="font-semibold truncate">{t.name}</h3>
                    <p className="text-sm text-muted-foreground line-clamp-2 mt-1">
                      {t.description || 'No description'}
                    </p>
                  </div>
                  <Badge variant="outline" className={FORMAT_COLORS[t.format] || ''}>
                    {t.format.toUpperCase()}
                  </Badge>
                </div>
                <div className="flex items-center justify-between mt-4">
                  <div className="flex items-center gap-2">
                    <Badge variant="secondary" className="text-xs">{t.category}</Badge>
                    <span className="text-xs text-muted-foreground">v{t.version}</span>
                  </div>
                  <div className="flex gap-1 opacity-0 group-hover:opacity-100 transition-opacity">
                    <Button variant="ghost" size="icon" className="h-7 w-7" onClick={() => openEdit(t)}>
                      <Pencil className="h-3.5 w-3.5" />
                    </Button>
                    <Button variant="ghost" size="icon" className="h-7 w-7 text-destructive hover:text-destructive/80" onClick={() => setDeleteConfirm(t.id)}>
                      <Trash2 className="h-3.5 w-3.5" />
                    </Button>
                  </div>
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
      )}

      {/* Create/Edit Modal */}
      <Dialog open={showCreateModal} onOpenChange={setShowCreateModal}>
        <DialogContent className="max-w-2xl max-h-[90vh] overflow-y-auto">
          <DialogHeader>
            <DialogTitle>{editTemplate ? 'Edit Template' : 'Create Template'}</DialogTitle>
          </DialogHeader>
          <div className="space-y-4">
            <div className="grid grid-cols-2 gap-4">
              <div>
                <Label>Name</Label>
                <Input value={formName} onChange={e => setFormName(e.target.value)} placeholder="My Report" />
              </div>
              <div>
                <Label>Format</Label>
                <Select value={formFormat} onValueChange={setFormFormat} disabled={!!editTemplate}>
                  <SelectTrigger><SelectValue /></SelectTrigger>
                  <SelectContent>
                    <SelectItem value="pdf">PDF</SelectItem>
                    <SelectItem value="docx">DOCX</SelectItem>
                    <SelectItem value="xlsx">XLSX</SelectItem>
                  </SelectContent>
                </Select>
              </div>
            </div>
            <div className="grid grid-cols-2 gap-4">
              <div>
                <Label>Category</Label>
                <Select value={formCategory} onValueChange={setFormCategory}>
                  <SelectTrigger><SelectValue /></SelectTrigger>
                  <SelectContent>
                    {CATEGORIES.map(c => (
                      <SelectItem key={c} value={c}>{c.charAt(0).toUpperCase() + c.slice(1)}</SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
              <div>
                <Label>Description</Label>
                <Input value={formDescription} onChange={e => setFormDescription(e.target.value)} placeholder="Brief description" />
              </div>
            </div>
            {formFormat === 'pdf' && (
              <div>
                <Label>Template HTML/CSS</Label>
                <Textarea
                  value={formContent}
                  onChange={e => setFormContent(e.target.value)}
                  placeholder="<!DOCTYPE html>..."
                  className="font-mono text-xs min-h-[200px]"
                />
              </div>
            )}
            <div>
              <Label>Data Schema (JSON)</Label>
              <Textarea
                value={formSchema}
                onChange={e => setFormSchema(e.target.value)}
                className="font-mono text-xs min-h-[100px]"
              />
            </div>
            <div>
              <Label>Sample Data (JSON)</Label>
              <Textarea
                value={formSampleData}
                onChange={e => setFormSampleData(e.target.value)}
                className="font-mono text-xs min-h-[100px]"
              />
            </div>
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={() => setShowCreateModal(false)}>Cancel</Button>
            <Button onClick={handleSave} disabled={!formName} className="bg-orange-500 hover:bg-orange-600 text-white">
              {editTemplate ? 'Update' : 'Create'}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Delete Confirmation */}
      <Dialog open={!!deleteConfirm} onOpenChange={() => setDeleteConfirm(null)}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Delete Template?</DialogTitle>
          </DialogHeader>
          <p className="text-sm text-muted-foreground">This will deactivate the template. It can be restored later.</p>
          <DialogFooter>
            <Button variant="outline" onClick={() => setDeleteConfirm(null)}>Cancel</Button>
            <Button variant="destructive" onClick={() => deleteConfirm && handleDelete(deleteConfirm)}>Delete</Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  )
}
