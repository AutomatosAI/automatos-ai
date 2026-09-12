'use client'

import React, { useEffect, useState } from 'react'
import Link from 'next/link'
import { Check, Copy, Download, ExternalLink, FileText, Loader2 } from 'lucide-react'
import { toast } from 'sonner'
import { Alert, AlertDescription } from '@/components/ui/alert'
import { Button } from '@/components/ui/button'
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle } from '@/components/ui/dialog'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { FieldHelp } from '@/components/ui/help-tooltip'
import { downloadGeneratedFile, templateBlocksApi } from './api'
import { PreviewDataForm } from './PreviewDataForm'
import type { GenerateDocumentResult, TemplateSummary, UnresolvedDetail } from './types'

interface GenerateDocumentDialogProps {
  open: boolean
  onOpenChange: (open: boolean) => void
  template: TemplateSummary | null
  // Prefill for the data.* fields (the editor's preview data, or the template's sample data).
  initialData?: Record<string, any>
  missingOnFile?: string[]
  onOpenBrandKit: () => void
}

const FORMATS = ['pdf', 'docx'] as const

function parseUnresolved(message: string): UnresolvedDetail | null {
  // The API client stringifies a JSON `detail`; recover the finalisation-gate shape.
  try {
    const parsed = JSON.parse(message)
    return parsed && typeof parsed === 'object' && 'message' in parsed ? (parsed as UnresolvedDetail) : null
  } catch {
    return null
  }
}

function CopyLink({ url }: { url: string }) {
  const [copied, setCopied] = useState(false)
  return (
    <Button
      type="button"
      size="sm"
      variant="outline"
      className="h-7 gap-1.5"
      onClick={async () => {
        try {
          await navigator.clipboard.writeText(url)
          setCopied(true)
          setTimeout(() => setCopied(false), 1500)
        } catch {
          toast.error('Could not copy the link')
        }
      }}
    >
      {copied ? <Check className="h-3.5 w-3.5" /> : <Copy className="h-3.5 w-3.5" />} Copy share link
    </Button>
  )
}

// Render a real document from a template, right here (PRD-242 S5). It is saved to
// Deliverables like an agent's output, and the dialog hands over the same links.
export function GenerateDocumentDialog({
  open,
  onOpenChange,
  template,
  initialData,
  missingOnFile = [],
  onOpenBrandKit,
}: GenerateDocumentDialogProps) {
  const [title, setTitle] = useState('')
  const [format, setFormat] = useState<string>('pdf')
  const [data, setData] = useState<Record<string, any>>({})
  const [busy, setBusy] = useState(false)
  const [result, setResult] = useState<GenerateDocumentResult | null>(null)
  const [blocked, setBlocked] = useState<UnresolvedDetail | null>(null)

  useEffect(() => {
    if (!open || !template) return
    setTitle(template.name)
    setFormat(FORMATS.includes(template.format as any) ? String(template.format) : 'pdf')
    const seed = initialData ?? (template.sample_data?.data as Record<string, any> | undefined) ?? template.sample_data ?? {}
    setData(seed && typeof seed === 'object' ? seed : {})
    setResult(null)
    setBlocked(null)
  }, [open, template, initialData])

  const generate = async () => {
    if (!template) return
    if (!title.trim()) {
      toast.error('Give the document a title')
      return
    }
    setBusy(true)
    setBlocked(null)
    try {
      const res = await templateBlocksApi.generateDocument({ title: title.trim(), format, template_id: template.id, data })
      setResult(res)
      toast.success('Document generated and saved to Deliverables')
    } catch (e: any) {
      const detail = parseUnresolved(String(e?.message || ''))
      if (detail) setBlocked(detail)
      else toast.error(e?.message || 'Generation failed')
    } finally {
      setBusy(false)
    }
  }

  const download = async () => {
    if (!result) return
    try {
      await downloadGeneratedFile(result.download_url, result.filename)
    } catch (e: any) {
      toast.error(e?.message || 'Download failed')
    }
  }

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-h-[90vh] max-w-2xl overflow-y-auto">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <FileText className="h-4 w-4" /> Generate from “{template?.name}”
          </DialogTitle>
          <DialogDescription>
            Fills the template with the values below, renders it, and saves it to Deliverables — exactly what an agent does with generate_document.
          </DialogDescription>
        </DialogHeader>

        {result ? (
          <div className="space-y-3">
            <Alert className="border-success/40 bg-success/5">
              <AlertDescription className="text-sm">
                <strong>{result.filename}</strong> ({result.size_kb} KB) is saved to Deliverables
                {result.template_name ? ` from “${result.template_name}”` : ''}.
              </AlertDescription>
            </Alert>
            <div className="flex flex-wrap items-center gap-2">
              <Button type="button" size="sm" className="h-8 gap-1.5" onClick={download}>
                <Download className="h-3.5 w-3.5" /> Download
              </Button>
              <Button asChild type="button" size="sm" variant="outline" className="h-8 gap-1.5">
                <Link href="/deliverables?tab=outputs">
                  <ExternalLink className="h-3.5 w-3.5" /> Open Deliverables
                </Link>
              </Button>
              {result.share_url && <CopyLink url={result.share_url} />}
            </div>
            <p className="text-xs text-muted-foreground">
              {result.share_url
                ? 'The share link opens without signing in and expires in 7 days — it is what an agent emails or posts.'
                : 'No share link: object storage holds no copy of this file, so only signed-in workspace members can download it.'}
            </p>
          </div>
        ) : (
          <div className="space-y-4">
            {blocked && (
              <Alert variant="destructive" className="py-2">
                <AlertDescription className="text-xs">
                  <p className="font-medium">Not delivered — some chips did not resolve.</p>
                  {blocked.unresolved && blocked.unresolved.length > 0 && (
                    <p>
                      Empty: <span className="font-mono">{blocked.unresolved.join(', ')}</span> — fill them below or in the Brand Kit.
                    </p>
                  )}
                  {blocked.unknown && blocked.unknown.length > 0 && (
                    <p>
                      Unknown chips (fix the template): <span className="font-mono">{blocked.unknown.join(', ')}</span>
                    </p>
                  )}
                </AlertDescription>
              </Alert>
            )}
            <div className="grid grid-cols-1 gap-3 sm:grid-cols-3">
              <div className="sm:col-span-2">
                <Label className="text-xs">Document title</Label>
                <Input value={title} onChange={(e) => setTitle(e.target.value)} />
              </div>
              <div>
                <Label className="flex items-center text-xs">
                  Format <FieldHelp id="deliverables.templates.editor.format" />
                </Label>
                <Select value={format} onValueChange={setFormat}>
                  <SelectTrigger><SelectValue /></SelectTrigger>
                  <SelectContent>
                    {FORMATS.map((f) => (
                      <SelectItem key={f} value={f}>{f.toUpperCase()}</SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
            </div>
            <PreviewDataForm
              fields={template?.data_fields ?? []}
              data={data}
              onChange={setData}
              missingOnFile={missingOnFile}
              onOpenBrandKit={onOpenBrandKit}
              title="Values for this document"
            />
          </div>
        )}

        <DialogFooter>
          <Button type="button" variant="outline" onClick={() => onOpenChange(false)}>{result ? 'Close' : 'Cancel'}</Button>
          {!result && (
            <Button type="button" onClick={generate} disabled={busy || !template}>
              {busy ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : <FileText className="mr-2 h-4 w-4" />}
              {busy ? 'Rendering…' : 'Generate'}
            </Button>
          )}
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}
