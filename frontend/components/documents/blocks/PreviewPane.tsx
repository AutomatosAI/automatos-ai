'use client'

import React, { useEffect, useRef, useState } from 'react'
import { AlertTriangle, Loader2 } from 'lucide-react'
import { Alert, AlertDescription } from '@/components/ui/alert'
import { FieldHelp } from '@/components/ui/help-tooltip'
import { templateBlocksApi } from './api'
import type { BlockDocument } from './types'

interface PreviewPaneProps {
  doc: BlockDocument
  data?: Record<string, any>
}

const DEBOUNCE_MS = 500

// Debounced server-side render of the block tree to HTML, shown in a sandboxed iframe
// (PRD-167 S5). Surfaces unresolved/unknown variable paths so authors fix them rather
// than ship blanks — split so the author knows WHO fills each gap (PRD-242 S5).
export function PreviewPane({ doc, data }: PreviewPaneProps) {
  const [html, setHtml] = useState('')
  const [unresolved, setUnresolved] = useState<string[]>([])
  const [unknown, setUnknown] = useState<string[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null)

  const serialized = JSON.stringify(doc) + '|' + JSON.stringify(data || {})

  useEffect(() => {
    if (timer.current) clearTimeout(timer.current)
    timer.current = setTimeout(async () => {
      setLoading(true)
      setError(null)
      try {
        const res = await templateBlocksApi.previewBlocks(doc, data || {})
        setHtml(res.html)
        setUnresolved(res.unresolved || [])
        setUnknown(res.unknown || [])
      } catch (e: any) {
        setError(e?.message || 'Preview failed')
      } finally {
        setLoading(false)
      }
    }, DEBOUNCE_MS)
    return () => {
      if (timer.current) clearTimeout(timer.current)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [serialized])

  const perDocument = unresolved.filter((p) => p.startsWith('data.'))
  const onFile = unresolved.filter((p) => !p.startsWith('data.'))

  return (
    <div className="flex h-full flex-col gap-2">
      <div className="flex items-center justify-between">
        <span className="flex items-center text-sm font-medium text-muted-foreground">
          Preview <FieldHelp id="deliverables.templates.editor.preview" />
        </span>
        {loading && <Loader2 className="h-4 w-4 animate-spin text-muted-foreground" />}
      </div>

      {(unresolved.length > 0 || unknown.length > 0) && (
        <Alert variant="default" className="border-warning/40 bg-warning/5 py-2">
          <AlertTriangle className="h-4 w-4 text-warning" />
          <AlertDescription className="space-y-1 text-xs">
            {perDocument.length > 0 && (
              <div>
                Filled per document (type a preview value or leave for the agent):{' '}
                <span className="font-mono">{perDocument.join(', ')}</span>
              </div>
            )}
            {onFile.length > 0 && (
              <div>
                Not on file yet — set in Brand Kit / your profile: <span className="font-mono">{onFile.join(', ')}</span>
              </div>
            )}
            {unknown.length > 0 && (
              <div>
                Unknown chips (typo? see “Insert variable” for the list): <span className="font-mono">{unknown.join(', ')}</span>
              </div>
            )}
            <div className="text-muted-foreground">Red <code>[[markers]]</code> below show where each gap lands. A document is never delivered with one.</div>
          </AlertDescription>
        </Alert>
      )}

      {error ? (
        <Alert variant="destructive" className="py-2">
          <AlertDescription className="text-xs">Preview failed: {error}</AlertDescription>
        </Alert>
      ) : (
        <iframe
          title="Template preview"
          className="w-full flex-1 rounded-md border bg-white"
          sandbox=""
          srcDoc={html}
        />
      )}
    </div>
  )
}
