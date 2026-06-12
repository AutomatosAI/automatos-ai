'use client'

import React, { useEffect, useRef, useState } from 'react'
import { AlertTriangle, Loader2 } from 'lucide-react'
import { Alert, AlertDescription } from '@/components/ui/alert'
import { templateBlocksApi } from './api'
import type { BlockDocument } from './types'

interface PreviewPaneProps {
  doc: BlockDocument
  data?: Record<string, any>
}

// Debounced server-side render of the block tree to HTML, shown in a sandboxed iframe
// (PRD-167 S5). Surfaces unresolved/unknown variable paths so authors fix them rather
// than ship blanks.
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
    }, 500)
    return () => {
      if (timer.current) clearTimeout(timer.current)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [serialized])

  return (
    <div className="flex h-full flex-col gap-2">
      <div className="flex items-center justify-between">
        <span className="text-sm font-medium text-muted-foreground">Preview</span>
        {loading && <Loader2 className="h-4 w-4 animate-spin text-muted-foreground" />}
      </div>

      {(unresolved.length > 0 || unknown.length > 0) && (
        <Alert variant="default" className="border-warning/40 bg-warning/5 py-2">
          <AlertTriangle className="h-4 w-4 text-warning" />
          <AlertDescription className="text-xs">
            {unresolved.length > 0 && (
              <div>
                Unresolved (will render as <code>[[path]]</code>):{' '}
                <span className="font-mono">{unresolved.join(', ')}</span>
              </div>
            )}
            {unknown.length > 0 && (
              <div>
                Unknown variable paths: <span className="font-mono">{unknown.join(', ')}</span>
              </div>
            )}
          </AlertDescription>
        </Alert>
      )}

      {error ? (
        <Alert variant="destructive" className="py-2">
          <AlertDescription className="text-xs">{error}</AlertDescription>
        </Alert>
      ) : (
        <iframe
          title="Template preview"
          className="flex-1 w-full rounded-md border bg-white"
          sandbox=""
          srcDoc={html}
        />
      )}
    </div>
  )
}
