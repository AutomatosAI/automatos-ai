'use client'

import { useMemo, useState } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { Download, Search } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { toast } from 'sonner'

interface PandasAIChart {
  filename: string
  mime_type: string
  base64: string
}

interface PandasAIInsight {
  summary?: string
  charts?: PandasAIChart[]
  error?: string
}

export interface SheetArtifactProps {
  content?: string
  metadata?: Record<string, any>
}

function toCSV(columns: string[], rows: any[]): string {
  const escape = (value: any) => {
    if (value === null || value === undefined) return ''
    const str = String(value)
    const needsQuotes = /[",\n\r]/.test(str)
    const escaped = str.replace(/"/g, '""')
    return needsQuotes ? `"${escaped}"` : escaped
  }

  const lines: string[] = []
  lines.push(columns.map(escape).join(','))
  for (const row of rows) {
    lines.push(columns.map((col) => escape(row?.[col])).join(','))
  }
  return lines.join('\n')
}

export function SheetArtifact({ content, metadata }: SheetArtifactProps) {
  const pandasAI = (metadata?.pandas_ai ?? null) as PandasAIInsight | null
  const status = metadata?.status as string | undefined
  const clarifications = metadata?.clarifications as string[] | undefined
  const infoMessage = metadata?.message as string | undefined

  const rawColumns = (metadata?.columns ?? []) as string[]
  const rawRows = (metadata?.data ?? []) as any[]

  const columns = useMemo(() => {
    if (Array.isArray(rawColumns) && rawColumns.length > 0) return rawColumns
    if (Array.isArray(rawRows) && rawRows.length > 0) return Object.keys(rawRows[0] ?? {})
    return []
  }, [rawColumns, rawRows])

  const [search, setSearch] = useState('')
  const [page, setPage] = useState(1)
  const pageSize = 25

  const filteredRows = useMemo(() => {
    if (!search.trim()) return rawRows
    const q = search.toLowerCase()
    return rawRows.filter((row) => {
      try {
        return JSON.stringify(row).toLowerCase().includes(q)
      } catch {
        return false
      }
    })
  }, [rawRows, search])

  const totalPages = Math.max(1, Math.ceil(filteredRows.length / pageSize))

  const pageRows = useMemo(() => {
    const safePage = Math.min(Math.max(1, page), totalPages)
    const start = (safePage - 1) * pageSize
    return filteredRows.slice(start, start + pageSize)
  }, [filteredRows, page, totalPages])

  const sql = (metadata?.sql ?? '') as string

  return (
    <div className="space-y-6">
      {/* Clarification flow */} 
      {status === 'needs_clarification' && clarifications && clarifications.length > 0 && (
        <div className="rounded-xl border border-orange-500/30 bg-orange-500/10 p-4">
          <div className="text-sm font-semibold text-orange-200">Clarification needed</div>
          {infoMessage && <div className="mt-1 text-sm text-orange-100/80">{infoMessage}</div>}
          <div className="mt-3 space-y-2">
            {clarifications.map((q, idx) => (
              <button
                key={idx}
                className="w-full rounded-lg border border-gray-800/60 bg-gray-900/40 px-3 py-2 text-left text-sm text-gray-200 hover:border-orange-400/40 hover:bg-orange-500/5"
                onClick={async () => {
                  if (!navigator.clipboard) {
                    toast.error('Clipboard API is not available')
                    return
                  }
                  try {
                    await navigator.clipboard.writeText(q)
                    toast.success('Question copied (paste into chat to answer)')
                  } catch {
                    toast.error('Failed to copy')
                  }
                }}
                type="button"
              >
                {q}
              </button>
            ))}
          </div>
          <div className="mt-3 text-xs text-orange-100/70">
            Tip: answer these in chat (you can paste), then re-run the query.
          </div>
        </div>
      )}

      {/* Summary markdown (optional) */}
      {content && (
        <ReactMarkdown
          remarkPlugins={[remarkGfm]}
          className="prose prose-invert prose-sm max-w-none"
        >
          {content}
        </ReactMarkdown>
      )}

      {/* Controls */}
      <div className="flex flex-col gap-3 rounded-xl border border-gray-800/60 bg-gray-900/30 p-4 md:flex-row md:items-center md:justify-between">
        <div className="flex items-center gap-2">
          <div className="relative w-full md:w-80">
            <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-gray-500" />
            <Input
              value={search}
              onChange={(e) => {
                setSearch(e.target.value)
                setPage(1)
              }}
              placeholder="Search rows…"
              className="pl-9 bg-gray-900/40 border-gray-800/40"
            />
          </div>
        </div>

        <div className="flex items-center gap-2">
          <Button
            variant="outline"
            className="border-emerald-500/30 bg-emerald-500/5 hover:bg-emerald-500/10"
            onClick={() => {
              if (!columns.length || !rawRows.length) {
                toast.error('No rows available to export')
                return
              }
              const csv = toCSV(columns, filteredRows)
              const blob = new Blob([csv], { type: 'text/csv;charset=utf-8' })
              const url = URL.createObjectURL(blob)
              const a = document.createElement('a')
              a.href = url
              a.download = `${(metadata?.database ?? 'database')}-export.csv`
              a.click()
              URL.revokeObjectURL(url)
              toast.success('CSV exported')
            }}
          >
            <Download className="mr-2 h-4 w-4" />
            Export CSV
          </Button>
        </div>
      </div>

      {/* Table */}
      <div className="rounded-xl border border-gray-800/60 bg-gray-900/40 overflow-hidden">
        <div className="overflow-x-auto">
          <table className="min-w-full divide-y divide-gray-800/70 text-sm text-gray-100">
            <thead className="bg-gray-900/60 text-xs uppercase tracking-wide text-gray-400">
              <tr>
                {columns.map((col) => (
                  <th key={col} className="px-4 py-3 text-left font-semibold">
                    {col}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-800/70">
              {pageRows.length === 0 ? (
                <tr>
                  <td className="px-4 py-6 text-gray-400" colSpan={Math.max(columns.length, 1)}>
                    No rows
                  </td>
                </tr>
              ) : (
                pageRows.map((row, idx) => (
                  <tr key={idx} className="hover:bg-gray-900/60 transition-colors">
                    {columns.map((col) => {
                      const value = row?.[col]
                      const display = value === null || value === undefined ? '—' : String(value)
                      return (
                        <td key={col} className="px-4 py-3 align-top text-gray-200">
                          <span className="block max-w-[420px] truncate" title={display}>
                            {display}
                          </span>
                        </td>
                      )
                    })}
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>

        {/* Pagination */}
        <div className="flex flex-col gap-2 border-t border-gray-800/60 bg-gray-900/30 px-4 py-3 md:flex-row md:items-center md:justify-between">
          <div className="text-xs text-gray-400">
            Showing {(page - 1) * pageSize + 1}-{Math.min(page * pageSize, filteredRows.length)} of {filteredRows.length} rows
          </div>
          <div className="flex items-center gap-2">
            <Button
              size="sm"
              variant="outline"
              disabled={page <= 1}
              onClick={() => setPage((p) => Math.max(1, p - 1))}
            >
              Prev
            </Button>
            <span className="text-xs text-gray-300 font-mono">
              {page}/{totalPages}
            </span>
            <Button
              size="sm"
              variant="outline"
              disabled={page >= totalPages}
              onClick={() => setPage((p) => Math.min(totalPages, p + 1))}
            >
              Next
            </Button>
          </div>
        </div>
      </div>

      {/* SQL */}
      {sql && (
        <details className="rounded-xl border border-gray-800/60 bg-gray-900/30 p-4">
          <summary className="cursor-pointer text-sm font-semibold text-gray-200">
            View SQL
          </summary>
          <pre className="mt-3 overflow-x-auto rounded-lg border border-gray-800/60 bg-gray-900/60 p-3 text-xs text-gray-200">
            {sql}
          </pre>
        </details>
      )}

      {/* PandasAI charts */}
      {pandasAI?.charts && pandasAI.charts.length > 0 && (
        <div className="space-y-4">
          <h4 className="text-sm font-semibold text-gray-300 uppercase tracking-wide">
            Charts
          </h4>
          <div className="grid gap-4 md:grid-cols-2">
            {pandasAI.charts.map((chart, idx) => (
              <div
                key={`${chart.filename}-${idx}`}
                className="rounded-lg border border-gray-800/60 bg-gray-900/40 p-4 flex flex-col items-center gap-3"
              >
                <img
                  src={`data:${chart.mime_type};base64,${chart.base64}`}
                  alt={chart.filename}
                  className="rounded-md border border-gray-800/40 max-h-72 w-full object-contain"
                />
                <div className="flex w-full items-center justify-between text-xs text-gray-500">
                  <span className="truncate">{chart.filename}</span>
                  <button
                    className="rounded border border-gray-700/60 px-2 py-1 text-[11px] uppercase tracking-wide text-gray-300 hover:border-orange-400/60 hover:text-orange-300"
                    onClick={() => {
                      const link = document.createElement('a')
                      link.href = `data:${chart.mime_type};base64,${chart.base64}`
                      link.download = chart.filename
                      document.body.appendChild(link)
                      link.click()
                      document.body.removeChild(link)
                    }}
                  >
                    Download
                  </button>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {pandasAI?.error && (
        <div className="text-sm text-red-400">PandasAI warning: {pandasAI.error}</div>
      )}
    </div>
  )
}

