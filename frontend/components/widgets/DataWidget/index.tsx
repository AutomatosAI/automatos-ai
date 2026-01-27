'use client'

/**
 * DataWidget Component for PRD-38.1 Widget Architecture
 *
 * Displays database query results with tables, charts, search,
 * pagination, and CSV export. Migrated from sheet-artifact.tsx.
 */

import { useState, useMemo, useCallback } from 'react'
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table'
import { Input } from '@/components/ui/input'
import { Button } from '@/components/ui/button'
import {
  Database,
  Search,
  ChevronLeft,
  ChevronRight,
  Download,
  BarChart3,
  Code,
  ChevronDown,
  ChevronUp,
} from 'lucide-react'
import { WidgetBase } from '../WidgetBase'
import { registerWidget } from '../registry'
import type { WidgetBaseProps, DataWidgetData, WidgetDefinition } from '../types'
import { toast } from 'sonner'
import { cn } from '@/lib/utils'

const ROWS_PER_PAGE = 25

/**
 * Convert data to CSV format
 */
function toCSV(columns: string[], rows: Record<string, unknown>[]): string {
  const escape = (value: unknown): string => {
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

export function DataWidget({
  id,
  title,
  data,
  metadata,
  isActive,
  isLoading,
  error,
  onClose,
  onMaximize,
  onRefresh,
}: WidgetBaseProps<DataWidgetData>) {
  const [searchTerm, setSearchTerm] = useState('')
  const [currentPage, setCurrentPage] = useState(1)
  const [showSql, setShowSql] = useState(false)
  const [showCharts, setShowCharts] = useState(true)

  // Ensure columns and rows are arrays
  const columns = useMemo(() => {
    if (Array.isArray(data.columns) && data.columns.length > 0) {
      return data.columns
    }
    if (Array.isArray(data.rows) && data.rows.length > 0) {
      return Object.keys(data.rows[0] || {})
    }
    return []
  }, [data.columns, data.rows])

  const rows = useMemo(() => data.rows || [], [data.rows])

  // Filter rows by search term
  const filteredRows = useMemo(() => {
    if (!searchTerm.trim()) return rows

    const term = searchTerm.toLowerCase()
    return rows.filter((row) =>
      Object.values(row).some((val) =>
        String(val ?? '').toLowerCase().includes(term)
      )
    )
  }, [rows, searchTerm])

  // Pagination
  const totalPages = Math.max(1, Math.ceil(filteredRows.length / ROWS_PER_PAGE))
  const paginatedRows = useMemo(() => {
    const safePage = Math.min(Math.max(1, currentPage), totalPages)
    const start = (safePage - 1) * ROWS_PER_PAGE
    return filteredRows.slice(start, start + ROWS_PER_PAGE)
  }, [filteredRows, currentPage, totalPages])

  // Export to CSV
  const handleExportCsv = useCallback(() => {
    if (!columns.length || !rows.length) {
      toast.error('No data available to export')
      return
    }

    const csv = toCSV(columns, filteredRows)
    const blob = new Blob([csv], { type: 'text/csv;charset=utf-8' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `${title.replace(/\s+/g, '_')}-export.csv`
    a.click()
    URL.revokeObjectURL(url)
    toast.success('CSV exported successfully')
  }, [columns, rows, filteredRows, title])

  return (
    <WidgetBase
      title={title}
      icon={<Database className="h-4 w-4" />}
      metadata={metadata}
      isActive={isActive}
      isLoading={isLoading}
      error={error}
      onClose={onClose}
      onMaximize={onMaximize}
      onRefresh={onRefresh}
      onDownload={handleExportCsv}
      canDownload
      canRefresh={!!onRefresh}
      customActions={[
        {
          label: showSql ? 'Hide SQL' : 'Show SQL',
          icon: <Code className="h-4 w-4 mr-2" />,
          onClick: () => setShowSql(!showSql),
          disabled: !data.sql,
        },
        {
          label: showCharts ? 'Hide Charts' : 'Show Charts',
          icon: <BarChart3 className="h-4 w-4 mr-2" />,
          onClick: () => setShowCharts(!showCharts),
          disabled: !data.charts || data.charts.length === 0,
        },
      ]}
    >
      <div className="flex flex-col h-full">
        {/* Stats bar */}
        <div className="flex items-center justify-between px-3 py-2 bg-muted/30 border-b border-border/30 text-xs">
          <div className="flex items-center gap-4 text-muted-foreground">
            <span>{data.rowCount.toLocaleString()} rows</span>
            <span>{columns.length} columns</span>
            {data.executionTime !== undefined && (
              <span>{data.executionTime}ms</span>
            )}
            {data.database && (
              <span className="font-mono">{data.database}</span>
            )}
          </div>
        </div>

        {/* SQL Preview */}
        {showSql && data.sql && (
          <div className="px-3 py-2 bg-muted/20 border-b border-border/30">
            <pre className="text-xs font-mono text-muted-foreground overflow-x-auto whitespace-pre-wrap">
              {data.sql}
            </pre>
          </div>
        )}

        {/* PandasAI Summary */}
        {data.pandasAiSummary && (
          <div className="px-3 py-2 bg-primary/5 border-b border-border/30">
            <p className="text-sm">{data.pandasAiSummary}</p>
          </div>
        )}

        {/* Charts */}
        {showCharts && data.charts && data.charts.length > 0 && (
          <div className="flex gap-2 p-3 overflow-x-auto border-b border-border/30 bg-muted/10">
            {data.charts.map((chart, i) => (
              <div
                key={i}
                className="flex-shrink-0 rounded border border-border/50 overflow-hidden bg-background"
              >
                <img
                  src={`data:${chart.mimeType || 'image/png'};base64,${chart.base64}`}
                  alt={chart.filename || `Chart ${i + 1}`}
                  className="max-h-40 object-contain"
                />
              </div>
            ))}
          </div>
        )}

        {/* Search */}
        <div className="p-2 border-b border-border/30">
          <div className="relative">
            <Search className="absolute left-2.5 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
            <Input
              placeholder="Search rows..."
              value={searchTerm}
              onChange={(e) => {
                setSearchTerm(e.target.value)
                setCurrentPage(1)
              }}
              className="pl-9 h-8"
            />
          </div>
        </div>

        {/* Table */}
        <div className="flex-1 overflow-auto">
          <Table>
            <TableHeader>
              <TableRow>
                {columns.map((col) => (
                  <TableHead
                    key={col}
                    className="bg-muted/50 sticky top-0 whitespace-nowrap font-semibold text-xs uppercase"
                  >
                    {col}
                  </TableHead>
                ))}
              </TableRow>
            </TableHeader>
            <TableBody>
              {paginatedRows.length === 0 ? (
                <TableRow>
                  <TableCell
                    colSpan={Math.max(columns.length, 1)}
                    className="text-center text-muted-foreground py-8"
                  >
                    {searchTerm ? 'No matching rows found' : 'No data available'}
                  </TableCell>
                </TableRow>
              ) : (
                paginatedRows.map((row, i) => (
                  <TableRow key={i} className="hover:bg-muted/30">
                    {columns.map((col) => {
                      const value = row?.[col]
                      const display =
                        value === null || value === undefined
                          ? '—'
                          : String(value)
                      return (
                        <TableCell
                          key={col}
                          className="whitespace-nowrap max-w-[300px] truncate text-sm"
                          title={display}
                        >
                          {display}
                        </TableCell>
                      )
                    })}
                  </TableRow>
                ))
              )}
            </TableBody>
          </Table>
        </div>

        {/* Pagination */}
        {totalPages > 1 && (
          <div className="flex items-center justify-between px-3 py-2 border-t border-border/30 bg-muted/20">
            <span className="text-xs text-muted-foreground">
              Showing {(currentPage - 1) * ROWS_PER_PAGE + 1}-
              {Math.min(currentPage * ROWS_PER_PAGE, filteredRows.length)} of{' '}
              {filteredRows.length}
            </span>
            <div className="flex items-center gap-1">
              <Button
                variant="ghost"
                size="sm"
                className="h-7 w-7 p-0"
                onClick={() => setCurrentPage((p) => Math.max(1, p - 1))}
                disabled={currentPage === 1}
              >
                <ChevronLeft className="h-4 w-4" />
              </Button>
              <span className="text-xs text-muted-foreground px-2 font-mono">
                {currentPage}/{totalPages}
              </span>
              <Button
                variant="ghost"
                size="sm"
                className="h-7 w-7 p-0"
                onClick={() =>
                  setCurrentPage((p) => Math.min(totalPages, p + 1))
                }
                disabled={currentPage === totalPages}
              >
                <ChevronRight className="h-4 w-4" />
              </Button>
            </div>
          </div>
        )}
      </div>
    </WidgetBase>
  )
}

/**
 * Widget definition for registry
 */
export const DataWidgetDef: WidgetDefinition<DataWidgetData> = {
  type: 'data',
  displayName: 'Data',
  description: 'Display database query results with tables and charts',
  icon: Database,
  component: DataWidget,
  defaultSize: { width: 6, height: 5 },
  minSize: { width: 4, height: 3 },
  capabilities: ['exportable', 'fullscreen', 'refreshable'],
}

// Register the widget
registerWidget(DataWidgetDef)
