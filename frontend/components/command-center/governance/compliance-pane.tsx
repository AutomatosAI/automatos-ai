'use client'

/**
 * CompliancePane — PRD-196 S7 (P2-15, governance I.4). GDPR self-service the
 * API already guards: export the bundle, erase a data subject (rendering the
 * honest gaps + untagged-history report — the point, not a footnote), and erase
 * the whole workspace behind the typed confirmation the API enforces.
 */

import { useEffect, useState } from 'react'
import { Download, ShieldOff, AlertTriangle, Loader2 } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { toast } from 'sonner'
import { apiClient } from '@/lib/api-client'
import { useGdprEraseSubject, useGdprEraseWorkspace } from '@/hooks/use-gdpr'

export function CompliancePane() {
  const [exporting, setExporting] = useState(false)
  const [subjectId, setSubjectId] = useState('')
  const [confirmWs, setConfirmWs] = useState('')
  const [currentWs, setCurrentWs] = useState<string | null>(null)

  const eraseSubject = useGdprEraseSubject()
  const eraseWorkspace = useGdprEraseWorkspace()

  useEffect(() => {
    if (typeof window !== 'undefined') {
      setCurrentWs(
        localStorage.getItem('last_active_workspace') || localStorage.getItem('last_active_org'),
      )
    }
  }, [])

  const handleExport = async () => {
    setExporting(true)
    try {
      const bundle = await apiClient.getGdprExport()
      const blob = new Blob([JSON.stringify(bundle, null, 2)], { type: 'application/json' })
      const url = URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = `gdpr-export-${new Date().toISOString().slice(0, 10)}.json`
      a.click()
      URL.revokeObjectURL(url)
      toast.success('Export downloaded')
    } catch {
      toast.error('Export failed — workspace admin only')
    } finally {
      setExporting(false)
    }
  }

  const handleEraseSubject = async () => {
    const sid = subjectId.trim()
    if (!sid) return
    try {
      await eraseSubject.mutateAsync(sid)
      toast.success('Subject erase ran — see the report below')
    } catch {
      toast.error('Subject erase failed')
    }
  }

  const handleEraseWorkspace = async () => {
    try {
      await eraseWorkspace.mutateAsync(confirmWs.trim())
      toast.success('Workspace erased')
    } catch {
      toast.error('Workspace erase failed')
    }
  }

  const result = eraseSubject.data
  const canEraseWorkspace = confirmWs.trim() !== '' && confirmWs.trim() === (currentWs ?? '')

  return (
    <div className="flex flex-col gap-6">
      {/* Export */}
      <section className="flex flex-col gap-2">
        <h3 className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
          Export
        </h3>
        <p className="text-xs text-muted-foreground">
          Download a portable JSON bundle of this workspace’s data (SQL + field + durable memory).
        </p>
        <div>
          <Button size="sm" variant="outline" disabled={exporting} onClick={handleExport}>
            {exporting ? <Loader2 className="h-4 w-4 mr-1 animate-spin" /> : <Download className="h-4 w-4 mr-1" />}
            Export data
          </Button>
        </div>
      </section>

      {/* Erase subject */}
      <section className="flex flex-col gap-2">
        <h3 className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
          Erase a data subject
        </h3>
        <p className="text-xs text-muted-foreground">
          Filter-deletes a subject’s tagged field + durable memories. The report shows exactly what
          was and was <strong>not</strong> deleted — a defensible answer includes the gaps.
        </p>
        <div className="flex gap-2">
          <input
            type="text"
            aria-label="Subject id"
            value={subjectId}
            onChange={(e) => setSubjectId(e.target.value)}
            placeholder="user:123"
            className="flex-1 rounded border border-border bg-background px-2 py-1 text-sm"
          />
          <Button size="sm" disabled={!subjectId.trim() || eraseSubject.isLoading} onClick={handleEraseSubject}>
            {eraseSubject.isLoading ? 'Erasing…' : 'Erase subject'}
          </Button>
        </div>

        {result && (
          <div className="mt-1 flex flex-col gap-2 rounded border border-border bg-background/50 p-3 text-xs">
            <p className="font-medium">Erase report for {result.subject_id}</p>
            <ul className="text-muted-foreground">
              <li>Field memory deleted: <span className="text-foreground">{result.derived.field_memory_deleted}</span></li>
              <li>Durable memory deleted: <span className="text-foreground">{result.derived.durable_memory_deleted}</span></li>
              <li>SQL deleted: <span className="text-foreground">{result.sql?.deleted ?? 0}</span></li>
            </ul>

            {result.gaps?.length > 0 && (
              <div className="rounded border border-amber-300 bg-amber-50 px-2 py-1.5 dark:border-amber-800 dark:bg-amber-950/40">
                <p className="flex items-center gap-1 text-[11px] font-medium text-amber-800 dark:text-amber-300">
                  <AlertTriangle className="h-3 w-3" /> Could not delete (documented gaps)
                </p>
                {result.gaps.map((g) => (
                  <p key={g.store} className="mt-0.5 text-[10px] text-amber-700 dark:text-amber-400">
                    <strong>{g.store}</strong>: {g.reason}
                  </p>
                ))}
              </div>
            )}

            {result.untagged_history && (
              <div className="rounded border border-border px-2 py-1.5">
                <p className="text-[11px] font-medium">Untagged history ({result.untagged_history.stores.join(', ')})</p>
                <p className="mt-0.5 text-[10px] text-muted-foreground">{result.untagged_history.reason}</p>
              </div>
            )}
          </div>
        )}
      </section>

      {/* Erase workspace */}
      <section className="flex flex-col gap-2">
        <h3 className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
          Erase this workspace
        </h3>
        <div className="flex items-start gap-2 rounded border border-red-300 bg-red-50 px-3 py-2 text-xs dark:border-red-900 dark:bg-red-950/40">
          <ShieldOff className="mt-0.5 h-3.5 w-3.5 shrink-0 text-red-600" />
          <span className="text-red-800 dark:text-red-300">
            Irreversible. Deletes every store for this workspace. Type the workspace id
            {currentWs ? <> (<code className="font-mono">{currentWs}</code>)</> : ''} to confirm.
          </span>
        </div>
        <div className="flex gap-2">
          <input
            type="text"
            aria-label="Confirm workspace id"
            value={confirmWs}
            onChange={(e) => setConfirmWs(e.target.value)}
            placeholder="workspace id"
            className="flex-1 rounded border border-border bg-background px-2 py-1 text-sm"
          />
          <Button
            size="sm"
            variant="destructive"
            disabled={!canEraseWorkspace || eraseWorkspace.isLoading}
            onClick={handleEraseWorkspace}
          >
            {eraseWorkspace.isLoading ? 'Erasing…' : 'Erase workspace'}
          </Button>
        </div>
      </section>
    </div>
  )
}
