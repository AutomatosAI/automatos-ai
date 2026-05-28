'use client'

import { useState } from 'react'
import { Loader2, Github, X, CheckCircle, AlertCircle, SkipForward } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Badge } from '@/components/ui/badge'
import { apiClient } from '@/lib/api-client'
import { toast } from 'sonner'

interface ImportResult {
  slug: string
  plugin_id: string | null
  security_status: string | null
  approval_status: string | null
  status: string
  error: string | null
}

interface GitHubImportModalProps {
  open: boolean
  onClose: () => void
  onImportComplete: () => void
}

export function GitHubImportModal({ open, onClose, onImportComplete }: GitHubImportModalProps) {
  const [url, setUrl] = useState('')
  const [isImporting, setIsImporting] = useState(false)
  const [results, setResults] = useState<ImportResult[] | null>(null)

  const handleImport = async () => {
    if (!url.trim() || isImporting) return

    setIsImporting(true)
    setResults(null)

    try {
      const data: any = await apiClient.post('/api/admin/plugins/import-github', {
        github_url: url.trim(),
      })
      const items = data.plugins || []
      setResults(items)
      const successCount = items.filter((r: ImportResult) => r.status === 'uploaded' || r.status === 'updated').length
      toast.success(`Imported ${successCount} item(s) from GitHub`)
      onImportComplete()

      // Auto-close after 2s on success
      if (successCount > 0) {
        setTimeout(() => {
          setUrl('')
          setResults(null)
          onClose()
        }, 2000)
      }
    } catch (error: any) {
      toast.error('Import failed', {
        description: error?.message || 'Failed to import from GitHub',
      })
    } finally {
      setIsImporting(false)
    }
  }

  const handleClose = () => {
    setUrl('')
    setResults(null)
    onClose()
  }

  if (!open) return null

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm">
      <div className="w-full max-w-lg mx-4 bg-card border border-border rounded-xl shadow-2xl">
        {/* Header */}
        <div className="flex items-center justify-between p-4 border-b border-border">
          <div className="flex items-center gap-2">
            <Github className="w-5 h-5" />
            <h2 className="text-lg font-semibold">Import from GitHub</h2>
          </div>
          <Button variant="ghost" size="sm" onClick={handleClose}>
            <X className="w-4 h-4" />
          </Button>
        </div>

        {/* Body */}
        <div className="p-4 space-y-4">
          <div className="space-y-2">
            <label className="text-sm text-muted-foreground">
              GitHub Repository URL
            </label>
            <div className="flex gap-2">
              <Input
                placeholder="https://github.com/user/repo"
                value={url}
                onChange={(e) => setUrl(e.target.value)}
                disabled={isImporting}
                onKeyDown={(e) => e.key === 'Enter' && handleImport()}
              />
              <Button
                onClick={handleImport}
                disabled={isImporting || !url.trim()}
                className="bg-primary hover:bg-primary/90 text-primary-foreground whitespace-nowrap"
              >
                {isImporting ? (
                  <>
                    <Loader2 className="w-4 h-4 mr-1 animate-spin" />
                    Importing...
                  </>
                ) : (
                  'Import'
                )}
              </Button>
            </div>
            {isImporting && (
              <p className="text-xs text-muted-foreground">
                Cloning repo and scanning for plugins and skills... this may take 30+ seconds.
              </p>
            )}
          </div>

          {/* Results */}
          {results && results.length > 0 && (
            <div className="space-y-2">
              <h3 className="text-sm font-medium">Results</h3>
              <div className="max-h-60 overflow-y-auto space-y-2 rounded-lg border border-border">
                {results.map((r) => (
                  <div
                    key={r.slug}
                    className="flex items-center justify-between p-3 border-b border-border last:border-b-0"
                  >
                    <div className="flex-1 min-w-0">
                      <p className="text-sm font-medium truncate">{r.slug}</p>
                      {r.error && (
                        <p className="text-xs text-destructive truncate">{r.error}</p>
                      )}
                    </div>
                    <div className="flex items-center gap-2 flex-shrink-0">
                      {r.security_status && (
                        <Badge
                          variant="outline"
                          className={
                            r.security_status === 'safe'
                              ? 'border-success/30 text-success text-xs'
                              : 'border-warning/30 text-warning text-xs'
                          }
                        >
                          {r.security_status}
                        </Badge>
                      )}
                      {(r.status === 'uploaded' || r.status === 'updated') && (
                        <CheckCircle className="w-4 h-4 text-success" />
                      )}
                      {r.status === 'already_exists' && (
                        <span title="Already exists"><SkipForward className="w-4 h-4 text-warning" /></span>
                      )}
                      {r.status === 'error' && (
                        <AlertCircle className="w-4 h-4 text-destructive" />
                      )}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="flex justify-end p-4 border-t border-border">
          <Button variant="outline" onClick={handleClose}>
            Close
          </Button>
        </div>
      </div>
    </div>
  )
}
