'use client'

import React, { useState, useEffect, useCallback } from 'react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Input } from '@/components/ui/input'
import { Textarea } from '@/components/ui/textarea'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from '@/components/ui/alert-dialog'
import {
  Star,
  Trash2,
  Upload,
  Edit3,
  Check,
  X,
  Database,
  BarChart,
  FileUp,
} from 'lucide-react'
import { toast } from 'sonner'

interface TrainingExample {
  id: number
  question: string
  sql: string
  tables_used: string[]
  is_verified: boolean
  verification_source: string
  usage_count: number
  last_used_at: string | null
  created_at: string | null
  metadata: Record<string, any>
}

interface ExampleStats {
  total_examples: number
  verified_count: number
  unverified_count: number
  auto_generated_count: number
  most_used_examples: any[]
}

interface TrainingExamplesManagerProps {
  sourceId: number
}

export function TrainingExamplesManager({ sourceId }: TrainingExamplesManagerProps) {
  const [examples, setExamples] = useState<TrainingExample[]>([])
  const [stats, setStats] = useState<ExampleStats | null>(null)
  const [loading, setLoading] = useState(true)
  const [filter, setFilter] = useState('all')
  const [editingId, setEditingId] = useState<number | null>(null)
  const [editSql, setEditSql] = useState('')
  const [deleteId, setDeleteId] = useState<number | null>(null)
  const [showImport, setShowImport] = useState(false)
  const [importText, setImportText] = useState('')

  const fetchExamples = useCallback(async () => {
    try {
      const verifiedOnly = filter === 'verified'
      const res = await fetch(
        `/api/knowledge/sources/database/${sourceId}/examples?verified_only=${verifiedOnly}`
      )
      if (res.ok) {
        let data = await res.json()
        if (filter === 'unverified') {
          data = data.filter((e: TrainingExample) => !e.is_verified)
        } else if (filter === 'auto') {
          data = data.filter(
            (e: TrainingExample) => e.verification_source === 'auto'
          )
        }
        setExamples(data)
      }
    } catch (err) {
      console.error('Failed to fetch examples:', err)
    }
  }, [sourceId, filter])

  const fetchStats = useCallback(async () => {
    try {
      const res = await fetch(
        `/api/knowledge/sources/database/${sourceId}/examples/stats`
      )
      if (res.ok) setStats(await res.json())
    } catch (err) {
      console.error('Failed to fetch stats:', err)
    }
  }, [sourceId])

  useEffect(() => {
    setLoading(true)
    Promise.all([fetchExamples(), fetchStats()]).finally(() => setLoading(false))
  }, [fetchExamples, fetchStats])

  const handleVerify = async (exampleId: number) => {
    try {
      const res = await fetch(
        `/api/knowledge/sources/database/${sourceId}/examples/${exampleId}/verify`,
        { method: 'PUT' }
      )
      if (res.ok) {
        toast.success('Example marked as Golden SQL')
        fetchExamples()
        fetchStats()
      }
    } catch (err) {
      toast.error('Failed to verify example')
    }
  }

  const handleDelete = async () => {
    if (!deleteId) return
    try {
      const res = await fetch(
        `/api/knowledge/sources/database/${sourceId}/examples/${deleteId}`,
        { method: 'DELETE' }
      )
      if (res.ok) {
        toast.success('Example deleted')
        setDeleteId(null)
        fetchExamples()
        fetchStats()
      }
    } catch (err) {
      toast.error('Failed to delete example')
    }
  }

  const handleSaveEdit = async (exampleId: number) => {
    try {
      const res = await fetch(
        `/api/knowledge/sources/database/${sourceId}/examples/${exampleId}`,
        {
          method: 'PUT',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ sql: editSql }),
        }
      )
      if (res.ok) {
        toast.success('SQL updated')
        setEditingId(null)
        fetchExamples()
      }
    } catch (err) {
      toast.error('Failed to update')
    }
  }

  const handleImport = async () => {
    try {
      const parsed = JSON.parse(importText)
      const importData = Array.isArray(parsed) ? parsed : [parsed]
      const res = await fetch(
        `/api/knowledge/sources/database/${sourceId}/examples/import`,
        {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ examples: importData }),
        }
      )
      if (res.ok) {
        const result = await res.json()
        toast.success(`Imported ${result.imported} of ${result.total} examples`)
        setShowImport(false)
        setImportText('')
        fetchExamples()
        fetchStats()
      }
    } catch (err) {
      toast.error('Invalid JSON. Expected array of {question, sql} objects.')
    }
  }

  return (
    <div className="space-y-4">
      {/* Stats Bar */}
      {stats && (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          <Card className="glass-card">
            <CardContent className="p-3 text-center">
              <div className="text-2xl font-bold">{stats.total_examples}</div>
              <div className="text-xs text-muted-foreground">Total Examples</div>
            </CardContent>
          </Card>
          <Card className="glass-card">
            <CardContent className="p-3 text-center">
              <div className="text-2xl font-bold text-yellow-500">
                {stats.verified_count}
              </div>
              <div className="text-xs text-muted-foreground">Golden SQL</div>
            </CardContent>
          </Card>
          <Card className="glass-card">
            <CardContent className="p-3 text-center">
              <div className="text-2xl font-bold text-muted-foreground">
                {stats.unverified_count}
              </div>
              <div className="text-xs text-muted-foreground">Unverified</div>
            </CardContent>
          </Card>
          <Card className="glass-card">
            <CardContent className="p-3 text-center">
              <div className="text-2xl font-bold text-blue-500">
                {stats.auto_generated_count}
              </div>
              <div className="text-xs text-muted-foreground">Auto-generated</div>
            </CardContent>
          </Card>
        </div>
      )}

      {/* Controls */}
      <div className="flex items-center gap-3">
        <Select value={filter} onValueChange={setFilter}>
          <SelectTrigger className="w-[160px]">
            <SelectValue placeholder="Filter" />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="all">All</SelectItem>
            <SelectItem value="verified">Verified Only</SelectItem>
            <SelectItem value="unverified">Unverified</SelectItem>
            <SelectItem value="auto">Auto-generated</SelectItem>
          </SelectContent>
        </Select>

        <Button
          variant="outline"
          size="sm"
          onClick={() => setShowImport(!showImport)}
        >
          <FileUp className="h-4 w-4 mr-1" />
          Import
        </Button>
      </div>

      {/* Import Panel */}
      {showImport && (
        <Card className="glass-card">
          <CardContent className="p-4 space-y-3">
            <p className="text-sm text-muted-foreground">
              Paste JSON array of examples: [{'"question"'}: ..., {'"sql"'}: ...]
            </p>
            <Textarea
              value={importText}
              onChange={(e) => setImportText(e.target.value)}
              placeholder='[{"question": "top customers", "sql": "SELECT ..."}]'
              rows={4}
              className="font-mono text-xs"
            />
            <div className="flex gap-2">
              <Button size="sm" onClick={handleImport}>
                Import
              </Button>
              <Button
                size="sm"
                variant="ghost"
                onClick={() => setShowImport(false)}
              >
                Cancel
              </Button>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Examples List */}
      {loading ? (
        <div className="space-y-3">
          {[1, 2, 3].map((i) => (
            <Card key={i} className="glass-card animate-pulse">
              <CardContent className="p-4">
                <div className="h-4 bg-muted rounded w-3/4 mb-2" />
                <div className="h-3 bg-muted rounded w-1/2" />
              </CardContent>
            </Card>
          ))}
        </div>
      ) : examples.length === 0 ? (
        <Card className="glass-card">
          <CardContent className="p-8 text-center text-muted-foreground">
            <Database className="h-10 w-10 mx-auto mb-3 opacity-50" />
            <p>No training examples yet.</p>
            <p className="text-xs mt-1">
              Query your database to auto-generate examples, or import them.
            </p>
          </CardContent>
        </Card>
      ) : (
        <div className="space-y-2">
          {examples.map((ex) => (
            <Card key={ex.id} className="glass-card">
              <CardContent className="p-3">
                <div className="flex items-start justify-between gap-2">
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2 mb-1">
                      {ex.is_verified ? (
                        <Star className="h-4 w-4 text-yellow-500 fill-yellow-500 shrink-0" />
                      ) : (
                        <Star className="h-4 w-4 text-muted-foreground shrink-0" />
                      )}
                      <span className="text-sm font-medium truncate">
                        {ex.question}
                      </span>
                      {ex.verification_source === 'auto' && (
                        <Badge variant="secondary" className="text-[10px] px-1">
                          auto
                        </Badge>
                      )}
                      {ex.usage_count > 0 && (
                        <Badge variant="outline" className="text-[10px] px-1">
                          used {ex.usage_count}x
                        </Badge>
                      )}
                    </div>

                    {editingId === ex.id ? (
                      <div className="space-y-2">
                        <Textarea
                          value={editSql}
                          onChange={(e) => setEditSql(e.target.value)}
                          className="font-mono text-xs"
                          rows={3}
                        />
                        <div className="flex gap-1">
                          <Button
                            size="sm"
                            variant="ghost"
                            onClick={() => handleSaveEdit(ex.id)}
                          >
                            <Check className="h-3 w-3" />
                          </Button>
                          <Button
                            size="sm"
                            variant="ghost"
                            onClick={() => setEditingId(null)}
                          >
                            <X className="h-3 w-3" />
                          </Button>
                        </div>
                      </div>
                    ) : (
                      <pre className="text-xs text-muted-foreground font-mono bg-muted/50 rounded p-2 overflow-x-auto whitespace-pre-wrap">
                        {ex.sql.length > 300
                          ? ex.sql.substring(0, 300) + '...'
                          : ex.sql}
                      </pre>
                    )}
                  </div>

                  <div className="flex flex-col gap-1 shrink-0">
                    {!ex.is_verified && (
                      <Button
                        size="sm"
                        variant="ghost"
                        className="h-7 w-7 p-0"
                        title="Verify (Golden SQL)"
                        onClick={() => handleVerify(ex.id)}
                      >
                        <Star className="h-3.5 w-3.5 text-yellow-500" />
                      </Button>
                    )}
                    <Button
                      size="sm"
                      variant="ghost"
                      className="h-7 w-7 p-0"
                      title="Edit SQL"
                      onClick={() => {
                        setEditingId(ex.id)
                        setEditSql(ex.sql)
                      }}
                    >
                      <Edit3 className="h-3.5 w-3.5" />
                    </Button>
                    <Button
                      size="sm"
                      variant="ghost"
                      className="h-7 w-7 p-0 text-destructive"
                      title="Delete"
                      onClick={() => setDeleteId(ex.id)}
                    >
                      <Trash2 className="h-3.5 w-3.5" />
                    </Button>
                  </div>
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
      )}

      {/* Delete Confirmation */}
      <AlertDialog
        open={deleteId !== null}
        onOpenChange={() => setDeleteId(null)}
      >
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Delete Training Example?</AlertDialogTitle>
            <AlertDialogDescription>
              This will permanently remove this training example. This cannot be
              undone.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>Cancel</AlertDialogCancel>
            <AlertDialogAction onClick={handleDelete}>Delete</AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </div>
  )
}
