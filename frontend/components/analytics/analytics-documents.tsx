'use client'

import { useState, useMemo } from 'react'
import { motion } from 'framer-motion'
import {
  FileText,
  HardDrive,
  Search,
  AlertTriangle,
  ArrowUpDown,
  Database,
  Eye,
  Tag,
} from 'lucide-react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Skeleton } from '@/components/ui/skeleton'
import { Progress } from '@/components/ui/progress'
import { StatsBar } from '@/components/shared/stats-bar'
import { useDocumentAnalyticsUnified } from '@/hooks/use-unified-analytics'

interface Props {
  days: number
}

type SortField = 'name' | 'size' | 'lastAccessed' | 'ragQueries'

function formatFileSize(bytes: number): string {
  if (!bytes) return '0 B'
  const k = 1024
  const sizes = ['B', 'KB', 'MB', 'GB']
  const i = Math.floor(Math.log(bytes) / Math.log(k))
  return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i]
}

export function AnalyticsDocuments({ days }: Props) {
  const { data, isLoading } = useDocumentAnalyticsUnified(days)
  const [sortField, setSortField] = useState<SortField>('lastAccessed')
  const [sortDir, setSortDir] = useState<'asc' | 'desc'>('asc')

  const toggleSort = (field: SortField) => {
    if (sortField === field) setSortDir(prev => prev === 'asc' ? 'desc' : 'asc')
    else { setSortField(field); setSortDir('desc') }
  }

  const sortedDocs = useMemo(() => {
    if (!data?.documents) return []
    return [...data.documents].sort((a: any, b: any) => {
      // Never-accessed docs float to top when sorting by lastAccessed ascending
      if (sortField === 'lastAccessed') {
        if (!a.lastAccessed && !b.lastAccessed) return 0
        if (!a.lastAccessed) return sortDir === 'asc' ? -1 : 1
        if (!b.lastAccessed) return sortDir === 'asc' ? 1 : -1
        return sortDir === 'asc'
          ? new Date(a.lastAccessed).getTime() - new Date(b.lastAccessed).getTime()
          : new Date(b.lastAccessed).getTime() - new Date(a.lastAccessed).getTime()
      }
      const aVal = a[sortField] ?? 0
      const bVal = b[sortField] ?? 0
      if (typeof aVal === 'string') return sortDir === 'asc' ? aVal.localeCompare(bVal as string) : (bVal as string).localeCompare(aVal)
      return sortDir === 'asc' ? (aVal as number) - (bVal as number) : (bVal as number) - (aVal as number)
    })
  }, [data?.documents, sortField, sortDir])

  const SortHeader = ({ field, children }: { field: SortField; children: React.ReactNode }) => (
    <button onClick={() => toggleSort(field)} className="flex items-center gap-1 text-xs font-medium text-muted-foreground hover:text-foreground transition-colors">
      {children}<ArrowUpDown className="w-3 h-3" />
    </button>
  )

  if (isLoading) {
    return (
      <div className="space-y-6">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          {Array.from({ length: 3 }).map((_, i) => (
            <Card key={i} className="glass-card"><CardContent className="p-6"><Skeleton className="h-8 w-16 mb-2" /><Skeleton className="h-4 w-24" /></CardContent></Card>
          ))}
        </div>
        <Skeleton className="h-64 w-full" />
      </div>
    )
  }

  return (
    <div className="space-y-6">
      <StatsBar stats={[
        { label: 'Total Documents', value: data?.summary?.totalDocuments || 0, icon: FileText, iconColor: 'text-primary', globalIconKey: 'global_document' },
        { label: 'Storage Used', value: `${(data?.summary?.totalStorageMb || 0).toFixed(1)} MB`, icon: HardDrive, iconColor: 'text-[hsl(var(--success))]', globalIconKey: 'global_storage' },
        { label: 'RAG Queries', value: data?.summary?.totalRagQueries || 0, change: 'This period', icon: Search, iconColor: 'text-[hsl(var(--info))]' },
        { label: 'Never Accessed', value: data?.neverAccessed || 0, change: 'Unused by RAG', icon: AlertTriangle, iconColor: data?.neverAccessed ? 'text-yellow-400' : 'text-[hsl(var(--success))]' },
      ]} loading={isLoading} />

      {/* Unused Documents Alert */}
      {(data?.neverAccessed || 0) > 0 && (
        <Card className="glass-card border-yellow-500/20 bg-yellow-500/5">
          <CardContent className="p-4">
            <div className="flex items-start gap-3">
              <AlertTriangle className="w-5 h-5 text-yellow-400 mt-0.5 shrink-0" />
              <div>
                <p className="font-medium text-sm">
                  {data?.neverAccessed} document{(data?.neverAccessed || 0) > 1 ? 's' : ''} never retrieved by RAG
                </p>
                <p className="text-sm text-muted-foreground mt-1">
                  These documents exist in your knowledge base but have never been used in any RAG query.
                  Consider adding better tags, keywords, or descriptions to help the AI find and use them.
                </p>
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {/* RAG Performance */}
      {data?.usage && (
        <Card className="glass-card">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Database className="w-5 h-5 text-purple-400" />
              RAG Performance
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div className="p-3 rounded-lg bg-secondary/20 text-center">
                <p className="text-2xl font-bold">{data.usage.total_events || 0}</p>
                <p className="text-sm text-muted-foreground">Total Events</p>
              </div>
              <div className="p-3 rounded-lg bg-secondary/20 text-center">
                <p className="text-2xl font-bold">{data.usage.event_counts?.rag_query || 0}</p>
                <p className="text-sm text-muted-foreground">RAG Queries</p>
              </div>
              <div className="p-3 rounded-lg bg-secondary/20 text-center">
                <p className="text-2xl font-bold">{data.usage.event_counts?.document_searched || 0}</p>
                <p className="text-sm text-muted-foreground">Document Searches</p>
              </div>
            </div>

            {/* Popular Search Terms */}
            {data.usage.popular_search_terms?.length > 0 && (
              <div className="mt-4">
                <p className="text-sm font-medium mb-2">Top Search Terms</p>
                <div className="flex flex-wrap gap-2">
                  {data.usage.popular_search_terms.slice(0, 8).map((term: any, i: number) => (
                    <Badge key={i} variant="secondary" className="text-xs">
                      {term.query || term.term} ({term.count || term.frequency})
                    </Badge>
                  ))}
                </div>
              </div>
            )}
          </CardContent>
        </Card>
      )}

      {/* Document Table */}
      <Card className="glass-card overflow-hidden">
        <div className="overflow-x-auto max-h-[500px] overflow-y-auto">
          <table className="w-full">
            <thead className="sticky top-0 bg-card z-10">
              <tr className="border-b border-border/50">
                <th className="text-left p-4"><SortHeader field="name">Name</SortHeader></th>
                <th className="text-left p-4 hidden md:table-cell"><span className="text-xs font-medium text-muted-foreground">Type</span></th>
                <th className="text-left p-4"><SortHeader field="size">Size</SortHeader></th>
                <th className="text-left p-4"><SortHeader field="lastAccessed">Last Accessed</SortHeader></th>
                <th className="text-left p-4 hidden lg:table-cell"><SortHeader field="ragQueries">RAG Queries</SortHeader></th>
                <th className="text-left p-4 hidden lg:table-cell"><span className="text-xs font-medium text-muted-foreground">Tags</span></th>
              </tr>
            </thead>
            <tbody>
              {sortedDocs.length === 0 ? (
                <tr>
                  <td colSpan={6}>
                    <div className="flex flex-col items-center justify-center py-12 gap-2 text-muted-foreground">
                      <FileText className="w-8 h-8 opacity-50" />
                      <p className="text-sm">No documents found</p>
                    </div>
                  </td>
                </tr>
              ) : (
                sortedDocs.map((doc) => (
                  <tr key={doc.id} className="border-b border-border/30 hover:bg-secondary/20 transition-colors">
                    <td className="p-4">
                      <span className="font-medium">{doc.name}</span>
                    </td>
                    <td className="p-4 hidden md:table-cell">
                      <Badge variant="outline" className="text-xs">{doc.type}</Badge>
                    </td>
                    <td className="p-4 text-sm">{formatFileSize(doc.size)}</td>
                    <td className="p-4 text-sm">
                      {doc.lastAccessed ? (
                        new Date(doc.lastAccessed).toLocaleDateString()
                      ) : (
                        <Badge className="text-xs bg-yellow-500/20 text-yellow-400 border-yellow-500/30">Never used</Badge>
                      )}
                    </td>
                    <td className="p-4 hidden lg:table-cell text-sm">{doc.ragQueries ?? '-'}</td>
                    <td className="p-4 hidden lg:table-cell">
                      {doc.tags?.length > 0 ? (
                        <div className="flex flex-wrap gap-1">
                          {doc.tags.slice(0, 3).map((tag: string, i: number) => (
                            <Badge key={i} variant="secondary" className="text-xs">{tag}</Badge>
                          ))}
                        </div>
                      ) : (
                        <span className="text-xs text-muted-foreground">No tags</span>
                      )}
                    </td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>
      </Card>
    </div>
  )
}
