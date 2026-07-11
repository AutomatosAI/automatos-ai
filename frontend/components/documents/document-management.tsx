
'use client'

import React, { useState, useRef, useMemo, useEffect } from 'react'
import { toast } from 'sonner'
import { motion } from 'framer-motion'
import { useInView } from 'react-intersection-observer'
import {
  Upload,
  Search,
  Filter,
  FileText,
  File,
  Image,
  Database,
  Trash2,
  Download,
  Eye,
  MoreVertical,
  FolderOpen,
  Plus,
  History,
  Cloud,
  ExternalLink,
  Brain,
  Network,
} from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import {
  DropdownMenu,
  DropdownMenuContent, 
  DropdownMenuItem, 
  DropdownMenuTrigger 
} from '@/components/ui/dropdown-menu'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { PageHeader } from '@/components/shared/page-header'
import { StatsBar } from '@/components/shared/stats-bar'
import { DeleteConfirmation } from '@/components/shared/delete-confirmation'
import { CodeGraphPanel } from '@/components/knowledge/CodeGraphPanel'
import { BusinessGraphPanel } from '@/components/knowledge/BusinessGraphPanel'
import { MemoryTab } from '@/components/knowledge/memory-tab'
import { GraphDiffBanner } from '@/components/knowledge/GraphDiffBanner'
import { MultimodalKnowledgePanel } from '@/components/knowledge/MultimodalKnowledgePanel'
import { DatabaseQueryExplorer } from '@/components/knowledge/DatabaseQueryExplorer'
import { QueryTemplatesGrid } from '@/components/knowledge/QueryTemplatesGrid'
import { SemanticLayerBuilder } from '@/components/knowledge/SemanticLayerBuilder'
import { AddDatabaseModal } from '@/components/knowledge/AddDatabaseModal'
import { TrainingExamplesManager } from '@/components/knowledge/TrainingExamplesManager'
// Document modals
import { DocumentDetailsModal } from './document-details-modal'
import { DeleteConfirmationModal } from './delete-confirmation-modal'
import { UploadProviderModal } from './upload-provider-modal'
import { SemanticSearch } from './semantic-search'
import { RAGContextBuilder } from '@/components/context/rag-context-builder'
import type { SearchResult } from '@/hooks/use-semantic-search-api'
import { DocumentProcessing } from './document-processing'
// DocumentAnalytics removed — analytics consolidated into /analytics
// Cloud Storage Components (PRD-42)
import { ProviderCards } from './provider-cards'
import { ProviderBrowser } from './provider-browser'
import { LocalStorageBrowser } from './local-storage-browser'
// API hooks
import { useDocuments, useDocumentStats, useUploadDocument, useDeleteDocument } from '@/hooks/use-document-api'
import { useTeams, useDocumentTeamCounts } from '@/hooks/use-teams'
import { useDatabaseKnowledge } from '@/hooks/use-database-knowledge'
import { useCloudConnections, useTriggerSync, useSelectRootFolder } from '@/hooks/use-cloud-storage'
import { useWorkspace } from '@/components/workspace-provider'

// Real document interface to match backend response
interface BackendDocument {
  id: number;
  filename: string;
  original_filename?: string;
  file_type?: string;
  file_size?: number;
  status?: string;
  chunk_count?: number;
  upload_date?: string;
  processed_date?: string | null;
  team_access?: string[];
}

// Stats will be calculated dynamically from real data

const statusStyles: Record<string, string> = {
  completed: 'bg-success/10 text-success border-success/20',
  processed: 'bg-success/10 text-success border-success/20',
  processing: 'bg-warning/10 text-warning border-warning/20',
  failed: 'bg-destructive/10 text-destructive border-destructive/20',
  pending: 'bg-secondary/50 text-muted-foreground border-border/30'
}

const typeIcons: Record<string, any> = {
  pdf: FileText,
  docx: FileText,
  md: File,
  txt: File,
  xlsx: FileText,
  csv: FileText,
  json: File,
  xml: File
}

// ── Schema Browser (inline) ──────────────────────────────────────────
function SchemaBrowser({ sourceId, getSchemaMetadata }: { sourceId?: number; getSchemaMetadata: (id: number) => Promise<any> }) {
  const [schema, setSchema] = useState<any>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [expandedTable, setExpandedTable] = useState<string | null>(null)

  useEffect(() => {
    if (!sourceId) return
    setLoading(true)
    setError(null)
    getSchemaMetadata(sourceId)
      .then((data) => setSchema(data))
      .catch((err) => setError(err?.message || 'Failed to load schema'))
      .finally(() => setLoading(false))
  }, [sourceId])

  if (!sourceId) {
    return (
      <Card className="glass-card">
        <CardContent className="p-8 text-center text-muted-foreground">
          <Database className="h-10 w-10 mx-auto mb-3 opacity-50" />
          <p>Connect a database first to browse the schema.</p>
        </CardContent>
      </Card>
    )
  }

  if (loading) {
    return (
      <Card className="glass-card">
        <CardContent className="p-8 text-center text-muted-foreground">
          <div className="animate-spin h-8 w-8 border-2 border-primary border-t-transparent rounded-full mx-auto mb-3" />
          <p>Loading schema...</p>
        </CardContent>
      </Card>
    )
  }

  if (error) {
    return (
      <Card className="glass-card">
        <CardContent className="p-8 text-center text-destructive">
          <p>{error}</p>
        </CardContent>
      </Card>
    )
  }

  const tables: any[] = schema?.tables || schema?.schema?.tables || []

  return (
    <div className="space-y-3">
      <Card className="glass-card">
        <CardHeader className="pb-3">
          <CardTitle className="text-lg flex items-center gap-2">
            <Database className="h-5 w-5" />
            Schema — {tables.length} table{tables.length !== 1 ? 's' : ''}
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-2">
          {tables.length === 0 && (
            <p className="text-muted-foreground text-sm">No tables found. Try syncing the schema.</p>
          )}
          {tables.map((table: any) => {
            const name = table.name || table.table_name
            const columns: any[] = table.columns || []
            const isExpanded = expandedTable === name
            return (
              <div key={name} className="border border-border/50 rounded-lg overflow-hidden">
                <button
                  className="w-full flex items-center justify-between p-3 hover:bg-secondary/30 transition-colors text-left"
                  onClick={() => setExpandedTable(isExpanded ? null : name)}
                >
                  <span className="font-medium text-sm">{name}</span>
                  <Badge variant="outline" className="text-xs">
                    {columns.length} col{columns.length !== 1 ? 's' : ''}
                  </Badge>
                </button>
                {isExpanded && columns.length > 0 && (
                  <div className="border-t border-border/50 bg-secondary/10">
                    <table className="w-full text-xs">
                      <thead>
                        <tr className="text-muted-foreground">
                          <th className="text-left p-2 pl-4">Column</th>
                          <th className="text-left p-2">Type</th>
                          <th className="text-left p-2">Nullable</th>
                          <th className="text-left p-2">Key</th>
                        </tr>
                      </thead>
                      <tbody>
                        {columns.map((col: any) => (
                          <tr key={col.name || col.column_name} className="border-t border-border/30">
                            <td className="p-2 pl-4 font-mono">{col.name || col.column_name}</td>
                            <td className="p-2 text-muted-foreground">{col.type || col.data_type}</td>
                            <td className="p-2 text-muted-foreground">{col.nullable ? 'Yes' : 'No'}</td>
                            <td className="p-2">
                              {col.is_primary_key && <Badge className="text-[10px] bg-warning/20 text-warning border-warning/30">PK</Badge>}
                              {col.is_foreign_key && <Badge className="text-[10px] bg-info/20 text-info border-info/30 ml-1">FK</Badge>}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                )}
              </div>
            )
          })}
        </CardContent>
      </Card>
    </div>
  )
}

// ── Audit History (inline) ───────────────────────────────────────────
function AuditHistory({ sourceId }: { sourceId?: number }) {
  const [entries, setEntries] = useState<any[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    if (!sourceId) return
    setLoading(true)
    setError(null)
    import('@/lib/api-client').then(({ default: apiClient }) =>
      apiClient.request(`/api/knowledge/sources/database/${sourceId}/audit`)
    )
      .then((data: any) => setEntries(Array.isArray(data) ? data : []))
      .catch((err: any) => setError(err?.message || 'Failed to load audit history'))
      .finally(() => setLoading(false))
  }, [sourceId])

  if (!sourceId) {
    return (
      <Card className="glass-card">
        <CardContent className="p-8 text-center text-muted-foreground">
          <History className="h-10 w-10 mx-auto mb-3 opacity-50" />
          <p>Connect a database first to view query audit history.</p>
        </CardContent>
      </Card>
    )
  }

  if (loading) {
    return (
      <Card className="glass-card">
        <CardContent className="p-8 text-center text-muted-foreground">
          <div className="animate-spin h-8 w-8 border-2 border-primary border-t-transparent rounded-full mx-auto mb-3" />
          <p>Loading audit history...</p>
        </CardContent>
      </Card>
    )
  }

  if (error) {
    return (
      <Card className="glass-card">
        <CardContent className="p-8 text-center text-destructive">
          <p>{error}</p>
        </CardContent>
      </Card>
    )
  }

  return (
    <Card className="glass-card">
      <CardHeader className="pb-3">
        <CardTitle className="text-lg flex items-center gap-2">
          <History className="h-5 w-5" />
          Query Audit History
        </CardTitle>
      </CardHeader>
      <CardContent>
        {entries.length === 0 ? (
          <p className="text-muted-foreground text-sm text-center py-4">No queries recorded yet.</p>
        ) : (
          <div className="space-y-2">
            {entries.map((entry: any) => (
              <div key={entry.id} className="border border-border/50 rounded-lg p-3 space-y-1">
                <div className="flex items-center justify-between">
                  <span className="text-sm font-medium truncate max-w-[70%]">
                    {entry.natural_language_query || '(direct SQL)'}
                  </span>
                  <div className="flex items-center gap-2">
                    {entry.execution_time_ms != null && (
                      <span className="text-xs text-muted-foreground">{entry.execution_time_ms}ms</span>
                    )}
                    <Badge
                      variant="outline"
                      className={entry.success
                        ? 'bg-success/10 text-success border-success/20 text-xs'
                        : 'bg-destructive/10 text-destructive border-destructive/20 text-xs'}
                    >
                      {entry.success ? 'Success' : 'Failed'}
                    </Badge>
                  </div>
                </div>
                {entry.generated_sql && (
                  <pre className="text-xs font-mono bg-secondary/30 rounded p-2 overflow-x-auto whitespace-pre-wrap">
                    {entry.generated_sql}
                  </pre>
                )}
                <div className="flex items-center gap-3 text-xs text-muted-foreground">
                  {entry.row_count != null && <span>{entry.row_count} rows</span>}
                  {entry.confidence_score != null && <span>Confidence: {(entry.confidence_score * 100).toFixed(0)}%</span>}
                  {entry.created_at && <span>{new Date(entry.created_at).toLocaleString()}</span>}
                </div>
                {entry.error_message && (
                  <p className="text-xs text-destructive mt-1">{entry.error_message}</p>
                )}
              </div>
            ))}
          </div>
        )}
      </CardContent>
    </Card>
  )
}

export function DocumentManagement() {
  const { canEdit } = useWorkspace()
  const [searchTerm, setSearchTerm] = useState('')
  const [selectedCategory, setSelectedCategory] = useState('all')
  const [dragActive, setDragActive] = useState(false)
  const fileInputRef = useRef<HTMLInputElement>(null)
  
  // Modal state management
  const [selectedDocumentId, setSelectedDocumentId] = useState<number | null>(null)
  const [showDetailsModal, setShowDetailsModal] = useState(false)
  const [showDeleteModal, setShowDeleteModal] = useState(false)
  const [documentToDelete, setDocumentToDelete] = useState<{id: number, filename: string} | null>(null)
  const [showAddDatabaseModal, setShowAddDatabaseModal] = useState(false)
  const [showUploadModal, setShowUploadModal] = useState(false)
  const [sourceToDelete, setSourceToDelete] = useState<{ id: number; name: string } | null>(null)
  const [graphImporting, setGraphImporting] = useState(false)
  const graphFileRef = useRef<HTMLInputElement>(null)

  // Cloud storage state
  const [selectedProvider, setSelectedProvider] = useState<any>(null)
  const [showProviderBrowser, setShowProviderBrowser] = useState(false)
  const [selectedSearchResult, setSelectedSearchResult] = useState<SearchResult | null>(null)
  // PRD-158 S3: page-level team filter (also the agent-eye-view scope).
  const [teamFilter, setTeamFilter] = useState<string | null>(null)

  // API hooks
  const { data: documents = [], isLoading, error } = useDocuments(teamFilter ?? undefined)
  const { data: teams = [] } = useTeams()
  const { data: teamCounts } = useDocumentTeamCounts()
  const { data: documentStats } = useDocumentStats()
  const uploadDocumentMutation = useUploadDocument()
  const deleteDocumentMutation = useDeleteDocument()
  
  // Database Knowledge hooks
  const {
    sources: databaseSources,
    templates,
    loading: dbLoading,
    createSource,
    deleteSource,
    executeQuery,
    syncSchema,
    getCacheStats,
    getSchemaMetadata,
    fetchSources: refreshDatabaseSources
  } = useDatabaseKnowledge()

  // PRD-160 S4: one source selection shared across all six Database tabs
  // (previously every tab hardcoded databaseSources[0]).
  const [selectedDbSourceId, setSelectedDbSourceId] = useState<string | number | null>(null)
  const activeDbSource = (databaseSources || []).find(
    (s: any) => String(s.id) === String(selectedDbSourceId)
  ) || databaseSources?.[0]

  // Cloud storage hooks
  const { data: cloudConnections = [], isLoading: cloudConnectionsLoading, error: cloudConnectionsError } = useCloudConnections()
  const triggerSyncMutation = useTriggerSync()
  const selectRootFolderMutation = useSelectRootFolder()

  // Type the documents array properly
  const typedDocuments = documents as BackendDocument[]
  
  const [ref, inView] = useInView({
    triggerOnce: true,
    threshold: 0.1,
  })

  // Calculate stats from real API data
  const stats = useMemo(() => {
    const totalDocs = typedDocuments.length
    const pendingDocs = typedDocuments.filter(d => {
      const s = (d.status || '').toLowerCase()
      return s === 'pending' || s === 'processing' || s === 'uploading'
    }).length
    const failedDocs = typedDocuments.filter(d => (d.status || '').toLowerCase() === 'failed').length
    const totalSizeBytes = typedDocuments.reduce((sum, d) => sum + (d.file_size || 0), 0)
    const sizeInMB = totalSizeBytes / (1024 * 1024)
    const sizeDisplay = sizeInMB < 1024 ? `${sizeInMB.toFixed(1)} MB` : `${(sizeInMB / 1024).toFixed(1)} GB`

    // Find last sync time from most recently updated document
    const lastSyncDoc = [...typedDocuments]
      .filter(d => d.updated_at || d.created_at)
      .sort((a, b) => new Date(b.updated_at || b.created_at || 0).getTime() - new Date(a.updated_at || a.created_at || 0).getTime())[0]
    const lastSyncDate = lastSyncDoc ? new Date(lastSyncDoc.updated_at || lastSyncDoc.created_at || 0) : null
    const lastSyncDisplay = lastSyncDate
      ? `${Math.floor((Date.now() - lastSyncDate.getTime()) / 3600000)}h ago`
      : 'Never'

    return [
      {
        label: 'Documents',
        value: totalDocs.toString(),
        change: `${totalDocs} total`,
        icon: FileText,
        iconColor: 'text-[hsl(var(--info))]',
        globalIconKey: 'global_document',
      },
      {
        label: 'Processing Status',
        value: pendingDocs > 0 ? `${pendingDocs} pending` : 'All done',
        change: failedDocs > 0 ? `${failedDocs} failed` : 'No errors',
        icon: Database,
        iconColor: pendingDocs > 0 ? 'text-[hsl(var(--warning))]' : 'text-[hsl(var(--success))]',
      },
      {
        label: 'Storage Used',
        value: sizeDisplay,
        change: `Across ${totalDocs} files`,
        icon: FolderOpen,
        iconColor: 'text-primary',
        globalIconKey: 'global_storage',
      },
      {
        label: 'Last Sync',
        value: lastSyncDisplay,
        change: lastSyncDate ? lastSyncDate.toLocaleDateString() : 'No documents',
        icon: History,
        iconColor: 'text-[hsl(var(--agent))]',
      }
    ]
  }, [typedDocuments])

  const handleFileUpload = async (files: FileList | null, teamAccess?: string[]) => {
    console.log('[DocumentManagement] handleFileUpload called with files:', files)

    if (!files || files.length === 0) {
      console.log('[DocumentManagement] No files selected')
      return
    }

    console.log('[DocumentManagement] Processing', files.length, 'file(s)')

    try {
      for (let i = 0; i < files.length; i++) {
        const file = files[i]
        console.log('[DocumentManagement] Uploading file:', file.name, 'size:', file.size, 'type:', file.type)

        await uploadDocumentMutation.mutateAsync({
          file,
          metadata: { description: '', tags: [], team_access: teamAccess || [] }
        })

        console.log('[DocumentManagement] File uploaded successfully:', file.name)
      }

      // Reset file input
      if (fileInputRef.current) {
        fileInputRef.current.value = ''
      }

      console.log('[DocumentManagement] All files uploaded successfully')
    } catch (error) {
      // Error handled by mutation hook
      console.error('[DocumentManagement] Upload error:', error)
    }
  }

  const handleProviderUpload = async (files: FileList, providerId: string, connectionId?: number, teamAccess?: string[]) => {
    console.log('[DocumentManagement] handleProviderUpload called', { providerId, connectionId, fileCount: files.length, teamAccess })

    if (providerId === 'manual') {
      // Upload to Automatos local storage
      await handleFileUpload(files, teamAccess)
    } else {
      // TODO: Upload to cloud provider via API
      // For now, fall back to local upload
      console.log('[DocumentManagement] Cloud provider upload not yet implemented, using local upload')
      await handleFileUpload(files, teamAccess)
    }
  }

  const handleUploadClick = () => {
    console.log('[DocumentManagement] Upload button clicked, triggering file picker')
    fileInputRef.current?.click()
  }

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    console.log('[DocumentManagement] File input changed, files:', e.target.files)
    handleFileUpload(e.target.files)
  }

  const handleGraphImport = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file || !file.name.endsWith('.json')) return
    if (graphFileRef.current) graphFileRef.current.value = ''
    setGraphImporting(true)
    try {
      const { default: apiClient } = await import('@/lib/api-client')
      const formData = new FormData()
      formData.append('file', file)
      formData.append('merge', 'false')
      const headers = await apiClient.getAuthHeaders()
      const wsId = localStorage.getItem('last_active_workspace') || localStorage.getItem('last_active_org')
      if (wsId) (headers as any)['X-Workspace-ID'] = wsId
      const res = await fetch(`${apiClient.getBaseUrl()}/api/knowledge/graph/import`, {
        method: 'POST',
        headers,
        body: formData,
      })
      if (!res.ok) throw new Error(await res.text())
    } catch (err: any) {
      console.error('[GraphImport]', err?.message || err)
    } finally {
      setGraphImporting(false)
    }
  }

  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true)
    } else if (e.type === 'dragleave') {
      setDragActive(false)
    }
  }

  const handleDrop = (e: React.DragEvent) => {
    console.log('[DocumentManagement] File dropped')
    e.preventDefault()
    e.stopPropagation()
    setDragActive(false)
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      console.log('[DocumentManagement] Processing dropped files:', e.dataTransfer.files)
      handleFileUpload(e.dataTransfer.files)
    } else {
      console.log('[DocumentManagement] No files in drop event')
    }
  }

  // Core User Functions - Following Testing Rules
  const handleViewDetails = async (documentId: number) => {
    try {
      // Find document in current list
      const document = typedDocuments.find(doc => doc.id === documentId)
      if (document) {
        setSelectedDocumentId(documentId)
        setShowDetailsModal(true)
      } else {
        console.error('Document not found:', documentId)
      }
    } catch (error) {
      console.error('Error viewing document details:', error)
    }
  }

  const handleDownload = async (documentId: number, filename: string) => {
    try {
      // Get pre-signed URL from backend (mode=url returns JSON, avoids CORS redirect issues)
      const { default: apiClient } = await import('@/lib/api-client')
      const headers = await apiClient.getAuthHeaders()
      const baseUrl = apiClient.getBaseUrl()
      const response = await fetch(`${baseUrl}/api/documents/${documentId}/download?mode=url`, {
        headers,
      })

      if (!response.ok) {
        throw new Error(`Download failed (${response.status})`)
      }

      const contentType = response.headers.get('content-type') || ''

      if (contentType.includes('application/json')) {
        // S3 path: got pre-signed URL — open directly (browser handles download)
        const data = await response.json()
        window.open(data.url, '_blank')
      } else {
        // Local file fallback: response IS the file bytes
        const blob = await response.blob()
        const url = window.URL.createObjectURL(blob)
        const link = document.createElement('a')
        link.href = url
        link.download = filename
        document.body.appendChild(link)
        link.click()
        document.body.removeChild(link)
        window.URL.revokeObjectURL(url)
      }
    } catch (error) {
      console.error('Error downloading document:', error)
      toast.error('Failed to download document')
    }
  }

  const handleDelete = async (documentId: number, filename: string) => {
    try {
      setDocumentToDelete({ id: documentId, filename })
      setShowDeleteModal(true)
    } catch (error) {
      console.error('Error preparing delete confirmation:', error)
    }
  }

  const confirmDelete = async (documentId: number) => {
    try {
      await deleteDocumentMutation.mutateAsync(documentId.toString())
      setShowDeleteModal(false)
      setDocumentToDelete(null)
    } catch (error) {
      // Error handled by mutation hook
      console.error('Error deleting document:', error)
    }
  }

  const filteredDocuments = typedDocuments.filter(doc =>
    (doc.filename || '').toLowerCase().includes(searchTerm.toLowerCase()) ||
    (doc.file_type || '').toLowerCase().includes(searchTerm.toLowerCase())
  )

  return (
    <div className="space-y-8">
      {/* Hidden file input */}
      <input
        ref={fileInputRef} data-testid="file-input"
        type="file"
        multiple
        accept=".pdf,.docx,.txt,.md,.xlsx,.csv,.json,.xml"
        onChange={handleFileChange}
        className="hidden"
      />
      {/* Header */}
      <div ref={ref}>
        <PageHeader
          title="Knowledge"
          titleAccent="Bases"
          eyebrow="Workforce · what they know"
          lede="Documents, code repositories, and references your agents can read. Add a source, scope who sees it, and your workforce becomes more accurate with every search."
          actions={
            <div className="flex items-center gap-2">
              <input ref={graphFileRef} type="file" accept=".json" onChange={handleGraphImport} className="hidden" />
              <Button
                variant="outline"
                onClick={() => graphFileRef.current?.click()}
                disabled={graphImporting}
              >
                <Network className={`w-4 h-4 mr-2 ${graphImporting ? 'animate-spin' : ''}`} />
                {graphImporting ? 'Importing...' : 'Import Graph'}
              </Button>
              <Button
                variant="outline"
                onClick={() => setShowUploadModal(true)}
                disabled={uploadDocumentMutation.isLoading}
              >
                <Upload className={`w-4 h-4 mr-2 ${uploadDocumentMutation.isLoading ? 'animate-spin' : ''}`} />
                {uploadDocumentMutation.isLoading ? 'Uploading...' : 'Upload Documents'}
              </Button>
            </div>
          }
        />
      </div>

      {/* Stats Overview */}
      <StatsBar stats={stats} />

      {/* PRD-158 S3: page-level team filter + per-team counts + agent-eye-view */}
      <div className="flex flex-wrap items-center gap-3">
        <div className="flex items-center gap-2">
          <Filter className="w-4 h-4 text-muted-foreground" />
          <Select
            value={teamFilter ?? 'all'}
            onValueChange={(v) => setTeamFilter(v === 'all' ? null : v)}
          >
            <SelectTrigger className="w-[240px]">
              <SelectValue placeholder="All teams" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="all">
                All teams ({teamCounts?.total ?? documents.length})
              </SelectItem>
              {teams.map((t) => (
                <SelectItem key={t.id} value={t.normalized_name}>
                  {t.name} ({teamCounts?.counts?.[t.normalized_name] ?? 0})
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
        {teamFilter && (
          <Badge variant="secondary" className="gap-1.5">
            <Eye className="w-3 h-3" />
            Agent-eye view: “{teams.find((t) => t.normalized_name === teamFilter)?.name ?? teamFilter}”
            sees public + its own documents
          </Badge>
        )}
      </div>

      {/* Document Management Tabs */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={inView ? { opacity: 1, y: 0 } : {}}
        transition={{ duration: 0.8, delay: 0.4 }}
      >
        <Tabs defaultValue="documents" className="space-y-6">
          <TabsList data-tour="documents-tabs" className="w-full lg:w-auto justify-start gap-1 bg-secondary/50">
            <TabsTrigger value="documents" className="flex items-center space-x-2">
              <FileText className="w-4 h-4" />
              <span>Documents</span>
            </TabsTrigger>
            <TabsTrigger value="database" className="flex items-center space-x-2">
              <Database className="w-4 h-4" />
              <span>Database</span>
            </TabsTrigger>
            <TabsTrigger value="codegraph" className="flex items-center space-x-2">
              <Database className="w-4 h-4" />
              <span>CodeGraph</span>
            </TabsTrigger>
            <TabsTrigger value="businessgraph" className="flex items-center space-x-2">
              <Network className="w-4 h-4" />
              <span>Knowledge Graph</span>
            </TabsTrigger>
            <TabsTrigger value="memory" className="flex items-center space-x-2">
              <Brain className="w-4 h-4" />
              <span>Memory</span>
            </TabsTrigger>
            {/* Analytics tab removed — see /analytics */}
          </TabsList>

          <TabsContent value="documents" className="space-y-6">
            {/* Provider Browser Views */}
            {showProviderBrowser && (selectedProvider?.type === 'manual' || selectedProvider?.type === 'manual-old') ? (
              <LocalStorageBrowser
                documents={typedDocuments}
                onBack={() => {
                  setShowProviderBrowser(false)
                  setSelectedProvider(null)
                }}
                onUpload={() => setShowUploadModal(true)}
                onViewDetails={handleViewDetails}
                onDownload={handleDownload}
                onDelete={handleDelete}
              />
            ) : showProviderBrowser && selectedProvider ? (
              <ProviderBrowser
                providerName={selectedProvider.name}
                providerType={selectedProvider.type}
                connectionId={selectedProvider.connectionId}
                rootFolder={selectedProvider.rootFolder}
                onBack={() => {
                  setShowProviderBrowser(false)
                  setSelectedProvider(null)
                }}
                onSync={async (path) => {
                  await triggerSyncMutation.mutateAsync(selectedProvider.connectionId)
                }}
                onSelectRootFolder={async (path) => {
                  console.log('[DocumentManagement] Selecting root folder:', path, 'for connection:', selectedProvider.connectionId)
                  await selectRootFolderMutation.mutateAsync({
                    connectionId: selectedProvider.connectionId,
                    rootFolderPath: path
                  })
                  setSelectedProvider({
                    ...selectedProvider,
                    rootFolder: path
                  })
                }}
              />
            ) : (
              /* Document sub-tabs */
              <Tabs defaultValue="library" className="space-y-6">
                <TabsList data-tour="documents-subtabs" className="bg-secondary/30">
                  <TabsTrigger value="library" className="flex items-center space-x-2">
                    <FileText className="w-4 h-4" />
                    <span>Library</span>
                  </TabsTrigger>
                  <TabsTrigger value="processing" className="flex items-center space-x-2">
                    <Database className="w-4 h-4" />
                    <span>Processing</span>
                  </TabsTrigger>
                  <TabsTrigger value="multimodal" className="flex items-center space-x-2">
                    <Image className="w-4 h-4" />
                    <span>Multimodal</span>
                  </TabsTrigger>
                  <TabsTrigger value="search" className="flex items-center space-x-2">
                    <Search className="w-4 h-4" />
                    <span>Search</span>
                  </TabsTrigger>
                  <TabsTrigger value="rag" className="flex items-center space-x-2">
                    <Brain className="w-4 h-4" />
                    <span>RAG Test</span>
                  </TabsTrigger>
                  <TabsTrigger value="upload" className="flex items-center space-x-2">
                    <Upload className="w-4 h-4" />
                    <span>Upload</span>
                  </TabsTrigger>
                </TabsList>

                <TabsContent value="library" className="space-y-6">
                  {/* Loading / Error States */}
                  {cloudConnectionsLoading && (
                    <Card className="glass-card p-4">
                      <p className="text-sm text-muted-foreground">Loading cloud connections...</p>
                    </Card>
                  )}
                  {cloudConnectionsError && (
                    <Card className="glass-card p-4 border-destructive/20">
                      <p className="text-sm text-destructive">Error loading connections: {cloudConnectionsError instanceof Error ? cloudConnectionsError.message : String(cloudConnectionsError)}</p>
                    </Card>
                  )}
                  {!cloudConnectionsLoading && cloudConnections.length === 0 && (
                    <Card className="glass-card p-4 border-warning/20">
                      <p className="text-sm text-warning">
                        No cloud connections found. Connect Google Drive or Dropbox in{' '}
                        <a href="/tools" className="underline font-medium">Tools</a> to see them here.
                      </p>
                    </Card>
                  )}

                  {/* Provider Cards */}
                  <ProviderCards
                    providers={[
                      {
                        id: 'manual',
                        name: 'Automatos Storage',
                        type: 'manual',
                        icon: Upload,
                        color: 'text-info',
                        connected: true,
                        stats: {
                          documentCount: typedDocuments.length,
                          chunkCount: typedDocuments.reduce((sum, d) => sum + (d.chunk_count || 0), 0),
                          syncStatus: 'idle',
                        }
                      },
                      ...cloudConnections.map(conn => ({
                        id: `cloud-${conn.id}`,
                        name: conn.app_name,
                        type: conn.app_name.toLowerCase(),
                        icon: Cloud,
                        color: 'text-success',
                        connected: true,
                        connectionId: conn.id,
                        rootFolder: conn.root_folder_path,
                        stats: {
                          documentCount: conn.total_documents_synced,
                          chunkCount: 0,
                          lastSyncedAt: conn.last_successful_sync
                            ? new Date(conn.last_successful_sync).toLocaleString()
                            : undefined,
                          syncStatus: 'idle' as const,
                          rootFolder: conn.root_folder_path
                        }
                      }))
                    ]}
                    onProviderClick={(provider) => {
                      setSelectedProvider(provider)
                      setShowProviderBrowser(true)
                    }}
                  />
                </TabsContent>

                <TabsContent value="processing" className="space-y-6">
                  <DocumentProcessing
                    documents={documents}
                    onDocumentSelect={(docId) => {
                      const doc = typedDocuments.find(d => d.id === parseInt(docId))
                      if (doc) {
                        setSelectedDocumentId(doc.id)
                        setShowDetailsModal(true)
                      }
                    }}
                  />
                </TabsContent>

                <TabsContent value="multimodal" className="space-y-6">
                  <MultimodalKnowledgePanel />
                </TabsContent>

                <TabsContent value="search" className="space-y-6">
                  <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                    <div className="lg:col-span-2">
                      <SemanticSearch
                        context="documents"
                        onResultSelect={(result) => setSelectedSearchResult(result)}
                        standalone={true}
                        showActions={true}
                        maxResults={20}
                      />
                    </div>
                    <div className="lg:col-span-1">
                      {selectedSearchResult ? (
                        <Card className="glass-card sticky top-6">
                          <CardHeader>
                            <div className="flex items-center gap-2">
                              <FileText className="w-5 h-5 text-info" />
                              <CardTitle className="text-base truncate">
                                {selectedSearchResult.source?.filename || 'Document'}
                              </CardTitle>
                            </div>
                            <div className="flex items-center gap-2 mt-2">
                              <Badge className="text-success bg-success/10 border-success/20">
                                {(selectedSearchResult.similarity * 100).toFixed(0)}% Match
                              </Badge>
                              {selectedSearchResult.source?.file_type && (
                                <Badge variant="outline">
                                  {selectedSearchResult.source.file_type.toUpperCase()}
                                </Badge>
                              )}
                            </div>
                          </CardHeader>
                          <CardContent className="space-y-4">
                            <div className="bg-muted/30 rounded-lg p-4 max-h-96 overflow-y-auto">
                              <p className="text-sm whitespace-pre-wrap">
                                {selectedSearchResult.preview || selectedSearchResult.excerpt || ''}
                              </p>
                            </div>
                            <div className="space-y-2 text-xs text-muted-foreground">
                              {selectedSearchResult.chunk_count && (
                                <div>{selectedSearchResult.chunk_count} chunks in document</div>
                              )}
                              {selectedSearchResult.source?.file_size != null && (
                                <div>{(selectedSearchResult.source.file_size / 1024).toFixed(1)} KB</div>
                              )}
                              {selectedSearchResult.source?.upload_date && (
                                <div>Uploaded: {new Date(selectedSearchResult.source.upload_date).toLocaleDateString()}</div>
                              )}
                            </div>
                            <div className="flex gap-2">
                              {selectedSearchResult.document_id && (
                                <Button
                                  size="sm"
                                  variant="outline"
                                  onClick={() => {
                                    setSelectedDocumentId(selectedSearchResult.document_id!)
                                    setShowDetailsModal(true)
                                  }}
                                >
                                  <ExternalLink className="w-3 h-3 mr-1" />
                                  Open Document
                                </Button>
                              )}
                            </div>
                          </CardContent>
                        </Card>
                      ) : (
                        <Card className="glass-card">
                          <CardContent className="p-12 text-center">
                            <FileText className="w-12 h-12 mx-auto text-muted-foreground mb-3" />
                            <p className="text-sm text-muted-foreground">
                              Select a search result to view details
                            </p>
                          </CardContent>
                        </Card>
                      )}
                    </div>
                  </div>
                </TabsContent>

                <TabsContent value="rag" className="space-y-6">
                  <RAGContextBuilder />
                </TabsContent>

                <TabsContent value="upload" className="space-y-6">
                  <Card className="glass-card">
                    <CardHeader>
                      <CardTitle>Upload Documents</CardTitle>
                    </CardHeader>
                    <CardContent>
                      <div
                        className={`border-2 border-dashed rounded-lg p-8 text-center transition-all duration-220 ${
                          dragActive
                            ? 'border-primary bg-primary/5'
                            : 'border-border/50 hover:border-primary/50'
                        }`}
                        onDragEnter={handleDrag}
                        onDragLeave={handleDrag}
                        onDragOver={handleDrag}
                        onDrop={handleDrop}
                      >
                        <Upload className={`w-12 h-12 mx-auto mb-4 ${
                          dragActive ? 'text-primary' : 'text-muted-foreground'
                        } ${uploadDocumentMutation.isLoading ? 'animate-bounce' : ''}`} />

                        {uploadDocumentMutation.isLoading ? (
                          <div className="space-y-4">
                            <h3 className="text-lg font-semibold">Uploading...</h3>
                            <div className="w-full bg-secondary rounded-full h-2">
                              <div
                                className="bg-gradient-to-r from-warning to-red-500 h-2 rounded-full transition-all duration-300 animate-pulse"
                              />
                            </div>
                            <p className="text-sm text-muted-foreground">Processing files...</p>
                          </div>
                        ) : (
                          <>
                            <h3 className="text-lg font-semibold mb-2">
                              {dragActive ? 'Drop files here' : 'Drag and drop files here'}
                            </h3>
                            <p className="text-muted-foreground mb-4">
                              Supports PDF, DOCX, TXT, MD, XLSX, CSV, JSON, XML and more
                            </p>
                            <Button
                              className="gradient-accent hover:opacity-90"
                              onClick={() => setShowUploadModal(true)}
                              disabled={uploadDocumentMutation.isLoading || !canEdit}
                              title={canEdit ? undefined : 'Viewers have read-only access'}
                            >
                              <Plus className="w-4 h-4 mr-2" />
                              Choose Files
                            </Button>
                          </>
                        )}
                      </div>
                    </CardContent>
                  </Card>
                </TabsContent>
              </Tabs>
            )}
          </TabsContent>

          <TabsContent value="database" className="space-y-6">
            {/* Database Knowledge Header */}
            <Card className="glass-card">
              <CardHeader>
                <CardTitle className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <Database className="w-5 h-5 text-info" />
                    Database Knowledge Sources
                  </div>
                  <Button 
                    variant="outline"
                    size="sm"
                    onClick={() => setShowAddDatabaseModal(true)}
                  >
                    <Plus className="w-4 h-4 mr-2" />
                    Add Database
                  </Button>
                </CardTitle>
              </CardHeader>
              <CardContent>
                {dbLoading ? (
                  <div className="text-center py-8 text-muted-foreground">
                    Loading database sources...
                  </div>
                ) : (!databaseSources || databaseSources.length === 0) ? (
                  <div className="text-center py-12">
                    <Database className="w-12 h-12 text-muted-foreground mx-auto mb-4" />
                    <h3 className="text-lg font-semibold mb-2">No database sources yet</h3>
                    <p className="text-muted-foreground mb-4">
                      Connect your databases to query them with natural language
                    </p>
                    <Button 
                      onClick={() => setShowAddDatabaseModal(true)}
                    >
                      <Database className="w-4 h-4 mr-2" />
                      Add Your First Database
                    </Button>
                  </div>
                ) : (
                  <div className="grid gap-4 md:grid-cols-2">
                    {databaseSources.map((source: any) => (
                      <div
                        key={source.id}
                        onClick={() => setSelectedDbSourceId(source.id)}
                        className={`p-4 border rounded-lg cursor-pointer transition-colors ${
                          String(activeDbSource?.id) === String(source.id)
                            ? 'border-primary ring-1 ring-primary bg-primary/5'
                            : 'hover:border-primary/50'
                        }`}
                      >
                        <div className="flex items-center justify-between mb-2">
                          <div className="flex items-center gap-2">
                            <Database className="w-5 h-5 text-success" />
                            <span className="font-medium">{source.name}</span>
                          </div>
                          <Badge variant="outline">{source.dialect}</Badge>
                        </div>
                        <p className="text-sm text-muted-foreground mb-2">
                          {source.tables_count} tables • Last synced {source.last_synced}
                        </p>
                        <div className="flex gap-2">
                          <Button 
                            variant="outline" 
                            size="sm"
                            onClick={() => syncSchema(source.id)}
                          >
                            Sync Schema
                          </Button>
                          <Button 
                            variant="outline" 
                            size="sm"
                            className="text-destructive hover:text-destructive hover:bg-destructive/10"
                            onClick={() => setSourceToDelete({ id: Number(source.id), name: String(source.name) })}
                          >
                            <Trash2 className="w-4 h-4" />
                          </Button>
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </CardContent>
            </Card>

            {/* Sub-tabs for Database Features */}
            <Tabs defaultValue="explorer" className="space-y-6">
              <TabsList className="w-full lg:w-auto justify-start gap-1 bg-secondary/30">
                <TabsTrigger value="explorer">SQL Explorer</TabsTrigger>
                <TabsTrigger value="semantic">Semantic Layer</TabsTrigger>
                <TabsTrigger value="templates">Query Templates</TabsTrigger>
                <TabsTrigger value="training">Training</TabsTrigger>
                <TabsTrigger value="schema">Schema Browser</TabsTrigger>
                <TabsTrigger value="audit">Audit History</TabsTrigger>
              </TabsList>
              
              <TabsContent value="explorer" className="space-y-6">
                <DatabaseQueryExplorer
                  selectedSource={activeDbSource}
                  sources={databaseSources || []}
                  onSourceDeleted={refreshDatabaseSources}
                />
              </TabsContent>
              
              <TabsContent value="semantic" className="space-y-6">
                {activeDbSource ? (
                  <SemanticLayerBuilder
                    sourceId={String(activeDbSource.id)}
                    sourceName={activeDbSource.name}
                    dialect={activeDbSource.dialect || 'postgresql'}
                  />
                ) : (
                  <Card className="glass-card">
                    <CardContent className="p-8 text-center text-muted-foreground">
                      <Database className="h-10 w-10 mx-auto mb-3 opacity-50" />
                      <p>Connect a database first to configure the semantic layer.</p>
                    </CardContent>
                  </Card>
                )}
              </TabsContent>
              
              <TabsContent value="templates" className="space-y-6">
                <QueryTemplatesGrid
                  templates={templates || []}
                  selectedSource={activeDbSource}
                />
              </TabsContent>

              <TabsContent value="training" className="space-y-6">
                {activeDbSource ? (
                  <TrainingExamplesManager sourceId={activeDbSource.id} />
                ) : (
                  <Card className="glass-card">
                    <CardContent className="p-8 text-center text-muted-foreground">
                      <Database className="h-10 w-10 mx-auto mb-3 opacity-50" />
                      <p>Connect a database first to manage training examples.</p>
                    </CardContent>
                  </Card>
                )}
              </TabsContent>

              <TabsContent value="schema" className="space-y-6">
                <SchemaBrowser
                  sourceId={activeDbSource?.id}
                  getSchemaMetadata={getSchemaMetadata}
                />
              </TabsContent>

              <TabsContent value="audit" className="space-y-6">
                <AuditHistory sourceId={activeDbSource?.id} />
              </TabsContent>
            </Tabs>
          </TabsContent>

          {/* Analytics tab removed — see /analytics */}

          <TabsContent value="codegraph" className="space-y-6">
            <CodeGraphPanel />
          </TabsContent>

          <TabsContent value="businessgraph" className="space-y-6">
            <GraphDiffBanner />
            <BusinessGraphPanel />
          </TabsContent>

          <TabsContent value="memory" className="space-y-6">
            <MemoryTab />
          </TabsContent>
        </Tabs>
      </motion.div>

      {/* Document Details Modal */}
      <DocumentDetailsModal
        documentId={selectedDocumentId}
        open={showDetailsModal}
        onClose={() => {
          setShowDetailsModal(false)
          setSelectedDocumentId(null)
        }}
        onDownload={handleDownload}
        onDelete={(id) => {
          const doc = typedDocuments.find(d => d.id === id)
          if (doc) {
            handleDelete(id, doc.filename)
          }
        }}
      />

      {/* Delete Confirmation Modal */}
      <DeleteConfirmationModal
        documentId={documentToDelete?.id || null}
        filename={documentToDelete?.filename || ''}
        open={showDeleteModal}
        onClose={() => {
          setShowDeleteModal(false)
          setDocumentToDelete(null)
        }}
        onConfirm={confirmDelete}
      />

      {/* Add Database Modal */}
      <AddDatabaseModal
        isOpen={showAddDatabaseModal}
        onClose={() => setShowAddDatabaseModal(false)}
        onSuccess={() => {
          // Refresh database sources after adding
          refreshDatabaseSources()
        }}
      />

      <DeleteConfirmation
        open={!!sourceToDelete}
        onOpenChange={(open) => !open && setSourceToDelete(null)}
        title="Delete database source?"
        itemName={sourceToDelete ? `the database source "${sourceToDelete.name}"` : undefined}
        onConfirm={async () => {
          if (sourceToDelete) await deleteSource(sourceToDelete.id)
        }}
      />

      {/* Upload Provider Modal */}
      <UploadProviderModal
        open={showUploadModal}
        onClose={() => setShowUploadModal(false)}
        providers={[
          // Manual upload provider
          {
            id: 'manual',
            name: 'Automatos Storage',
            type: 'manual',
            icon: Upload,
            color: 'text-info',
            connected: true,
          },
          // Cloud providers from connections
          ...cloudConnections.map(conn => ({
            id: `cloud-${conn.id}`,
            name: conn.app_name,
            type: conn.app_name.toLowerCase() as any,
            icon: Cloud,
            color: 'text-success',
            connected: true,
            connectionId: conn.id,
          }))
        ]}
        onUpload={handleProviderUpload}
      />
    </div>
  )
}
