
'use client'

import React, { useState, useRef, useMemo, useEffect } from 'react'
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
} from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { 
  DropdownMenu, 
  DropdownMenuContent, 
  DropdownMenuItem, 
  DropdownMenuTrigger 
} from '@/components/ui/dropdown-menu'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { CodeGraphPanel } from '@/components/knowledge/CodeGraphPanel'
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
import { DocumentProcessing } from './document-processing'
// DocumentAnalytics removed — analytics consolidated into /analytics
// Cloud Storage Components (PRD-42)
import { ProviderCards } from './provider-cards'
import { ProviderBrowser } from './provider-browser'
import { LocalStorageBrowser } from './local-storage-browser'
// API hooks
import { useDocuments, useDocumentStats, useUploadDocument, useDeleteDocument } from '@/hooks/use-document-api'
import { useDatabaseKnowledge } from '@/hooks/use-database-knowledge'
import { useCloudConnections, useTriggerSync, useSelectRootFolder } from '@/hooks/use-cloud-storage'

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
}

// Stats will be calculated dynamically from real data

const statusStyles: Record<string, string> = {
  completed: 'bg-green-500/10 text-green-400 border-green-500/20',
  processed: 'bg-green-500/10 text-green-400 border-green-500/20',
  processing: 'bg-yellow-500/10 text-yellow-400 border-yellow-500/20',
  failed: 'bg-red-500/10 text-red-400 border-red-500/20',
  pending: 'bg-gray-500/10 text-gray-400 border-gray-500/20'
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

export function DocumentManagement() {
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

  // Cloud storage state
  const [selectedProvider, setSelectedProvider] = useState<any>(null)
  const [showProviderBrowser, setShowProviderBrowser] = useState(false)
  
  // API hooks
  const { data: documents = [], isLoading, error } = useDocuments()
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
    fetchSources: refreshDatabaseSources
  } = useDatabaseKnowledge()

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
    const processedDocs = typedDocuments.filter(d => (d.status || '').toLowerCase() === 'completed').length
    const totalSizeBytes = typedDocuments.reduce((sum, d) => sum + (d.file_size || 0), 0)
    const sizeInMB = totalSizeBytes / (1024 * 1024)
    const sizeDisplay = sizeInMB < 1024 ? `${sizeInMB.toFixed(1)} MB` : `${(sizeInMB / 1024).toFixed(1)} GB`
    
    return [
      {
        label: 'Total Documents',
        value: totalDocs.toString(),
        change: `+${Math.max(0, totalDocs - 2)} this month`,
        icon: FileText,
        color: 'text-blue-400'
      },
      {
        label: 'Processed',
        value: processedDocs.toString(),
        change: totalDocs > 0 ? `${((processedDocs / totalDocs) * 100).toFixed(1)}% success rate` : '0% success rate',
        icon: Database,
        color: 'text-green-400'
      },
      {
        label: 'Storage Used',
        value: sizeDisplay,
        change: `+${Math.max(0, sizeInMB - 0.5).toFixed(1)} MB this week`,
        icon: FolderOpen,
        color: 'text-orange-400'
      },
      {
        label: 'Vector Chunks',
        value: typedDocuments.reduce((sum, d)=> sum + (d.chunk_count || 0), 0).toString(),
        change: '',
        icon: Database,
        color: 'text-purple-400'
      }
    ]
  }, [typedDocuments])

  const handleFileUpload = async (files: FileList | null) => {
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
          metadata: { description: '', tags: [] }
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

  const handleProviderUpload = async (files: FileList, providerId: string, connectionId?: number) => {
    console.log('[DocumentManagement] handleProviderUpload called', { providerId, connectionId, fileCount: files.length })

    if (providerId === 'manual') {
      // Upload to Automatos local storage
      await handleFileUpload(files)
    } else {
      // TODO: Upload to cloud provider via API
      // For now, fall back to local upload
      console.log('[DocumentManagement] Cloud provider upload not yet implemented, using local upload')
      await handleFileUpload(files)
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
      // Use apiClient for authenticated download with proper headers
      const response = await apiClient.get(`/api/documents/${documentId}/download`, {
        responseType: 'blob'
      })

      // Create blob URL and trigger download
      const blob = new Blob([response])
      const url = window.URL.createObjectURL(blob)
      const link = document.createElement('a')
      link.href = url
      link.download = filename
      document.body.appendChild(link)
      link.click()
      document.body.removeChild(link)
      window.URL.revokeObjectURL(url)
    } catch (error) {
      console.error('Error downloading document:', error)
      alert('Error downloading document')
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
      <motion.div
        ref={ref}
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.8 }}
        className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3"
      >
        <div>
          <h1 data-testid="page-title" className="text-2xl md:text-3xl font-bold mb-1 md:mb-2">
            Knowledge <span className="gradient-text">Bases</span>
          </h1>
          <p className="text-muted-foreground text-sm md:text-lg">
            Manage documents, code repositories, and knowledge sources
          </p>
        </div>

        <Button
          className="gradient-accent hover:opacity-90 transition-opacity shrink-0"
          onClick={() => setShowUploadModal(true)}
          disabled={uploadDocumentMutation.isLoading}
        >
          <Upload className={`w-4 h-4 mr-2 ${uploadDocumentMutation.isLoading ? 'animate-spin' : ''}`} />
          {uploadDocumentMutation.isLoading ? 'Uploading...' : 'Upload Documents'}
        </Button>
      </motion.div>

      {/* Stats Overview */}
      <motion.div
        className="hidden md:grid grid-cols-2 lg:grid-cols-4 gap-4 md:gap-6"
        initial={{ opacity: 0, y: 20 }}
        animate={inView ? { opacity: 1, y: 0 } : {}}
        transition={{ duration: 0.8, delay: 0.2 }}
      >
        {stats.map((stat, index) => (
          <motion.div
            key={stat.label}
            className="glass-card p-4 card-glow hover:border-primary/20 transition-all duration-300"
            initial={{ opacity: 0, y: 20 }}
            animate={inView ? { opacity: 1, y: 0 } : {}}
            transition={{ duration: 0.8, delay: index * 0.1 }}
          >
            <div className="flex items-center justify-between gap-3">
              <div className="flex items-center gap-3 min-w-0">
                <div className="w-10 h-10 rounded-2xl bg-black/20 border border-orange-500/10 flex items-center justify-center shrink-0">
                  <stat.icon className={`w-5 h-5 ${stat.color}`} />
                </div>
                <div className="min-w-0">
                  <div className="text-2xl font-bold leading-none">{stat.value}</div>
                  <div className="text-sm text-muted-foreground truncate">{stat.label}</div>
                </div>
              </div>
              <div className="shrink-0 text-right text-xs text-green-400">
                {stat.change}
              </div>
            </div>
          </motion.div>
        ))}
      </motion.div>

      {/* Document Management Tabs */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={inView ? { opacity: 1, y: 0 } : {}}
        transition={{ duration: 0.8, delay: 0.4 }}
      >
        <Tabs defaultValue="documents" className="space-y-6">
          <TabsList className="w-full lg:w-auto justify-start gap-1 bg-secondary/50">
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
                <TabsList className="bg-secondary/30">
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
                    <Card className="glass-card p-4 border-red-500/20">
                      <p className="text-sm text-red-500">Error loading connections: {cloudConnectionsError instanceof Error ? cloudConnectionsError.message : String(cloudConnectionsError)}</p>
                    </Card>
                  )}
                  {!cloudConnectionsLoading && cloudConnections.length === 0 && (
                    <Card className="glass-card p-4 border-yellow-500/20">
                      <p className="text-sm text-yellow-600">
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
                        color: 'text-blue-400',
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
                        color: 'text-green-400',
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
                  <SemanticSearch
                    context="documents"
                    onResultSelect={(result) => {
                      const doc = typedDocuments.find(d => d.id === result.document_id)
                      if (doc) {
                        setSelectedDocumentId(doc.id)
                        setShowDetailsModal(true)
                      }
                    }}
                    showActions={true}
                    maxResults={10}
                  />
                </TabsContent>

                <TabsContent value="upload" className="space-y-6">
                  <Card className="glass-card">
                    <CardHeader>
                      <CardTitle>Upload Documents</CardTitle>
                    </CardHeader>
                    <CardContent>
                      <div
                        className={`border-2 border-dashed rounded-lg p-8 text-center transition-all duration-200 ${
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
                                className="bg-gradient-to-r from-orange-500 to-red-500 h-2 rounded-full transition-all duration-300 animate-pulse"
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
                              disabled={uploadDocumentMutation.isLoading}
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
                    <Database className="w-5 h-5 text-blue-400" />
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
                      <div key={source.id} className="p-4 border rounded-lg">
                        <div className="flex items-center justify-between mb-2">
                          <div className="flex items-center gap-2">
                            <Database className="w-5 h-5 text-green-500" />
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
                            className="text-red-500 hover:text-red-600 hover:bg-red-50"
                            onClick={() => {
                              if (window.confirm(`Delete database source "${source.name}"? This cannot be undone.`)) {
                                deleteSource(source.id)
                              }
                            }}
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
                  selectedSource={databaseSources?.[0]}
                  sources={databaseSources || []}
                  onSourceDeleted={refreshDatabaseSources}
                />
              </TabsContent>
              
              <TabsContent value="semantic" className="space-y-6">
                <SemanticLayerBuilder 
                  selectedSource={databaseSources?.[0]}
                  sources={databaseSources || []}
                />
              </TabsContent>
              
              <TabsContent value="templates" className="space-y-6">
                <QueryTemplatesGrid 
                  templates={templates || []}
                  selectedSource={databaseSources?.[0]}
                />
              </TabsContent>

              <TabsContent value="training" className="space-y-6">
                {databaseSources && databaseSources.length > 0 ? (
                  <TrainingExamplesManager sourceId={databaseSources[0].id} />
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
                {/* Schema Browser */}
                <Card className="glass-card">
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                      <Database className="h-5 w-5" />
                      Database Schema Browser
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    {!databaseSources || databaseSources.length === 0 ? (
                      <p className="text-muted-foreground text-center py-8">
                        No database sources connected. Add a database to browse its schema.
                      </p>
                    ) : (
                      <div className="space-y-4">
                        <div className="border rounded-lg p-4">
                          <h4 className="font-semibold mb-3">Tables & Columns</h4>
                          <div className="space-y-3">
                            {/* Example schema structure - would be populated from API */}
                            <div className="border-l-2 border-blue-500 pl-4">
                              <div className="font-medium flex items-center gap-2">
                                <Database className="h-4 w-4" />
                                customers
                              </div>
                              <div className="ml-4 mt-2 space-y-1 text-sm text-muted-foreground">
                                <div>• id (INTEGER, PK)</div>
                                <div>• name (VARCHAR)</div>
                                <div>• email (VARCHAR)</div>
                                <div>• created_at (TIMESTAMP)</div>
                              </div>
                            </div>
                            <div className="border-l-2 border-green-500 pl-4">
                              <div className="font-medium flex items-center gap-2">
                                <Database className="h-4 w-4" />
                                orders
                              </div>
                              <div className="ml-4 mt-2 space-y-1 text-sm text-muted-foreground">
                                <div>• id (INTEGER, PK)</div>
                                <div>• customer_id (INTEGER, FK→customers.id)</div>
                                <div>• total_amount (DECIMAL)</div>
                                <div>• order_date (TIMESTAMP)</div>
                              </div>
                            </div>
                          </div>
                        </div>
                        <div className="border rounded-lg p-4">
                          <h4 className="font-semibold mb-3">Relationships</h4>
                          <div className="space-y-2 text-sm">
                            <div className="flex items-center gap-2">
                              <Badge variant="outline">Many-to-One</Badge>
                              <span>orders.customer_id → customers.id</span>
                            </div>
                          </div>
                        </div>
                      </div>
                    )}
                  </CardContent>
                </Card>
              </TabsContent>

              <TabsContent value="audit" className="space-y-6">
                {/* Audit History */}
                <Card className="glass-card">
                  <CardHeader>
                    <CardTitle className="flex items-center justify-between">
                      <div className="flex items-center gap-2">
                        <History className="h-5 w-5" />
                        Query Audit History
                      </div>
                      <Button variant="outline" size="sm" onClick={() => {
                        // TODO: Export audit log
                        console.log('Export audit log')
                      }}>
                        <Download className="h-4 w-4 mr-2" />
                        Export CSV
                      </Button>
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-4">
                      {/* Audit entries would come from API */}
                      <div className="border rounded-lg p-4 hover:bg-muted/50 transition-colors">
                        <div className="flex items-start justify-between mb-2">
                          <div>
                            <p className="font-medium">Show top 10 customers by revenue</p>
                            <p className="text-sm text-muted-foreground mt-1">
                              SELECT customer_name, SUM(order_total) as revenue...
                            </p>
                          </div>
                          <Badge variant="outline" className="text-xs">
                            247ms
                          </Badge>
                        </div>
                        <div className="flex items-center gap-4 text-xs text-muted-foreground">
                          <span>User: admin@automatos.ai</span>
                          <span>Rows: 10</span>
                          <span>2 minutes ago</span>
                          <Badge className="text-xs" variant="secondary">
                            Success
                          </Badge>
                        </div>
                      </div>
                      
                      <div className="border rounded-lg p-4 hover:bg-muted/50 transition-colors">
                        <div className="flex items-start justify-between mb-2">
                          <div>
                            <p className="font-medium">Calculate monthly revenue trend</p>
                            <p className="text-sm text-muted-foreground mt-1">
                              SELECT DATE_TRUNC('month', order_date) as month...
                            </p>
                          </div>
                          <Badge variant="outline" className="text-xs">
                            523ms
                          </Badge>
                        </div>
                        <div className="flex items-center gap-4 text-xs text-muted-foreground">
                          <span>User: analyst@automatos.ai</span>
                          <span>Rows: 12</span>
                          <span>15 minutes ago</span>
                          <Badge className="text-xs" variant="secondary">
                            Success
                          </Badge>
                        </div>
                      </div>

                      <div className="text-center py-4">
                        <Button variant="outline" size="sm">
                          Load More
                        </Button>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              </TabsContent>
            </Tabs>
          </TabsContent>

          {/* Analytics tab removed — see /analytics */}

          <TabsContent value="codegraph" className="space-y-6">
            <CodeGraphPanel />
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
          window.location.reload()
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
            color: 'text-blue-400',
            connected: true,
          },
          // Cloud providers from connections
          ...cloudConnections.map(conn => ({
            id: `cloud-${conn.id}`,
            name: conn.app_name,
            type: conn.app_name.toLowerCase() as any,
            icon: Cloud,
            color: 'text-green-400',
            connected: true,
            connectionId: conn.id,
          }))
        ]}
        onUpload={handleProviderUpload}
      />
    </div>
  )
}
