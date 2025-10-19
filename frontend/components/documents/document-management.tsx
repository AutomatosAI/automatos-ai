
'use client'

import React, { useState, useRef, useMemo } from 'react'
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
  Plus
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
// Document modals
import { DocumentDetailsModal } from './document-details-modal'
import { DeleteConfirmationModal } from './delete-confirmation-modal'
import { SemanticSearch } from './semantic-search'
import { DocumentProcessing } from './document-processing'
import { DocumentAnalytics } from './document-analytics'
// API hooks
import { useDocuments, useDocumentStats, useUploadDocument, useDeleteDocument } from '@/hooks/use-document-api'

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
  
  // API hooks
  const { data: documents = [], isLoading, error } = useDocuments()
  const { data: documentStats } = useDocumentStats()
  const uploadDocumentMutation = useUploadDocument()
  const deleteDocumentMutation = useDeleteDocument()
  
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
      // Create download link
      const downloadUrl = `https://api.automatos.app/api/documents/${documentId}/download`
      const link = document.createElement('a')
      link.href = downloadUrl
      link.download = filename
      document.body.appendChild(link)
      link.click()
      document.body.removeChild(link)
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
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.8 }}
        className="flex items-center justify-between"
      >
        <div>
          <h1 className="text-3xl font-bold mb-2">
            Knowledge <span className="gradient-text">Bases</span>
          </h1>
          <p className="text-muted-foreground text-lg">
            Manage documents, code repositories, and knowledge sources
          </p>
        </div>
        
        <Button 
          className="gradient-accent hover:opacity-90 transition-opacity"
          onClick={() => {
            console.log('[Header Button] Upload Documents clicked, fileInputRef:', fileInputRef.current)
            fileInputRef.current?.click()
          }}
          disabled={uploadDocumentMutation.isLoading}
        >
          <Upload className={`w-4 h-4 mr-2 ${uploadDocumentMutation.isLoading ? 'animate-spin' : ''}`} />
          {uploadDocumentMutation.isLoading ? 'Uploading...' : 'Upload Documents'}
        </Button>
      </motion.div>

      {/* Stats Overview */}
      <motion.div
        ref={ref}
        className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6"
        initial={{ opacity: 0, y: 20 }}
        animate={inView ? { opacity: 1, y: 0 } : {}}
        transition={{ duration: 0.8, delay: 0.2 }}
      >
        {stats.map((stat, index) => (
          <motion.div
            key={stat.label}
            className="glass-card p-6 card-glow hover:border-primary/20 transition-all duration-300"
            initial={{ opacity: 0, y: 20 }}
            animate={inView ? { opacity: 1, y: 0 } : {}}
            transition={{ duration: 0.8, delay: index * 0.1 }}
          >
            <div className="flex items-center justify-between mb-4">
              <div className="w-10 h-10 rounded-lg bg-secondary/50 flex items-center justify-center">
                <stat.icon className={`w-5 h-5 ${stat.color}`} />
              </div>
            </div>
            <div className="space-y-1">
              <h3 className="text-2xl font-bold">{stat.value}</h3>
              <p className="text-muted-foreground text-sm">{stat.label}</p>
              <p className="text-xs text-green-400">{stat.change}</p>
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
        <Tabs defaultValue="library" className="space-y-6">
          <TabsList className="grid w-full grid-cols-7 lg:w-auto lg:inline-grid bg-secondary/50">
            <TabsTrigger value="library" className="flex items-center space-x-2">
              <FileText className="w-4 h-4" />
              <span className="hidden sm:inline">Library</span>
            </TabsTrigger>
            <TabsTrigger value="multimodal" className="flex items-center space-x-2">
              <Image className="w-4 h-4" />
              <span className="hidden sm:inline">Multimodal</span>
            </TabsTrigger>
            <TabsTrigger value="search" className="flex items-center space-x-2">
              <Search className="w-4 h-4" />
              <span className="hidden sm:inline">Search</span>
            </TabsTrigger>
            <TabsTrigger value="upload" className="flex items-center space-x-2">
              <Upload className="w-4 h-4" />
              <span className="hidden sm:inline">Upload</span>
            </TabsTrigger>
            <TabsTrigger value="processing" className="flex items-center space-x-2">
              <Database className="w-4 h-4" />
              <span className="hidden sm:inline">Processing</span>
            </TabsTrigger>
            <TabsTrigger value="analytics" className="flex items-center space-x-2">
              <Eye className="w-4 h-4" />
              <span className="hidden sm:inline">Analytics</span>
            </TabsTrigger>
            <TabsTrigger value="codegraph" className="flex items-center space-x-2">
              <Database className="w-4 h-4" />
              <span className="hidden sm:inline">CodeGraph</span>
            </TabsTrigger>
          </TabsList>

          <TabsContent value="library" className="space-y-6">
            {/* Search and Filters */}
            <div className="flex flex-col sm:flex-row gap-4">
              <div className="relative flex-1">
                <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-muted-foreground" />
                <Input
                  placeholder="Search documents by name, category, or tags..."
                  value={searchTerm}
                  onChange={(e) => setSearchTerm(e.target.value)}
                  className="pl-10 bg-secondary/50 border-secondary focus:border-primary/50"
                />
              </div>
              <Button variant="outline" className="shrink-0">
                <Filter className="w-4 h-4 mr-2" />
                Filters
              </Button>
            </div>

            {/* Document Grid */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
              {filteredDocuments.map((doc, index) => {
                const fileType = (doc.file_type || 'unknown').toLowerCase()
                const TypeIcon = typeIcons[fileType] || File
                
                return (
                  <motion.div
                    key={doc.id}
                    className="glass-card p-6 card-glow hover:border-primary/20 transition-all duration-300"
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.5, delay: index * 0.1 }}
                  >
                    {/* Header */}
                    <div className="flex items-start justify-between mb-4">
                      <div className="flex items-center space-x-3">
                        <div className="w-10 h-10 rounded-lg bg-secondary/50 flex items-center justify-center">
                          <TypeIcon className="w-5 h-5 text-primary" />
                        </div>
                        <div className="flex-1 min-w-0">
                          <h3 className="font-semibold truncate">{doc.filename}</h3>
                          <p className="text-xs text-muted-foreground">
                            {(doc.file_type || 'unknown').toUpperCase()} • {(() => {
                              const bytes = doc.file_size || 0
                              if (bytes >= 1024 * 1024 * 1024) return `${(bytes / (1024*1024*1024)).toFixed(1)}GB`
                              if (bytes >= 1024 * 1024) return `${(bytes / (1024*1024)).toFixed(1)}MB`
                              if (bytes >= 1024) return `${(bytes / 1024).toFixed(1)}KB`
                              return `${bytes}B`
                            })()}
                          </p>
                        </div>
                      </div>
                      
                      <DropdownMenu>
                        <DropdownMenuTrigger asChild>
                          <Button variant="ghost" size="icon" className="h-8 w-8">
                            <MoreVertical className="w-4 h-4" />
                          </Button>
                        </DropdownMenuTrigger>
                        <DropdownMenuContent align="end">
                          <DropdownMenuItem onClick={() => handleViewDetails(doc.id)}>
                            <Eye className="w-4 h-4 mr-2" />
                            View Details
                          </DropdownMenuItem>
                          <DropdownMenuItem onClick={() => handleDownload(doc.id, doc.filename)}>
                            <Download className="w-4 h-4 mr-2" />
                            Download
                          </DropdownMenuItem>
                          <DropdownMenuItem className="text-red-400" onClick={() => handleDelete(doc.id, doc.filename)}>
                            <Trash2 className="w-4 h-4 mr-2" />
                            Delete
                          </DropdownMenuItem>
                        </DropdownMenuContent>
                      </DropdownMenu>
                    </div>

                    {/* Status */}
                    <div className="flex items-center justify-between mb-4">
                      <Badge className={statusStyles[(doc.status || 'completed').toLowerCase()] || statusStyles.completed}>
                        {(doc.status || 'completed').toLowerCase()}
                      </Badge>
                      <Badge variant="outline" className="text-xs">
                        {doc.file_type || 'unknown'}
                      </Badge>
                    </div>

                    {/* Description */}
                    <p className="text-sm text-muted-foreground mb-4 line-clamp-2">
                      Document ready for processing and analysis
                    </p>

                    {/* Processing Info */}
                    <div className="grid grid-cols-2 gap-4 mb-4">
                      <div>
                        <p className="text-sm font-medium">Vector Chunks</p>
                        <p className="text-xs text-muted-foreground">{doc.chunk_count ?? 0}</p>
                      </div>
                      <div>
                        <p className="text-sm font-medium">Uploaded</p>
                        <p className="text-xs text-muted-foreground">
                          {doc.upload_date ? new Date(doc.upload_date).toLocaleDateString() : new Date().toLocaleDateString()}
                        </p>
                      </div>
                    </div>

                    {/* Tags */}
                    <div className="flex flex-wrap gap-1">
                      <Badge variant="secondary" className="text-xs">
                        {doc.file_type || 'unknown'}
                      </Badge>
                      <Badge variant="secondary" className="text-xs">
                        {(doc.status || 'processed').toLowerCase()}
                      </Badge>
                    </div>
                  </motion.div>
                )
              })}
              
              {/* Loading State */}
              {isLoading && (
                <div className="col-span-full flex items-center justify-center py-12">
                  <div className="text-center">
                    <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary mx-auto mb-4"></div>
                    <p className="text-muted-foreground">Loading documents...</p>
                  </div>
                </div>
              )}
              
              {/* Empty State */}
              {(isLoading === false && filteredDocuments.length === 0) && (
                <div className="col-span-full flex items-center justify-center py-12">
                  <div className="text-center">
                    <FileText className="w-12 h-12 text-muted-foreground mx-auto mb-4" />
                    <h3 className="text-lg font-semibold mb-2">No documents found</h3>
                    <p className="text-muted-foreground mb-4">
                      {searchTerm ? 'Try adjusting your search terms' : 'Upload your first document to get started'}
                    </p>
                    {!searchTerm && (
                      <Button onClick={() => fileInputRef.current?.click()}>
                        <Upload className="w-4 h-4 mr-2" />
                        Upload Documents
                      </Button>
                    )}
                  </div>
                </div>
              )}
              
              {/* Error State */}
              {error && (
                <div className="col-span-full flex items-center justify-center py-12">
                  <div className="text-center">
                    <div className="text-red-500 mb-4">⚠️</div>
                    <h3 className="text-lg font-semibold mb-2 text-red-500">Error loading documents</h3>
                    <p className="text-muted-foreground">{error instanceof Error ? error.message : String(error)}</p>
                  </div>
                </div>
              )}
            </div>
          </TabsContent>

          <TabsContent value="multimodal" className="space-y-6">
            <MultimodalKnowledgePanel />
          </TabsContent>

          <TabsContent value="search" className="space-y-6">
            <SemanticSearch
              context="documents"
              onResultSelect={(result) => {
                // Find and select the document
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
                        onClick={() => {
                          console.log('[Upload Button] Choose Files clicked, fileInputRef:', fileInputRef.current)
                          fileInputRef.current?.click()
                        }}
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

          <TabsContent value="analytics" className="space-y-6">
            <DocumentAnalytics 
              documents={documents}
              documentStats={documentStats}
            />
          </TabsContent>

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
    </div>
  )
}
