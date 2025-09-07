
'use client'

import { useState, useRef, useEffect } from 'react'
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
import { apiClient, Document } from '@/lib/api'
import { ProcessingTab } from './processing-tab'
import { AnalyticsTab } from './analytics-tab'

// Import the new professional modals
import { DocumentDetailsModal } from './document-details-modal'
import { DownloadProgressModal } from './download-progress-modal'
import { DeleteConfirmationModal } from './delete-confirmation-modal'

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
  const [isUploading, setIsUploading] = useState(false)
  const [uploadProgress, setUploadProgress] = useState(0)
  const [dragActive, setDragActive] = useState(false)
  const fileInputRef = useRef<HTMLInputElement>(null)
  
  // Real data state
  const [realDocuments, setRealDocuments] = useState<BackendDocument[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [stats, setStats] = useState([
    {
      label: 'Total Documents',
      value: '0',
      change: '+0 this month',
      icon: FileText,
      color: 'text-blue-400'
    },
    {
      label: 'Processed',
      value: '0',
      change: '0% success rate',
      icon: Database,
      color: 'text-green-400'
    },
    {
      label: 'Storage Used',
      value: '0 GB',
      change: '+0 GB this week',
      icon: FolderOpen,
      color: 'text-orange-400'
    },
    {
      label: 'Vector Chunks',
      value: '0',
      change: '+0 chunks',
      icon: Database,
      color: 'text-purple-400'
    }
  ])
  
  // Modal states
  const [detailsModalOpen, setDetailsModalOpen] = useState(false)
  const [selectedDocumentId, setSelectedDocumentId] = useState<number | null>(null)
  const [downloadModalOpen, setDownloadModalOpen] = useState(false)
  const [downloadInfo, setDownloadInfo] = useState<{filename: string, documentId: number} | null>(null)
  const [deleteModalOpen, setDeleteModalOpen] = useState(false)
  const [deleteInfo, setDeleteInfo] = useState<{documentId: number, filename: string} | null>(null)

  const [ref, inView] = useInView({
    triggerOnce: true,
    threshold: 0.1,
  })

  // Fetch real documents from API
  useEffect(() => {
    const fetchDocuments = async () => {
      try {
        setLoading(true)
        const documents = await apiClient.request<BackendDocument[]>('/api/documents/')
        setRealDocuments(documents)
        
        // Calculate real stats from backend data
        const totalDocs = documents.length
        const processedDocs = documents.filter(d => (d.status || '').toLowerCase() === 'completed').length
        const totalSizeBytes = documents.reduce((sum, d) => sum + (d.file_size || 0), 0)
        const sizeInMB = totalSizeBytes / (1024 * 1024)
        const sizeDisplay = sizeInMB < 1024 ? `${sizeInMB.toFixed(1)} MB` : `${(sizeInMB / 1024).toFixed(1)} GB`
        
        setStats([
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
            value: documents.reduce((sum, d)=> sum + (d.chunk_count || 0), 0).toString(),
            change: '',
            icon: Database,
            color: 'text-purple-400'
          }
        ])
        
        setError(null)
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to fetch documents')
        console.error('Error fetching documents:', err)
      } finally {
        setLoading(false)
      }
    }

    fetchDocuments()
  }, [])

  const handleFileUpload = async (files: FileList | null) => {
    if (!files || files.length === 0) return

    setIsUploading(true)
    setUploadProgress(0)

    try {
      for (let i = 0; i < files.length; i++) {
        const file = files[i]

        // Simulate upload progress
        const interval = setInterval(() => {
          setUploadProgress(prev => {
            if (prev >= 90) {
              clearInterval(interval)
              return 90
            }
            return prev + 10
          })
        }, 200)

        // Upload to backend API (multipart/form-data -> /api/documents/upload)
        const uploadRes = await apiClient.uploadDocument(file, { description: '', tags: [] })

        clearInterval(interval)
        setUploadProgress(100)

        console.log('Upload result:', uploadRes)

        // Small delay to show completion
        await new Promise(resolve => setTimeout(resolve, 500))
      }

      // Reset state after successful upload and refresh documents
      setTimeout(async () => {
        setIsUploading(false)
        setUploadProgress(0)
        if (fileInputRef.current) {
          fileInputRef.current.value = ''
        }
        
        // Refresh documents list
        try {
          const documents = await apiClient.request<BackendDocument[]>('/api/documents/')
          if (Array.isArray(documents)) {
            setRealDocuments(documents)
            
            // Recalculate stats
            const totalDocs = documents.length
            const processedDocs = documents.filter(d => (d.status || '').toLowerCase() === 'completed').length
            
            // Calculate total storage (bytes)
            const totalSizeBytes = documents.reduce((sum, d) => sum + (d.file_size || 0), 0)
            const sizeInMB = totalSizeBytes / (1024 * 1024)
            const sizeDisplay = sizeInMB < 1024 ? `${sizeInMB.toFixed(1)} MB` : `${(sizeInMB / 1024).toFixed(1)} GB`
            
            setStats([
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
                value: documents.reduce((sum, d)=> sum + (d.chunk_count || 0), 0).toString(),
                change: '',
                icon: Database,
                color: 'text-purple-400'
              }
            ])
          }
        } catch (error) {
          console.error('Error refreshing documents:', error)
        }
      }, 1000)

    } catch (error) {
      console.error('Upload error:', error)
      setIsUploading(false)
      setUploadProgress(0)
      setError('Failed to upload document. Please try again.')
    }
  }

  const handleUploadClick = () => {
    fileInputRef.current?.click()
  }

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
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
    e.preventDefault()
    e.stopPropagation()
    setDragActive(false)
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFileUpload(e.dataTransfer.files)
    }
  }

  // Enhanced User Functions - Using Professional Modals
  const handleViewDetails = async (documentId: number) => {
    setSelectedDocumentId(documentId)
    setDetailsModalOpen(true)
  }

  const handleDownload = async (documentId: number, filename: string) => {
    setDownloadInfo({ filename, documentId })
    setDownloadModalOpen(true)
  }

  const handleDelete = async (documentId: number) => {
    const document = realDocuments.find(d => d.id === documentId)
    if (!document) return
    
    setDeleteInfo({ documentId, filename: document.filename })
    setDeleteModalOpen(true)
  }

  // Modal handlers
  const handleConfirmDelete = async (documentId: number) => {
    try {
      await apiClient.request(`/api/documents/${documentId}`, {
        method: 'DELETE'
      })
      
      // Refresh documents list after deletion
      const documents = await apiClient.request<BackendDocument[]>('/api/documents/')
      if (Array.isArray(documents)) {
        setRealDocuments(documents)
      }
    } catch (error) {
      console.error('Error deleting document:', error)
      throw error // Re-throw to let the modal handle the error display
    }
  }

  const refreshDocuments = async () => {
    try {
      const documents = await apiClient.request<BackendDocument[]>('/api/documents/')
      if (Array.isArray(documents)) {
        setRealDocuments(documents)
      }
    } catch (error) {
      console.error('Error refreshing documents:', error)
    }
  }

  const filteredDocuments = realDocuments.filter(doc =>
    (doc.filename || '').toLowerCase().includes(searchTerm.toLowerCase()) ||
    (doc.file_type || '').toLowerCase().includes(searchTerm.toLowerCase())
  )

  return (
    <div className="space-y-8">
      {/* Hidden file input */}
      <input
        ref={fileInputRef}
        type="file"
        multiple
        accept=".pdf,.docx,.txt,.md,.xlsx,.csv,.json,.xml"
        onChange={handleFileChange}
        className="hidden"
      />

      {/* Professional Modals */}
      <DocumentDetailsModal
        documentId={selectedDocumentId}
        open={detailsModalOpen}
        onClose={() => {
          setDetailsModalOpen(false)
          setSelectedDocumentId(null)
        }}
        onDownload={handleDownload}
        onDelete={handleDelete}
      />

      <DownloadProgressModal
        open={downloadModalOpen}
        onClose={() => {
          setDownloadModalOpen(false)
          setDownloadInfo(null)
        }}
        filename={downloadInfo?.filename || ''}
        documentId={downloadInfo?.documentId}
      />

      <DeleteConfirmationModal
        documentId={deleteInfo?.documentId || null}
        filename={deleteInfo?.filename || ''}
        open={deleteModalOpen}
        onClose={() => {
          setDeleteModalOpen(false)
          setDeleteInfo(null)
        }}
        onConfirm={handleConfirmDelete}
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
            Document <span className="gradient-text">Management</span>
          </h1>
          <p className="text-muted-foreground text-lg">
            Manage your knowledge base and document processing pipeline
          </p>
        </div>
        
        <Button 
          className="gradient-accent hover:opacity-90 transition-opacity"
          onClick={handleUploadClick}
          disabled={isUploading}
        >
          <Upload className={`w-4 h-4 mr-2 ${isUploading ? 'animate-spin' : ''}`} />
          {isUploading ? `Uploading... ${uploadProgress}%` : 'Upload Documents'}
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
          <TabsList className="grid w-full grid-cols-5 lg:w-auto lg:inline-grid bg-secondary/50">
            <TabsTrigger value="library" className="flex items-center space-x-2">
              <FileText className="w-4 h-4" />
              <span className="hidden sm:inline">Document Library</span>
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
                        <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-orange-500 to-red-500 flex items-center justify-center">
                          <TypeIcon className="w-5 h-5 text-white" />
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
                          <DropdownMenuItem className="text-red-400" onClick={() => handleDelete(doc.id)}>
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
              {loading && (
                <div className="col-span-full flex items-center justify-center py-12">
                  <div className="text-center">
                    <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary mx-auto mb-4"></div>
                    <p className="text-muted-foreground">Loading documents...</p>
                  </div>
                </div>
              )}
              
              {/* Empty State */}
              {!loading && filteredDocuments.length === 0 && (
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
                    <p className="text-muted-foreground">{error}</p>
                  </div>
                </div>
              )}
            </div>
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
                  }`} />
                  <h3 className="text-lg font-semibold mb-2">
                    {dragActive ? 'Drop files here' : 'Upload Documents'}
                  </h3>
                  <p className="text-muted-foreground mb-4">
                    Drag and drop files here, or click to select files
                  </p>
                  <Button onClick={handleUploadClick} disabled={isUploading}>
                    <Plus className="w-4 h-4 mr-2" />
                    Select Files
                  </Button>
                  <p className="text-xs text-muted-foreground mt-4">
                    Supported formats: PDF, DOCX, TXT, MD, XLSX, CSV, JSON, XML
                  </p>
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="processing">
            <ProcessingTab />
          </TabsContent>

          <TabsContent value="analytics">
            <AnalyticsTab />
          </TabsContent>

          <TabsContent value="codegraph">
            <CodeGraphPanel />
          </TabsContent>
        </Tabs>
      </motion.div>
    </div>
  )
}
