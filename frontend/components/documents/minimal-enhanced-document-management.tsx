'use client'

import { useState } from 'react'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Badge } from '@/components/ui/badge'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
  DropdownMenuSeparator
} from '@/components/ui/dropdown-menu'
import { Upload, FileText, BarChart, Search, Eye, Settings, Plus, Filter, Download, RefreshCw, HardDrive, Clock, CheckCircle, MoreVertical, Edit, Trash2, Database, File } from 'lucide-react'

// Import the professional modals
import { DocumentDetailsModal } from './document-details-modal'
import { DownloadProgressModal } from './download-progress-modal'
import { DeleteConfirmationModal } from './delete-confirmation-modal'

export function MinimalEnhancedDocumentManagement() {
  const [activeTab, setActiveTab] = useState('library')
  const [searchTerm, setSearchTerm] = useState('')
  
  // Modal states
  const [detailsModalOpen, setDetailsModalOpen] = useState(false)
  const [selectedDocumentId, setSelectedDocumentId] = useState<number | null>(null)
  const [downloadModalOpen, setDownloadModalOpen] = useState(false)
  const [downloadInfo, setDownloadInfo] = useState<{filename: string, documentId: number} | null>(null)
  const [deleteModalOpen, setDeleteModalOpen] = useState(false)
  const [deleteInfo, setDeleteInfo] = useState<{documentId: number, filename: string} | null>(null)

  // Document stats matching Agent Management pattern
  const documentStats = [
    {
      label: 'Total Documents',
      value: '247',
      change: '+12 this week',
      icon: FileText,
      color: 'text-orange-400'
    },
    {
      label: 'Storage Used',
      value: '1.8 GB',
      change: '45% capacity',
      icon: HardDrive,
      color: 'text-orange-400'
    },
    {
      label: 'Processing Queue',
      value: '3',
      change: 'Active jobs',
      icon: Clock,
      color: 'text-orange-400'
    },
    {
      label: 'Vector Chunks',
      value: '24,567',
      change: '+1,247 chunks',
      icon: Database,
      color: 'text-orange-400'
    }
  ]

  // Enhanced document data with marketing site structure
  const documents = [
    {
      id: 1,
      name: 'API Documentation v2.1',
      type: 'PDF',
      size: '2.4 MB',
      status: 'processed',
      category: 'Technical',
      description: 'Comprehensive API documentation covering all endpoints and authentication methods.',
      tags: ['api', 'documentation', 'v2.1'],
      chunks: 45,
      uploadedAt: '2024-01-15'
    },
    {
      id: 2,
      name: 'Security Guidelines',
      type: 'DOCX',
      size: '1.8 MB',
      status: 'processed',
      category: 'Security',
      description: 'Security best practices and guidelines for development teams.',
      tags: ['security', 'guidelines', 'best-practices'],
      chunks: 32,
      uploadedAt: '2024-01-14'
    },
    {
      id: 3,
      name: 'Code Review Checklist',
      type: 'MD',
      size: '124 KB',
      status: 'processed',
      category: 'Development',
      description: 'Comprehensive checklist for conducting effective code reviews.',
      tags: ['code-review', 'checklist', 'quality'],
      chunks: 15,
      uploadedAt: '2024-01-13'
    },
    {
      id: 4,
      name: 'Performance Metrics Report',
      type: 'XLSX',
      size: '3.2 MB',
      status: 'processing',
      category: 'Analytics',
      description: 'Detailed performance metrics and analysis for Q4 2023.',
      tags: ['performance', 'metrics', 'report'],
      chunks: 0,
      uploadedAt: '2024-01-12'
    },
    {
      id: 5,
      name: 'User Manual Draft',
      type: 'PDF',
      size: '5.1 MB',
      status: 'failed',
      category: 'Documentation',
      description: 'Draft version of the user manual with updated screenshots.',
      tags: ['manual', 'user-guide', 'draft'],
      chunks: 0,
      uploadedAt: '2024-01-11'
    }
  ]

  const typeIcons = {
    PDF: FileText,
    DOCX: FileText,
    MD: FileText,
    XLSX: FileText,
    TXT: FileText
  }

  const statusStyles = {
    processed: 'bg-green-500/10 text-green-400 border-green-500/20',
    processing: 'bg-blue-500/10 text-blue-400 border-blue-500/20',
    failed: 'bg-red-500/10 text-red-400 border-red-500/20'
  }

  const filteredDocuments = documents.filter(doc =>
    doc.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
    doc.category.toLowerCase().includes(searchTerm.toLowerCase()) ||
    doc.tags.some(tag => tag.toLowerCase().includes(searchTerm.toLowerCase()))
  )

  // Modal handlers
  const handleViewDetails = (documentId: number) => {
    setSelectedDocumentId(documentId)
    setDetailsModalOpen(true)
  }

  const handleDownload = (documentId: number, filename: string) => {
    setDownloadInfo({ filename, documentId })
    setDownloadModalOpen(true)
  }

  const handleDelete = (documentId: number) => {
    const document = documents.find(d => d.id === documentId)
    if (!document) return
    
    setDeleteInfo({ documentId, filename: document.name })
    setDeleteModalOpen(true)
  }

  const handleConfirmDelete = async (documentId: number) => {
    // Mock delete - replace with actual API call
    console.log('Deleting document:', documentId)
    // Remove from local state for demo
    // In real app, this would call API and refresh data
  }

  return (
    <div className="space-y-6">
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

      {/* Header Section - Matching Agent Management exactly */}
      <div className="flex justify-between items-start">
        <div>
          <h1 className="text-3xl font-bold gradient-text">Document Management</h1>
          <p className="text-muted-foreground mt-1">
            Knowledge base and documents
          </p>
        </div>

        <div className="flex items-center gap-3">
          <Badge variant="outline" className="text-brand-primary border-brand-primary/30">
            <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse mr-2" />
            All Systems Operational
          </Badge>
          <Button variant="outline" size="sm">
            <RefreshCw className="w-4 h-4 mr-2" />
            Refresh
          </Button>
          <Button size="sm">
            <Upload className="w-4 h-4 mr-2" />
            Upload Document
          </Button>
        </div>
      </div>

      {/* Metrics Cards - Matching Agent Management exactly */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        {documentStats.map((stat, index) => (
          <Card key={index} className="relative overflow-hidden">
            <CardContent className="p-6">
              <div className="flex items-center justify-between">
                <div className="space-y-2">
                  <p className="text-sm font-medium text-muted-foreground">
                    {stat.label}
                  </p>
                  <div className="space-y-1">
                    <p className="text-2xl font-bold tracking-tight">
                      {stat.value}
                    </p>
                    <p className={`text-xs ${stat.color}`}>
                      {stat.change}
                    </p>
                  </div>
                </div>
                <div className="p-3 rounded-xl bg-gradient-to-br from-orange-500 to-red-500">
                  <stat.icon className="w-5 h-5 text-white" />
                </div>
              </div>
            </CardContent>
          </Card>
        ))}
      </div>

      {/* Search Bar - Matching Agent Management */}
      <div className="flex items-center gap-4">
        <div className="relative flex-1 max-w-md">
          <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-muted-foreground w-4 h-4" />
          <Input
            placeholder="Search documents by name, content, or tags..."
            className="pl-10"
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
          />
        </div>
        <div className="flex items-center gap-2">
          <Button variant="outline" size="sm">
            <Filter className="w-4 h-4 mr-2" />
            Filter
          </Button>
        </div>
      </div>

      <Tabs value={activeTab} onValueChange={setActiveTab} className="space-y-4">
        <TabsList className="grid w-full grid-cols-6">
          <TabsTrigger value="library" className="flex items-center gap-2">
            <FileText className="w-4 h-4" />
            <span className="hidden sm:inline">Library</span>
          </TabsTrigger>
          <TabsTrigger value="upload" className="flex items-center gap-2">
            <Upload className="w-4 h-4" />
            <span className="hidden sm:inline">Upload</span>
          </TabsTrigger>
          <TabsTrigger value="processing" className="flex items-center gap-2">
            <Settings className="w-4 h-4" />
            <span className="hidden sm:inline">Processing</span>
          </TabsTrigger>
          <TabsTrigger value="search" className="flex items-center gap-2">
            <Search className="w-4 h-4" />
            <span className="hidden sm:inline">Search</span>
          </TabsTrigger>
          <TabsTrigger value="analytics" className="flex items-center gap-2">
            <BarChart className="w-4 h-4" />
            <span className="hidden sm:inline">Analytics</span>
          </TabsTrigger>
          <TabsTrigger value="preview" className="flex items-center gap-2">
            <Eye className="w-4 h-4" />
            <span className="hidden sm:inline">Preview</span>
          </TabsTrigger>
        </TabsList>

        {/* Document Library Tab */}
        <TabsContent value="library" className="space-y-6">
          <div className="flex justify-between items-center">
            <div>
              <h3 className="text-lg font-semibold">Document Library</h3>
              <p className="text-sm text-muted-foreground">Manage your document collection</p>
            </div>
            <div className="flex gap-2">
              <Button variant="outline" size="sm">
                <Filter className="w-4 h-4 mr-2" />
                Filter
              </Button>
              <Button size="sm">
                <Plus className="w-4 h-4 mr-2" />
                Add Document
              </Button>
            </div>
          </div>

          {/* Document Cards - Marketing Site Structure */}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {filteredDocuments.map((doc, index) => {
              const TypeIcon = typeIcons[doc.type] || File

              return (
                <Card
                  key={doc.id}
                  className="glass-card p-6 card-glow hover:border-primary/20 transition-all duration-300"
                >
                  {/* Header */}
                  <div className="flex items-start justify-between mb-4">
                    <div className="flex items-center space-x-3">
                      <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-orange-500 to-red-500 flex items-center justify-center">
                        <TypeIcon className="w-5 h-5 text-white" />
                      </div>
                      <div className="flex-1 min-w-0">
                        <h3 className="font-semibold truncate">{doc.name}</h3>
                        <p className="text-xs text-muted-foreground">
                          {doc.type.toUpperCase()} • {doc.size}
                        </p>
                      </div>
                    </div>

                    <DropdownMenu>
                      <DropdownMenuTrigger asChild>
                        <Button variant="ghost" size="sm" className="h-8 w-8 p-0">
                          <MoreVertical className="w-4 h-4" />
                        </Button>
                      </DropdownMenuTrigger>
                      <DropdownMenuContent align="end" className="w-48">
                        <DropdownMenuItem onClick={() => handleViewDetails(doc.id)}>
                          <Eye className="w-4 h-4 mr-2" />
                          View Details
                        </DropdownMenuItem>
                        <DropdownMenuItem onClick={() => handleDownload(doc.id, doc.name)}>
                          <Download className="w-4 h-4 mr-2" />
                          Download
                        </DropdownMenuItem>
                        <DropdownMenuItem>
                          <Edit className="w-4 h-4 mr-2" />
                          Edit Document
                        </DropdownMenuItem>
                        <DropdownMenuSeparator />
                        <DropdownMenuItem 
                          className="text-red-600"
                          onClick={() => handleDelete(doc.id)}
                        >
                          <Trash2 className="w-4 h-4 mr-2" />
                          Delete Document
                        </DropdownMenuItem>
                      </DropdownMenuContent>
                    </DropdownMenu>
                  </div>

                  {/* Status and Category */}
                  <div className="flex items-center justify-between mb-4">
                    <Badge className={statusStyles[doc.status]}>
                      {doc.status}
                    </Badge>
                    <Badge variant="outline" className="text-xs">
                      {doc.category}
                    </Badge>
                  </div>

                  {/* Description */}
                  <p className="text-sm text-muted-foreground mb-4 line-clamp-2">
                    {doc.description}
                  </p>

                  {/* Processing Info */}
                  <div className="grid grid-cols-2 gap-4 mb-4">
                    <div>
                      <p className="text-sm font-medium">Vector Chunks</p>
                      <p className="text-xs text-muted-foreground">{doc.chunks}</p>
                    </div>
                    <div>
                      <p className="text-sm font-medium">Uploaded</p>
                      <p className="text-xs text-muted-foreground">{doc.uploadedAt}</p>
                    </div>
                  </div>

                  {/* Tags */}
                  <div className="flex flex-wrap gap-1">
                    {doc.tags.slice(0, 3).map(tag => (
                      <Badge key={tag} variant="secondary" className="text-xs">
                        {tag}
                      </Badge>
                    ))}
                    {doc.tags.length > 3 && (
                      <Badge variant="secondary" className="text-xs">
                        +{doc.tags.length - 3} more
                      </Badge>
                    )}
                  </div>
                </Card>
              )
            })}
          </div>
        </TabsContent>

        {/* Other tabs */}
        <TabsContent value="upload" className="space-y-6">
          <Card className="glass-card">
            <CardContent className="p-12 text-center">
              <Upload className="w-16 h-16 mx-auto text-muted-foreground mb-4" />
              <h3 className="text-lg font-semibold mb-2">Upload Documents</h3>
              <p className="text-muted-foreground mb-4">
                Drag and drop files here or click to browse
              </p>
              <Button>
                <Upload className="w-4 h-4 mr-2" />
                Choose Files
              </Button>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="processing" className="space-y-6">
          <Card className="glass-card">
            <CardContent className="p-12 text-center">
              <Settings className="w-16 h-16 mx-auto text-muted-foreground mb-4" />
              <h3 className="text-lg font-semibold mb-2">Processing Pipeline</h3>
              <p className="text-muted-foreground">
                Monitor document processing status and configure processing settings
              </p>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="search" className="space-y-6">
          <Card className="glass-card">
            <CardContent className="p-12 text-center">
              <Search className="w-16 h-16 mx-auto text-muted-foreground mb-4" />
              <h3 className="text-lg font-semibold mb-2">Semantic Search</h3>
              <p className="text-muted-foreground">
                Search through document content using AI-powered semantic search
              </p>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="analytics" className="space-y-6">
          <Card className="glass-card">
            <CardContent className="p-12 text-center">
              <BarChart className="w-16 h-16 mx-auto text-muted-foreground mb-4" />
              <h3 className="text-lg font-semibold mb-2">Document Analytics</h3>
              <p className="text-muted-foreground">
                View insights and analytics about your document collection
              </p>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="preview" className="space-y-6">
          <Card className="glass-card">
            <CardContent className="p-12 text-center">
              <Eye className="w-16 h-16 mx-auto text-muted-foreground mb-4" />
              <h3 className="text-lg font-semibold mb-2">Document Preview</h3>
              <p className="text-muted-foreground">
                Preview and view document content
              </p>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  )
}
