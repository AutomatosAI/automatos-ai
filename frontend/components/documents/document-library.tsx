
'use client'

import { useState, useMemo } from 'react'
import { motion } from 'framer-motion'
import { formatDistanceToNow } from 'date-fns'
import { 
  FileText, 
  FileImage, 
  FileVideo, 
  FileAudio, 
  Archive,
  MoreVertical, 
  Download, 
  Edit, 
  Trash2, 
  Eye,
  Share,
  Copy,
  Tag,
  Folder,
  Search,
  Filter,
  Grid3X3,
  List,
  Calendar,
  User,
  HardDrive,
  Zap,
} from 'lucide-react'
import { Card, CardContent, CardHeader } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { 
  DropdownMenu, 
  DropdownMenuContent, 
  DropdownMenuItem, 
  DropdownMenuTrigger,
  DropdownMenuSeparator
} from '@/components/ui/dropdown-menu'
import { Avatar, AvatarFallback } from '@/components/ui/avatar'
import { Progress } from '@/components/ui/progress'
import { Skeleton } from '@/components/ui/skeleton'
import { Checkbox } from '@/components/ui/checkbox'
import { toast } from 'sonner'

// API hooks
import { useDeleteDocument, useTagDocument, useStartProcessing } from '@/hooks/use-document-api'
import { ViewToggle } from '@/components/shared/view-toggle'
import { useViewMode } from '@/hooks/use-view-mode'

// Document type configurations
const documentTypes = {
  pdf: {
    icon: FileText,
    color: 'text-destructive',
    bgColor: 'bg-red-50',
    name: 'PDF Document'
  },
  image: {
    icon: FileImage,
    color: 'text-success',
    bgColor: 'bg-green-50',
    name: 'Image'
  },
  video: {
    icon: FileVideo,
    color: 'text-agent',
    bgColor: 'bg-purple-50',
    name: 'Video'
  },
  audio: {
    icon: FileAudio,
    color: 'text-info',
    bgColor: 'bg-blue-50',
    name: 'Audio'
  },
  archive: {
    icon: Archive,
    color: 'text-orange-500',
    bgColor: 'bg-orange-50',
    name: 'Archive'
  },
  text: {
    icon: FileText,
    color: 'text-muted-foreground',
    bgColor: 'bg-gray-50',
    name: 'Text Document'
  }
}

const statusStyles = {
  completed: 'bg-success/10 text-success border-success/20',
  processing: 'bg-info/10 text-info border-info/20',
  failed: 'bg-destructive/10 text-destructive border-destructive/20',
  pending: 'bg-warning/10 text-warning border-warning/20'
}

interface DocumentLibraryProps {
  documents: any[]
  loading: boolean
  searchTerm: string
  categoryFilter: string
  typeFilter: string
  statusFilter: string
  selectedDocumentId: string | null
  onDocumentSelect: (documentId: string | null) => void
  onRefresh: () => void
}

export function DocumentLibrary({
  documents,
  loading,
  searchTerm,
  categoryFilter,
  typeFilter,
  statusFilter,
  selectedDocumentId,
  onDocumentSelect,
  onRefresh
}: DocumentLibraryProps) {
  const [viewMode, setViewMode] = useViewMode('documents')
  const [sortBy, setSortBy] = useState('created_at')
  const [sortOrder, setSortOrder] = useState<'asc' | 'desc'>('desc')
  const [selectedDocuments, setSelectedDocuments] = useState<string[]>([])

  // API mutations
  const deleteDocumentMutation = useDeleteDocument()
  const tagDocumentMutation = useTagDocument()
  const startProcessingMutation = useStartProcessing()

  // Filter and sort documents
  const filteredAndSortedDocuments = useMemo(() => {
    let filtered = documents.filter(doc => {
      const matchesSearch = !searchTerm || 
        doc.name?.toLowerCase().includes(searchTerm.toLowerCase()) ||
        doc.content?.toLowerCase().includes(searchTerm.toLowerCase()) ||
        doc.tags?.some((tag: string) => tag.toLowerCase().includes(searchTerm.toLowerCase()))
      
      const matchesCategory = categoryFilter === 'all' || doc.category === categoryFilter
      const matchesType = typeFilter === 'all' || doc.file_type === typeFilter
      const matchesStatus = statusFilter === 'all' || doc.status === statusFilter
      
      return matchesSearch && matchesCategory && matchesType && matchesStatus
    })

    // Sort documents
    filtered.sort((a, b) => {
      const aValue = a[sortBy]
      const bValue = b[sortBy]
      
      if (sortOrder === 'desc') {
        return aValue > bValue ? -1 : 1
      } else {
        return aValue > bValue ? 1 : -1
      }
    })

    return filtered
  }, [documents, searchTerm, categoryFilter, typeFilter, statusFilter, sortBy, sortOrder])

  // Handle document actions
  const handleDelete = async (documentId: string, documentName: string) => {
    if (!confirm(`Are you sure you want to delete "${documentName}"? This action cannot be undone.`)) {
      return
    }

    try {
      await deleteDocumentMutation.mutateAsync(documentId)
      onRefresh()
    } catch (error) {
      // Error handled by mutation hook
    }
  }

  const handleProcess = async (documentId: string) => {
    try {
      await startProcessingMutation.mutateAsync({ documentId })
    } catch (error) {
      // Error handled by mutation hook
    }
  }

  const handleSelectDocument = (documentId: string) => {
    setSelectedDocuments(prev => 
      prev.includes(documentId) 
        ? prev.filter(id => id !== documentId)
        : [...prev, documentId]
    )
  }

  const handleSelectAll = () => {
    if (selectedDocuments.length === filteredAndSortedDocuments.length) {
      setSelectedDocuments([])
    } else {
      setSelectedDocuments(filteredAndSortedDocuments.map(doc => doc.id))
    }
  }

  // Format file size
  const formatFileSize = (bytes: number) => {
    if (!bytes) return '0 B'
    const k = 1024
    const sizes = ['B', 'KB', 'MB', 'GB']
    const i = Math.floor(Math.log(bytes) / Math.log(k))
    return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i]
  }

  if (loading) {
    return (
      <div className="space-y-6">
        {/* Loading skeleton for controls */}
        <div className="flex justify-between items-center">
          <Skeleton className="h-10 w-48" />
          <div className="flex gap-2">
            <Skeleton className="h-10 w-24" />
            <Skeleton className="h-10 w-24" />
          </div>
        </div>

        {/* Loading skeleton for documents */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {Array.from({ length: 6 }).map((_, i) => (
            <Card key={i} className="glass-card">
              <CardHeader className="pb-2">
                <div className="flex items-center justify-between">
                  <Skeleton className="h-8 w-8 rounded" />
                  <Skeleton className="h-6 w-16" />
                </div>
              </CardHeader>
              <CardContent>
                <Skeleton className="h-5 w-3/4 mb-2" />
                <Skeleton className="h-4 w-1/2 mb-4" />
                <div className="flex justify-between">
                  <Skeleton className="h-4 w-16" />
                  <Skeleton className="h-4 w-20" />
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
      </div>
    )
  }

  return (
    <div className="space-y-6">
      {/* Controls */}
      <div className="flex flex-col sm:flex-row gap-4 justify-between">
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-2">
            <Checkbox
              checked={selectedDocuments.length === filteredAndSortedDocuments.length && filteredAndSortedDocuments.length > 0}
              onCheckedChange={handleSelectAll}
            />
            <span className="text-sm text-muted-foreground">
              {selectedDocuments.length > 0 
                ? `${selectedDocuments.length} selected` 
                : `${filteredAndSortedDocuments.length} documents`
              }
            </span>
          </div>

          {selectedDocuments.length > 0 && (
            <div className="flex items-center gap-2">
              <Button size="sm" variant="outline">
                <Tag className="w-4 h-4 mr-1" />
                Tag
              </Button>
              <Button size="sm" variant="outline">
                <Folder className="w-4 h-4 mr-1" />
                Move
              </Button>
              <Button size="sm" variant="destructive">
                <Trash2 className="w-4 h-4 mr-1" />
                Delete
              </Button>
            </div>
          )}
        </div>

        <div className="flex items-center gap-2">
          {/* Sort dropdown */}
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button variant="outline" size="sm">
                <Filter className="w-4 h-4 mr-2" />
                Sort
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent>
              <DropdownMenuItem onClick={() => setSortBy('name')}>
                Name
              </DropdownMenuItem>
              <DropdownMenuItem onClick={() => setSortBy('created_at')}>
                Date Created
              </DropdownMenuItem>
              <DropdownMenuItem onClick={() => setSortBy('size')}>
                Size
              </DropdownMenuItem>
              <DropdownMenuItem onClick={() => setSortBy('status')}>
                Status
              </DropdownMenuItem>
              <DropdownMenuSeparator />
              <DropdownMenuItem onClick={() => setSortOrder(sortOrder === 'asc' ? 'desc' : 'asc')}>
                {sortOrder === 'asc' ? 'Descending' : 'Ascending'}
              </DropdownMenuItem>
            </DropdownMenuContent>
          </DropdownMenu>

          {/* View mode toggle */}
          <ViewToggle value={viewMode} onChange={setViewMode} />
        </div>
      </div>

      {/* Documents */}
      {viewMode === 'grid' ? (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {filteredAndSortedDocuments.map((document, index) => {
            const docType = documentTypes[document.file_type as keyof typeof documentTypes] || documentTypes.text
            const IconComponent = docType.icon
            
            return (
              <motion.div
                key={document.id}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.3, delay: index * 0.05 }}
              >
                <Card
                  data-testid="doc-card"
                  className={`glass-card card-glow cursor-pointer transition-all duration-300 hover:border-primary/20 ${
                    selectedDocumentId === document.id ? 'border-primary' : ''
                  }`}
                >
                  <CardHeader className="pb-2">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-3">
                        <Checkbox
                          checked={selectedDocuments.includes(document.id)}
                          onCheckedChange={() => handleSelectDocument(document.id)}
                          onClick={(e) => e.stopPropagation()}
                        />
                        <div className={`p-3 rounded-lg ${docType.bgColor}`}>
                          <IconComponent className={`w-6 h-6 ${docType.color}`} />
                        </div>
                      </div>

                      <div className="flex items-center gap-2">
                        <Badge 
                          variant="outline" 
                          className={`text-xs ${statusStyles[document.status as keyof typeof statusStyles] || statusStyles.pending}`}
                        >
                          {document.status}
                        </Badge>

                        <DropdownMenu>
                          <DropdownMenuTrigger asChild onClick={(e) => e.stopPropagation()}>
                            <Button variant="ghost" size="sm" className="h-8 w-8 p-0">
                              <MoreVertical className="w-4 h-4" />
                            </Button>
                          </DropdownMenuTrigger>
                          <DropdownMenuContent align="end" className="w-48">
                            <DropdownMenuItem onClick={() => onDocumentSelect(document.id)}>
                              <Eye className="w-4 h-4 mr-2" />
                              View Details
                            </DropdownMenuItem>
                            
                            <DropdownMenuItem>
                              <Download className="w-4 h-4 mr-2" />
                              Download
                            </DropdownMenuItem>
                            
                            <DropdownMenuItem>
                              <Edit className="w-4 h-4 mr-2" />
                              Edit
                            </DropdownMenuItem>
                            
                            <DropdownMenuItem>
                              <Share className="w-4 h-4 mr-2" />
                              Share
                            </DropdownMenuItem>
                            
                            <DropdownMenuItem>
                              <Copy className="w-4 h-4 mr-2" />
                              Duplicate
                            </DropdownMenuItem>
                            
                            <DropdownMenuSeparator />
                            
                            {document.status !== 'processing' && (
                              <DropdownMenuItem onClick={() => handleProcess(document.id)}>
                                <Zap className="w-4 h-4 mr-2" />
                                Process
                              </DropdownMenuItem>
                            )}
                            
                            <DropdownMenuSeparator />
                            
                            <DropdownMenuItem 
                              className="text-destructive"
                              onClick={() => handleDelete(document.id, document.name)}
                            >
                              <Trash2 className="w-4 h-4 mr-2" />
                              Delete
                            </DropdownMenuItem>
                          </DropdownMenuContent>
                        </DropdownMenu>
                      </div>
                    </div>
                  </CardHeader>

                  <CardContent onClick={() => onDocumentSelect(document.id)} className="pt-0">
                    <div className="space-y-3">
                      <div>
                        <h3 className="font-semibold text-lg leading-tight line-clamp-2">
                          {document.name}
                        </h3>
                        <p className="text-sm text-muted-foreground capitalize">
                          {docType.name}
                        </p>
                      </div>

                      {document.description && (
                        <p className="text-sm text-muted-foreground line-clamp-2">
                          {document.description}
                        </p>
                      )}

                      {/* Tags */}
                      {document.tags && document.tags.length > 0 && (
                        <div className="flex flex-wrap gap-1">
                          {document.tags.slice(0, 3).map((tag: string) => (
                            <Badge key={tag} variant="secondary" className="text-xs">
                              {tag}
                            </Badge>
                          ))}
                          {document.tags.length > 3 && (
                            <Badge variant="secondary" className="text-xs">
                              +{document.tags.length - 3}
                            </Badge>
                          )}
                        </div>
                      )}

                      {/* Processing progress */}
                      {document.status === 'processing' && document.processing_progress && (
                        <div className="space-y-1">
                          <div className="flex justify-between text-xs">
                            <span className="text-muted-foreground">Processing...</span>
                            <span className="font-medium">{document.processing_progress}%</span>
                          </div>
                          <Progress value={document.processing_progress} className="h-2" />
                        </div>
                      )}

                      {/* Document info */}
                      <div className="flex items-center justify-between pt-2 text-xs text-muted-foreground border-t border-border/50">
                        <div className="flex items-center gap-4">
                          <div className="flex items-center gap-1">
                            <HardDrive className="w-3 h-3" />
                            {formatFileSize(document.size)}
                          </div>
                          <div className="flex items-center gap-1">
                            <Calendar className="w-3 h-3" />
                            {document.created_at ? formatDistanceToNow(new Date(document.created_at), { addSuffix: true }) : 'Unknown'}
                          </div>
                        </div>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              </motion.div>
            )
          })}
        </div>
      ) : (
        /* List View */
        <div className="space-y-2">
          {filteredAndSortedDocuments.map((document, index) => {
            const docType = documentTypes[document.file_type as keyof typeof documentTypes] || documentTypes.text
            const IconComponent = docType.icon
            
            return (
              <motion.div
                key={document.id}
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ duration: 0.3, delay: index * 0.02 }}
              >
                <Card
                  data-testid="doc-card"
                  className={`glass-card card-glow cursor-pointer transition-all duration-300 hover:border-primary/20 ${
                    selectedDocumentId === document.id ? 'border-primary' : ''
                  }`}
                  onClick={() => onDocumentSelect(document.id)}
                >
                  <CardContent className="p-4">
                    <div className="flex items-center gap-4">
                      <Checkbox
                        checked={selectedDocuments.includes(document.id)}
                        onCheckedChange={() => handleSelectDocument(document.id)}
                        onClick={(e) => e.stopPropagation()}
                      />
                      
                      <div className={`p-2 rounded ${docType.bgColor}`}>
                        <IconComponent className={`w-5 h-5 ${docType.color}`} />
                      </div>

                      <div className="flex-1 min-w-0">
                        <div className="flex items-center gap-2 mb-1">
                          <h3 className="font-semibold truncate">{document.name}</h3>
                          <Badge 
                            variant="outline" 
                            className={`text-xs ${statusStyles[document.status as keyof typeof statusStyles] || statusStyles.pending}`}
                          >
                            {document.status}
                          </Badge>
                        </div>
                        <div className="flex items-center gap-4 text-sm text-muted-foreground">
                          <span>{docType.name}</span>
                          <span>{formatFileSize(document.size)}</span>
                          <span>{document.created_at ? formatDistanceToNow(new Date(document.created_at), { addSuffix: true }) : 'Unknown'}</span>
                        </div>
                      </div>

                      <DropdownMenu>
                        <DropdownMenuTrigger asChild onClick={(e) => e.stopPropagation()}>
                          <Button variant="ghost" size="sm" className="h-8 w-8 p-0">
                            <MoreVertical className="w-4 h-4" />
                          </Button>
                        </DropdownMenuTrigger>
                        <DropdownMenuContent align="end" className="w-48">
                          <DropdownMenuItem onClick={() => onDocumentSelect(document.id)}>
                            <Eye className="w-4 h-4 mr-2" />
                            View Details
                          </DropdownMenuItem>
                          <DropdownMenuItem>
                            <Download className="w-4 h-4 mr-2" />
                            Download
                          </DropdownMenuItem>
                          <DropdownMenuItem>
                            <Share className="w-4 h-4 mr-2" />
                            Share
                          </DropdownMenuItem>
                          <DropdownMenuSeparator />
                          <DropdownMenuItem 
                            className="text-destructive"
                            onClick={() => handleDelete(document.id, document.name)}
                          >
                            <Trash2 className="w-4 h-4 mr-2" />
                            Delete
                          </DropdownMenuItem>
                        </DropdownMenuContent>
                      </DropdownMenu>
                    </div>
                  </CardContent>
                </Card>
              </motion.div>
            )
          })}
        </div>
      )}

      {/* No documents */}
      {filteredAndSortedDocuments.length === 0 && (
        <Card className="glass-card">
          <CardContent className="p-12 text-center">
            <FileText className="w-16 h-16 mx-auto text-muted-foreground mb-4" />
            <h3 className="text-lg font-semibold mb-2">No documents found</h3>
            <p className="text-muted-foreground mb-4">
              {searchTerm || categoryFilter !== 'all' || statusFilter !== 'all'
                ? 'Try adjusting your search or filters' 
                : 'Upload your first document to get started'
              }
            </p>
            <div className="flex items-center justify-center gap-3">
              <Button onClick={() => onRefresh()}>
                <RefreshCw className="w-4 h-4 mr-2" />
                Refresh
              </Button>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  )
}

