'use client'

import * as React from 'react'
import { useState, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { 
  X, 
  FileText, 
  Calendar, 
  Database, 
  HardDrive, 
  Clock, 
  CheckCircle, 
  AlertTriangle, 
  Info, 
  Download,
  Eye,
  Trash2,
  RefreshCw
} from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Separator } from '@/components/ui/separator'
import { Progress } from '@/components/ui/progress'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'

interface DocumentDetailsModalProps {
  documentId: number | null
  open: boolean
  onClose: () => void
  onDownload?: (documentId: number, filename: string) => void
  onDelete?: (documentId: number) => void
}

interface DocumentDetails {
  id: number
  filename: string
  original_filename?: string
  file_type?: string
  file_size?: number
  status?: string
  chunk_count?: number
  upload_date?: string
  processed_date?: string | null
  metadata?: {
    description?: string
    tags?: string[]
    source?: string
    language?: string
    page_count?: number
    word_count?: number
  }
  processing_info?: {
    embeddings_created: number
    total_tokens: number
    processing_time: string
    success_rate: number
  }
  access_info?: {
    view_count: number
    last_accessed?: string
    download_count: number
  }
}

const statusStyles: Record<string, string> = {
  completed: 'bg-green-500/10 text-green-400 border-green-500/20',
  processed: 'bg-green-500/10 text-green-400 border-green-500/20',
  processing: 'bg-yellow-500/10 text-yellow-400 border-yellow-500/20',
  failed: 'bg-red-500/10 text-red-400 border-red-500/20',
  pending: 'bg-gray-500/10 text-gray-400 border-gray-500/20'
}

export function DocumentDetailsModal({ 
  documentId, 
  open, 
  onClose, 
  onDownload, 
  onDelete 
}: DocumentDetailsModalProps) {
  const [document, setDocument] = useState<DocumentDetails | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [activeTab, setActiveTab] = useState('overview')

  useEffect(() => {
    if (open && documentId) {
      loadDocumentDetails()
    }
  }, [open, documentId])

  const loadDocumentDetails = async () => {
    if (!documentId) return
    
    setLoading(true)
    setError(null)
    
    try {
      // Mock data for now - replace with actual API call
      setDocument({
        id: documentId,
        filename: 'API Documentation v2.1.pdf',
        file_type: 'pdf',
        file_size: 2400000,
        status: 'completed',
        chunk_count: 45,
        upload_date: '2024-01-15T10:30:00Z',
        processed_date: '2024-01-15T10:32:00Z',
        metadata: {
          description: 'Comprehensive API documentation covering all endpoints',
          tags: ['api', 'documentation', 'v2.1'],
          language: 'English',
          page_count: 24,
          word_count: 12500
        },
        processing_info: {
          embeddings_created: 45,
          total_tokens: 15000,
          processing_time: '2.4s',
          success_rate: 100
        },
        access_info: {
          view_count: 12,
          download_count: 3,
          last_accessed: '2024-01-16T09:15:00Z'
        }
      })
    } catch (err) {
      console.error('Error loading document details:', err)
      setError('Failed to load document details')
    } finally {
      setLoading(false)
    }
  }

  const formatFileSize = (bytes?: number) => {
    if (!bytes) return '0 B'
    const sizes = ['B', 'KB', 'MB', 'GB']
    const i = Math.floor(Math.log(bytes) / Math.log(1024))
    return `${(bytes / Math.pow(1024, i)).toFixed(1)} ${sizes[i]}`
  }

  const formatDate = (dateString?: string) => {
    if (!dateString) return 'N/A'
    return new Date(dateString).toLocaleString()
  }

  const handleDownload = () => {
    if (document && onDownload) {
      onDownload(document.id, document.filename)
    }
  }

  const handleDelete = () => {
    if (document && onDelete) {
      onDelete(document.id)
      onClose()
    }
  }

  if (!open || !documentId) return null

  return (
    <AnimatePresence>
      <motion.div 
        className="fixed inset-0 bg-black/50 backdrop-blur-sm z-50" 
        initial={{ opacity: 0 }} 
        animate={{ opacity: 1 }} 
        exit={{ opacity: 0 }} 
        onClick={onClose}
      />
      <motion.div 
        className="fixed inset-0 z-50 flex items-center justify-center p-4" 
        initial={{ opacity: 0, scale: 0.95 }} 
        animate={{ opacity: 1, scale: 1 }} 
        exit={{ opacity: 0, scale: 0.95 }}
      >
        <Card className="glass-card w-full max-w-4xl max-h-[90vh] overflow-hidden">
          <CardHeader className="flex flex-row items-center justify-between border-b border-border/30">
            <CardTitle className="flex items-center space-x-2">
              <FileText className="w-6 h-6 text-orange-400" />
              <span>Document Details</span>
            </CardTitle>
            <div className="flex items-center space-x-2">
              {document && (
                <>
                  <Button 
                    variant="outline" 
                    size="sm"
                    onClick={handleDownload}
                    className="hover:border-blue-500/50"
                  >
                    <Download className="w-4 h-4 mr-2" />
                    Download
                  </Button>
                  <Button 
                    variant="outline" 
                    size="sm"
                    onClick={handleDelete}
                    className="hover:border-red-500/50 text-red-400"
                  >
                    <Trash2 className="w-4 h-4 mr-2" />
                    Delete
                  </Button>
                </>
              )}
              <Button variant="ghost" size="icon" onClick={onClose}>
                <X className="w-5 h-5" />
              </Button>
            </div>
          </CardHeader>
          
          <CardContent className="overflow-y-auto p-6">
            {loading && (
              <div className="flex items-center justify-center py-12">
                <div className="text-center">
                  <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary mx-auto mb-4"></div>
                  <p className="text-muted-foreground">Loading document details...</p>
                </div>
              </div>
            )}

            {error && (
              <div className="flex items-center justify-center py-12">
                <div className="text-center">
                  <AlertTriangle className="h-8 w-8 text-red-400 mx-auto mb-4" />
                  <p className="text-red-400 mb-4">Error: {error}</p>
                  <Button onClick={loadDocumentDetails} variant="outline">
                    Try Again
                  </Button>
                </div>
              </div>
            )}

            {document && (
              <Tabs value={activeTab} onValueChange={setActiveTab}>
                <TabsList className="grid w-full grid-cols-3 bg-secondary/50">
                  <TabsTrigger value="overview" className="flex items-center space-x-2">
                    <Info className="w-4 h-4" />
                    <span>Overview</span>
                  </TabsTrigger>
                  <TabsTrigger value="processing" className="flex items-center space-x-2">
                    <Database className="w-4 h-4" />
                    <span>Processing</span>
                  </TabsTrigger>
                  <TabsTrigger value="activity" className="flex items-center space-x-2">
                    <Eye className="w-4 h-4" />
                    <span>Activity</span>
                  </TabsTrigger>
                </TabsList>

                <TabsContent value="overview" className="space-y-6 mt-6">
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <Card className="bg-secondary/30 border-border/30">
                      <CardHeader>
                        <CardTitle className="text-base">File Information</CardTitle>
                      </CardHeader>
                      <CardContent className="space-y-4">
                        <div>
                          <p className="text-sm font-medium text-muted-foreground">Filename</p>
                          <p className="font-semibold">{document?.filename}</p>
                        </div>
                        <div>
                          <p className="text-sm font-medium text-muted-foreground">Type</p>
                          <Badge variant="outline">{document?.file_type?.toUpperCase() || 'Unknown'}</Badge>
                        </div>
                        <div>
                          <p className="text-sm font-medium text-muted-foreground">Size</p>
                          <p className="font-semibold">{formatFileSize(document?.file_size)}</p>
                        </div>
                        <div>
                          <p className="text-sm font-medium text-muted-foreground">Status</p>
                          <Badge className={statusStyles[(document?.status || 'completed').toLowerCase()]}>
                            {document?.status || 'completed'}
                          </Badge>
                        </div>
                      </CardContent>
                    </Card>

                    <Card className="bg-secondary/30 border-border/30">
                      <CardHeader>
                        <CardTitle className="text-base">Processing Info</CardTitle>
                      </CardHeader>
                      <CardContent className="space-y-4">
                        <div>
                          <p className="text-sm font-medium text-muted-foreground">Upload Date</p>
                          <p className="font-semibold">{formatDate(document?.upload_date)}</p>
                        </div>
                        <div>
                          <p className="text-sm font-medium text-muted-foreground">Vector Chunks</p>
                          <p className="font-semibold">{document?.chunk_count ?? 0}</p>
                        </div>
                        <div>
                          <p className="text-sm font-medium text-muted-foreground">Document ID</p>
                          <p className="font-semibold text-orange-400">#{document?.id}</p>
                        </div>
                      </CardContent>
                    </Card>
                  </div>

                  {document?.metadata && (
                    <Card className="bg-secondary/30 border-border/30">
                      <CardHeader>
                        <CardTitle className="text-base">Metadata</CardTitle>
                      </CardHeader>
                      <CardContent>
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                          {document?.metadata?.description && (
                            <div>
                              <p className="text-sm font-medium text-muted-foreground">Description</p>
                              <p className="text-sm">{document?.metadata?.description}</p>
                            </div>
                          )}
                          {document?.metadata?.language && (
                            <div>
                              <p className="text-sm font-medium text-muted-foreground">Language</p>
                              <Badge variant="outline">{document?.metadata?.language}</Badge>
                            </div>
                          )}
                        </div>
                        {document?.metadata?.tags && document?.metadata?.tags?.length > 0 && (
                          <div className="mt-4">
                            <p className="text-sm font-medium text-muted-foreground mb-2">Tags</p>
                            <div className="flex flex-wrap gap-2">
                              {document?.metadata?.tags?.map((tag, index) => (
                                <Badge key={index} variant="secondary">{tag}</Badge>
                              ))}
                            </div>
                          </div>
                        )}
                      </CardContent>
                    </Card>
                  )}
                </TabsContent>

                <TabsContent value="processing" className="space-y-6 mt-6">
                  {document?.processing_info && (
                    <Card className="bg-secondary/30 border-border/30">
                      <CardHeader>
                        <CardTitle className="text-base">Processing Metrics</CardTitle>
                      </CardHeader>
                      <CardContent>
                        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                          <div className="text-center p-3 bg-background/50 rounded-lg">
                            <p className="text-2xl font-bold text-green-400">{document?.processing_info?.embeddings_created}</p>
                            <p className="text-sm text-muted-foreground">Embeddings</p>
                          </div>
                          <div className="text-center p-3 bg-background/50 rounded-lg">
                            <p className="text-2xl font-bold text-blue-400">{document?.processing_info?.total_tokens?.toLocaleString()}</p>
                            <p className="text-sm text-muted-foreground">Tokens</p>
                          </div>
                          <div className="text-center p-3 bg-background/50 rounded-lg">
                            <p className="text-2xl font-bold text-purple-400">{document?.processing_info?.processing_time}</p>
                            <p className="text-sm text-muted-foreground">Process Time</p>
                          </div>
                          <div className="text-center p-3 bg-background/50 rounded-lg">
                            <p className="text-2xl font-bold text-orange-400">{document?.processing_info?.success_rate}%</p>
                            <p className="text-sm text-muted-foreground">Success Rate</p>
                          </div>
                        </div>
                      </CardContent>
                    </Card>
                  )}
                </TabsContent>

                <TabsContent value="activity" className="space-y-6 mt-6">
                  {document?.access_info && (
                    <Card className="bg-secondary/30 border-border/30">
                      <CardHeader>
                        <CardTitle className="text-base">Access Statistics</CardTitle>
                      </CardHeader>
                      <CardContent>
                        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                          <div className="text-center p-4 bg-background/50 rounded-lg">
                            <p className="text-2xl font-bold text-blue-400">{document?.access_info?.view_count || 0}</p>
                            <p className="text-sm text-muted-foreground">Total Views</p>
                          </div>
                          <div className="text-center p-4 bg-background/50 rounded-lg">
                            <p className="text-2xl font-bold text-green-400">{document?.access_info?.download_count || 0}</p>
                            <p className="text-sm text-muted-foreground">Downloads</p>
                          </div>
                          <div className="text-center p-4 bg-background/50 rounded-lg">
                            <p className="text-sm font-bold text-orange-400">{formatDate(document?.access_info?.last_accessed)}</p>
                            <p className="text-sm text-muted-foreground">Last Accessed</p>
                          </div>
                        </div>
                      </CardContent>
                    </Card>
                  )}
                </TabsContent>
              </Tabs>
            )}
          </CardContent>
        </Card>
      </motion.div>
    </AnimatePresence>
  )
}
