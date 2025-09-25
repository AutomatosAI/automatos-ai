
'use client'

import { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { 
  X, 
  Download, 
  Share, 
  Edit, 
  FileText, 
  Eye, 
  Brain,
  Tag,
  Clock,
  User,
  HardDrive,
  Zap,
  Star
} from 'lucide-react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Separator } from '@/components/ui/separator'
import { Progress } from '@/components/ui/progress'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Skeleton } from '@/components/ui/skeleton'

// API hooks
import { useDocument, useDocumentInsights } from '@/hooks/use-document-api'

interface DocumentPreviewProps {
  documentId: string
  onClose: () => void
}

export function DocumentPreview({ documentId, onClose }: DocumentPreviewProps) {
  const [activeTab, setActiveTab] = useState('preview')

  // Fetch document data
  const { data: document, isLoading: documentLoading } = useDocument(documentId)
  const { data: insights, isLoading: insightsLoading } = useDocumentInsights(documentId)

  const formatFileSize = (bytes: number) => {
    if (!bytes) return '0 B'
    const k = 1024
    const sizes = ['B', 'KB', 'MB', 'GB']
    const i = Math.floor(Math.log(bytes) / Math.log(k))
    return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i]
  }

  if (documentLoading) {
    return (
      <AnimatePresence>
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          className="fixed inset-0 bg-black/50 z-50 flex items-center justify-center p-4"
          onClick={onClose}
        >
          <motion.div
            initial={{ scale: 0.95, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            exit={{ scale: 0.95, opacity: 0 }}
            className="bg-background rounded-lg shadow-xl w-full max-w-6xl max-h-[90vh] overflow-hidden"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="p-6">
              <div className="flex justify-between items-center mb-6">
                <Skeleton className="h-8 w-48" />
                <Skeleton className="h-10 w-10 rounded-full" />
              </div>
              <div className="space-y-4">
                <Skeleton className="h-64 w-full" />
                <Skeleton className="h-4 w-3/4" />
                <Skeleton className="h-4 w-1/2" />
              </div>
            </div>
          </motion.div>
        </motion.div>
      </AnimatePresence>
    )
  }

  if (!document) return null

  return (
    <AnimatePresence>
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
        className="fixed inset-0 bg-black/50 z-50 flex items-center justify-center p-4"
        onClick={onClose}
      >
        <motion.div
          initial={{ scale: 0.95, opacity: 0 }}
          animate={{ scale: 1, opacity: 1 }}
          exit={{ scale: 0.95, opacity: 0 }}
          className="bg-background rounded-lg shadow-xl w-full max-w-6xl max-h-[90vh] overflow-hidden"
          onClick={(e) => e.stopPropagation()}
        >
          {/* Header */}
          <div className="flex items-center justify-between p-6 border-b">
            <div className="flex items-center gap-4">
              <div className="p-3 rounded-xl bg-gradient-to-br from-blue-500 to-blue-600">
                <FileText className="w-6 h-6 text-white" />
              </div>
              <div>
                <h2 className="text-xl font-bold">{document.name}</h2>
                <p className="text-sm text-muted-foreground capitalize">
                  {document.file_type?.replace('_', ' ')} • {formatFileSize(document.size)}
                </p>
              </div>
            </div>
            
            <div className="flex items-center gap-2">
              <Badge variant={document.status === 'completed' ? 'default' : 'secondary'}>
                {document.status}
              </Badge>
              <Button variant="outline" size="sm">
                <Download className="w-4 h-4 mr-2" />
                Download
              </Button>
              <Button variant="outline" size="sm">
                <Share className="w-4 h-4 mr-2" />
                Share
              </Button>
              <Button variant="ghost" size="sm" onClick={onClose}>
                <X className="w-4 h-4" />
              </Button>
            </div>
          </div>

          {/* Content */}
          <div className="flex-1 overflow-y-auto max-h-[calc(90vh-120px)]">
            <Tabs value={activeTab} onValueChange={setActiveTab} className="h-full">
              <div className="px-6 py-4 border-b">
                <TabsList>
                  <TabsTrigger value="preview">Preview</TabsTrigger>
                  <TabsTrigger value="details">Details</TabsTrigger>
                  <TabsTrigger value="insights">AI Insights</TabsTrigger>
                  <TabsTrigger value="history">History</TabsTrigger>
                </TabsList>
              </div>

              <div className="p-6">
                <TabsContent value="preview" className="space-y-6 mt-0">
                  {/* Document Preview */}
                  <Card className="glass-card">
                    <CardHeader>
                      <CardTitle className="flex items-center gap-2">
                        <Eye className="w-5 h-5" />
                        Document Preview
                      </CardTitle>
                    </CardHeader>
                    <CardContent>
                      {document.status === 'completed' ? (
                        <div className="bg-muted/30 rounded-lg p-6 min-h-[400px]">
                          <div className="prose prose-sm max-w-none">
                            {document.extracted_text ? (
                              <div className="whitespace-pre-wrap">{document.extracted_text}</div>
                            ) : (
                              <div className="text-center text-muted-foreground py-12">
                                <FileText className="w-16 h-16 mx-auto mb-4" />
                                <h3 className="text-lg font-semibold mb-2">Preview not available</h3>
                                <p>This document type doesn't support text preview</p>
                              </div>
                            )}
                          </div>
                        </div>
                      ) : (
                        <div className="text-center py-12">
                          <Zap className="w-16 h-16 mx-auto text-muted-foreground mb-4" />
                          <h3 className="text-lg font-semibold mb-2">Processing Document</h3>
                          <p className="text-muted-foreground mb-4">
                            Document is being processed. Preview will be available once completed.
                          </p>
                          {document.processing_progress && (
                            <div className="max-w-sm mx-auto">
                              <Progress value={document.processing_progress} className="h-2" />
                              <p className="text-sm text-muted-foreground mt-2">
                                {document.processing_progress}% complete
                              </p>
                            </div>
                          )}
                        </div>
                      )}
                    </CardContent>
                  </Card>
                </TabsContent>

                <TabsContent value="details" className="space-y-6 mt-0">
                  <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                    {/* Basic Information */}
                    <Card className="glass-card">
                      <CardHeader>
                        <CardTitle>Basic Information</CardTitle>
                      </CardHeader>
                      <CardContent className="space-y-4">
                        <div className="flex justify-between">
                          <span className="text-muted-foreground">Name:</span>
                          <span className="font-medium">{document.name}</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-muted-foreground">Type:</span>
                          <span className="font-medium capitalize">{document.file_type?.replace('_', ' ')}</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-muted-foreground">Size:</span>
                          <span className="font-medium">{formatFileSize(document.size)}</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-muted-foreground">Status:</span>
                          <Badge variant={document.status === 'completed' ? 'default' : 'secondary'}>
                            {document.status}
                          </Badge>
                        </div>
                        {document.category && (
                          <div className="flex justify-between">
                            <span className="text-muted-foreground">Category:</span>
                            <span className="font-medium capitalize">{document.category}</span>
                          </div>
                        )}
                      </CardContent>
                    </Card>

                    {/* Metadata */}
                    <Card className="glass-card">
                      <CardHeader>
                        <CardTitle>Metadata</CardTitle>
                      </CardHeader>
                      <CardContent className="space-y-4">
                        <div className="flex justify-between">
                          <span className="text-muted-foreground">Created:</span>
                          <span className="font-medium">
                            {document.created_at ? new Date(document.created_at).toLocaleString() : 'Unknown'}
                          </span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-muted-foreground">Modified:</span>
                          <span className="font-medium">
                            {document.updated_at ? new Date(document.updated_at).toLocaleString() : 'Unknown'}
                          </span>
                        </div>
                        {document.uploaded_by && (
                          <div className="flex justify-between">
                            <span className="text-muted-foreground">Uploaded by:</span>
                            <span className="font-medium">{document.uploaded_by}</span>
                          </div>
                        )}
                        {document.processed_at && (
                          <div className="flex justify-between">
                            <span className="text-muted-foreground">Processed:</span>
                            <span className="font-medium">
                              {new Date(document.processed_at).toLocaleString()}
                            </span>
                          </div>
                        )}
                      </CardContent>
                    </Card>
                  </div>

                  {/* Tags */}
                  {document.tags && document.tags.length > 0 && (
                    <Card className="glass-card">
                      <CardHeader>
                        <CardTitle className="flex items-center gap-2">
                          <Tag className="w-5 h-5" />
                          Tags
                        </CardTitle>
                      </CardHeader>
                      <CardContent>
                        <div className="flex flex-wrap gap-2">
                          {document.tags.map((tag: string) => (
                            <Badge key={tag} variant="secondary">
                              {tag}
                            </Badge>
                          ))}
                        </div>
                      </CardContent>
                    </Card>
                  )}

                  {/* Description */}
                  {document.description && (
                    <Card className="glass-card">
                      <CardHeader>
                        <CardTitle>Description</CardTitle>
                      </CardHeader>
                      <CardContent>
                        <p className="text-muted-foreground">{document.description}</p>
                      </CardContent>
                    </Card>
                  )}
                </TabsContent>

                <TabsContent value="insights" className="space-y-6 mt-0">
                  {insightsLoading ? (
                    <div className="space-y-4">
                      {Array.from({ length: 3 }).map((_, i) => (
                        <Card key={i} className="glass-card">
                          <CardContent className="p-6">
                            <Skeleton className="h-6 w-32 mb-4" />
                            <Skeleton className="h-4 w-full mb-2" />
                            <Skeleton className="h-4 w-3/4" />
                          </CardContent>
                        </Card>
                      ))}
                    </div>
                  ) : insights ? (
                    <div className="space-y-6">
                      {/* Summary */}
                      {insights.summary && (
                        <Card className="glass-card">
                          <CardHeader>
                            <CardTitle className="flex items-center gap-2">
                              <Brain className="w-5 h-5" />
                              AI Summary
                            </CardTitle>
                          </CardHeader>
                          <CardContent>
                            <p className="text-muted-foreground">{insights.summary}</p>
                          </CardContent>
                        </Card>
                      )}

                      {/* Key Topics */}
                      {insights.key_topics && (
                        <Card className="glass-card">
                          <CardHeader>
                            <CardTitle>Key Topics</CardTitle>
                          </CardHeader>
                          <CardContent>
                            <div className="flex flex-wrap gap-2">
                              {insights.key_topics.map((topic: any, index: number) => (
                                <Badge key={index} variant="outline">
                                  {topic.topic || topic} 
                                  {topic.confidence && (
                                    <span className="ml-1 text-xs">
                                      ({Math.round(topic.confidence * 100)}%)
                                    </span>
                                  )}
                                </Badge>
                              ))}
                            </div>
                          </CardContent>
                        </Card>
                      )}

                      {/* Entities */}
                      {insights.entities && (
                        <Card className="glass-card">
                          <CardHeader>
                            <CardTitle>Named Entities</CardTitle>
                          </CardHeader>
                          <CardContent>
                            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                              {Object.entries(insights.entities).map(([type, entities]: [string, any]) => (
                                <div key={type}>
                                  <h4 className="font-medium mb-2 capitalize">{type.replace('_', ' ')}</h4>
                                  <div className="flex flex-wrap gap-1">
                                    {(entities as string[]).map((entity: string, index: number) => (
                                      <Badge key={index} variant="secondary" className="text-xs">
                                        {entity}
                                      </Badge>
                                    ))}
                                  </div>
                                </div>
                              ))}
                            </div>
                          </CardContent>
                        </Card>
                      )}

                      {/* Sentiment */}
                      {insights.sentiment && (
                        <Card className="glass-card">
                          <CardHeader>
                            <CardTitle>Sentiment Analysis</CardTitle>
                          </CardHeader>
                          <CardContent>
                            <div className="flex items-center gap-4">
                              <div className="text-2xl">
                                {insights.sentiment.label === 'positive' ? '😊' : 
                                 insights.sentiment.label === 'negative' ? '😔' : '😐'}
                              </div>
                              <div>
                                <div className="font-medium capitalize">{insights.sentiment.label}</div>
                                <div className="text-sm text-muted-foreground">
                                  Confidence: {Math.round(insights.sentiment.score * 100)}%
                                </div>
                              </div>
                            </div>
                          </CardContent>
                        </Card>
                      )}
                    </div>
                  ) : document.status !== 'completed' ? (
                    <Card className="glass-card">
                      <CardContent className="p-12 text-center">
                        <Brain className="w-16 h-16 mx-auto text-muted-foreground mb-4" />
                        <h3 className="text-lg font-semibold mb-2">AI Insights Pending</h3>
                        <p className="text-muted-foreground">
                          AI insights will be available once the document is fully processed
                        </p>
                      </CardContent>
                    </Card>
                  ) : (
                    <Card className="glass-card">
                      <CardContent className="p-12 text-center">
                        <Brain className="w-16 h-16 mx-auto text-muted-foreground mb-4" />
                        <h3 className="text-lg font-semibold mb-2">No Insights Available</h3>
                        <p className="text-muted-foreground mb-4">
                          AI insights are not available for this document
                        </p>
                        <Button variant="outline">
                          Generate Insights
                        </Button>
                      </CardContent>
                    </Card>
                  )}
                </TabsContent>

                <TabsContent value="history" className="space-y-6 mt-0">
                  <Card className="glass-card">
                    <CardHeader>
                      <CardTitle className="flex items-center gap-2">
                        <Clock className="w-5 h-5" />
                        Document History
                      </CardTitle>
                    </CardHeader>
                    <CardContent>
                      <div className="space-y-4">
                        <div className="flex items-start gap-4 pb-4 border-b">
                          <div className="w-3 h-3 bg-green-500 rounded-full mt-2" />
                          <div className="flex-1">
                            <div className="font-medium">Document uploaded</div>
                            <div className="text-sm text-muted-foreground">
                              {document.created_at ? new Date(document.created_at).toLocaleString() : 'Recently'}
                            </div>
                          </div>
                        </div>

                        {document.processed_at && (
                          <div className="flex items-start gap-4 pb-4 border-b">
                            <div className="w-3 h-3 bg-blue-500 rounded-full mt-2" />
                            <div className="flex-1">
                              <div className="font-medium">Processing completed</div>
                              <div className="text-sm text-muted-foreground">
                                {new Date(document.processed_at).toLocaleString()}
                              </div>
                            </div>
                          </div>
                        )}

                        {document.last_accessed && (
                          <div className="flex items-start gap-4">
                            <div className="w-3 h-3 bg-gray-500 rounded-full mt-2" />
                            <div className="flex-1">
                              <div className="font-medium">Last accessed</div>
                              <div className="text-sm text-muted-foreground">
                                {new Date(document.last_accessed).toLocaleString()}
                              </div>
                            </div>
                          </div>
                        )}
                      </div>
                    </CardContent>
                  </Card>
                </TabsContent>
              </div>
            </Tabs>
          </div>
        </motion.div>
      </motion.div>
    </AnimatePresence>
  )
}

