
'use client'

import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import { 
  Zap, 
  FileText, 
  Clock, 
  CheckCircle, 
  XCircle, 
  AlertCircle,
  Play,
  Pause,
  RefreshCw,
  BarChart3,
  Brain,
  Settings,
  Table,
  Image,
  Calculator,
  FileSpreadsheet
} from 'lucide-react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Progress } from '@/components/ui/progress'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Skeleton } from '@/components/ui/skeleton'

// API hooks
import { useProcessingQueue as useProcessingQueueOld, useStartProcessing } from '@/hooks/use-document-api'
import { useProcessingQueueWebSocket } from '@/hooks/use-processing-queue-websocket'
// Processing components
import { LiveIndicator } from './processing/live-indicator'
import { ProcessingSteps } from './processing/processing-steps'
import { LiveProgressBar } from './processing/live-progress-bar'

interface DocumentProcessingProps {
  documents: any[]
  onDocumentSelect: (documentId: string) => void
}

export function DocumentProcessing({ documents, onDocumentSelect }: DocumentProcessingProps) {
  // Use WebSocket hook for live updates
  const { 
    queueStatus, 
    isConnected, 
    lastUpdate,
    isLoading 
  } = useProcessingQueueWebSocket()
  
  const startProcessingMutation = useStartProcessing()
  
  // Extract real stats from API
  const processingStats = {
    total_processed_today: queueStatus?.total_processed_today || 0,
    currently_processing: queueStatus?.currently_processing || 0,
    average_processing_time: queueStatus?.average_processing_time || 0,
    success_rate: queueStatus?.success_rate || 0
  }
  
  // Update stats from live queue data
  const liveProcessingDocuments = queueStatus?.processing || []
  const livePendingDocuments = queueStatus?.pending || []
  const liveFailedDocuments = queueStatus?.failed || []

  const processingDocuments = documents.filter(doc => doc.status === 'processing')
  const completedDocuments = documents.filter(doc => doc.status === 'completed')
  const failedDocuments = documents.filter(doc => doc.status === 'failed')
  const pendingDocuments = documents.filter(doc => doc.status === 'pending')

  const handleStartProcessing = async (documentId: string) => {
    try {
      await startProcessingMutation.mutateAsync(documentId)
      await refetch()
    } catch (error) {
      // Error handled by hook
    }
  }

  if (isLoading) {
    return (
      <div className="space-y-6">
        {Array.from({ length: 3 }).map((_, i) => (
          <Card key={i} className="glass-card">
            <CardContent className="p-6">
              <div className="flex items-center justify-between mb-4">
                <Skeleton className="h-6 w-32" />
                <Skeleton className="h-6 w-16" />
              </div>
              <Skeleton className="h-4 w-full mb-2" />
              <Skeleton className="h-2 w-full" />
            </CardContent>
          </Card>
        ))}
      </div>
    )
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold">Document Processing</h2>
          <p className="text-muted-foreground">Real-time processing pipeline and queue management</p>
        </div>
        <LiveIndicator isConnected={isConnected} lastUpdate={lastUpdate} />
      </div>

      {/* Processing Stats */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-5 gap-6">
        <Card className="glass-card">
          <CardContent className="p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-muted-foreground">Processed Today</p>
                <p className="text-2xl font-bold">{processingStats.total_processed_today}</p>
              </div>
              <div className="w-10 h-10 rounded-lg bg-secondary/50 flex items-center justify-center">
                <CheckCircle className="w-5 h-5 text-green-400" />
              </div>
            </div>
          </CardContent>
        </Card>

        <Card className="glass-card">
          <CardContent className="p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-muted-foreground">Currently Processing</p>
                <p className="text-2xl font-bold">{liveProcessingDocuments.length || processingDocuments.length}</p>
              </div>
              <div className="w-10 h-10 rounded-lg bg-secondary/50 flex items-center justify-center">
                <Zap className="w-5 h-5 text-blue-400" />
              </div>
            </div>
          </CardContent>
        </Card>

        <Card className="glass-card">
          <CardContent className="p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-muted-foreground">Avg Time</p>
                <p className="text-2xl font-bold">{processingStats.average_processing_time}s</p>
              </div>
              <div className="w-10 h-10 rounded-lg bg-secondary/50 flex items-center justify-center">
                <Clock className="w-5 h-5 text-purple-400" />
              </div>
            </div>
          </CardContent>
        </Card>

        <Card className="glass-card">
          <CardContent className="p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-muted-foreground">Success Rate</p>
                <p className="text-2xl font-bold">{processingStats.success_rate}%</p>
              </div>
              <div className="w-10 h-10 rounded-lg bg-secondary/50 flex items-center justify-center">
                <BarChart3 className="w-5 h-5 text-orange-400" />
              </div>
            </div>
          </CardContent>
        </Card>

        <Card className="glass-card">
          <CardContent className="p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-muted-foreground">Multimodal Items</p>
                <p className="text-2xl font-bold">{queueStatus?.multimodal_items_extracted || 0}</p>
              </div>
              <div className="w-10 h-10 rounded-lg bg-secondary/50 flex items-center justify-center">
                <Brain className="w-5 h-5 text-purple-400" />
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      <Tabs defaultValue="queue" className="space-y-6">
        <TabsList>
          <TabsTrigger value="queue">Processing Queue</TabsTrigger>
          <TabsTrigger value="active">Active Processing</TabsTrigger>
          <TabsTrigger value="completed">Completed</TabsTrigger>
          <TabsTrigger value="failed">Failed</TabsTrigger>
        </TabsList>

        <TabsContent value="queue" className="space-y-4">
          <div className="flex justify-between items-center">
            <h3 className="text-lg font-semibold">Processing Queue ({pendingDocuments.length})</h3>
            <Button onClick={() => refetch()} size="sm" variant="outline">
              <RefreshCw className="w-4 h-4 mr-2" />
              Refresh
            </Button>
          </div>

          {pendingDocuments.map(doc => (
            <Card key={doc.id} className="glass-card">
              <CardContent className="p-4">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-3">
                    <FileText className="w-8 h-8 text-blue-500" />
                    <div>
                      <h4 className="font-medium">{doc.name}</h4>
                      <p className="text-sm text-muted-foreground">
                        {doc.file_type} • {doc.size ? `${(doc.size / 1024 / 1024).toFixed(1)}MB` : 'Unknown size'}
                      </p>
                    </div>
                  </div>
                  <div className="flex items-center gap-2">
                    <Badge variant="outline">Pending</Badge>
                    <Button 
                      size="sm" 
                      onClick={() => handleStartProcessing(doc.id)}
                      disabled={startProcessingMutation.isLoading}
                    >
                      <Play className="w-4 h-4 mr-1" />
                      Start
                    </Button>
                  </div>
                </div>
              </CardContent>
            </Card>
          ))}

          {pendingDocuments.length === 0 && (
            <Card className="glass-card">
              <CardContent className="p-12 text-center">
                <Clock className="w-16 h-16 mx-auto text-muted-foreground mb-4" />
                <h3 className="text-lg font-semibold mb-2">No documents in queue</h3>
                <p className="text-muted-foreground">
                  All documents have been processed or are currently being processed
                </p>
              </CardContent>
            </Card>
          )}
        </TabsContent>

        <TabsContent value="active" className="space-y-4">
          <h3 className="text-lg font-semibold">
            Currently Processing ({liveProcessingDocuments.length || processingDocuments.length})
          </h3>

          {/* Show live processing documents if available */}
          {(liveProcessingDocuments.length > 0 ? liveProcessingDocuments : processingDocuments).map((doc: any, index: number) => (
            <motion.div
              key={doc.document_id || doc.id}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: index * 0.05 }}
            >
              <Card className="glass-card">
                <CardContent className="p-4">
                  <div className="space-y-4">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-3">
                        <div className="relative">
                          <Zap className="w-8 h-8 text-blue-500 animate-pulse" />
                        </div>
                        <div>
                          <h4 className="font-medium">{doc.filename || doc.name}</h4>
                          <p className="text-sm text-muted-foreground">
                            Processing • Started {new Date(doc.started_at || doc.processing_started_at || Date.now()).toLocaleTimeString()}
                          </p>
                        </div>
                      </div>
                      <Badge variant="outline" className="text-blue-600 border-blue-600">
                        Processing
                      </Badge>
                    </div>

                    {/* Live Progress Bar */}
                    <LiveProgressBar 
                      progress={doc.progress || doc.processing_progress || 45}
                      step={doc.step_name || doc.processing_step}
                      eta_seconds={doc.eta_seconds}
                    />

                    {/* Processing Steps Visualization */}
                    <ProcessingSteps 
                      steps={['Upload', 'Extract', 'Multimodal', 'Chunk', 'Embed', 'Store', 'Complete']}
                      currentStep={doc.current_step || 'text_extraction'}
                    />

                    {/* Multimodal Extraction Status */}
                    {(doc.current_step === 'multimodal_extraction' || doc.multimodal_status) && (
                      <div className="space-y-3">
                        <div className="flex items-center gap-2">
                          <Brain className="w-4 h-4 text-purple-400" />
                          <span className="text-sm font-medium">Multimodal Extraction</span>
                        </div>
                        <div className="grid grid-cols-3 gap-3">
                          <div className="flex items-center gap-2 p-2 rounded-lg bg-secondary/30">
                            <Table className="w-4 h-4 text-blue-400" />
                            <span className="text-xs">Tables: {doc.extracted_tables || 0}</span>
                          </div>
                          <div className="flex items-center gap-2 p-2 rounded-lg bg-secondary/30">
                            <Calculator className="w-4 h-4 text-green-400" />
                            <span className="text-xs">Formulas: {doc.extracted_formulas || 0}</span>
                          </div>
                          <div className="flex items-center gap-2 p-2 rounded-lg bg-secondary/30">
                            <Image className="w-4 h-4 text-orange-400" />
                            <span className="text-xs">Images: {doc.extracted_images || 0}</span>
                          </div>
                        </div>
                      </div>
                    )}
                  </div>
                </CardContent>
              </Card>
            </motion.div>
          ))}

          {processingDocuments.length === 0 && (
            <Card className="glass-card">
              <CardContent className="p-12 text-center">
                <Zap className="w-16 h-16 mx-auto text-muted-foreground mb-4" />
                <h3 className="text-lg font-semibold mb-2">No active processing</h3>
                <p className="text-muted-foreground">
                  No documents are currently being processed
                </p>
              </CardContent>
            </Card>
          )}
        </TabsContent>

        <TabsContent value="completed" className="space-y-4">
          <h3 className="text-lg font-semibold">Recently Completed ({completedDocuments.length})</h3>

          {completedDocuments.slice(0, 10).map(doc => (
            <Card key={doc.id} className="glass-card cursor-pointer hover:shadow-lg" onClick={() => onDocumentSelect(doc.id)}>
              <CardContent className="p-4">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-3">
                    <CheckCircle className="w-8 h-8 text-green-500" />
                    <div>
                      <h4 className="font-medium">{doc.name}</h4>
                      <p className="text-sm text-muted-foreground">
                        Completed {new Date(doc.processed_at || Date.now()).toLocaleString()}
                      </p>
                    </div>
                  </div>
                  <div className="flex items-center gap-2">
                    <Badge variant="outline" className="text-green-600 border-green-600">
                      Completed
                    </Badge>
                    <Badge variant="secondary">
                      {doc.processing_time || '2.1'}s
                    </Badge>
                  </div>
                </div>
              </CardContent>
            </Card>
          ))}

          {completedDocuments.length === 0 && (
            <Card className="glass-card">
              <CardContent className="p-12 text-center">
                <CheckCircle className="w-16 h-16 mx-auto text-muted-foreground mb-4" />
                <h3 className="text-lg font-semibold mb-2">No completed documents</h3>
                <p className="text-muted-foreground">
                  Completed documents will appear here
                </p>
              </CardContent>
            </Card>
          )}
        </TabsContent>

        <TabsContent value="failed" className="space-y-4">
          <h3 className="text-lg font-semibold">Failed Processing ({failedDocuments.length})</h3>

          {failedDocuments.map(doc => (
            <Card key={doc.id} className="glass-card">
              <CardContent className="p-4">
                <div className="space-y-3">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-3">
                      <XCircle className="w-8 h-8 text-red-500" />
                      <div>
                        <h4 className="font-medium">{doc.name}</h4>
                        <p className="text-sm text-red-600">
                          {doc.processing_error || 'Processing failed'}
                        </p>
                      </div>
                    </div>
                    <div className="flex items-center gap-2">
                      <Badge variant="destructive">Failed</Badge>
                      <Button 
                        size="sm" 
                        variant="outline"
                        onClick={() => handleStartProcessing(doc.id)}
                      >
                        <RefreshCw className="w-4 h-4 mr-1" />
                        Retry
                      </Button>
                    </div>
                  </div>

                  <p className="text-sm text-muted-foreground">
                    Failed at {new Date(doc.failed_at || Date.now()).toLocaleString()}
                  </p>
                </div>
              </CardContent>
            </Card>
          ))}

          {failedDocuments.length === 0 && (
            <Card className="glass-card">
              <CardContent className="p-12 text-center">
                <CheckCircle className="w-16 h-16 mx-auto text-green-500 mb-4" />
                <h3 className="text-lg font-semibold mb-2">No failed processing</h3>
                <p className="text-muted-foreground">
                  All document processing has been successful
                </p>
              </CardContent>
            </Card>
          )}
        </TabsContent>
      </Tabs>
    </div>
  )
}

