"use client"

import React from "react"

import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import { 
  Activity, 
  Clock, 
  CheckCircle, 
  AlertTriangle, 
  Play, 
  Pause,
  RefreshCw,
  FileText,
  Zap,
  TrendingUp,
  Users,
  Database
} from 'lucide-react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Progress } from '@/components/ui/progress'
import { Button } from '@/components/ui/button'
import { apiClient } from '@/lib/api'

interface ProcessingStage {
  stage: string;
  status: 'active' | 'idle' | 'error';
  documents_count: number;
  avg_duration: string;
  success_rate: number;
}

interface ProcessingPipeline {
  pipeline_status: 'active' | 'idle';
  total_documents: number;
  processing_documents: number;
  completed_documents: number;
  failed_documents: number;
  success_rate: number;
  avg_processing_time: string;
  queue_status: {
    pending: number;
    active_workers: number;
    estimated_completion: string;
  };
  processing_stages: ProcessingStage[];
  recent_activity: any[];
  last_updated: string;
}

interface LiveProcessingJob {
  document_id: number;
  filename: string;
  status: string;
  progress: number;
  current_stage: string;
  estimated_completion: string;
  started_at: string;
}

export function ProcessingTab() {
  const [pipelineData, setPipelineData] = useState<ProcessingPipeline | null>(null)
  const [liveJobs, setLiveJobs] = useState<LiveProcessingJob[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [autoRefresh, setAutoRefresh] = useState(true)

  useEffect(() => {
    loadProcessingData()
    
    if (autoRefresh) {
      const interval = setInterval(() => {
        loadProcessingData()
        loadLiveStatus()
      }, 5000) // Refresh every 5 seconds
      
      return () => clearInterval(interval)
    }
  }, [autoRefresh])

  const loadProcessingData = async () => {
    try {
      setError(null)
      const data = await apiClient.request<ProcessingPipeline>('/api/documents/processing/pipeline')
      setPipelineData(data)
    } catch (err) {
      console.error('Error loading processing data:', err)
      setError(err instanceof Error ? err.message : 'Failed to load processing data')
    } finally {
      setLoading(false)
    }
  }

  const loadLiveStatus = async () => {
    try {
      const data = await apiClient.request<{active_jobs: LiveProcessingJob[]}>('/api/documents/processing/live-status')
      setLiveJobs(data.active_jobs)
    } catch (err) {
      console.error('Error loading live status:', err)
    }
  }

  const handleReprocessAll = async () => {
    try {
      await apiClient.request('/api/documents/processing/reprocess-all', {
        method: 'POST'
      })
      // Reload data after starting reprocessing
      setTimeout(() => loadProcessingData(), 1000)
    } catch (err) {
      console.error('Error starting reprocessing:', err)
      setError(err instanceof Error ? err.message : 'Failed to start reprocessing')
    }
  }

  if (loading) {
    return (
      <div className="flex items-center justify-center py-12">
        <div className="text-center">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary mx-auto mb-4"></div>
          <p className="text-muted-foreground">Loading processing pipeline...</p>
        </div>
      </div>
    )
  }

  if (error) {
    return (
      <div className="flex items-center justify-center py-12">
        <div className="text-center">
          <AlertTriangle className="h-8 w-8 text-red-400 mx-auto mb-4" />
          <p className="text-red-400 mb-4">Error: {error}</p>
          <Button onClick={loadProcessingData} variant="outline">
            Try Again
          </Button>
        </div>
      </div>
    )
  }

  if (!pipelineData) {
    return (
      <div className="text-center py-12">
        <p className="text-muted-foreground">No processing data available</p>
      </div>
    )
  }

  const statusColor = pipelineData.pipeline_status === 'active' ? 'text-green-400' : 'text-gray-400'
  const statusIcon = pipelineData.pipeline_status === 'active' ? Activity : Clock

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold mb-2">Document Processing Pipeline</h2>
          <p className="text-muted-foreground">
            Real-time monitoring of document processing and ingestion
          </p>
        </div>
        
        <div className="flex items-center space-x-2">
          <Button
            variant="outline"
            size="sm"
            onClick={() => setAutoRefresh(!autoRefresh)}
            className={autoRefresh ? 'bg-green-500/10 border-green-500/20' : ''}
          >
            <RefreshCw className={`w-4 h-4 mr-2 ${autoRefresh ? 'animate-spin' : ''}`} />
            Auto Refresh
          </Button>
          
          <Button
            variant="outline"
            size="sm"
            onClick={handleReprocessAll}
            disabled={pipelineData.processing_documents > 0}
          >
            <Play className="w-4 h-4 mr-2" />
            Reprocess All
          </Button>
        </div>
      </div>

      {/* Pipeline Status Overview */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <motion.div
          className="glass-card p-6"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
        >
          <div className="flex items-center justify-between mb-4">
            <div className="w-10 h-10 rounded-lg bg-secondary/50 flex items-center justify-center">
              {React.createElement(statusIcon, { className: `w-5 h-5 ${statusColor}` })}
            </div>
            <Badge className={pipelineData.pipeline_status === 'active' ? 'bg-green-500/10 text-green-400 border-green-500/20' : 'bg-gray-500/10 text-gray-400 border-gray-500/20'}>
              {pipelineData.pipeline_status}
            </Badge>
          </div>
          <div className="space-y-1">
            <h3 className="text-2xl font-bold">{pipelineData.total_documents}</h3>
            <p className="text-muted-foreground text-sm">Total Documents</p>
            <p className="text-xs text-green-400">
              {pipelineData.success_rate}% success rate
            </p>
          </div>
        </motion.div>

        <motion.div
          className="glass-card p-6"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.1 }}
        >
          <div className="flex items-center justify-between mb-4">
            <div className="w-10 h-10 rounded-lg bg-secondary/50 flex items-center justify-center">
              <Clock className="w-5 h-5 text-orange-400" />
            </div>
          </div>
          <div className="space-y-1">
            <h3 className="text-2xl font-bold">{pipelineData.processing_documents}</h3>
            <p className="text-muted-foreground text-sm">Processing</p>
            <p className="text-xs text-orange-400">
              ETA: {pipelineData.queue_status.estimated_completion}
            </p>
          </div>
        </motion.div>

        <motion.div
          className="glass-card p-6"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.2 }}
        >
          <div className="flex items-center justify-between mb-4">
            <div className="w-10 h-10 rounded-lg bg-secondary/50 flex items-center justify-center">
              <CheckCircle className="w-5 h-5 text-green-400" />
            </div>
          </div>
          <div className="space-y-1">
            <h3 className="text-2xl font-bold">{pipelineData.completed_documents}</h3>
            <p className="text-muted-foreground text-sm">Completed</p>
            <p className="text-xs text-green-400">
              Avg: {pipelineData.avg_processing_time}
            </p>
          </div>
        </motion.div>

        <motion.div
          className="glass-card p-6"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.3 }}
        >
          <div className="flex items-center justify-between mb-4">
            <div className="w-10 h-10 rounded-lg bg-secondary/50 flex items-center justify-center">
              <Users className="w-5 h-5 text-blue-400" />
            </div>
          </div>
          <div className="space-y-1">
            <h3 className="text-2xl font-bold">{pipelineData.queue_status.active_workers}</h3>
            <p className="text-muted-foreground text-sm">Active Workers</p>
            <p className="text-xs text-blue-400">
              {pipelineData.queue_status.pending} in queue
            </p>
          </div>
        </motion.div>
      </div>

      {/* Processing Stages */}
      <motion.div
        className="glass-card p-6"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5, delay: 0.4 }}
      >
        <h3 className="text-lg font-semibold mb-4">Processing Stages</h3>
        <div className="space-y-4">
          {pipelineData.processing_stages.map((stage, index) => (
            <div key={stage.stage} className="flex items-center justify-between p-4 bg-secondary/30 rounded-lg">
              <div className="flex items-center space-x-4">
                <div className={`w-3 h-3 rounded-full ${
                  stage.status === 'active' ? 'bg-green-400' : 
                  stage.status === 'error' ? 'bg-red-400' : 'bg-gray-400'
                }`} />
                <div>
                  <h4 className="font-medium">{stage.stage}</h4>
                  <p className="text-sm text-muted-foreground">
                    {stage.documents_count} documents • Avg: {stage.avg_duration}
                  </p>
                </div>
              </div>
              <div className="text-right">
                <p className="text-sm font-medium">{stage.success_rate}%</p>
                <p className="text-xs text-muted-foreground">Success Rate</p>
              </div>
            </div>
          ))}
        </div>
      </motion.div>

      {/* Live Processing Jobs */}
      {liveJobs.length > 0 && (
        <motion.div
          className="glass-card p-6"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.5 }}
        >
          <h3 className="text-lg font-semibold mb-4">Live Processing Jobs</h3>
          <div className="space-y-4">
            {liveJobs.map((job) => (
              <div key={job.document_id} className="p-4 bg-secondary/30 rounded-lg">
                <div className="flex items-center justify-between mb-2">
                  <div>
                    <h4 className="font-medium">{job.filename}</h4>
                    <p className="text-sm text-muted-foreground">
                      {job.current_stage} • ETA: {job.estimated_completion}
                    </p>
                  </div>
                  <Badge className="bg-blue-500/10 text-blue-400 border-blue-500/20">
                    {job.progress}%
                  </Badge>
                </div>
                <Progress value={job.progress} className="h-2" />
              </div>
            ))}
          </div>
        </motion.div>
      )}

      {/* Recent Activity */}
      <motion.div
        className="glass-card p-6"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5, delay: 0.6 }}
      >
        <h3 className="text-lg font-semibold mb-4">Recent Activity</h3>
        <div className="space-y-3">
          {pipelineData.recent_activity.slice(0, 10).map((activity, index) => (
            <div key={activity.id} className="flex items-center justify-between p-3 bg-secondary/20 rounded-lg">
              <div className="flex items-center space-x-3">
                <FileText className="w-4 h-4 text-muted-foreground" />
                <div>
                  <p className="font-medium text-sm">{activity.filename}</p>
                  <p className="text-xs text-muted-foreground">
                    {activity.chunk_count} chunks • {(activity.file_size / 1024).toFixed(1)}KB
                  </p>
                </div>
              </div>
              <div className="text-right">
                <Badge className={
                  activity.status === 'processed' ? 'bg-green-500/10 text-green-400 border-green-500/20' :
                  activity.status === 'failed' ? 'bg-red-500/10 text-red-400 border-red-500/20' :
                  'bg-orange-500/10 text-orange-400 border-orange-500/20'
                }>
                  {activity.status}
                </Badge>
                <p className="text-xs text-muted-foreground mt-1">
                  {activity.processed_date ? new Date(activity.processed_date).toLocaleTimeString() : 'Processing...'}
                </p>
              </div>
            </div>
          ))}
        </div>
      </motion.div>
    </div>
  )
}
