
'use client'

import * as React from 'react'
import { useState, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { 
  X, 
  Download, 
  CheckCircle, 
  AlertTriangle, 
  Loader2,
  FileDown,
  Pause,
  Play,
  StopCircle
} from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Progress } from '@/components/ui/progress'
import { Badge } from '@/components/ui/badge'

interface DownloadProgressModalProps {
  open: boolean
  onClose: () => void
  filename: string
  downloadUrl?: string
  documentId?: number
}

type DownloadStatus = 'preparing' | 'downloading' | 'completed' | 'error' | 'cancelled' | 'paused'

export function DownloadProgressModal({ 
  open, 
  onClose, 
  filename, 
  downloadUrl,
  documentId 
}: DownloadProgressModalProps) {
  const [status, setStatus] = useState<DownloadStatus>('preparing')
  const [progress, setProgress] = useState(0)
  const [speed, setSpeed] = useState('')
  const [timeRemaining, setTimeRemaining] = useState('')
  const [downloadedSize, setDownloadedSize] = useState('')
  const [totalSize, setTotalSize] = useState('')
  const [error, setError] = useState<string | null>(null)
  const [canCancel, setCanCancel] = useState(true)
  const [abortController, setAbortController] = useState<AbortController | null>(null)

  useEffect(() => {
    if (open && (downloadUrl || documentId)) {
      startDownload()
    }
    
    return () => {
      // Cleanup abort controller on unmount
      if (abortController) {
        abortController.abort()
      }
    }
  }, [open, downloadUrl, documentId])

  const startDownload = async () => {
    const controller = new AbortController()
    setAbortController(controller)
    setStatus('preparing')
    setProgress(0)
    setError(null)
    
    try {
      // Simulate preparation phase
      await new Promise(resolve => setTimeout(resolve, 500))
      
      if (controller.signal.aborted) {
        setStatus('cancelled')
        return
      }

      setStatus('downloading')
      
      // Build download URL
      const url = downloadUrl || `https://api.automatos.app/api/documents/${documentId}/download`
      
      const response = await fetch(url, {
        signal: controller.signal,
        headers: {
          'Accept': '*/*',
        }
      })

      if (!response.ok) {
        throw new Error(`Download failed: ${response.status} ${response.statusText}`)
      }

      const contentLength = response.headers.get('content-length')
      const total = contentLength ? parseInt(contentLength, 10) : 0
      
      if (total) {
        setTotalSize(formatBytes(total))
      }

      const reader = response.body?.getReader()
      if (!reader) {
        throw new Error('Failed to get response reader')
      }

      let received = 0
      const chunks: Uint8Array[] = []
      const startTime = Date.now()

      while (true) {
        const { done, value } = await reader.read()

        if (controller.signal.aborted) {
          setStatus('cancelled')
          return
        }

        if (done) break

        if (value) {
          chunks.push(value)
          received += value.length

          const progressPercent = total ? Math.round((received / total) * 100) : 0
          setProgress(progressPercent)
          setDownloadedSize(formatBytes(received))

          // Calculate speed and time remaining
          const elapsed = (Date.now() - startTime) / 1000
          if (elapsed > 1) {
            const speedBps = received / elapsed
            setSpeed(`${formatBytes(speedBps)}/s`)
            
            if (total && speedBps > 0) {
              const remaining = (total - received) / speedBps
              setTimeRemaining(formatTime(remaining))
            }
          }

          // Small delay to make progress visible
          await new Promise(resolve => setTimeout(resolve, 50))
        }
      }

      if (controller.signal.aborted) {
        setStatus('cancelled')
        return
      }

      // Create blob and download
      const blob = new Blob(chunks)
      const blobUrl = URL.createObjectURL(blob)
      
      const link = document.createElement('a')
      link.href = blobUrl
      link.download = filename
      document.body.appendChild(link)
      link.click()
      document.body.removeChild(link)
      URL.revokeObjectURL(blobUrl)

      setStatus('completed')
      setCanCancel(false)
      setProgress(100)

    } catch (err) {
      if (err instanceof Error && err.name === 'AbortError') {
        setStatus('cancelled')
      } else {
        console.error('Download error:', err)
        setError(err instanceof Error ? err.message : 'Download failed')
        setStatus('error')
      }
      setCanCancel(false)
    }
  }

  const handleCancel = () => {
    if (abortController && canCancel) {
      abortController.abort()
      setStatus('cancelled')
      setCanCancel(false)
    }
  }

  const handleRetry = () => {
    setError(null)
    startDownload()
  }

  const handleClose = () => {
    if (canCancel && abortController) {
      abortController.abort()
    }
    onClose()
  }

  const formatBytes = (bytes: number): string => {
    if (bytes === 0) return '0 Bytes'
    const k = 1024
    const sizes = ['Bytes', 'KB', 'MB', 'GB']
    const i = Math.floor(Math.log(bytes) / Math.log(k))
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i]
  }

  const formatTime = (seconds: number): string => {
    if (seconds < 60) return `${Math.round(seconds)}s`
    if (seconds < 3600) return `${Math.round(seconds / 60)}m ${Math.round(seconds % 60)}s`
    return `${Math.round(seconds / 3600)}h ${Math.round((seconds % 3600) / 60)}m`
  }

  const getStatusInfo = () => {
    switch (status) {
      case 'preparing':
        return {
          icon: Loader2,
          color: 'text-blue-400',
          bgColor: 'bg-blue-500/10',
          borderColor: 'border-blue-500/20',
          message: 'Preparing download...'
        }
      case 'downloading':
        return {
          icon: Download,
          color: 'text-orange-400',
          bgColor: 'bg-orange-500/10',
          borderColor: 'border-orange-500/20',
          message: 'Downloading...'
        }
      case 'completed':
        return {
          icon: CheckCircle,
          color: 'text-green-400',
          bgColor: 'bg-green-500/10',
          borderColor: 'border-green-500/20',
          message: 'Download completed successfully!'
        }
      case 'error':
        return {
          icon: AlertTriangle,
          color: 'text-red-400',
          bgColor: 'bg-red-500/10',
          borderColor: 'border-red-500/20',
          message: 'Download failed'
        }
      case 'cancelled':
        return {
          icon: StopCircle,
          color: 'text-gray-400',
          bgColor: 'bg-gray-500/10',
          borderColor: 'border-gray-500/20',
          message: 'Download cancelled'
        }
      case 'paused':
        return {
          icon: Pause,
          color: 'text-yellow-400',
          bgColor: 'bg-yellow-500/10',
          borderColor: 'border-yellow-500/20',
          message: 'Download paused'
        }
      default:
        return {
          icon: Loader2,
          color: 'text-gray-400',
          bgColor: 'bg-gray-500/10',
          borderColor: 'border-gray-500/20',
          message: 'Initializing...'
        }
    }
  }

  if (!open) return null

  const statusInfo = getStatusInfo()
  const StatusIcon = statusInfo.icon

  return (
    <AnimatePresence>
      <motion.div 
        className="fixed inset-0 bg-black/50 backdrop-blur-sm z-50" 
        initial={{ opacity: 0 }} 
        animate={{ opacity: 1 }} 
        exit={{ opacity: 0 }} 
        onClick={status === 'completed' || status === 'error' || status === 'cancelled' ? handleClose : undefined}
      />
      <motion.div 
        className="fixed inset-0 z-50 flex items-center justify-center p-4" 
        initial={{ opacity: 0, scale: 0.95 }} 
        animate={{ opacity: 1, scale: 1 }} 
        exit={{ opacity: 0, scale: 0.95 }}
      >
        <Card className="glass-card w-full max-w-md">
          <CardHeader className="flex flex-row items-center justify-between border-b border-border/30">
            <CardTitle className="flex items-center space-x-2">
              <FileDown className="w-5 h-5 text-orange-400" />
              <span className="text-lg">Download Progress</span>
            </CardTitle>
            {(status === 'completed' || status === 'error' || status === 'cancelled') && (
              <Button variant="ghost" size="icon" onClick={handleClose}>
                <X className="w-4 h-4" />
              </Button>
            )}
          </CardHeader>
          
          <CardContent className="pt-6 space-y-6">
            {/* File Info */}
            <div className="text-center">
              <h3 className="font-semibold text-lg mb-2 truncate">{filename}</h3>
              <Badge className={`${statusInfo.bgColor} ${statusInfo.color} ${statusInfo.borderColor}`}>
                <StatusIcon className={`w-4 h-4 mr-2 ${status === 'preparing' || status === 'downloading' ? 'animate-spin' : ''}`} />
                {statusInfo.message}
              </Badge>
            </div>

            {/* Progress Bar */}
            <div className="space-y-3">
              <Progress 
                value={progress} 
                className="h-3 bg-secondary/50" 
              />
              <div className="flex justify-between text-sm text-muted-foreground">
                <span>{progress}%</span>
                <span>
                  {downloadedSize} {totalSize && `/ ${totalSize}`}
                </span>
              </div>
            </div>

            {/* Download Stats */}
            {(status === 'downloading' || status === 'paused') && (
              <div className="grid grid-cols-2 gap-4 text-center">
                {speed && (
                  <div className="bg-secondary/30 rounded-lg p-3">
                    <p className="text-lg font-bold text-blue-400">{speed}</p>
                    <p className="text-xs text-muted-foreground">Download Speed</p>
                  </div>
                )}
                {timeRemaining && (
                  <div className="bg-secondary/30 rounded-lg p-3">
                    <p className="text-lg font-bold text-green-400">{timeRemaining}</p>
                    <p className="text-xs text-muted-foreground">Time Remaining</p>
                  </div>
                )}
              </div>
            )}

            {/* Error Message */}
            {error && (
              <div className="bg-red-500/10 border border-red-500/20 rounded-lg p-3">
                <p className="text-red-400 text-sm">{error}</p>
              </div>
            )}

            {/* Action Buttons */}
            <div className="flex justify-center space-x-2">
              {status === 'error' && (
                <Button onClick={handleRetry} className="gradient-accent">
                  <Download className="w-4 h-4 mr-2" />
                  Retry Download
                </Button>
              )}
              
              {canCancel && (status === 'preparing' || status === 'downloading') && (
                <Button variant="outline" onClick={handleCancel}>
                  <StopCircle className="w-4 h-4 mr-2" />
                  Cancel
                </Button>
              )}

              {(status === 'completed' || status === 'cancelled') && (
                <Button onClick={handleClose} variant="outline">
                  Close
                </Button>
              )}
            </div>

            {/* Completed Success Message */}
            {status === 'completed' && (
              <div className="text-center bg-green-500/10 border border-green-500/20 rounded-lg p-4">
                <CheckCircle className="w-8 h-8 text-green-400 mx-auto mb-2" />
                <p className="text-green-400 font-medium">Download Complete!</p>
                <p className="text-sm text-muted-foreground mt-1">
                  The file has been saved to your Downloads folder.
                </p>
              </div>
            )}
          </CardContent>
        </Card>
      </motion.div>
    </AnimatePresence>
  )
}
