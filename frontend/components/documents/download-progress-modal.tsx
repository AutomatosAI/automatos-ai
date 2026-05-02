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

  useEffect(() => {
    if (open && (downloadUrl || documentId)) {
      startDownload()
    }
  }, [open, downloadUrl, documentId])

  const startDownload = async () => {
    setStatus('preparing')
    setProgress(0)
    setError(null)
    
    try {
      // Simulate preparation phase
      await new Promise(resolve => setTimeout(resolve, 500))
      
      setStatus('downloading')
      
      // Simulate download progress
      const interval = setInterval(() => {
        setProgress(prev => {
          if (prev >= 100) {
            clearInterval(interval)
            setStatus('completed')
            setCanCancel(false)
            return 100
          }
          
          // Simulate realistic download progress
          const increment = Math.random() * 15 + 5
          const newProgress = Math.min(prev + increment, 100)
          
          // Update download stats
          setDownloadedSize(`${(newProgress * 2.4 / 100).toFixed(1)} MB`)
          setTotalSize('2.4 MB')
          setSpeed(`${(Math.random() * 500 + 200).toFixed(0)} KB/s`)
          setTimeRemaining(`${Math.max(0, Math.ceil((100 - newProgress) / 10))}s`)
          
          return newProgress
        })
      }, 200)

    } catch (err) {
      console.error('Download error:', err)
      setError(err instanceof Error ? err.message : 'Download failed')
      setStatus('error')
      setCanCancel(false)
    }
  }

  const handleCancel = () => {
    if (canCancel) {
      setStatus('cancelled')
      setCanCancel(false)
    }
  }

  const handleRetry = () => {
    setError(null)
    startDownload()
  }

  const handleClose = () => {
    onClose()
  }

  const getStatusInfo = () => {
    switch (status) {
      case 'preparing':
        return {
          icon: Loader2,
          color: 'text-info',
          bgColor: 'bg-info/10',
          borderColor: 'border-info/20',
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
          color: 'text-success',
          bgColor: 'bg-success/10',
          borderColor: 'border-success/20',
          message: 'Download completed successfully!'
        }
      case 'error':
        return {
          icon: AlertTriangle,
          color: 'text-destructive',
          bgColor: 'bg-destructive/10',
          borderColor: 'border-destructive/20',
          message: 'Download failed'
        }
      case 'cancelled':
        return {
          icon: StopCircle,
          color: 'text-muted-foreground',
          bgColor: 'bg-secondary/50',
          borderColor: 'border-border/30',
          message: 'Download cancelled'
        }
      default:
        return {
          icon: Loader2,
          color: 'text-muted-foreground',
          bgColor: 'bg-secondary/50',
          borderColor: 'border-border/30',
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
        <Card className="glass-card card-glow w-full max-w-md">
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
                <span>{progress.toFixed(0)}%</span>
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
                    <p className="text-lg font-bold text-info">{speed}</p>
                    <p className="text-xs text-muted-foreground">Download Speed</p>
                  </div>
                )}
                {timeRemaining && (
                  <div className="bg-secondary/30 rounded-lg p-3">
                    <p className="text-lg font-bold text-success">{timeRemaining}</p>
                    <p className="text-xs text-muted-foreground">Time Remaining</p>
                  </div>
                )}
              </div>
            )}

            {/* Error Message */}
            {error && (
              <div className="bg-destructive/10 border border-destructive/20 rounded-lg p-3">
                <p className="text-destructive text-sm">{error}</p>
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
              <div className="text-center bg-success/10 border border-success/20 rounded-lg p-4">
                <CheckCircle className="w-8 h-8 text-success mx-auto mb-2" />
                <p className="text-success font-medium">Download Complete!</p>
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
