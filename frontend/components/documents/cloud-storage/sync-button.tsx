'use client'

import React, { useState, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import {
  RefreshCw,
  CheckCircle2,
  XCircle,
  FileText,
  SkipForward,
  AlertTriangle,
  X,
} from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import {
  useTriggerSync,
  useSyncJob,
  type SyncJob,
} from '@/hooks/use-cloud-documents-api'

interface Props {
  connectionId: number
  syncEnabled: boolean
  onSyncComplete?: () => void
}

export function SyncButton({ connectionId, syncEnabled, onSyncComplete }: Props) {
  const [activeJobId, setActiveJobId] = useState<number | null>(null)
  const [showModal, setShowModal] = useState(false)

  const triggerMutation = useTriggerSync()
  const { data: job } = useSyncJob(activeJobId, showModal)

  // When job completes, notify parent
  useEffect(() => {
    if (job && (job.status === 'completed' || job.status === 'failed')) {
      onSyncComplete?.()
    }
  }, [job?.status])

  const handleSync = async () => {
    try {
      const result = await triggerMutation.mutateAsync(connectionId)
      setActiveJobId(result.job_id)
      setShowModal(true)
    } catch {
      // Error handled by mutation hook
    }
  }

  const isRunning = triggerMutation.isLoading || job?.status === 'running'

  return (
    <>
      <Button
        onClick={handleSync}
        disabled={!syncEnabled || isRunning}
        variant="outline"
        size="sm"
      >
        <RefreshCw className={`w-3.5 h-3.5 mr-1 ${isRunning ? 'animate-spin' : ''}`} />
        {isRunning ? 'Syncing...' : 'Sync Now'}
      </Button>

      {/* Sync Progress Modal */}
      <AnimatePresence>
        {showModal && activeJobId && (
          <SyncProgressModal
            job={job ?? null}
            isTriggering={triggerMutation.isLoading}
            onClose={() => {
              setShowModal(false)
              setActiveJobId(null)
            }}
          />
        )}
      </AnimatePresence>
    </>
  )
}

// ---------------------------------------------------------------------------
// Progress modal
// ---------------------------------------------------------------------------

function SyncProgressModal({
  job,
  isTriggering,
  onClose,
}: {
  job: SyncJob | null
  isTriggering: boolean
  onClose: () => void
}) {
  const isRunning = isTriggering || job?.status === 'running'
  const isCompleted = job?.status === 'completed'
  const isFailed = job?.status === 'failed'

  return (
    <motion.div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      onClick={onClose}
    >
      <motion.div
        className="glass-card p-6 w-full max-w-md mx-4"
        initial={{ scale: 0.95, opacity: 0 }}
        animate={{ scale: 1, opacity: 1 }}
        exit={{ scale: 0.95, opacity: 0 }}
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-semibold">
            {isRunning ? 'Syncing Documents...' : isCompleted ? 'Sync Complete' : 'Sync Failed'}
          </h3>
          <Button variant="ghost" size="icon" className="h-8 w-8" onClick={onClose}>
            <X className="w-4 h-4" />
          </Button>
        </div>

        {/* Status icon */}
        <div className="flex justify-center mb-4">
          {isRunning ? (
            <div className="w-16 h-16 rounded-full bg-primary/10 flex items-center justify-center">
              <RefreshCw className="w-8 h-8 text-primary animate-spin" />
            </div>
          ) : isCompleted ? (
            <div className="w-16 h-16 rounded-full bg-success/10 flex items-center justify-center">
              <CheckCircle2 className="w-8 h-8 text-success" />
            </div>
          ) : (
            <div className="w-16 h-16 rounded-full bg-destructive/10 flex items-center justify-center">
              <XCircle className="w-8 h-8 text-destructive" />
            </div>
          )}
        </div>

        {/* Progress bar */}
        {isRunning && (
          <div className="w-full bg-secondary rounded-full h-2 mb-4">
            <div className="bg-gradient-to-r from-orange-500 to-red-500 h-2 rounded-full animate-pulse w-2/3" />
          </div>
        )}

        {/* Stats */}
        {job && (
          <div className="grid grid-cols-3 gap-3 mb-4">
            <div className="text-center p-2 rounded-lg bg-secondary/30">
              <div className="flex items-center justify-center gap-1 text-success mb-1">
                <FileText className="w-3.5 h-3.5" />
              </div>
              <div className="text-lg font-bold">{job.files_synced}</div>
              <div className="text-xs text-muted-foreground">Synced</div>
            </div>
            <div className="text-center p-2 rounded-lg bg-secondary/30">
              <div className="flex items-center justify-center gap-1 text-warning mb-1">
                <SkipForward className="w-3.5 h-3.5" />
              </div>
              <div className="text-lg font-bold">{job.files_skipped}</div>
              <div className="text-xs text-muted-foreground">Skipped</div>
            </div>
            <div className="text-center p-2 rounded-lg bg-secondary/30">
              <div className="flex items-center justify-center gap-1 text-destructive mb-1">
                <AlertTriangle className="w-3.5 h-3.5" />
              </div>
              <div className="text-lg font-bold">{job.files_errored}</div>
              <div className="text-xs text-muted-foreground">Errors</div>
            </div>
          </div>
        )}

        {/* Chunks created */}
        {job && job.total_chunks_created > 0 && (
          <div className="text-center text-sm text-muted-foreground mb-4">
            {job.total_chunks_created} vector chunks created
          </div>
        )}

        {/* Error message */}
        {isFailed && job?.error_message && (
          <div className="p-3 rounded-lg bg-destructive/10 border border-destructive/20 text-sm text-destructive mb-4">
            {job.error_message}
          </div>
        )}

        {/* Time info */}
        {job?.started_at && (
          <div className="text-xs text-muted-foreground text-center">
            Started: {new Date(job.started_at).toLocaleTimeString()}
            {job.completed_at && ` | Completed: ${new Date(job.completed_at).toLocaleTimeString()}`}
          </div>
        )}

        {/* Close button for completed/failed */}
        {!isRunning && (
          <Button className="w-full mt-4" variant="outline" onClick={onClose}>
            Close
          </Button>
        )}
      </motion.div>
    </motion.div>
  )
}
