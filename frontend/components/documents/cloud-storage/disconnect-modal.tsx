'use client'

import React, { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import {
  AlertTriangle,
  Trash2,
  Database,
  X,
  RefreshCw,
} from 'lucide-react'
import { Button } from '@/components/ui/button'
import { useDisconnectConnection } from '@/hooks/use-cloud-documents-api'

interface Props {
  open: boolean
  connectionId: number
  appName: string
  totalDocuments: number
  onClose: () => void
  onDisconnected?: () => void
}

export function DisconnectModal({
  open,
  connectionId,
  appName,
  totalDocuments,
  onClose,
  onDisconnected,
}: Props) {
  const [deleteVectors, setDeleteVectors] = useState(false)
  const disconnectMutation = useDisconnectConnection()

  const handleDisconnect = async () => {
    await disconnectMutation.mutateAsync({ connectionId, deleteVectors })
    onDisconnected?.()
    onClose()
  }

  return (
    <AnimatePresence>
      {open && (
        <motion.div
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-sm"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          onClick={onClose}
        >
          <motion.div
            className="glass-card card-glow p-6 w-full max-w-md mx-4"
            initial={{ scale: 0.95, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            exit={{ scale: 0.95, opacity: 0 }}
            onClick={(e) => e.stopPropagation()}
          >
            {/* Header */}
            <div className="flex items-center justify-between mb-4">
              <div className="flex items-center gap-2 text-red-400">
                <AlertTriangle className="w-5 h-5" />
                <h3 className="text-lg font-semibold">Disconnect {appName}</h3>
              </div>
              <Button variant="ghost" size="icon" className="h-8 w-8" onClick={onClose}>
                <X className="w-4 h-4" />
              </Button>
            </div>

            <p className="text-sm text-muted-foreground mb-4">
              This will disconnect the {appName} cloud storage provider.
              {totalDocuments > 0 && (
                <> There are <strong className="text-foreground">{totalDocuments}</strong> synced documents.</>
              )}
            </p>

            {/* Vector deletion option */}
            {totalDocuments > 0 && (
              <div className="space-y-3 mb-6">
                <button
                  className={`w-full p-3 rounded-lg border text-left transition-colors text-sm ${
                    !deleteVectors
                      ? 'border-primary/40 bg-primary/5'
                      : 'border-border hover:border-primary/20'
                  }`}
                  onClick={() => setDeleteVectors(false)}
                >
                  <div className="flex items-center gap-2 font-medium mb-1">
                    <Database className="w-4 h-4 text-green-400" />
                    Keep vector embeddings
                  </div>
                  <div className="text-xs text-muted-foreground">
                    Documents stay searchable via RAG. You can reconnect later.
                  </div>
                </button>

                <button
                  className={`w-full p-3 rounded-lg border text-left transition-colors text-sm ${
                    deleteVectors
                      ? 'border-red-500/40 bg-red-500/5'
                      : 'border-border hover:border-red-500/20'
                  }`}
                  onClick={() => setDeleteVectors(true)}
                >
                  <div className="flex items-center gap-2 font-medium mb-1 text-red-400">
                    <Trash2 className="w-4 h-4" />
                    Delete all vectors
                  </div>
                  <div className="text-xs text-muted-foreground">
                    Permanently removes {totalDocuments} documents and their vector embeddings from S3.
                  </div>
                </button>
              </div>
            )}

            {/* Actions */}
            <div className="flex gap-3">
              <Button
                variant="outline"
                className="flex-1"
                onClick={onClose}
                disabled={disconnectMutation.isLoading}
              >
                Cancel
              </Button>
              <Button
                variant="destructive"
                className="flex-1"
                onClick={handleDisconnect}
                disabled={disconnectMutation.isLoading}
              >
                {disconnectMutation.isLoading ? (
                  <>
                    <RefreshCw className="w-4 h-4 mr-2 animate-spin" />
                    Disconnecting...
                  </>
                ) : (
                  <>
                    <AlertTriangle className="w-4 h-4 mr-2" />
                    Disconnect
                  </>
                )}
              </Button>
            </div>
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  )
}
