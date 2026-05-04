'use client'

import React, { useState } from 'react'
import {
  AlertTriangle,
  Trash2,
  Database,
  RefreshCw,
} from 'lucide-react'
import { Button } from '@/components/ui/button'
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogFooter,
} from '@/components/ui/dialog'
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
    <Dialog open={open} onOpenChange={onClose}>
      <DialogContent size="sm">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2 text-destructive">
            <AlertTriangle className="w-5 h-5" />
            Disconnect {appName}
          </DialogTitle>
        </DialogHeader>

        <div className="space-y-4">
          <p className="text-sm text-muted-foreground">
            This will disconnect the {appName} cloud storage provider.
            {totalDocuments > 0 && (
              <> There are <strong className="text-foreground">{totalDocuments}</strong> synced documents.</>
            )}
          </p>

          {/* Vector deletion option */}
          {totalDocuments > 0 && (
            <div className="space-y-3">
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
        </div>

        <DialogFooter>
          <Button
            variant="outline"
            onClick={onClose}
            disabled={disconnectMutation.isLoading}
          >
            Cancel
          </Button>
          <Button
            variant="destructive"
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
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}
