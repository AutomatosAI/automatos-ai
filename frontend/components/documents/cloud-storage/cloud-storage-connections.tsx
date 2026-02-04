'use client'

import React from 'react'
import { motion } from 'framer-motion'
import {
  Cloud,
  CheckCircle2,
  XCircle,
  RefreshCw,
  FolderSync,
  FileText,
  Unlink,
} from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import {
  useCloudConnections,
  type CloudConnection,
} from '@/hooks/use-cloud-documents-api'

// Provider display config
const PROVIDER_META: Record<string, { label: string; color: string }> = {
  GOOGLEDRIVE: { label: 'Google Drive', color: 'text-blue-400' },
  DROPBOX: { label: 'Dropbox', color: 'text-sky-400' },
  ONEDRIVE: { label: 'OneDrive', color: 'text-indigo-400' },
  BOX: { label: 'Box', color: 'text-cyan-400' },
  SHAREPOINT: { label: 'SharePoint', color: 'text-purple-400' },
}

function providerLabel(appName: string) {
  return PROVIDER_META[appName.toUpperCase()]?.label ?? appName
}

function providerColor(appName: string) {
  return PROVIDER_META[appName.toUpperCase()]?.color ?? 'text-muted-foreground'
}

interface Props {
  onSelect: (connection: CloudConnection) => void
  onDisconnect: (connection: CloudConnection) => void
  selectedConnectionId?: number | null
}

export function CloudStorageConnections({
  onSelect,
  onDisconnect,
  selectedConnectionId,
}: Props) {
  const { data: connections = [], isLoading, error } = useCloudConnections()

  if (isLoading) {
    return (
      <div className="flex items-center justify-center py-12">
        <RefreshCw className="w-6 h-6 animate-spin text-muted-foreground" />
        <span className="ml-3 text-muted-foreground">Loading connections...</span>
      </div>
    )
  }

  if (error) {
    return (
      <div className="text-center py-12 text-red-400">
        Failed to load cloud connections.
      </div>
    )
  }

  if (connections.length === 0) {
    return (
      <div className="text-center py-12">
        <Cloud className="w-12 h-12 text-muted-foreground mx-auto mb-4" />
        <h3 className="text-lg font-semibold mb-2">No Cloud Storage Connected</h3>
        <p className="text-muted-foreground mb-4">
          Connect Google Drive, Dropbox, OneDrive, or Box via the Tools page to get started.
        </p>
      </div>
    )
  }

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
      {connections.map((conn, idx) => {
        const isSelected = selectedConnectionId === conn.id
        const isActive = conn.status === 'active' || conn.status === 'connected'

        return (
          <motion.div
            key={conn.id}
            className={`glass-card p-5 card-glow cursor-pointer transition-all duration-200 ${
              isSelected ? 'border-primary/40 ring-1 ring-primary/30' : 'hover:border-primary/20'
            }`}
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.4, delay: idx * 0.08 }}
            onClick={() => onSelect(conn)}
          >
            {/* Header */}
            <div className="flex items-center justify-between mb-3">
              <div className="flex items-center gap-2">
                <Cloud className={`w-5 h-5 ${providerColor(conn.app_name)}`} />
                <span className="font-semibold">{providerLabel(conn.app_name)}</span>
              </div>
              {isActive ? (
                <Badge className="bg-green-500/10 text-green-400 border-green-500/20">
                  <CheckCircle2 className="w-3 h-3 mr-1" />
                  Connected
                </Badge>
              ) : (
                <Badge className="bg-red-500/10 text-red-400 border-red-500/20">
                  <XCircle className="w-3 h-3 mr-1" />
                  {conn.status}
                </Badge>
              )}
            </div>

            {/* Stats */}
            <div className="grid grid-cols-2 gap-3 mb-4 text-sm">
              <div>
                <div className="flex items-center gap-1 text-muted-foreground">
                  <FileText className="w-3.5 h-3.5" />
                  <span>Documents</span>
                </div>
                <div className="font-medium">{conn.total_documents_synced}</div>
              </div>
              <div>
                <div className="flex items-center gap-1 text-muted-foreground">
                  <FolderSync className="w-3.5 h-3.5" />
                  <span>Sync</span>
                </div>
                <div className="font-medium">
                  {conn.sync_enabled ? (
                    <span className="text-green-400">Enabled</span>
                  ) : (
                    <span className="text-gray-400">Off</span>
                  )}
                </div>
              </div>
            </div>

            {/* Root folder */}
            {conn.root_folder_path && (
              <div className="text-xs text-muted-foreground truncate mb-3">
                Root: {conn.root_folder_path}
              </div>
            )}

            {/* Last synced */}
            {conn.last_successful_sync && (
              <div className="text-xs text-muted-foreground mb-3">
                Last synced: {new Date(conn.last_successful_sync).toLocaleString()}
              </div>
            )}

            {/* Actions */}
            <div className="flex gap-2">
              <Button
                size="sm"
                variant="outline"
                className="flex-1"
                onClick={(e) => {
                  e.stopPropagation()
                  onSelect(conn)
                }}
              >
                <FolderSync className="w-3.5 h-3.5 mr-1" />
                Manage
              </Button>
              <Button
                size="sm"
                variant="outline"
                className="text-red-400 hover:text-red-300 hover:border-red-500/30"
                onClick={(e) => {
                  e.stopPropagation()
                  onDisconnect(conn)
                }}
              >
                <Unlink className="w-3.5 h-3.5" />
              </Button>
            </div>
          </motion.div>
        )
      })}
    </div>
  )
}
