'use client'

import React, { useState } from 'react'
import { motion } from 'framer-motion'
import {
  Cloud,
  FolderTree,
  FileText,
  RefreshCw,
  ArrowLeft,
} from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import {
  CloudStorageConnections,
  FolderNavigator,
  SyncButton,
  CloudDocumentList,
  DisconnectModal,
} from './cloud-storage'
import { useCloudConnections, type CloudConnection } from '@/hooks/use-cloud-documents-api'

type View = 'connections' | 'manage'

export function CloudStoragePanel() {
  const [view, setView] = useState<View>('connections')
  const [selectedConnection, setSelectedConnection] = useState<CloudConnection | null>(null)
  const [disconnectTarget, setDisconnectTarget] = useState<CloudConnection | null>(null)
  const [showFolderPicker, setShowFolderPicker] = useState(false)

  const { refetch } = useCloudConnections()

  const handleSelect = (conn: CloudConnection) => {
    setSelectedConnection(conn)
    setView('manage')
  }

  const handleBack = () => {
    setSelectedConnection(null)
    setView('connections')
    setShowFolderPicker(false)
  }

  return (
    <div className="space-y-6">
      {/* Connections grid view */}
      {view === 'connections' && (
        <motion.div
          initial={{ opacity: 0, y: 12 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.4 }}
        >
          <Card className="glass-card">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Cloud className="w-5 h-5 text-primary" />
                Cloud Storage Connections
              </CardTitle>
            </CardHeader>
            <CardContent>
              <CloudStorageConnections
                onSelect={handleSelect}
                onDisconnect={(conn) => setDisconnectTarget(conn)}
                selectedConnectionId={selectedConnection?.id}
              />
            </CardContent>
          </Card>
        </motion.div>
      )}

      {/* Manage view for a selected connection */}
      {view === 'manage' && selectedConnection && (
        <motion.div
          className="space-y-6"
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.3 }}
        >
          {/* Header bar */}
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <Button variant="ghost" size="icon" onClick={handleBack}>
                <ArrowLeft className="w-4 h-4" />
              </Button>
              <div>
                <h2 className="text-lg font-semibold flex items-center gap-2">
                  <Cloud className="w-5 h-5 text-primary" />
                  {selectedConnection.app_name}
                </h2>
                <p className="text-sm text-muted-foreground">
                  {selectedConnection.root_folder_path
                    ? `Root: ${selectedConnection.root_folder_path}`
                    : 'No root folder selected'}
                </p>
              </div>
            </div>

            <div className="flex items-center gap-2">
              <Button
                variant="outline"
                size="sm"
                onClick={() => setShowFolderPicker(!showFolderPicker)}
              >
                <FolderTree className="w-3.5 h-3.5 mr-1" />
                {selectedConnection.root_folder_path ? 'Change Folder' : 'Select Folder'}
              </Button>
              <SyncButton
                connectionId={selectedConnection.id}
                syncEnabled={selectedConnection.sync_enabled}
                onSyncComplete={() => refetch()}
              />
            </div>
          </div>

          {/* Folder picker (conditionally shown) */}
          {showFolderPicker && (
            <Card className="glass-card">
              <CardHeader>
                <CardTitle className="flex items-center gap-2 text-base">
                  <FolderTree className="w-4 h-4 text-primary" />
                  Select Sync Root Folder
                </CardTitle>
              </CardHeader>
              <CardContent>
                <FolderNavigator
                  connectionId={selectedConnection.id}
                  currentRootPath={selectedConnection.root_folder_path}
                  onFolderSelected={() => {
                    setShowFolderPicker(false)
                    refetch()
                  }}
                />
              </CardContent>
            </Card>
          )}

          {/* File list */}
          {selectedConnection.root_folder_path ? (
            <Card className="glass-card">
              <CardHeader>
                <CardTitle className="flex items-center gap-2 text-base">
                  <FileText className="w-4 h-4 text-primary" />
                  Synced Documents
                </CardTitle>
              </CardHeader>
              <CardContent>
                <CloudDocumentList
                  connectionId={selectedConnection.id}
                  rootPath={selectedConnection.root_folder_path}
                  appName={selectedConnection.app_name}
                />
              </CardContent>
            </Card>
          ) : (
            <Card className="glass-card">
              <CardContent className="py-12">
                <div className="text-center">
                  <FolderTree className="w-10 h-10 text-muted-foreground mx-auto mb-3" />
                  <h3 className="font-semibold mb-1">Select a root folder</h3>
                  <p className="text-sm text-muted-foreground mb-4">
                    Choose which folder to sync from {selectedConnection.app_name}
                  </p>
                  <Button
                    variant="outline"
                    onClick={() => setShowFolderPicker(true)}
                  >
                    <FolderTree className="w-4 h-4 mr-2" />
                    Browse Folders
                  </Button>
                </div>
              </CardContent>
            </Card>
          )}
        </motion.div>
      )}

      {/* Disconnect Modal */}
      {disconnectTarget && (
        <DisconnectModal
          open={!!disconnectTarget}
          connectionId={disconnectTarget.id}
          appName={disconnectTarget.app_name}
          totalDocuments={disconnectTarget.total_documents_synced}
          onClose={() => setDisconnectTarget(null)}
          onDisconnected={() => {
            setDisconnectTarget(null)
            if (selectedConnection?.id === disconnectTarget.id) {
              handleBack()
            }
            refetch()
          }}
        />
      )}
    </div>
  )
}
