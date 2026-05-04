'use client'

import React, { useState, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import {
  ArrowLeft,
  Folder,
  File,
  RefreshCw,
  Settings,
  CheckCircle,
  Clock,
  XCircle,
  Loader2,
  ChevronRight,
  Search,
  AlertCircle
} from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Input } from '@/components/ui/input'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import { useCloudFolders, useCloudFiles } from '@/hooks/use-cloud-storage'

interface CloudFile {
  external_file_id: string
  name: string
  path: string
  mime_type?: string
  size?: number
  modified_at?: string
  is_synced: boolean
  sync_status: 'pending' | 'syncing' | 'synced' | 'error'
  chunk_count: number
  last_synced_at?: string
}

interface CloudFolder {
  name: string
  path: string
  has_children: boolean
}

interface ProviderBrowserProps {
  providerName: string
  providerType: string
  connectionId: number
  rootFolder?: string
  onBack: () => void
  onSync: (path?: string) => void
  onSelectRootFolder: (path: string) => void
}

export function ProviderBrowser({
  providerName,
  providerType,
  connectionId,
  rootFolder,
  onBack,
  onSync,
  onSelectRootFolder
}: ProviderBrowserProps) {
  const [currentPath, setCurrentPath] = useState(rootFolder || '/')
  const [currentFolderName, setCurrentFolderName] = useState<string>('Root')
  const [pathHistory, setPathHistory] = useState<Array<{name: string, path: string}>>([{name: 'Root', path: '/'}])
  const [syncing, setSyncing] = useState(false)
  const [searchTerm, setSearchTerm] = useState('')
  const [filterStatus, setFilterStatus] = useState<string>('all')

  // Use react-query hooks to fetch folders and files
  const { data: folders = [], isLoading: foldersLoading, error: foldersError } = useCloudFolders(connectionId, currentPath)
  const { data: files = [], isLoading: filesLoading, error: filesError } = useCloudFiles(connectionId, currentPath, false)

  const loading = foldersLoading || filesLoading

  // Log when rootFolder prop changes
  useEffect(() => {
    console.log('[ProviderBrowser] rootFolder prop changed:', rootFolder)
  }, [rootFolder])

  console.log('[ProviderBrowser] Data:', {
    connectionId,
    currentPath,
    rootFolder,
    folders: folders.length,
    files: files.length,
    loading,
    foldersError,
    filesError
  })

  const handleFolderClick = (folder: CloudFolder) => {
    setCurrentPath(folder.path)
    setCurrentFolderName(folder.name)
    setPathHistory([...pathHistory, {name: folder.name, path: folder.path}])
  }

  const handleSelectAsRoot = async () => {
    console.log('[ProviderBrowser] Selecting root folder:', currentPath)
    try {
      await onSelectRootFolder(currentPath)
      console.log('[ProviderBrowser] Root folder selected successfully')
    } catch (error) {
      console.error('[ProviderBrowser] Error selecting root folder:', error)
    }
  }

  const handleSyncNow = async () => {
    setSyncing(true)
    try {
      await onSync(currentPath)
      // React-query will auto-refresh the files
    } catch (error) {
      console.error('Sync error:', error)
    } finally {
      setSyncing(false)
    }
  }

  const navigateUp = () => {
    if (pathHistory.length > 1) {
      const newHistory = pathHistory.slice(0, -1)
      const parent = newHistory[newHistory.length - 1]
      setPathHistory(newHistory)
      setCurrentPath(parent.path)
      setCurrentFolderName(parent.name)
    }
  }

  const navigateToRoot = () => {
    setCurrentPath('/')
    setCurrentFolderName('Root')
    setPathHistory([{name: 'Root', path: '/'}])
  }

  const filteredFiles = files.filter(file => {
    const matchesSearch = file.name.toLowerCase().includes(searchTerm.toLowerCase())
    const matchesFilter =
      filterStatus === 'all' ||
      (filterStatus === 'synced' && file.is_synced) ||
      (filterStatus === 'pending' && !file.is_synced)
    return matchesSearch && matchesFilter
  })

  const syncStats = {
    total: files.length,
    synced: files.filter(f => f.is_synced).length,
    pending: files.filter(f => !f.is_synced).length,
    chunks: files.reduce((sum, f) => sum + f.chunk_count, 0)
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-4">
          <Button
            variant="ghost"
            size="icon"
            onClick={onBack}
            className="hover:bg-secondary"
          >
            <ArrowLeft className="w-5 h-5" />
          </Button>
          <div>
            <h2 className="text-2xl font-bold">{providerName}</h2>
            <p className="text-sm text-muted-foreground mt-1">
              Browse and sync documents from your cloud storage
            </p>
          </div>
        </div>
        <div className="flex items-center gap-2">
          {currentPath !== '/' && (
            <Button
              variant="outline"
              size="sm"
              onClick={navigateToRoot}
            >
              <Folder className="w-4 h-4 mr-2" />
              Back to Root
            </Button>
          )}
          <Button
            variant="outline"
            size="sm"
            onClick={() => {
              // React-query will auto-refresh on focus/mount
              window.location.reload()
            }}
            disabled={loading}
          >
            <RefreshCw className={`w-4 h-4 mr-2 ${loading ? 'animate-spin' : ''}`} />
            Refresh
          </Button>
          <Button
            onClick={handleSyncNow}
            disabled={syncing || !rootFolder || rootFolder === '/'}
            className="gradient-accent"
            title={!rootFolder || rootFolder === '/' ? 'Select a specific folder first' : 'Sync all documents from root folder'}
          >
            {syncing ? (
              <>
                <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                Syncing...
              </>
            ) : (
              <>
                <RefreshCw className="w-4 h-4 mr-2" />
                {rootFolder && rootFolder !== '/' ? 'Sync Folder' : 'Select Folder First'}
              </>
            )}
          </Button>
        </div>
      </div>

      {/* Stats Cards */}
      <div className="grid grid-cols-4 gap-4">
        <Card className="glass-card">
          <CardContent className="p-4">
            <div className="text-2xl font-bold">{syncStats.total}</div>
            <div className="text-xs text-muted-foreground">Total Files</div>
          </CardContent>
        </Card>
        <Card className="glass-card">
          <CardContent className="p-4">
            <div className="text-2xl font-bold text-success">{syncStats.synced}</div>
            <div className="text-xs text-muted-foreground">Synced</div>
          </CardContent>
        </Card>
        <Card className="glass-card">
          <CardContent className="p-4">
            <div className="text-2xl font-bold text-orange-400">{syncStats.pending}</div>
            <div className="text-xs text-muted-foreground">Pending</div>
          </CardContent>
        </Card>
        <Card className="glass-card">
          <CardContent className="p-4">
            <div className="text-2xl font-bold text-agent">{syncStats.chunks.toLocaleString()}</div>
            <div className="text-xs text-muted-foreground">Chunks</div>
          </CardContent>
        </Card>
      </div>

      {/* Folder Selection Notice & Current Path */}
      {!rootFolder || rootFolder === '/' ? (
        <Card className="glass-card border-orange-500/30 bg-orange-500/5">
          <CardContent className="p-4">
            <div className="flex items-start gap-3">
              <div className="w-10 h-10 rounded-full bg-orange-500/10 flex items-center justify-center shrink-0">
                <AlertCircle className="w-5 h-5 text-orange-400" />
              </div>
              <div className="flex-1">
                <h4 className="font-semibold text-orange-400 mb-1">Select a Folder to Sync</h4>
                <p className="text-sm text-muted-foreground mb-3">
                  Navigate to the folder you want to sync, then click "Set as Root Folder" to select it.
                  This prevents syncing your entire {providerName} which may take a long time.
                </p>
                <div className="flex items-center gap-2 mb-3">
                  <Folder className="w-4 h-4 text-muted-foreground" />
                  <span className="text-xs text-muted-foreground">Path:</span>
                  <div className="flex items-center gap-1">
                    {pathHistory.map((item, index) => (
                      <React.Fragment key={index}>
                        {index > 0 && <ChevronRight className="w-3 h-3 text-muted-foreground" />}
                        <button
                          onClick={() => {
                            const newHistory = pathHistory.slice(0, index + 1)
                            setPathHistory(newHistory)
                            setCurrentPath(item.path)
                            setCurrentFolderName(item.name)
                          }}
                          className="text-xs text-primary hover:underline"
                        >
                          {item.name}
                        </button>
                      </React.Fragment>
                    ))}
                  </div>
                </div>
                {currentPath !== '/' && (
                  <Button
                    onClick={handleSelectAsRoot}
                    className="gradient-accent"
                    size="sm"
                  >
                    <Settings className="w-4 h-4 mr-2" />
                    Set "{currentFolderName}" as Root Folder
                  </Button>
                )}
              </div>
            </div>
          </CardContent>
        </Card>
      ) : (
        <Card className="glass-card border-success/30 bg-success/5">
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div className="flex items-start gap-3 flex-1">
                <div className="w-10 h-10 rounded-full bg-success/10 flex items-center justify-center shrink-0">
                  <CheckCircle className="w-5 h-5 text-success" />
                </div>
                <div className="flex-1">
                  <h4 className="font-semibold text-success mb-1">Root Folder Selected</h4>
                  <div className="flex items-center gap-2 mb-2">
                    <Folder className="w-4 h-4 text-muted-foreground" />
                    <span className="text-xs text-muted-foreground">Path:</span>
                    <div className="flex items-center gap-1">
                      {pathHistory.map((item, index) => (
                        <React.Fragment key={index}>
                          {index > 0 && <ChevronRight className="w-3 h-3 text-muted-foreground" />}
                          <button
                            onClick={() => {
                              const newHistory = pathHistory.slice(0, index + 1)
                              setPathHistory(newHistory)
                              setCurrentPath(item.path)
                              setCurrentFolderName(item.name)
                            }}
                            className="text-xs text-primary hover:underline"
                          >
                            {item.name}
                          </button>
                        </React.Fragment>
                      ))}
                    </div>
                  </div>
                  <div className="flex items-center gap-2 text-sm text-success">
                    <CheckCircle className="w-4 h-4" />
                    <span>Will sync all files from this folder and its subfolders</span>
                  </div>
                </div>
              </div>
              <div className="flex items-center gap-2 ml-4">
                {currentPath !== '/' && (
                  <Button variant="outline" size="sm" onClick={navigateUp}>
                    <ArrowLeft className="w-4 h-4 mr-2" />
                    Up
                  </Button>
                )}
                {currentPath !== rootFolder && (
                  <Button
                    onClick={handleSelectAsRoot}
                    className="gradient-accent"
                    size="sm"
                  >
                    <Settings className="w-4 h-4 mr-2" />
                    Set as Root Folder
                  </Button>
                )}
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Search and Filter */}
      <div className="flex gap-4">
        <div className="relative flex-1">
          <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-muted-foreground" />
          <Input
            placeholder="Search files..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="pl-10 bg-secondary/50"
          />
        </div>
        <Select value={filterStatus} onValueChange={setFilterStatus}>
          <SelectTrigger className="w-[180px] bg-secondary/50">
            <SelectValue placeholder="Filter by status" />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="all">All Files</SelectItem>
            <SelectItem value="synced">Synced Only</SelectItem>
            <SelectItem value="pending">Pending Only</SelectItem>
          </SelectContent>
        </Select>
      </div>

      {/* Folders List */}
      {folders.length > 0 && (
        <Card className="glass-card">
          <CardHeader>
            <CardTitle className="text-lg flex items-center gap-2">
              <Folder className="w-5 h-5 text-info" />
              Folders
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-2">
              {folders.map((folder, index) => (
                <motion.div
                  key={folder.path}
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ duration: 0.3, delay: index * 0.05 }}
                  className="flex items-center justify-between p-3 rounded-lg bg-secondary/30 hover:bg-secondary/50 cursor-pointer transition-colors group"
                  onClick={() => handleFolderClick(folder)}
                >
                  <div className="flex items-center gap-3">
                    <Folder className="w-5 h-5 text-info" />
                    <span className="font-medium">{folder.name}</span>
                  </div>
                  <ChevronRight className="w-5 h-5 text-muted-foreground group-hover:text-primary transition-colors" />
                </motion.div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}

      {/* Files List */}
      <Card className="glass-card">
        <CardHeader>
          <CardTitle className="text-lg flex items-center gap-2">
            <File className="w-5 h-5 text-orange-400" />
            Files ({filteredFiles.length})
          </CardTitle>
        </CardHeader>
        <CardContent>
          {loading ? (
            <div className="text-center py-12">
              <Loader2 className="w-8 h-8 animate-spin mx-auto mb-4 text-primary" />
              <p className="text-muted-foreground">Loading files...</p>
            </div>
          ) : filteredFiles.length === 0 ? (
            <div className="text-center py-12">
              <File className="w-12 h-12 text-muted-foreground mx-auto mb-4" />
              <p className="text-muted-foreground">
                {searchTerm ? 'No files match your search' : 'No files in this folder'}
              </p>
            </div>
          ) : (
            <div className="space-y-2">
              <AnimatePresence>
                {filteredFiles.map((file, index) => (
                  <motion.div
                    key={file.external_file_id}
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, y: -20 }}
                    transition={{ duration: 0.3, delay: index * 0.05 }}
                    className="flex items-center justify-between p-4 rounded-lg bg-secondary/30 hover:bg-secondary/50 transition-colors"
                  >
                    <div className="flex items-center gap-3 flex-1 min-w-0">
                      <File className="w-5 h-5 text-orange-400 shrink-0" />
                      <div className="flex-1 min-w-0">
                        <p className="font-medium truncate">{file.name}</p>
                        <div className="flex items-center gap-4 text-xs text-muted-foreground mt-1">
                          <span>{file.mime_type}</span>
                          {file.size && (
                            <span>{(file.size / 1024).toFixed(1)} KB</span>
                          )}
                          {file.modified_at && (
                            <span>{new Date(file.modified_at).toLocaleDateString()}</span>
                          )}
                        </div>
                      </div>
                    </div>
                    <div className="flex items-center gap-3">
                      {file.is_synced ? (
                        <>
                          <Badge variant="outline" className="text-xs bg-success/10 text-success border-success/20">
                            <CheckCircle className="w-3 h-3 mr-1" />
                            {file.chunk_count} chunks
                          </Badge>
                          {file.last_synced_at && (
                            <span className="text-xs text-muted-foreground">
                              {new Date(file.last_synced_at).toLocaleTimeString()}
                            </span>
                          )}
                        </>
                      ) : file.sync_status === 'error' ? (
                        <Badge variant="destructive" className="text-xs">
                          <XCircle className="w-3 h-3 mr-1" />
                          Error
                        </Badge>
                      ) : (
                        <Badge variant="outline" className="text-xs">
                          <Clock className="w-3 h-3 mr-1" />
                          Pending
                        </Badge>
                      )}
                    </div>
                  </motion.div>
                ))}
              </AnimatePresence>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  )
}
