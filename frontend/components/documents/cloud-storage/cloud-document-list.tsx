'use client'

import React, { useState } from 'react'
import { motion } from 'framer-motion'
import {
  FileText,
  Cloud,
  CheckCircle2,
  Clock,
  AlertTriangle,
  Search,
  Filter,
  RefreshCw,
  Database,
} from 'lucide-react'
import { Input } from '@/components/ui/input'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu'
import {
  useCloudFiles,
  useSyncStatus,
  type CloudFile,
} from '@/hooks/use-cloud-documents-api'

const STATUS_STYLES: Record<string, string> = {
  synced: 'bg-success/10 text-success border-success/20',
  pending: 'bg-secondary/50 text-muted-foreground border-border/30',
  error: 'bg-destructive/10 text-destructive border-destructive/20',
  processing: 'bg-warning/10 text-warning border-warning/20',
}

const STATUS_ICONS: Record<string, React.ElementType> = {
  synced: CheckCircle2,
  pending: Clock,
  error: AlertTriangle,
  processing: RefreshCw,
}

function formatFileSize(bytes: number): string {
  if (bytes >= 1024 * 1024 * 1024) return `${(bytes / (1024 * 1024 * 1024)).toFixed(1)} GB`
  if (bytes >= 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
  if (bytes >= 1024) return `${(bytes / 1024).toFixed(1)} KB`
  return `${bytes} B`
}

interface Props {
  connectionId: number
  rootPath: string
  appName: string
}

export function CloudDocumentList({ connectionId, rootPath, appName }: Props) {
  const [searchTerm, setSearchTerm] = useState('')
  const [statusFilter, setStatusFilter] = useState<string | null>(null)

  const { data: files = [], isLoading } = useCloudFiles(connectionId, rootPath)
  const { data: syncStatus } = useSyncStatus(connectionId)

  const filteredFiles = files.filter((f) => {
    const matchesSearch =
      f.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
      f.path.toLowerCase().includes(searchTerm.toLowerCase())
    const matchesStatus = !statusFilter || f.sync_status === statusFilter
    return matchesSearch && matchesStatus
  })

  return (
    <div className="space-y-4">
      {/* Sync summary bar */}
      {syncStatus && (
        <motion.div
          className="grid grid-cols-5 gap-3"
          initial={{ opacity: 0, y: 8 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.4 }}
        >
          {[
            { label: 'Total', value: syncStatus.total_documents, color: 'text-info' },
            { label: 'Synced', value: syncStatus.synced, color: 'text-success' },
            { label: 'Pending', value: syncStatus.pending, color: 'text-muted-foreground' },
            { label: 'Errors', value: syncStatus.errors, color: 'text-destructive' },
            { label: 'Chunks', value: syncStatus.total_chunks, color: 'text-agent' },
          ].map((stat) => (
            <div key={stat.label} className="text-center p-2 rounded-lg bg-secondary/30">
              <div className={`text-lg font-bold ${stat.color}`}>{stat.value}</div>
              <div className="text-xs text-muted-foreground">{stat.label}</div>
            </div>
          ))}
        </motion.div>
      )}

      {/* Search + Filters */}
      <div className="flex flex-col sm:flex-row gap-3">
        <div className="relative flex-1">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" />
          <Input
            placeholder="Search files..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="pl-10 bg-secondary/50 border-secondary focus:border-primary/50"
          />
        </div>
        <DropdownMenu>
          <DropdownMenuTrigger asChild>
            <Button variant="outline" className="shrink-0">
              <Filter className="w-4 h-4 mr-2" />
              {statusFilter ? statusFilter : 'All Status'}
            </Button>
          </DropdownMenuTrigger>
          <DropdownMenuContent>
            <DropdownMenuItem onClick={() => setStatusFilter(null)}>
              All
            </DropdownMenuItem>
            <DropdownMenuItem onClick={() => setStatusFilter('synced')}>
              Synced
            </DropdownMenuItem>
            <DropdownMenuItem onClick={() => setStatusFilter('pending')}>
              Pending
            </DropdownMenuItem>
            <DropdownMenuItem onClick={() => setStatusFilter('error')}>
              Errors
            </DropdownMenuItem>
          </DropdownMenuContent>
        </DropdownMenu>
      </div>

      {/* File table */}
      {isLoading ? (
        <div className="flex items-center justify-center py-12">
          <RefreshCw className="w-5 h-5 animate-spin text-muted-foreground mr-2" />
          <span className="text-muted-foreground">Loading files...</span>
        </div>
      ) : filteredFiles.length === 0 ? (
        <div className="text-center py-12">
          <FileText className="w-10 h-10 text-muted-foreground mx-auto mb-3" />
          <p className="text-muted-foreground">
            {searchTerm || statusFilter ? 'No files match your filters' : 'No files found in this folder'}
          </p>
        </div>
      ) : (
        <div className="border rounded-lg overflow-hidden">
          {/* Header */}
          <div className="grid grid-cols-12 gap-2 px-4 py-2 text-xs font-medium text-muted-foreground bg-secondary/30 border-b">
            <div className="col-span-5">File</div>
            <div className="col-span-2">Status</div>
            <div className="col-span-1 text-right">Size</div>
            <div className="col-span-2 text-right">Chunks</div>
            <div className="col-span-2 text-right">Last Synced</div>
          </div>

          {/* Rows */}
          {filteredFiles.map((file, idx) => {
            const StatusIcon = STATUS_ICONS[file.sync_status] ?? Clock
            return (
              <motion.div
                key={file.external_file_id}
                className="grid grid-cols-12 gap-2 px-4 py-2.5 items-center text-sm border-b last:border-b-0 hover:bg-secondary/20 transition-colors"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ duration: 0.2, delay: idx * 0.02 }}
              >
                <div className="col-span-5 flex items-center gap-2 min-w-0">
                  <FileText className="w-4 h-4 shrink-0 text-muted-foreground" />
                  <span className="truncate">{file.name}</span>
                </div>
                <div className="col-span-2">
                  <Badge className={`text-xs ${STATUS_STYLES[file.sync_status] ?? STATUS_STYLES.pending}`}>
                    <StatusIcon className="w-3 h-3 mr-1" />
                    {file.sync_status}
                  </Badge>
                </div>
                <div className="col-span-1 text-right text-xs text-muted-foreground">
                  {formatFileSize(file.size)}
                </div>
                <div className="col-span-2 text-right">
                  {file.chunk_count > 0 ? (
                    <span className="flex items-center justify-end gap-1 text-xs">
                      <Database className="w-3 h-3 text-agent" />
                      {file.chunk_count}
                    </span>
                  ) : (
                    <span className="text-xs text-muted-foreground">-</span>
                  )}
                </div>
                <div className="col-span-2 text-right text-xs text-muted-foreground">
                  {file.last_synced_at
                    ? new Date(file.last_synced_at).toLocaleDateString()
                    : '-'}
                </div>
              </motion.div>
            )
          })}
        </div>
      )}

      {/* Source badge */}
      <div className="flex items-center gap-2 text-xs text-muted-foreground">
        <Cloud className="w-3.5 h-3.5" />
        <span>Source: {appName}</span>
        {rootPath !== '/' && <span>| Path: {rootPath}</span>}
        <span>| {filteredFiles.length} files</span>
      </div>
    </div>
  )
}
