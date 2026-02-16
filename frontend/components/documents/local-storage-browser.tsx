'use client'

import React, { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import {
  ArrowLeft,
  File,
  FileText,
  Search,
  Upload,
  Download,
  Trash2,
  Eye,
  CheckCircle,
  Clock,
  MoreVertical
} from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Input } from '@/components/ui/input'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu'
import { ViewToggle } from '@/components/shared/view-toggle'
import { useViewMode } from '@/hooks/use-view-mode'

interface LocalDocument {
  id: number
  filename: string
  file_type?: string
  file_size?: number
  status?: string
  chunk_count?: number
  upload_date?: string
}

interface LocalStorageBrowserProps {
  documents: LocalDocument[]
  onBack: () => void
  onUpload: () => void
  onViewDetails: (id: number) => void
  onDownload: (id: number, filename: string) => void
  onDelete: (id: number, filename: string) => void
}

const typeIcons: Record<string, any> = {
  pdf: FileText,
  docx: FileText,
  md: File,
  txt: File,
  xlsx: FileText,
  csv: FileText,
  json: File,
  xml: File
}

export function LocalStorageBrowser({
  documents,
  onBack,
  onUpload,
  onViewDetails,
  onDownload,
  onDelete
}: LocalStorageBrowserProps) {
  const [viewMode, setViewMode] = useViewMode('documents')
  const [searchTerm, setSearchTerm] = useState('')
  const [filterStatus, setFilterStatus] = useState<string>('all')

  const filteredDocuments = documents.filter(doc => {
    const matchesSearch = doc.filename.toLowerCase().includes(searchTerm.toLowerCase())
    const matchesFilter =
      filterStatus === 'all' ||
      (doc.status || 'completed').toLowerCase() === filterStatus
    return matchesSearch && matchesFilter
  })

  const stats = {
    total: documents.length,
    processed: documents.filter(d => (d.status || 'completed').toLowerCase() === 'completed').length,
    chunks: documents.reduce((sum, d) => sum + (d.chunk_count || 0), 0),
    size: documents.reduce((sum, d) => sum + (d.file_size || 0), 0)
  }

  const formatSize = (bytes: number) => {
    if (bytes >= 1024 * 1024 * 1024) return `${(bytes / (1024*1024*1024)).toFixed(1)}GB`
    if (bytes >= 1024 * 1024) return `${(bytes / (1024*1024)).toFixed(1)}MB`
    if (bytes >= 1024) return `${(bytes / 1024).toFixed(1)}KB`
    return `${bytes}B`
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
            <h2 className="text-2xl font-bold">Automatos Storage</h2>
            <p className="text-sm text-muted-foreground mt-1">
              Documents uploaded directly to Automatos
            </p>
          </div>
        </div>
      </div>

      {/* Search + View Toggle */}
      <div className="flex items-center gap-3">
        <div className="relative flex-1">
          <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-muted-foreground" />
          <Input
            placeholder="Search files..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="pl-10 bg-secondary/50"
          />
        </div>
        <ViewToggle value={viewMode} onChange={setViewMode} />
      </div>

      {/* Files List */}
      <Card className="glass-card">
        <CardHeader>
          <CardTitle className="text-lg flex items-center gap-2">
            <File className="w-5 h-5 text-orange-400" />
            Documents ({filteredDocuments.length})
          </CardTitle>
        </CardHeader>
        <CardContent>
          {filteredDocuments.length === 0 ? (
            <div className="text-center py-12">
              <File className="w-12 h-12 text-muted-foreground mx-auto mb-4" />
              <p className="text-muted-foreground">
                {searchTerm ? 'No files match your search' : 'No documents uploaded yet'}
              </p>
              {!searchTerm && (
                <Button onClick={onUpload} className="mt-4">
                  <Upload className="w-4 h-4 mr-2" />
                  Upload Your First Document
                </Button>
              )}
            </div>
          ) : viewMode === 'list' ? (
            <div className="space-y-2">
              <AnimatePresence>
                {filteredDocuments.map((doc, index) => {
                  const fileType = (doc.file_type || 'unknown').toLowerCase()
                  const TypeIcon = typeIcons[fileType] || File
                  const status = (doc.status || 'completed').toLowerCase()

                  return (
                    <motion.div
                      key={doc.id}
                      initial={{ opacity: 0, y: 20 }}
                      animate={{ opacity: 1, y: 0 }}
                      exit={{ opacity: 0, y: -20 }}
                      transition={{ duration: 0.3, delay: index * 0.05 }}
                      className="flex items-center justify-between p-4 rounded-lg bg-secondary/30 hover:bg-secondary/50 transition-colors"
                    >
                      <div className="flex items-center gap-3 flex-1 min-w-0">
                        <TypeIcon className="w-5 h-5 text-orange-400 shrink-0" />
                        <div className="flex-1 min-w-0">
                          <p className="font-medium truncate">{doc.filename}</p>
                          <div className="flex items-center gap-4 text-xs text-muted-foreground mt-1">
                            <span>{doc.file_type?.toUpperCase()}</span>
                            {doc.file_size && <span>{formatSize(doc.file_size)}</span>}
                            {doc.upload_date && (
                              <span>{new Date(doc.upload_date).toLocaleDateString()}</span>
                            )}
                          </div>
                        </div>
                      </div>
                      <div className="flex items-center gap-3">
                        <Badge
                          variant={status === 'completed' ? 'outline' : 'secondary'}
                          className={`text-xs ${
                            status === 'completed'
                              ? 'bg-green-500/10 text-green-400 border-green-500/20'
                              : ''
                          }`}
                        >
                          {status === 'completed' ? (
                            <CheckCircle className="w-3 h-3 mr-1" />
                          ) : (
                            <Clock className="w-3 h-3 mr-1" />
                          )}
                          {doc.chunk_count || 0} chunks
                        </Badge>

                        <DropdownMenu>
                          <DropdownMenuTrigger asChild>
                            <Button variant="ghost" size="icon" className="h-8 w-8">
                              <MoreVertical className="w-4 h-4" />
                            </Button>
                          </DropdownMenuTrigger>
                          <DropdownMenuContent align="end">
                            <DropdownMenuItem onClick={() => onViewDetails(doc.id)}>
                              <Eye className="w-4 h-4 mr-2" />
                              View Details
                            </DropdownMenuItem>
                            <DropdownMenuItem onClick={() => onDownload(doc.id, doc.filename)}>
                              <Download className="w-4 h-4 mr-2" />
                              Download
                            </DropdownMenuItem>
                            <DropdownMenuItem
                              className="text-red-400"
                              onClick={() => onDelete(doc.id, doc.filename)}
                            >
                              <Trash2 className="w-4 h-4 mr-2" />
                              Delete
                            </DropdownMenuItem>
                          </DropdownMenuContent>
                        </DropdownMenu>
                      </div>
                    </motion.div>
                  )
                })}
              </AnimatePresence>
            </div>
          ) : (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
              <AnimatePresence>
                {filteredDocuments.map((doc, index) => {
                  const fileType = (doc.file_type || 'unknown').toLowerCase()
                  const TypeIcon = typeIcons[fileType] || File
                  const status = (doc.status || 'completed').toLowerCase()

                  return (
                    <motion.div
                      key={doc.id}
                      initial={{ opacity: 0, y: 20 }}
                      animate={{ opacity: 1, y: 0 }}
                      exit={{ opacity: 0, y: -20 }}
                      transition={{ duration: 0.3, delay: index * 0.05 }}
                      className="glass-card p-4 card-glow hover:border-primary/20 transition-all duration-300 cursor-pointer"
                      onClick={() => onViewDetails(doc.id)}
                    >
                      <div className="flex items-start justify-between mb-3">
                        <TypeIcon className="w-8 h-8 text-orange-400" />
                        <DropdownMenu>
                          <DropdownMenuTrigger asChild>
                            <Button variant="ghost" size="icon" className="h-7 w-7" onClick={(e) => e.stopPropagation()}>
                              <MoreVertical className="w-4 h-4" />
                            </Button>
                          </DropdownMenuTrigger>
                          <DropdownMenuContent align="end">
                            <DropdownMenuItem onClick={() => onViewDetails(doc.id)}>
                              <Eye className="w-4 h-4 mr-2" />
                              View Details
                            </DropdownMenuItem>
                            <DropdownMenuItem onClick={() => onDownload(doc.id, doc.filename)}>
                              <Download className="w-4 h-4 mr-2" />
                              Download
                            </DropdownMenuItem>
                            <DropdownMenuItem
                              className="text-red-400"
                              onClick={() => onDelete(doc.id, doc.filename)}
                            >
                              <Trash2 className="w-4 h-4 mr-2" />
                              Delete
                            </DropdownMenuItem>
                          </DropdownMenuContent>
                        </DropdownMenu>
                      </div>
                      <p className="font-semibold text-sm truncate mb-1">{doc.filename}</p>
                      <div className="flex items-center gap-2 text-xs text-muted-foreground mb-3">
                        <span>{doc.file_type?.toUpperCase()}</span>
                        {doc.file_size && (
                          <>
                            <span>&middot;</span>
                            <span>{formatSize(doc.file_size)}</span>
                          </>
                        )}
                      </div>
                      <div className="flex items-center justify-between">
                        <Badge
                          variant={status === 'completed' ? 'outline' : 'secondary'}
                          className={`text-[10px] ${
                            status === 'completed'
                              ? 'bg-green-500/10 text-green-400 border-green-500/20'
                              : ''
                          }`}
                        >
                          {status === 'completed' ? (
                            <CheckCircle className="w-2.5 h-2.5 mr-0.5" />
                          ) : (
                            <Clock className="w-2.5 h-2.5 mr-0.5" />
                          )}
                          {doc.chunk_count || 0} chunks
                        </Badge>
                        {doc.upload_date && (
                          <span className="text-[10px] text-muted-foreground">
                            {new Date(doc.upload_date).toLocaleDateString()}
                          </span>
                        )}
                      </div>
                    </motion.div>
                  )
                })}
              </AnimatePresence>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  )
}
