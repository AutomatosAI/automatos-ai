'use client'

/**
 * FileList Component for PRD-38.2 Extended Widgets
 *
 * Directory listing with file/folder icons, name, size, and date per item.
 * Supports sorting by name, size, or modified date.
 */

import { useState, useMemo } from 'react'
import { ScrollArea } from '@/components/ui/scroll-area'
import { Button } from '@/components/ui/button'
import {
  Folder,
  ChevronUp,
  ArrowUpDown,
} from 'lucide-react'
import { cn, formatFileSize } from '@/lib/utils'
import { getFileIcon } from './FileInfo'
import { format } from 'date-fns'
import type { FileInfo } from '../types'

interface FileListProps {
  files: FileInfo[]
  currentPath?: string
  onFileSelect?: (file: FileInfo) => void
  onNavigate?: (path: string) => void
  className?: string
}

type SortField = 'name' | 'size' | 'modifiedAt'
type SortOrder = 'asc' | 'desc'

/**
 * Format date for compact listing display
 */
function formatListDate(dateStr?: string): string {
  if (!dateStr) return '-'
  try {
    return format(new Date(dateStr), 'MMM d, yyyy')
  } catch {
    return '-'
  }
}

export function FileList({
  files,
  currentPath,
  onFileSelect,
  onNavigate,
  className,
}: FileListProps) {
  const [sortField, setSortField] = useState<SortField>('name')
  const [sortOrder, setSortOrder] = useState<SortOrder>('asc')

  // Sort files (directories first, then by selected field)
  const sortedFiles = useMemo(() => {
    return [...files].sort((a, b) => {
      // Directories always first
      if (a.type === 'directory' && b.type !== 'directory') return -1
      if (a.type !== 'directory' && b.type === 'directory') return 1

      // Sort by field
      let comparison = 0
      switch (sortField) {
        case 'name':
          comparison = a.name.localeCompare(b.name)
          break
        case 'size':
          comparison = a.size - b.size
          break
        case 'modifiedAt': {
          const aDate = a.modifiedAt ? new Date(a.modifiedAt).getTime() : 0
          const bDate = b.modifiedAt ? new Date(b.modifiedAt).getTime() : 0
          comparison = aDate - bDate
          break
        }
      }

      return sortOrder === 'asc' ? comparison : -comparison
    })
  }, [files, sortField, sortOrder])

  // Toggle sort
  const toggleSort = (field: SortField) => {
    if (sortField === field) {
      setSortOrder((prev) => (prev === 'asc' ? 'desc' : 'asc'))
    } else {
      setSortField(field)
      setSortOrder('asc')
    }
  }

  // Navigate to parent directory
  const handleNavigateUp = () => {
    if (!currentPath || !onNavigate) return
    const parentPath = currentPath.split('/').slice(0, -1).join('/') || '/'
    onNavigate(parentPath)
  }

  // Handle file/folder click
  const handleItemClick = (file: FileInfo) => {
    if (file.type === 'directory' && onNavigate) {
      onNavigate(file.path)
    } else if (onFileSelect) {
      onFileSelect(file)
    }
  }

  return (
    <div className={cn('flex flex-col h-full', className)}>
      {/* Path bar */}
      {currentPath && (
        <div className="flex items-center gap-2 px-3 py-2 border-b border-border/30 bg-muted/20">
          {onNavigate && currentPath !== '/' && (
            <Button
              variant="ghost"
              size="icon"
              className="h-7 w-7"
              onClick={handleNavigateUp}
              title="Go up"
            >
              <ChevronUp className="h-4 w-4" />
            </Button>
          )}
          <code className="text-xs text-muted-foreground truncate flex-1">
            {currentPath}
          </code>
        </div>
      )}

      {/* Header with sort options */}
      <div className="flex items-center px-3 py-1.5 border-b border-border/30 text-xs text-muted-foreground">
        <button
          className="flex items-center gap-1 hover:text-foreground flex-1"
          onClick={() => toggleSort('name')}
        >
          Name
          {sortField === 'name' && (
            <ArrowUpDown className="h-3 w-3" />
          )}
        </button>
        <button
          className="flex items-center gap-1 hover:text-foreground w-20 justify-end"
          onClick={() => toggleSort('size')}
        >
          Size
          {sortField === 'size' && (
            <ArrowUpDown className="h-3 w-3" />
          )}
        </button>
        <button
          className="flex items-center gap-1 hover:text-foreground w-28 justify-end"
          onClick={() => toggleSort('modifiedAt')}
        >
          Date
          {sortField === 'modifiedAt' && (
            <ArrowUpDown className="h-3 w-3" />
          )}
        </button>
      </div>

      {/* File list */}
      <ScrollArea className="flex-1">
        {sortedFiles.length === 0 ? (
          <div className="flex items-center justify-center h-32 text-sm text-muted-foreground">
            Empty directory
          </div>
        ) : (
          <div className="divide-y divide-border/30">
            {sortedFiles.map((file) => {
              const FileIcon = file.type === 'directory' ? Folder : getFileIcon(file)
              const isDirectory = file.type === 'directory'

              return (
                <button
                  key={file.path}
                  className={cn(
                    'w-full text-left px-3 py-2 hover:bg-muted/50 transition-colors',
                    'focus:outline-none focus:bg-muted/50',
                    'flex items-center gap-3'
                  )}
                  onClick={() => handleItemClick(file)}
                >
                  <FileIcon
                    className={cn(
                      'h-4 w-4 flex-shrink-0',
                      isDirectory ? 'text-blue-500' : 'text-muted-foreground'
                    )}
                  />
                  <span
                    className={cn(
                      'text-sm truncate flex-1',
                      isDirectory && 'font-medium'
                    )}
                  >
                    {file.name}
                  </span>
                  <span className="text-xs text-muted-foreground w-20 text-right flex-shrink-0">
                    {isDirectory ? '-' : formatFileSize(file.size)}
                  </span>
                  <span className="text-xs text-muted-foreground w-28 text-right flex-shrink-0">
                    {formatListDate(file.modifiedAt)}
                  </span>
                </button>
              )
            })}
          </div>
        )}
      </ScrollArea>

      {/* Footer with count */}
      <div className="px-3 py-1.5 border-t border-border/30 bg-muted/20 text-xs text-muted-foreground">
        {files.filter((f) => f.type === 'directory').length} folders,{' '}
        {files.filter((f) => f.type === 'file').length} files
      </div>
    </div>
  )
}
