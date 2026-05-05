'use client'

/**
 * FileExplorer — Recursive file tree for workspace browsing
 * PRD-66 Phase 1: Code Viewer Widget
 */

import { useCallback, useState } from 'react'
import {
  ChevronRight,
  ChevronDown,
  Folder,
  FolderOpen,
  FileCode,
  FileText,
  FileJson,
  File as FileIcon,
  Loader2,
} from 'lucide-react'
import { cn } from '@/lib/utils'
import type { WorkspaceFileEntry } from '../types'

interface FileExplorerProps {
  entries: WorkspaceFileEntry[]
  isLoading?: boolean
  error?: string | null
  onFileSelect: (path: string) => void
  onDirectoryToggle: (path: string) => void
  selectedPath?: string | null
}

export function FileExplorer({
  entries,
  isLoading,
  error,
  onFileSelect,
  onDirectoryToggle,
  selectedPath,
}: FileExplorerProps) {
  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-full p-4">
        <Loader2 className="h-5 w-5 animate-spin text-muted-foreground" />
      </div>
    )
  }

  if (error) {
    return (
      <div className="p-3 text-xs text-destructive">
        {error}
      </div>
    )
  }

  if (entries.length === 0) {
    return (
      <div className="p-3 text-xs text-muted-foreground">
        No files in workspace
      </div>
    )
  }

  return (
    <div className="py-1 overflow-y-auto h-full text-xs select-none">
      {entries.map((entry) => (
        <TreeNode
          key={entry.path}
          entry={entry}
          depth={0}
          onFileSelect={onFileSelect}
          onDirectoryToggle={onDirectoryToggle}
          selectedPath={selectedPath}
        />
      ))}
    </div>
  )
}

// ---------------------------------------------------------------------------
// TreeNode — recursive node component
// ---------------------------------------------------------------------------

interface TreeNodeProps {
  entry: WorkspaceFileEntry
  depth: number
  onFileSelect: (path: string) => void
  onDirectoryToggle: (path: string) => void
  selectedPath?: string | null
}

function TreeNode({ entry, depth, onFileSelect, onDirectoryToggle, selectedPath }: TreeNodeProps) {
  const [expanded, setExpanded] = useState(false)
  const isDir = entry.type === 'directory'
  const isSelected = entry.path === selectedPath

  const handleClick = useCallback(() => {
    if (isDir) {
      const willExpand = !expanded
      setExpanded(willExpand)
      if (willExpand) {
        onDirectoryToggle(entry.path)
      }
    } else {
      onFileSelect(entry.path)
    }
  }, [isDir, expanded, entry.path, onFileSelect, onDirectoryToggle])

  const Icon = getFileIcon(entry)

  return (
    <>
      <div
        className={cn(
          'flex items-center gap-1 px-2 py-[3px] cursor-pointer',
          'hover:bg-muted/50 transition-colors',
          isSelected && 'bg-primary/10 text-primary'
        )}
        style={{ paddingLeft: `${depth * 14 + 8}px` }}
        onClick={handleClick}
      >
        {/* Chevron for directories */}
        {isDir ? (
          entry.isLoading ? (
            <Loader2 className="h-3.5 w-3.5 animate-spin text-muted-foreground flex-shrink-0" />
          ) : expanded ? (
            <ChevronDown className="h-3.5 w-3.5 text-muted-foreground flex-shrink-0" />
          ) : (
            <ChevronRight className="h-3.5 w-3.5 text-muted-foreground flex-shrink-0" />
          )
        ) : (
          <span className="w-3.5 flex-shrink-0" />
        )}

        {/* File/folder icon */}
        <Icon className={cn('h-3.5 w-3.5 flex-shrink-0', isDir ? 'text-warning' : 'text-muted-foreground')} />

        {/* Name */}
        <span className="truncate">{entry.name}</span>
      </div>

      {/* Children (expanded directories) */}
      {isDir && expanded && entry.children && (
        <>
          {entry.children.map((child) => (
            <TreeNode
              key={child.path}
              entry={child}
              depth={depth + 1}
              onFileSelect={onFileSelect}
              onDirectoryToggle={onDirectoryToggle}
              selectedPath={selectedPath}
            />
          ))}
        </>
      )}
    </>
  )
}

// ---------------------------------------------------------------------------
// Icon helpers
// ---------------------------------------------------------------------------

function getFileIcon(entry: WorkspaceFileEntry) {
  if (entry.type === 'directory') {
    return Folder
  }

  const ext = entry.name.split('.').pop()?.toLowerCase()

  switch (ext) {
    case 'py':
    case 'js':
    case 'jsx':
    case 'ts':
    case 'tsx':
    case 'rs':
    case 'go':
    case 'java':
    case 'c':
    case 'cpp':
    case 'rb':
    case 'php':
    case 'swift':
    case 'kt':
    case 'sh':
    case 'bash':
      return FileCode
    case 'json':
    case 'yaml':
    case 'yml':
    case 'toml':
      return FileJson
    case 'md':
    case 'txt':
    case 'csv':
    case 'log':
      return FileText
    default:
      return FileIcon
  }
}
