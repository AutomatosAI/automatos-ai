'use client'

import React, { useState } from 'react'
import {
  Folder,
  FolderOpen,
  ChevronRight,
  ChevronDown,
  Check,
  RefreshCw,
} from 'lucide-react'
import { Button } from '@/components/ui/button'
import {
  useCloudFolders,
  useSelectFolder,
  type CloudFolder,
} from '@/hooks/use-cloud-documents-api'

interface FolderNodeProps {
  connectionId: number
  folder: CloudFolder
  depth: number
  selectedPath: string | null
  onSelect: (path: string) => void
}

function FolderNode({ connectionId, folder, depth, selectedPath, onSelect }: FolderNodeProps) {
  const [expanded, setExpanded] = useState(false)
  const isSelected = selectedPath === folder.path

  const {
    data: children = [],
    isLoading,
  } = useCloudFolders(connectionId, folder.path, expanded && folder.has_children)

  return (
    <div>
      <div
        className={`flex items-center gap-1 px-2 py-1.5 rounded-md cursor-pointer transition-colors text-sm ${
          isSelected
            ? 'bg-primary/10 text-primary border border-primary/20'
            : 'hover:bg-secondary/60'
        }`}
        style={{ paddingLeft: `${depth * 16 + 8}px` }}
        onClick={() => {
          onSelect(folder.path)
          if (folder.has_children) setExpanded(!expanded)
        }}
      >
        {/* Expand icon */}
        {folder.has_children ? (
          expanded ? (
            <ChevronDown className="w-3.5 h-3.5 shrink-0 text-muted-foreground" />
          ) : (
            <ChevronRight className="w-3.5 h-3.5 shrink-0 text-muted-foreground" />
          )
        ) : (
          <span className="w-3.5" />
        )}

        {/* Folder icon */}
        {expanded ? (
          <FolderOpen className="w-4 h-4 shrink-0 text-primary" />
        ) : (
          <Folder className="w-4 h-4 shrink-0 text-muted-foreground" />
        )}

        <span className="truncate">{folder.name}</span>

        {isSelected && <Check className="w-3.5 h-3.5 ml-auto text-primary" />}
      </div>

      {/* Children */}
      {expanded && (
        <div>
          {isLoading ? (
            <div
              className="flex items-center gap-1 px-2 py-1 text-xs text-muted-foreground"
              style={{ paddingLeft: `${(depth + 1) * 16 + 8}px` }}
            >
              <RefreshCw className="w-3 h-3 animate-spin" />
              Loading...
            </div>
          ) : (
            children.map((child) => (
              <FolderNode
                key={child.path}
                connectionId={connectionId}
                folder={child}
                depth={depth + 1}
                selectedPath={selectedPath}
                onSelect={onSelect}
              />
            ))
          )}
        </div>
      )}
    </div>
  )
}

interface Props {
  connectionId: number
  currentRootPath: string | null
  onFolderSelected?: () => void
}

export function FolderNavigator({ connectionId, currentRootPath, onFolderSelected }: Props) {
  const [selectedPath, setSelectedPath] = useState<string | null>(currentRootPath)

  const { data: rootFolders = [], isLoading } = useCloudFolders(connectionId, '/')
  const selectFolderMutation = useSelectFolder()

  const handleConfirm = async () => {
    if (!selectedPath) return
    await selectFolderMutation.mutateAsync({
      connectionId,
      rootFolderPath: selectedPath,
    })
    onFolderSelected?.()
  }

  return (
    <div className="space-y-4">
      <div className="text-sm text-muted-foreground">
        Select a root folder to sync. All files within this folder will be indexed.
      </div>

      {/* Tree */}
      <div className="border rounded-lg max-h-72 overflow-y-auto p-2 bg-secondary/20">
        {isLoading ? (
          <div className="flex items-center justify-center py-8 text-muted-foreground">
            <RefreshCw className="w-4 h-4 animate-spin mr-2" />
            Loading folders...
          </div>
        ) : rootFolders.length === 0 ? (
          <div className="text-center py-8 text-muted-foreground text-sm">
            No folders found. The cloud drive may be empty.
          </div>
        ) : (
          rootFolders.map((folder) => (
            <FolderNode
              key={folder.path}
              connectionId={connectionId}
              folder={folder}
              depth={0}
              selectedPath={selectedPath}
              onSelect={setSelectedPath}
            />
          ))
        )}
      </div>

      {/* Selected path display */}
      {selectedPath && (
        <div className="text-xs text-muted-foreground">
          Selected: <span className="text-foreground font-medium">{selectedPath}</span>
        </div>
      )}

      {/* Confirm button */}
      <Button
        onClick={handleConfirm}
        disabled={!selectedPath || selectFolderMutation.isLoading}
        className="w-full gradient-accent hover:opacity-90"
      >
        {selectFolderMutation.isLoading ? (
          <>
            <RefreshCw className="w-4 h-4 mr-2 animate-spin" />
            Setting root folder...
          </>
        ) : (
          <>
            <Check className="w-4 h-4 mr-2" />
            Set as Sync Root
          </>
        )}
      </Button>
    </div>
  )
}
