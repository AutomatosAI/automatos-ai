'use client'

/**
 * FileWidget Component for PRD-38.2 Extended Widgets
 *
 * Display file information, preview content, and provide file operations.
 * Switches between single/list/preview modes based on FileWidgetData.mode.
 * - Copy path button copies file.path to clipboard via navigator.clipboard.writeText
 * - Download button triggers onAction callback
 */

import { useState, useCallback } from 'react'
import { File, Download, Copy, Check, FolderOpen } from 'lucide-react'
import { WidgetBase } from '../WidgetBase'
import { registerWidget } from '../registry'
import { FileInfo as FileInfoComponent } from './FileInfo'
import { FileList } from './FileList'
import { FilePreview } from './FilePreview'
import type {
  WidgetBaseProps,
  FileWidgetData,
  FileInfo,
  WidgetDefinition,
} from '../types'
import { toast } from 'sonner'

export function FileWidget({
  id,
  title,
  data,
  metadata,
  isActive,
  isLoading,
  error,
  onClose,
  onMaximize,
  onRefresh,
  onAction,
}: WidgetBaseProps<FileWidgetData>) {
  const [copied, setCopied] = useState(false)
  const [selectedFile, setSelectedFile] = useState<FileInfo | undefined>(
    data.file
  )

  // Copy file path to clipboard using navigator.clipboard.writeText
  const handleCopyPath = useCallback(async () => {
    const path = selectedFile?.path || data.file?.path || data.currentPath
    if (!path) return

    try {
      await navigator.clipboard.writeText(path)
      setCopied(true)
      toast.success('Path copied to clipboard')
      setTimeout(() => setCopied(false), 2000)
    } catch {
      toast.error('Failed to copy path')
    }
  }, [selectedFile, data.file, data.currentPath])

  // Download file — dispatches through onAction callback
  const handleDownload = useCallback(() => {
    const targetFile = selectedFile || data.file
    if (onAction) {
      onAction({
        id: 'file-download',
        label: 'Download',
        handler: async () => {},
      })
    } else {
      toast.info('Download functionality requires backend integration')
    }
  }, [selectedFile, data.file, onAction])

  // Navigate directory (placeholder - requires backend)
  const handleNavigate = useCallback((path: string) => {
    toast.info(`Navigate to ${path} requires backend integration`)
  }, [])

  // Select file from list
  const handleFileSelect = useCallback((file: FileInfo) => {
    setSelectedFile(file)
  }, [])

  // Determine display mode from FileWidgetData.mode
  const mode = data.mode || (data.files ? 'list' : 'single')

  // Display title
  const displayTitle =
    title ||
    selectedFile?.name ||
    data.file?.name ||
    (mode === 'list' ? 'Files' : 'File')

  return (
    <WidgetBase
      title={displayTitle}
      icon={mode === 'list' ? <FolderOpen className="h-4 w-4" /> : <File className="h-4 w-4" />}
      metadata={metadata}
      isActive={isActive}
      isLoading={isLoading}
      error={error}
      onClose={onClose}
      onMaximize={onMaximize}
      onRefresh={onRefresh}
      canCopy
      onCopy={handleCopyPath}
      canDownload
      onDownload={handleDownload}
      customActions={[
        {
          label: copied ? 'Copied!' : 'Copy Path',
          icon: copied ? (
            <Check className="h-4 w-4 mr-2 text-green-500" />
          ) : (
            <Copy className="h-4 w-4 mr-2" />
          ),
          onClick: handleCopyPath,
        },
        {
          label: 'Download',
          icon: <Download className="h-4 w-4 mr-2" />,
          onClick: handleDownload,
        },
      ]}
    >
      <div className="flex flex-col h-full">
        {/* List mode */}
        {mode === 'list' && data.files && (
          <FileList
            files={data.files}
            currentPath={data.currentPath}
            onFileSelect={handleFileSelect}
            onNavigate={handleNavigate}
            className="flex-1"
          />
        )}

        {/* Single file mode */}
        {mode === 'single' && data.file && (
          <div className="flex flex-col h-full">
            {/* File info */}
            <div className="px-3 py-2 border-b border-border/30">
              <FileInfoComponent file={data.file} />
            </div>

            {/* Preview if available */}
            {data.previewContent && (
              <FilePreview
                content={data.previewContent}
                previewType={data.previewType}
                filename={data.file.name}
                className="flex-1"
              />
            )}
          </div>
        )}

        {/* Preview mode */}
        {mode === 'preview' && (
          <div className="flex flex-col h-full">
            {/* File info header */}
            {data.file && (
              <div className="px-3 py-2 border-b border-border/30 bg-muted/20">
                <div className="flex items-center gap-2 text-sm">
                  <File className="h-4 w-4 text-muted-foreground" />
                  <span className="font-medium truncate">{data.file.name}</span>
                  <span className="text-muted-foreground text-xs">
                    {data.file.path}
                  </span>
                </div>
              </div>
            )}

            {/* Preview content */}
            <FilePreview
              content={data.previewContent}
              previewType={data.previewType}
              filename={data.file?.name}
              className="flex-1"
            />
          </div>
        )}

        {/* Selected file overlay for list mode */}
        {mode === 'list' && selectedFile && (
          <div className="absolute inset-0 bg-background flex flex-col">
            <div className="flex items-center justify-between px-3 py-2 border-b border-border/30">
              <button
                className="text-sm text-primary hover:underline"
                onClick={() => setSelectedFile(undefined)}
              >
                ← Back to list
              </button>
            </div>
            <div className="flex-1 p-3 overflow-auto">
              <FileInfoComponent file={selectedFile} />
            </div>
          </div>
        )}
      </div>
    </WidgetBase>
  )
}

/**
 * Widget definition for registry
 */
export const FileWidgetDef: WidgetDefinition<FileWidgetData> = {
  type: 'file',
  displayName: 'File',
  description: 'View file information and preview content',
  icon: File,
  component: FileWidget,
  defaultSize: { width: 5, height: 4 },
  minSize: { width: 3, height: 3 },
  capabilities: ['copyable', 'downloadable', 'refreshable', 'fullscreen'],
}

// Register the widget
registerWidget(FileWidgetDef)
