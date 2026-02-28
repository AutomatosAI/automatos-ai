'use client'

/**
 * FileInfo Component for PRD-38.2 Extended Widgets
 *
 * Displays file metadata: name, path, human-readable size, mime type,
 * modified date, and file type icon.
 */

import {
  File,
  Folder,
  Clock,
  HardDrive,
  FileCode,
  FileText,
  FileImage,
  FileVideo,
  FileAudio,
  FileArchive,
  FileSpreadsheet,
  Lock,
} from 'lucide-react'
import { cn, formatFileSize } from '@/lib/utils'
import { format } from 'date-fns'
import type { FileInfo as FileInfoType } from '../types'

interface FileInfoProps {
  file: FileInfoType
  className?: string
}

/**
 * Get file icon based on mime type or extension.
 * Maps common mime types and file extensions to lucide-react icons:
 * File, FileText, FileImage, FileCode, Folder, FileArchive, etc.
 */
function getFileIcon(file: FileInfoType) {
  if (file.type === 'directory') {
    return Folder
  }

  const mimeType = file.mimeType?.toLowerCase() || ''
  const ext = file.name.split('.').pop()?.toLowerCase() || ''

  // Code files
  if (
    mimeType.includes('javascript') ||
    mimeType.includes('typescript') ||
    mimeType.includes('python') ||
    mimeType.includes('java') ||
    ['js', 'ts', 'tsx', 'jsx', 'py', 'java', 'go', 'rs', 'c', 'cpp', 'h'].includes(ext)
  ) {
    return FileCode
  }

  // Text files
  if (
    mimeType.includes('text') ||
    ['txt', 'md', 'json', 'yaml', 'yml', 'xml', 'html', 'css'].includes(ext)
  ) {
    return FileText
  }

  // Images
  if (mimeType.includes('image')) {
    return FileImage
  }

  // Video
  if (mimeType.includes('video')) {
    return FileVideo
  }

  // Audio
  if (mimeType.includes('audio')) {
    return FileAudio
  }

  // Archives
  if (
    mimeType.includes('zip') ||
    mimeType.includes('tar') ||
    mimeType.includes('gzip') ||
    ['zip', 'tar', 'gz', 'rar', '7z'].includes(ext)
  ) {
    return FileArchive
  }

  // Spreadsheets
  if (
    mimeType.includes('spreadsheet') ||
    mimeType.includes('excel') ||
    ['xlsx', 'xls', 'csv'].includes(ext)
  ) {
    return FileSpreadsheet
  }

  return File
}

/**
 * Format date string to human-readable format
 */
function formatDate(dateStr: string): string {
  try {
    return format(new Date(dateStr), 'MMM d, yyyy h:mm a')
  } catch {
    return dateStr
  }
}

export function FileInfo({ file, className }: FileInfoProps) {
  const FileIcon = getFileIcon(file)

  return (
    <div className={cn('space-y-3', className)}>
      {/* File name and icon */}
      <div className="flex items-center gap-3">
        <div className="p-2 rounded-lg bg-muted/50">
          <FileIcon className="h-8 w-8 text-muted-foreground" />
        </div>
        <div className="flex-1 min-w-0">
          <h3 className="font-medium truncate">{file.name}</h3>
          <p className="text-sm text-muted-foreground truncate">{file.path}</p>
        </div>
      </div>

      {/* Metadata grid */}
      <div className="grid grid-cols-2 gap-3 text-sm">
        {/* Size */}
        <div className="flex items-center gap-2">
          <HardDrive className="h-4 w-4 text-muted-foreground" />
          <span className="text-muted-foreground">Size:</span>
          <span>{formatFileSize(file.size)}</span>
        </div>

        {/* Mime Type */}
        <div className="flex items-center gap-2">
          <File className="h-4 w-4 text-muted-foreground" />
          <span className="text-muted-foreground">Type:</span>
          <span className="truncate">{file.mimeType || file.type}</span>
        </div>

        {/* Created */}
        {file.createdAt && (
          <div className="flex items-center gap-2 col-span-2">
            <Clock className="h-4 w-4 text-muted-foreground" />
            <span className="text-muted-foreground">Created:</span>
            <span>{formatDate(file.createdAt)}</span>
          </div>
        )}

        {/* Modified */}
        {file.modifiedAt && (
          <div className="flex items-center gap-2 col-span-2">
            <Clock className="h-4 w-4 text-muted-foreground" />
            <span className="text-muted-foreground">Modified:</span>
            <span>{formatDate(file.modifiedAt)}</span>
          </div>
        )}

        {/* Permissions */}
        {file.permissions && (
          <div className="flex items-center gap-2 col-span-2">
            <Lock className="h-4 w-4 text-muted-foreground" />
            <span className="text-muted-foreground">Permissions:</span>
            <code className="text-xs bg-muted px-1 rounded">{file.permissions}</code>
          </div>
        )}
      </div>
    </div>
  )
}

export { getFileIcon, formatFileSize as formatSize }
