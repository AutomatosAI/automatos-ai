'use client'

/**
 * The New mission form's file upload zone and the list of files dropped on it. Moved out of
 * create-mission-modal.tsx; the markup is unchanged.
 */

import { Loader2, Upload, X, FileText, Paperclip } from 'lucide-react'
import { Label } from '@/components/ui/label'
import { cn } from '@/lib/utils'
import { formatSize, type UploadingFile } from './create-mission-options'
import type { MissionAttachments } from './use-mission-attachments'

function rowClass(status: UploadingFile['status']): string {
  if (status === 'error') return 'border-destructive/30 bg-destructive/5'
  if (status === 'done') return 'border-[hsl(var(--success))]/30 bg-[hsl(var(--success))]/5'
  return 'border-border bg-secondary/10'
}

function AttachmentRow({ f, onRemove }: { f: UploadingFile; onRemove: () => void }) {
  return (
    <div className={cn('flex items-center gap-2 rounded-md border px-2.5 py-1.5 text-xs', rowClass(f.status))}>
      <FileText className="w-3.5 h-3.5 text-muted-foreground shrink-0" />
      <span className="truncate flex-1">{f.file.name}</span>
      <span className="text-muted-foreground shrink-0">
        {formatSize(f.file.size)}
      </span>
      {f.status === 'uploading' && (
        <Loader2 className="w-3 h-3 animate-spin text-primary shrink-0" />
      )}
      {f.status === 'error' && (
        <span className="text-destructive text-[10px] shrink-0">Failed</span>
      )}
      <button
        type="button"
        onClick={(e) => {
          e.stopPropagation()
          onRemove()
        }}
        className="text-muted-foreground hover:text-foreground shrink-0"
      >
        <X className="w-3 h-3" />
      </button>
    </div>
  )
}

/* File upload zone */
export function AttachmentsField({ uploads }: { uploads: MissionAttachments }) {
  const { getRootProps, getInputProps, isDragActive } = uploads.dropzone
  return (
    <div className="space-y-2">
      <Label className="flex items-center gap-1.5">
        <Paperclip className="w-3.5 h-3.5" />
        Attachments
        <span className="text-muted-foreground font-normal">(optional)</span>
      </Label>
      <div
        {...getRootProps()}
        className={cn(
          'border-2 border-dashed rounded-lg p-4 text-center cursor-pointer transition-colors',
          isDragActive
            ? 'border-primary bg-primary/5'
            : 'border-muted-foreground/20 hover:border-muted-foreground/40',
        )}
      >
        <input {...getInputProps()} />
        <Upload className="w-5 h-5 mx-auto mb-1.5 text-muted-foreground" />
        <p className="text-xs text-muted-foreground">
          {isDragActive
            ? 'Drop files here...'
            : 'Drop files or click to browse'}
        </p>
        <p className="text-[10px] text-muted-foreground/60 mt-1">
          PDF, Markdown, Text, Word, JSON, CSV, Excel (max 20MB each)
        </p>
      </div>

      {/* File list */}
      {uploads.files.length > 0 && (
        <div className="space-y-2">
          {uploads.files.map((f, i) => (
            <AttachmentRow key={i} f={f} onRemove={() => uploads.removeFile(i)} />
          ))}
        </div>
      )}
    </div>
  )
}
