'use client'

/**
 * PRD-127 attachments for the create-task dialog: upload, remove, and the bar
 * that lists them. Split out of CreateTaskDialog (PRD-252), which had grown past
 * the component size limit; the behaviour is unchanged.
 */

import { useCallback, useRef, useState } from 'react'
import { toast } from 'sonner'
import { Paperclip, X, FileText, Image } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { apiClient } from '@/lib/api-client'

export interface AttachmentMeta {
  attachment_id: string
  filename: string
  mime: string
  media_type: 'image' | 'document'
}

const ACCEPTED_FILES = 'image/*,.pdf,.doc,.docx,.xls,.xlsx,.txt,.csv,.md,.json,.py,.js,.ts,.tsx'

export function useTaskAttachments() {
  const fileInputRef = useRef<HTMLInputElement>(null)
  const [attachments, setAttachments] = useState<AttachmentMeta[]>([])
  const [isUploading, setIsUploading] = useState(false)

  const select = useCallback(async (event: React.ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(event.target.files || [])
    if (files.length === 0) return

    setIsUploading(true)
    try {
      const results = await Promise.all(files.map((file) => apiClient.uploadAttachment(file)))
      setAttachments((prev) => [
        ...prev,
        ...results.map((r) => ({
          attachment_id: r.attachment_id,
          filename: r.filename,
          mime: r.mime,
          media_type: r.media_type,
        })),
      ])
      toast.success(`Uploaded ${files.length} file${files.length === 1 ? '' : 's'}`)
    } catch (error: unknown) {
      toast.error(error instanceof Error ? error.message : 'Upload failed')
    } finally {
      setIsUploading(false)
      if (fileInputRef.current) fileInputRef.current.value = ''
    }
  }, [])

  const remove = useCallback(async (attachmentId: string) => {
    try {
      await apiClient.deleteAttachment(attachmentId)
    } catch {
      // Ignore delete errors — just remove from UI
    }
    setAttachments((prev) => prev.filter((a) => a.attachment_id !== attachmentId))
  }, [])

  const reset = useCallback(() => setAttachments([]), [])

  return { fileInputRef, attachments, isUploading, select, remove, reset }
}

export type TaskAttachments = ReturnType<typeof useTaskAttachments>

/** The hidden file input the Attach button opens. */
export function AttachmentInput({ files }: { files: TaskAttachments }) {
  return (
    <input
      ref={files.fileInputRef}
      type="file"
      className="hidden"
      multiple
      accept={ACCEPTED_FILES}
      onChange={files.select}
    />
  )
}

/** Attach, and a chip per attached file with its remove button. */
export function AttachmentBar({ files }: { files: TaskAttachments }) {
  return (
    <div className="flex items-center gap-2 pb-2 border-b border-border/50">
      <Button
        type="button"
        variant="ghost"
        size="sm"
        className="h-8 gap-1.5 text-muted-foreground hover:text-foreground"
        disabled={files.isUploading}
        onClick={() => files.fileInputRef.current?.click()}
      >
        <Paperclip className="w-4 h-4" />
        <span className="text-xs">Attach</span>
      </Button>
      {files.isUploading && <span className="text-xs text-muted-foreground animate-pulse">Uploading...</span>}
      {files.attachments.length > 0 && (
        <div className="flex flex-wrap gap-1.5">
          {files.attachments.map((att) => (
            <div
              key={att.attachment_id}
              className="inline-flex items-center gap-1.5 rounded-full border border-border bg-muted/30 px-2.5 py-0.5 text-xs"
            >
              {att.media_type === 'image' ? (
                <Image className="w-3 h-3 text-info" />
              ) : (
                <FileText className="w-3 h-3 text-warning" />
              )}
              <span className="max-w-[120px] truncate">{att.filename}</span>
              <button
                type="button"
                onClick={() => files.remove(att.attachment_id)}
                className="text-muted-foreground hover:text-destructive"
              >
                <X className="w-3 h-3" />
              </button>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
