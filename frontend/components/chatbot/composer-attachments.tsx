'use client'

/**
 * The chat composer's attachments (PRD-127: ephemeral attachments): the hidden file picker, the upload of what the
 * member picks, and the strip above the box that shows each uploaded file (removable) and each file still uploading.
 * MultimodalInput (multimodal-input.tsx) holds them through useComposerAttachments and sends their ids with the
 * message.
 */
import { type ChangeEvent, type RefObject, useRef, useState } from 'react'
import { toast } from 'sonner'

import { apiClient } from '@/lib/api-client'

/** An uploaded file, as the message carries it. */
export interface ComposerAttachment {
  attachment_id: string
  filename: string
  media_type: 'image' | 'document'
}

/** The files the picker offers. */
const ACCEPTED_FILES = 'image/*,.pdf,.doc,.docx,.xls,.xlsx,.txt,.csv,.md,.json,.py,.js,.ts,.tsx'
const UPLOAD_FAILED = 'Upload failed'

export interface ComposerAttachments {
  readonly fileInputRef: RefObject<HTMLInputElement>
  /** The names of the files uploading now. */
  readonly uploadQueue: string[]
  readonly uploaded: ComposerAttachment[]
  /** The picker's change: uploads what was picked. */
  readonly upload: (event: ChangeEvent<HTMLInputElement>) => Promise<void>
  readonly remove: (attachmentId: string) => void
  /** Empties the strip once the message is sent. */
  readonly clear: () => void
}

/** Uploads every file (PRD-127: uploadAttachment, not uploadDocument) and returns them as the message carries them. */
async function uploadAll(files: File[]): Promise<ComposerAttachment[]> {
  const results = await Promise.all(files.map((file) => apiClient.uploadAttachment(file)))
  return results.map((r) => ({ attachment_id: r.attachment_id, filename: r.filename, media_type: r.media_type }))
}

function uploadedMessage(count: number): string {
  return `Uploaded ${count} file${count === 1 ? '' : 's'}`
}

/** The composer's attachments: the picker's ref, the uploads under way, and the uploaded files. */
export function useComposerAttachments(): ComposerAttachments {
  const fileInputRef = useRef<HTMLInputElement>(null)
  const [uploadQueue, setUploadQueue] = useState<string[]>([])
  const [uploaded, setUploaded] = useState<ComposerAttachment[]>([])

  const upload = async (event: ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(event.target.files || [])
    if (files.length === 0) return
    setUploadQueue(files.map((f) => f.name))
    try {
      const added = await uploadAll(files)
      setUploaded((prev) => [...prev, ...added])
      toast.success(uploadedMessage(files.length))
    } catch (error: unknown) {
      toast.error(error instanceof Error ? error.message : UPLOAD_FAILED)
    } finally {
      setUploadQueue([])
      // Reset file input so selecting the same file again works
      if (fileInputRef.current) fileInputRef.current.value = ''
    }
  }

  return {
    fileInputRef,
    uploadQueue,
    uploaded,
    upload,
    remove: (attachmentId) => setUploaded((prev) => prev.filter((a) => a.attachment_id !== attachmentId)),
    clear: () => setUploaded([]),
  }
}

/** The hidden file picker; the composer's attach button opens it. */
export function AttachmentPicker({ attachments }: { readonly attachments: ComposerAttachments }) {
  return (
    <input
      ref={attachments.fileInputRef}
      type="file"
      className="hidden"
      multiple
      accept={ACCEPTED_FILES}
      onChange={attachments.upload}
    />
  )
}

/** The uploaded files (each removable) and the files still uploading, above the box; nothing while there are none. */
export function AttachmentStrip({ attachments }: { readonly attachments: ComposerAttachments }) {
  const { uploaded, uploadQueue, remove } = attachments
  if (uploaded.length === 0 && uploadQueue.length === 0) return null
  return (
    <div className="mb-3 flex flex-wrap gap-2">
      {uploaded.map((att) => (
        <div
          key={att.attachment_id}
          className={`inline-flex items-center gap-2 rounded-full border px-3 py-1 text-xs ${
            att.media_type === 'image'
              ? 'border-info/20 bg-info/10 text-info/70'
              : 'border-warning/20 bg-warning/10 text-amber-200'
          }`}
        >
          <span className="truncate max-w-[240px]">{att.filename}</span>
          <button
            type="button"
            className="ml-1 opacity-70 hover:opacity-100"
            onClick={() => remove(att.attachment_id)}
            aria-label="Remove attachment"
          >
            ×
          </button>
        </div>
      ))}

      {uploadQueue.map((name) => (
        <div
          key={name}
          className="inline-flex items-center gap-2 rounded-full border border-primary/20 bg-primary/10 px-3 py-1 text-xs text-primary"
        >
          <span className="truncate max-w-[240px]">{name}</span>
          <span className="text-primary/70">(uploading…)</span>
        </div>
      ))}
    </div>
  )
}
