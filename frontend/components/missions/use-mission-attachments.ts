'use client'

/**
 * The New mission form's attachments: the drop zone, each file's upload as it is dropped, and the
 * list the form shows and sends. Moved out of create-mission-modal.tsx.
 */

import { useCallback, useState } from 'react'
import { useDropzone, type FileRejection } from 'react-dropzone'
import { toast } from 'sonner'
import { apiClient } from '@/lib/api-client'
import {
  ACCEPT_MAP,
  MAX_FILE_SIZE,
  type MissionAttachment,
  type UploadingFile,
} from './create-mission-options'

// PRD-127: Use ephemeral attachment upload instead of document upload
async function uploadFile(file: File): Promise<MissionAttachment> {
  const result = await apiClient.uploadAttachment(file)
  return {
    attachment_id: result.attachment_id,
    filename: result.filename,
    mime: result.mime,
    media_type: result.media_type,
  }
}

function markDone(files: UploadingFile[], file: File, attachment: MissionAttachment): UploadingFile[] {
  return files.map((f) =>
    f.file === file ? { ...f, status: 'done' as const, attachment } : f,
  )
}

function markFailed(files: UploadingFile[], file: File, error: string): UploadingFile[] {
  return files.map((f) =>
    f.file === file
      ? { ...f, status: 'error' as const, error }
      : f,
  )
}

function toastRejections(rejections: FileRejection[]) {
  rejections.forEach((r) => {
    const msg = r.errors.map((e) => e.message).join(', ')
    toast.error(`${r.file.name}: ${msg}`)
  })
}

export function useMissionAttachments() {
  const [files, setFiles] = useState<UploadingFile[]>([])

  const onDrop = useCallback((acceptedFiles: File[]) => {
    const newFiles: UploadingFile[] = acceptedFiles.map((file) => ({
      file,
      status: 'uploading' as const,
    }))
    setFiles((prev) => [...prev, ...newFiles])

    // Upload each file
    acceptedFiles.forEach((file) => {
      uploadFile(file)
        .then((attachment) => setFiles((prev) => markDone(prev, file, attachment)))
        .catch((err) => {
          setFiles((prev) => markFailed(prev, file, err.message))
          toast.error(`Failed to upload ${file.name}: ${err.message}`)
        })
    })
  }, [])

  const removeFile = useCallback((index: number) => {
    setFiles((prev) => prev.filter((_, i) => i !== index))
  }, [])

  const clearFiles = useCallback(() => setFiles([]), [])

  const dropzone = useDropzone({
    onDrop,
    accept: ACCEPT_MAP,
    maxSize: MAX_FILE_SIZE,
    onDropRejected: toastRejections,
  })

  const attachments = files
    .filter((f) => f.status === 'done' && f.attachment)
    .map((f) => f.attachment!)

  return {
    files,
    attachments,
    isUploading: files.some((f) => f.status === 'uploading'),
    removeFile,
    clearFiles,
    dropzone,
  }
}

export type MissionAttachments = ReturnType<typeof useMissionAttachments>
