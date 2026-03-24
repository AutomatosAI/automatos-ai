'use client'

import { useState, useCallback } from 'react'
import { useAuth } from '@clerk/nextjs'
import { useDropzone } from 'react-dropzone'
import { Target, Loader2, Upload, X, FileText, Paperclip } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Textarea } from '@/components/ui/textarea'
import { Label } from '@/components/ui/label'
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
  DialogFooter,
} from '@/components/ui/dialog'
import { useCreateMission } from '@/hooks/use-missions-api'
import { useMissionStore } from '@/stores/mission-store'
import { toast } from 'sonner'
import { useRouter } from 'next/navigation'
import { cn } from '@/lib/utils'

interface MissionAttachment {
  document_id: number
  filename: string
  size: number
  content_type: string
}

interface UploadingFile {
  file: File
  status: 'uploading' | 'done' | 'error'
  attachment?: MissionAttachment
  error?: string
}

const ALLOWED_TYPES: Record<string, string[]> = {
  'application/pdf': ['.pdf'],
  'text/plain': ['.txt'],
  'text/markdown': ['.md'],
  'application/json': ['.json'],
  'text/csv': ['.csv'],
  'application/vnd.openxmlformats-officedocument.wordprocessingml.document': ['.docx'],
  'application/msword': ['.doc'],
  'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': ['.xlsx'],
}

const ACCEPT_MAP = Object.fromEntries(
  Object.entries(ALLOWED_TYPES).map(([mime, exts]) => [mime, exts]),
)

const MAX_FILE_SIZE = 20 * 1024 * 1024 // 20MB

interface CreateMissionModalProps {
  open: boolean
  onOpenChange: (open: boolean) => void
}

export function CreateMissionModal({ open, onOpenChange }: CreateMissionModalProps) {
  const router = useRouter()
  const { getToken } = useAuth()
  const createMission = useCreateMission()
  const setActivePlanningMissionId = useMissionStore((s) => s.setActivePlanningMissionId)

  const [name, setName] = useState('')
  const [description, setDescription] = useState('')
  const [tags, setTags] = useState('')
  const [files, setFiles] = useState<UploadingFile[]>([])
  const [budgetPauseEnabled, setBudgetPauseEnabled] = useState(true)

  const isSubmitting = createMission.isLoading
  const isUploading = files.some((f) => f.status === 'uploading')
  const attachments = files
    .filter((f) => f.status === 'done' && f.attachment)
    .map((f) => f.attachment!)

  const uploadFile = useCallback(async (file: File): Promise<MissionAttachment> => {
    const BACKEND_URL = process.env.NEXT_PUBLIC_API_URL || ''
    const formData = new FormData()
    formData.append('file', file)

    const headers: Record<string, string> = {}
    const workspaceId = typeof window !== 'undefined'
      ? localStorage.getItem('automatos_workspace_id')
      : null
    if (workspaceId) headers['X-Workspace-ID'] = workspaceId

    const token = await getToken()
    if (token) headers['Authorization'] = `Bearer ${token}`

    const res = await fetch(`${BACKEND_URL}/api/missions/upload`, {
      method: 'POST',
      headers,
      body: formData,
    })

    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: 'Upload failed' }))
      throw new Error(err.detail || `Upload failed (${res.status})`)
    }

    return res.json()
  }, [getToken])

  const onDrop = useCallback(
    (acceptedFiles: File[]) => {
      const newFiles: UploadingFile[] = acceptedFiles.map((file) => ({
        file,
        status: 'uploading' as const,
      }))

      setFiles((prev) => [...prev, ...newFiles])

      // Upload each file
      acceptedFiles.forEach((file, i) => {
        uploadFile(file)
          .then((attachment) => {
            setFiles((prev) =>
              prev.map((f) =>
                f.file === file ? { ...f, status: 'done' as const, attachment } : f,
              ),
            )
          })
          .catch((err) => {
            setFiles((prev) =>
              prev.map((f) =>
                f.file === file
                  ? { ...f, status: 'error' as const, error: err.message }
                  : f,
              ),
            )
            toast.error(`Failed to upload ${file.name}: ${err.message}`)
          })
      })
    },
    [uploadFile],
  )

  const removeFile = useCallback((index: number) => {
    setFiles((prev) => prev.filter((_, i) => i !== index))
  }, [])

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: ACCEPT_MAP,
    maxSize: MAX_FILE_SIZE,
    onDropRejected: (rejections) => {
      rejections.forEach((r) => {
        const msg = r.errors.map((e) => e.message).join(', ')
        toast.error(`${r.file.name}: ${msg}`)
      })
    },
  })

  const handleSubmit = () => {
    const goalParts: string[] = []
    if (name.trim()) goalParts.push(name.trim())
    if (description.trim()) goalParts.push(description.trim())

    const goal = goalParts.join(': ')
    if (!goal) {
      toast.error('Please enter a mission name or description')
      return
    }

    // Build config from optional fields
    const config: Record<string, unknown> = {}
    const tagList = tags
      .split(',')
      .map((t) => t.trim())
      .filter(Boolean)
    if (tagList.length > 0) config.tags = tagList
    if (name.trim()) config.name = name.trim()
    if (attachments.length > 0) config.attachments = attachments
    if (!budgetPauseEnabled) config.budget_pause_disabled = true

    createMission.mutate(
      { goal, ...(Object.keys(config).length > 0 ? { config } : {}) },
      {
        onSuccess: (mission) => {
          setActivePlanningMissionId(mission.id)
          toast.success('Mission created — plan is being generated')
          onOpenChange(false)
          resetForm()
          router.push(`/missions/${mission.id}` as any)
        },
        onError: (err) => {
          toast.error(err.message || 'Failed to create mission')
        },
      },
    )
  }

  const resetForm = () => {
    setName('')
    setDescription('')
    setTags('')
    setFiles([])
    setBudgetPauseEnabled(true)
  }

  const formatSize = (bytes: number) => {
    if (bytes < 1024) return `${bytes}B`
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)}KB`
    return `${(bytes / (1024 * 1024)).toFixed(1)}MB`
  }

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-lg max-h-[90vh] overflow-y-auto">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <Target className="w-5 h-5 text-primary" />
            New Mission
          </DialogTitle>
          <DialogDescription>
            Define a goal for your AI workforce. Attach files like PRDs, design docs,
            or data to give agents context.
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-4 py-2">
          <div className="space-y-2">
            <Label htmlFor="mission-name">Mission Name</Label>
            <Input
              id="mission-name"
              placeholder="e.g. Research top AI agent frameworks"
              value={name}
              onChange={(e) => setName(e.target.value)}
              autoFocus
            />
          </div>

          <div className="space-y-2">
            <Label htmlFor="mission-description">Description</Label>
            <Textarea
              id="mission-description"
              placeholder="Describe what you want to accomplish, any constraints, output format..."
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              rows={4}
            />
          </div>

          {/* File upload zone */}
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
            {files.length > 0 && (
              <div className="space-y-1.5">
                {files.map((f, i) => (
                  <div
                    key={i}
                    className={cn(
                      'flex items-center gap-2 rounded-md border px-2.5 py-1.5 text-xs',
                      f.status === 'error'
                        ? 'border-destructive/30 bg-destructive/5'
                        : f.status === 'done'
                          ? 'border-[hsl(var(--success))]/30 bg-[hsl(var(--success))]/5'
                          : 'border-border bg-secondary/10',
                    )}
                  >
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
                        removeFile(i)
                      }}
                      className="text-muted-foreground hover:text-foreground shrink-0"
                    >
                      <X className="w-3 h-3" />
                    </button>
                  </div>
                ))}
              </div>
            )}
          </div>

          {/* Budget pause toggle */}
          <div className="flex items-center justify-between rounded-lg border border-border px-3 py-2.5">
            <div className="space-y-0.5">
              <Label htmlFor="budget-pause" className="text-sm cursor-pointer">
                Pause on budget exceeded
              </Label>
              <p className="text-[11px] text-muted-foreground">
                Pause the mission if token usage exceeds the estimated budget
              </p>
            </div>
            <button
              id="budget-pause"
              type="button"
              role="switch"
              aria-checked={budgetPauseEnabled}
              onClick={() => setBudgetPauseEnabled((v) => !v)}
              className={cn(
                'relative inline-flex h-5 w-9 shrink-0 cursor-pointer rounded-full border-2 border-transparent transition-colors',
                budgetPauseEnabled ? 'bg-primary' : 'bg-muted',
              )}
            >
              <span
                className={cn(
                  'pointer-events-none inline-block h-4 w-4 rounded-full bg-background shadow-lg ring-0 transition-transform',
                  budgetPauseEnabled ? 'translate-x-4' : 'translate-x-0',
                )}
              />
            </button>
          </div>

          <div className="space-y-2">
            <Label htmlFor="mission-tags">Tags</Label>
            <Input
              id="mission-tags"
              placeholder="e.g. research, competitive-analysis, urgent"
              value={tags}
              onChange={(e) => setTags(e.target.value)}
            />
            <p className="text-xs text-muted-foreground">Comma-separated</p>
          </div>
        </div>

        <DialogFooter>
          <Button variant="outline" onClick={() => onOpenChange(false)} disabled={isSubmitting}>
            Cancel
          </Button>
          <Button
            onClick={handleSubmit}
            disabled={isSubmitting || isUploading || (!name.trim() && !description.trim())}
          >
            {isSubmitting ? (
              <>
                <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                Creating...
              </>
            ) : (
              <>
                <Target className="w-4 h-4 mr-2" />
                Create Mission
                {attachments.length > 0 && (
                  <span className="ml-1 text-[10px] opacity-70">
                    ({attachments.length} file{attachments.length !== 1 ? 's' : ''})
                  </span>
                )}
              </>
            )}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}
