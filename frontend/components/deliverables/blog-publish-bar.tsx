'use client'

import { useEffect, useState } from 'react'
import { Save } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Switch } from '@/components/ui/switch'
import { Label } from '@/components/ui/label'
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from '@/components/ui/tooltip'
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from '@/components/ui/alert-dialog'
import { usePublishPost, useUnpublishPost } from '@/hooks/use-blogs-api'
import type { BlogPost } from '@/hooks/use-blogs-api'
import { useSystemRole } from '@/contexts/role-context'

interface BlogPublishBarProps {
  postId: string | null
  post: BlogPost | undefined
  slug: string
  /** Saves the form's pending edits to the post (edit mode). */
  saveEdits: () => Promise<void>
  onSave: () => void
  isSaving: boolean
}

function PublishConfirm({ open, onOpenChange, slug, onConfirm }: {
  open: boolean
  onOpenChange: (open: boolean) => void
  slug: string
  onConfirm: () => void
}) {
  return (
    <AlertDialog open={open} onOpenChange={onOpenChange}>
      <AlertDialogContent>
        <AlertDialogHeader>
          <AlertDialogTitle>Publish this post?</AlertDialogTitle>
          <AlertDialogDescription>
            This will make your blog visible at <span className="font-mono text-xs">/blog/{slug || '...'}</span>. Publish?
          </AlertDialogDescription>
        </AlertDialogHeader>
        <AlertDialogFooter>
          <AlertDialogCancel>Cancel</AlertDialogCancel>
          <AlertDialogAction onClick={onConfirm}>
            Publish
          </AlertDialogAction>
        </AlertDialogFooter>
      </AlertDialogContent>
    </AlertDialog>
  )
}

/** Draft / Publish toggle (edit mode, workspace admin) and Save, with a confirmation on first publish. */
export function BlogPublishBar({ postId, post, slug, saveEdits, onSave, isSaving }: BlogPublishBarProps) {
  const { isAdmin } = useSystemRole()
  const publishMutation = usePublishPost()
  const unpublishMutation = useUnpublishPost()
  const [isPublished, setIsPublished] = useState(false)
  const [showPublishConfirm, setShowPublishConfirm] = useState(false)
  // Whether the post was already published when opened (for first-publish confirmation)
  const wasPublished = post?.status === 'published'
  const busy = isSaving || publishMutation.isLoading || unpublishMutation.isLoading

  useEffect(() => {
    if (post && postId) setIsPublished(post.status === 'published')
  }, [post, postId])

  const handlePublishAction = async () => {
    if (!postId) return
    await saveEdits() // save pending edits first, then publish
    await publishMutation.mutateAsync(postId)
    setIsPublished(true)
  }

  const handleUnpublishAction = async () => {
    if (!postId) return
    await unpublishMutation.mutateAsync(postId)
    setIsPublished(false)
  }

  const handleTogglePublish = (checked: boolean) => {
    if (checked && !wasPublished) setShowPublishConfirm(true) // first publish: confirm
    else if (checked) handlePublishAction() // re-publishing: no confirmation needed
    else handleUnpublishAction()
  }

  const handleConfirmPublish = async () => {
    setShowPublishConfirm(false)
    await handlePublishAction()
  }

  return (
    <div className="flex items-center justify-between pt-2">
      <div className="flex items-center gap-3">
        {postId ? (
          <TooltipProvider>
            <Tooltip>
              <TooltipTrigger asChild>
                <div className="flex items-center gap-2">
                  <Switch
                    id="publish-toggle"
                    checked={isPublished}
                    onCheckedChange={handleTogglePublish}
                    disabled={!isAdmin || busy}
                  />
                  <Label
                    htmlFor="publish-toggle"
                    className={`text-sm font-medium ${isPublished ? 'text-[hsl(var(--success))]' : 'text-muted-foreground'}`}
                  >
                    {isPublished ? 'Published' : 'Draft'}
                  </Label>
                </div>
              </TooltipTrigger>
              {!isAdmin && (
                <TooltipContent>
                  <p>Workspace admin can publish</p>
                </TooltipContent>
              )}
            </Tooltip>
          </TooltipProvider>
        ) : (
          <span className="text-xs text-muted-foreground">New post will be saved as draft</span>
        )}
      </div>

      <Button onClick={onSave} disabled={busy}>
        <Save className="w-4 h-4 mr-1" />
        Save
      </Button>

      <PublishConfirm
        open={showPublishConfirm}
        onOpenChange={setShowPublishConfirm}
        slug={slug || post?.slug || ''}
        onConfirm={handleConfirmPublish}
      />
    </div>
  )
}
