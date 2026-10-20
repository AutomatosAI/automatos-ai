'use client'

import { useState } from 'react'
import { MarkdownView } from '@/components/shared/markdown-view'
import { Badge } from '@/components/ui/badge'
import {
  Sheet,
  SheetContent,
  SheetHeader,
  SheetTitle,
} from '@/components/ui/sheet'
import { toast } from 'sonner'
import { useBlogPost, useCreatePost, useUpdatePost } from '@/hooks/use-blogs-api'
import { BlogModeToggle, BlogMissionForm, type BlogCreateMode } from './blog-mission-form'
import { BlogPostFields } from './blog-post-fields'
import { BlogPublishBar } from './blog-publish-bar'
import { BlogReviewControl } from './blog-review-control'
import { formMatchesPost, postDataFromForm, useBlogPostForm } from './blog-post-form'

interface BlogEditorProps {
  postId: string | null
  onClose: () => void
}

function BlogPostPreview({ title, tags, content }: { title: string; tags: string[]; content: string }) {
  return (
    <div className="hidden lg:block">
      <label className="text-xs font-medium text-muted-foreground">Preview</label>
      <div className="mt-1 glass-card p-4 min-h-[400px] overflow-y-auto">
        {title && <h1 className="text-lg font-bold mb-2">{title}</h1>}
        {tags.length > 0 && (
          <div className="flex gap-1 mb-3">
            {tags.map((tag) => (
              <Badge key={tag} variant="secondary" className="text-[10px]">
                {tag}
              </Badge>
            ))}
          </div>
        )}
        {content ? (
          <MarkdownView>{content}</MarkdownView>
        ) : (
          <p className="text-muted-foreground italic">Start writing to see preview...</p>
        )}
      </div>
    </div>
  )
}

export function BlogEditor({ postId, onClose }: BlogEditorProps) {
  const isEditMode = !!postId
  const { data: existingPost } = useBlogPost(postId)
  const { form, setField, tags } = useBlogPostForm(existingPost, isEditMode)
  // Mode toggle — only relevant in New Post (no postId). Edit always shows manual.
  const [mode, setMode] = useState<BlogCreateMode>('manual')

  const createMutation = useCreatePost()
  const updateMutation = useUpdatePost()

  const saveEdits = async () => {
    if (postId) await updateMutation.mutateAsync({ postId, data: postDataFromForm(form) })
  }

  const handleSave = async () => {
    if (!form.title.trim() || !form.content.trim()) {
      toast.error('Title and content are required')
      return
    }
    if (isEditMode) {
      await saveEdits()
    } else {
      await createMutation.mutateAsync({ ...postDataFromForm(form), status: 'draft' })
    }
    onClose()
  }

  const isSaving = createMutation.isLoading || updateMutation.isLoading

  return (
    <Sheet open onOpenChange={() => onClose()}>
      <SheetContent side="right" className="w-full sm:max-w-[900px] overflow-y-auto">
        <SheetHeader>
          <SheetTitle>{isEditMode ? 'Edit Post' : 'New Post'}</SheetTitle>
        </SheetHeader>

        {!isEditMode && <BlogModeToggle mode={mode} onChange={setMode} />}

        {!isEditMode && mode === 'mission' && <BlogMissionForm onClose={onClose} />}

        {/* Manual mode — full editor (always shown for Edit, conditional for New) */}
        {(isEditMode || mode === 'manual') && (
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mt-6">
            <div className="space-y-4">
              {existingPost && isEditMode && (
                <BlogReviewControl post={existingPost} hasUnsavedEdits={!formMatchesPost(form, existingPost)} />
              )}
              <BlogPostFields form={form} setField={setField} />
              <BlogPublishBar
                postId={postId}
                post={existingPost}
                slug={form.slug}
                saveEdits={saveEdits}
                onSave={handleSave}
                isSaving={isSaving}
              />
            </div>

            <BlogPostPreview title={form.title} tags={tags} content={form.content} />
          </div>
        )}
      </SheetContent>
    </Sheet>
  )
}
