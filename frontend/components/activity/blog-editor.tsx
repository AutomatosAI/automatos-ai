'use client'

import { useState, useEffect, useMemo } from 'react'
import DOMPurify from 'dompurify'
import ReactMarkdown from 'react-markdown'
import { X, Save, ArrowUpFromLine } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Textarea } from '@/components/ui/textarea'
import { Badge } from '@/components/ui/badge'
import {
  Sheet,
  SheetContent,
  SheetHeader,
  SheetTitle,
} from '@/components/ui/sheet'
import { toast } from 'react-hot-toast'
import { useBlogPost, useCreatePost, useUpdatePost, usePublishPost } from '@/hooks/use-blogs-api'

interface BlogEditorProps {
  postId: string | null
  onClose: () => void
}

export function BlogEditor({ postId, onClose }: BlogEditorProps) {
  const isEditMode = !!postId
  const { data: existingPost } = useBlogPost(postId)

  const [title, setTitle] = useState('')
  const [content, setContent] = useState('')
  const [excerpt, setExcerpt] = useState('')
  const [coverImageUrl, setCoverImageUrl] = useState('')
  const [category, setCategory] = useState('')
  const [tagsInput, setTagsInput] = useState('')
  const [seoTitle, setSeoTitle] = useState('')
  const [seoDescription, setSeoDescription] = useState('')

  const createMutation = useCreatePost()
  const updateMutation = useUpdatePost()
  const publishMutation = usePublishPost()

  useEffect(() => {
    if (existingPost && isEditMode) {
      setTitle(existingPost.title || '')
      setContent(existingPost.content || '')
      setExcerpt(existingPost.excerpt || '')
      setCoverImageUrl(existingPost.cover_image_url || '')
      setCategory(existingPost.category || '')
      setTagsInput((existingPost.tags || []).join(', '))
      setSeoTitle(existingPost.seo_title || '')
      setSeoDescription(existingPost.seo_description || '')
    }
  }, [existingPost, isEditMode])

  const tags = useMemo(
    () => tagsInput.split(',').map((t) => t.trim()).filter(Boolean),
    [tagsInput]
  )

  const sanitizedPreview = useMemo(
    () => DOMPurify.sanitize(content),
    [content]
  )

  const handleSaveDraft = async () => {
    if (!title.trim() || !content.trim()) {
      toast.error('Title and content are required')
      return
    }

    const data = {
      title: title.trim(),
      content: content.trim(),
      excerpt: excerpt.trim() || undefined,
      cover_image_url: coverImageUrl.trim() || undefined,
      category: category.trim() || undefined,
      tags,
      status: 'draft' as const,
      seo_title: seoTitle.trim() || undefined,
      seo_description: seoDescription.trim() || undefined,
    }

    if (isEditMode && postId) {
      await updateMutation.mutateAsync({ postId, data })
    } else {
      await createMutation.mutateAsync(data)
    }
    onClose()
  }

  const handlePublish = async () => {
    if (!title.trim() || !content.trim()) {
      toast.error('Title and content are required')
      return
    }

    if (isEditMode && postId) {
      await updateMutation.mutateAsync({
        postId,
        data: {
          title: title.trim(),
          content: content.trim(),
          excerpt: excerpt.trim() || undefined,
          cover_image_url: coverImageUrl.trim() || undefined,
          category: category.trim() || undefined,
          tags,
          seo_title: seoTitle.trim() || undefined,
          seo_description: seoDescription.trim() || undefined,
        },
      })
      await publishMutation.mutateAsync(postId)
    } else {
      const post = await createMutation.mutateAsync({
        title: title.trim(),
        content: content.trim(),
        excerpt: excerpt.trim() || undefined,
        cover_image_url: coverImageUrl.trim() || undefined,
        category: category.trim() || undefined,
        tags,
        status: 'published',
        seo_title: seoTitle.trim() || undefined,
        seo_description: seoDescription.trim() || undefined,
      })
    }
    onClose()
  }

  const isSaving = createMutation.isLoading || updateMutation.isLoading || publishMutation.isLoading

  return (
    <Sheet open onOpenChange={() => onClose()}>
      <SheetContent side="right" className="w-full sm:max-w-[900px] overflow-y-auto">
        <SheetHeader>
          <SheetTitle>{isEditMode ? 'Edit Post' : 'New Post'}</SheetTitle>
        </SheetHeader>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mt-6">
          {/* Left: Form */}
          <div className="space-y-4">
            <div>
              <label className="text-xs font-medium text-muted-foreground">Title</label>
              <Input
                value={title}
                onChange={(e) => setTitle(e.target.value)}
                placeholder="Post title"
                className="mt-1"
              />
            </div>

            <div>
              <label className="text-xs font-medium text-muted-foreground">Content (Markdown)</label>
              <Textarea
                value={content}
                onChange={(e) => setContent(e.target.value)}
                placeholder="Write your post in markdown..."
                className="mt-1 min-h-[300px] font-mono text-sm"
              />
            </div>

            <div>
              <label className="text-xs font-medium text-muted-foreground">
                Excerpt ({excerpt.length}/300)
              </label>
              <Textarea
                value={excerpt}
                onChange={(e) => setExcerpt(e.target.value.slice(0, 300))}
                placeholder="Short summary (auto-generated if empty)"
                className="mt-1"
                rows={2}
              />
            </div>

            <div>
              <label className="text-xs font-medium text-muted-foreground">Cover Image URL</label>
              <Input
                value={coverImageUrl}
                onChange={(e) => setCoverImageUrl(e.target.value)}
                placeholder="https://..."
                className="mt-1"
              />
            </div>

            <div className="grid grid-cols-2 gap-3">
              <div>
                <label className="text-xs font-medium text-muted-foreground">Category</label>
                <Input
                  value={category}
                  onChange={(e) => setCategory(e.target.value)}
                  placeholder="e.g. Research"
                  className="mt-1"
                />
              </div>
              <div>
                <label className="text-xs font-medium text-muted-foreground">Tags (comma-separated)</label>
                <Input
                  value={tagsInput}
                  onChange={(e) => setTagsInput(e.target.value)}
                  placeholder="ai, research"
                  className="mt-1"
                />
              </div>
            </div>

            <details className="text-xs">
              <summary className="cursor-pointer text-muted-foreground font-medium">SEO Fields</summary>
              <div className="space-y-3 mt-2">
                <div>
                  <label className="text-xs text-muted-foreground">SEO Title</label>
                  <Input
                    value={seoTitle}
                    onChange={(e) => setSeoTitle(e.target.value)}
                    placeholder="Override title for search engines"
                    className="mt-1"
                  />
                </div>
                <div>
                  <label className="text-xs text-muted-foreground">SEO Description</label>
                  <Textarea
                    value={seoDescription}
                    onChange={(e) => setSeoDescription(e.target.value)}
                    placeholder="Meta description for search engines"
                    className="mt-1"
                    rows={2}
                  />
                </div>
              </div>
            </details>

            <div className="flex gap-2 pt-2">
              <Button
                variant="outline"
                onClick={handleSaveDraft}
                disabled={isSaving}
              >
                <Save className="w-4 h-4 mr-1" />
                Save Draft
              </Button>
              <Button
                onClick={handlePublish}
                disabled={isSaving}
              >
                <ArrowUpFromLine className="w-4 h-4 mr-1" />
                Publish
              </Button>
            </div>
          </div>

          {/* Right: Preview */}
          <div className="hidden lg:block">
            <label className="text-xs font-medium text-muted-foreground">Preview</label>
            <div className="mt-1 glass-card p-4 min-h-[400px] overflow-y-auto prose prose-sm dark:prose-invert max-w-none">
              {title && <h1 className="text-lg font-bold mb-2">{title}</h1>}
              {tags.length > 0 && (
                <div className="flex gap-1 mb-3 not-prose">
                  {tags.map((tag) => (
                    <Badge key={tag} variant="secondary" className="text-[10px]">
                      {tag}
                    </Badge>
                  ))}
                </div>
              )}
              {content ? (
                <ReactMarkdown>{content}</ReactMarkdown>
              ) : (
                <p className="text-muted-foreground italic">Start writing to see preview...</p>
              )}
            </div>
          </div>
        </div>
      </SheetContent>
    </Sheet>
  )
}
