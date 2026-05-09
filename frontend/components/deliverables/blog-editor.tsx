'use client'

import { useState, useEffect, useMemo, useRef } from 'react'
import DOMPurify from 'dompurify'
import ReactMarkdown from 'react-markdown'
import { Save, Sparkles, Pencil, Upload, Loader2 } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Textarea } from '@/components/ui/textarea'
import { Badge } from '@/components/ui/badge'
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
import {
  Sheet,
  SheetContent,
  SheetHeader,
  SheetTitle,
} from '@/components/ui/sheet'
import { toast } from 'react-hot-toast'
import {
  useBlogPost,
  useCreatePost,
  useUpdatePost,
  usePublishPost,
  useUnpublishPost,
  useCreateBlogMission,
  useUploadCoverImage,
} from '@/hooks/use-blogs-api'
import { useSystemRole } from '@/contexts/role-context'

type CreateMode = 'manual' | 'mission'

interface BlogEditorProps {
  postId: string | null
  onClose: () => void
}

export function BlogEditor({ postId, onClose }: BlogEditorProps) {
  const isEditMode = !!postId
  const { data: existingPost } = useBlogPost(postId)
  const { isAdmin } = useSystemRole()

  const [title, setTitle] = useState('')
  const [slug, setSlug] = useState('')
  const [content, setContent] = useState('')
  const [excerpt, setExcerpt] = useState('')
  const [coverImageUrl, setCoverImageUrl] = useState('')
  const [category, setCategory] = useState('')
  const [tagsInput, setTagsInput] = useState('')
  const [seoTitle, setSeoTitle] = useState('')
  const [seoDescription, setSeoDescription] = useState('')
  const [isPublished, setIsPublished] = useState(false)
  const [showPublishConfirm, setShowPublishConfirm] = useState(false)

  const createMutation = useCreatePost()
  const updateMutation = useUpdatePost()
  const publishMutation = usePublishPost()
  const unpublishMutation = useUnpublishPost()
  const missionMutation = useCreateBlogMission()
  const uploadMutation = useUploadCoverImage()

  // Mode toggle — only relevant in New Post (no postId). Edit always shows manual.
  const [mode, setMode] = useState<CreateMode>('manual')
  const [missionTopic, setMissionTopic] = useState('')
  const [missionCategory, setMissionCategory] = useState('AI & Automation')
  const fileInputRef = useRef<HTMLInputElement | null>(null)

  // Track whether post was already published when opened (for first-publish confirmation)
  const wasPublished = existingPost?.status === 'published'

  useEffect(() => {
    if (existingPost && isEditMode) {
      setTitle(existingPost.title || '')
      setSlug(existingPost.slug || '')
      setContent(existingPost.content || '')
      setExcerpt(existingPost.excerpt || '')
      setCoverImageUrl(existingPost.cover_image_url || '')
      setCategory(existingPost.category || '')
      setTagsInput((existingPost.tags || []).join(', '))
      setSeoTitle(existingPost.seo_title || '')
      setSeoDescription(existingPost.seo_description || '')
      setIsPublished(existingPost.status === 'published')
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

  const buildPostData = () => ({
    title: title.trim(),
    slug: slug.trim() || undefined,
    content: content.trim(),
    excerpt: excerpt.trim() || undefined,
    cover_image_url: coverImageUrl.trim() || undefined,
    category: category.trim() || undefined,
    tags,
    seo_title: seoTitle.trim() || undefined,
    seo_description: seoDescription.trim() || undefined,
  })

  const handleSave = async () => {
    if (!title.trim() || !content.trim()) {
      toast.error('Title and content are required')
      return
    }

    const data = buildPostData()

    if (isEditMode && postId) {
      await updateMutation.mutateAsync({ postId, data })
    } else {
      await createMutation.mutateAsync({ ...data, status: isPublished ? 'published' : 'draft' })
    }
    onClose()
  }

  const handleTogglePublish = (checked: boolean) => {
    if (checked && !wasPublished) {
      // First publish — show confirmation modal
      setShowPublishConfirm(true)
    } else if (checked && wasPublished) {
      // Re-publishing (was published before) — no confirmation needed
      handlePublishAction()
    } else {
      // Unpublishing
      handleUnpublishAction()
    }
  }

  const handlePublishAction = async () => {
    if (isEditMode && postId) {
      // Save pending edits first, then publish
      const data = buildPostData()
      await updateMutation.mutateAsync({ postId, data })
      await publishMutation.mutateAsync(postId)
      setIsPublished(true)
    }
  }

  const handleUnpublishAction = async () => {
    if (isEditMode && postId) {
      await unpublishMutation.mutateAsync(postId)
      setIsPublished(false)
    }
  }

  const handleConfirmPublish = async () => {
    setShowPublishConfirm(false)
    await handlePublishAction()
  }

  const handleStartMission = async () => {
    const topic = missionTopic.trim()
    if (!topic) {
      toast.error('Topic is required')
      return
    }
    await missionMutation.mutateAsync({
      topic,
      category: missionCategory.trim() || undefined,
    })
    onClose()
  }

  const handleCoverFileSelected = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    e.target.value = '' // allow re-selecting same file
    if (!file) return
    try {
      const result = await uploadMutation.mutateAsync(file)
      setCoverImageUrl(result.cover_image_url)
      toast.success('Cover image uploaded')
    } catch {
      // error toast surfaced by hook
    }
  }

  const isSaving = createMutation.isLoading || updateMutation.isLoading || publishMutation.isLoading || unpublishMutation.isLoading
  const isMissionStarting = missionMutation.isLoading
  const isUploading = uploadMutation.isLoading

  return (
    <Sheet open onOpenChange={() => onClose()}>
      <SheetContent side="right" className="w-full sm:max-w-[900px] overflow-y-auto">
        <SheetHeader>
          <SheetTitle>{isEditMode ? 'Edit Post' : 'New Post'}</SheetTitle>
        </SheetHeader>

        {/* Mode toggle — only when creating a new post */}
        {!isEditMode && (
          <div className="mt-4 flex gap-2 p-1 bg-muted/40 rounded-lg w-fit">
            <button
              type="button"
              onClick={() => setMode('manual')}
              className={`flex items-center gap-2 px-4 py-1.5 rounded-md text-sm font-medium transition ${
                mode === 'manual'
                  ? 'bg-background shadow-sm'
                  : 'text-muted-foreground hover:text-foreground'
              }`}
            >
              <Pencil className="w-4 h-4" /> Write Manually
            </button>
            <button
              type="button"
              onClick={() => setMode('mission')}
              className={`flex items-center gap-2 px-4 py-1.5 rounded-md text-sm font-medium transition ${
                mode === 'mission'
                  ? 'bg-background shadow-sm'
                  : 'text-muted-foreground hover:text-foreground'
              }`}
            >
              <Sparkles className="w-4 h-4" /> Have Agents Write It
            </button>
          </div>
        )}

        {/* Mission mode — simple topic + category form */}
        {!isEditMode && mode === 'mission' && (
          <div className="mt-6 space-y-4 max-w-xl">
            <div className="text-sm text-muted-foreground">
              Pick a topic. Agents will research, write, edit, generate a cover image,
              and queue the draft for your review. Takes 5-10 minutes.
            </div>

            <div className="space-y-2">
              <Label htmlFor="mission-topic">Topic</Label>
              <Input
                id="mission-topic"
                placeholder="e.g. Multi-agent orchestration for Shopify stores"
                value={missionTopic}
                onChange={(e) => setMissionTopic(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' && missionTopic.trim()) {
                    e.preventDefault()
                    handleStartMission()
                  }
                }}
                autoFocus
              />
              <p className="text-xs text-muted-foreground">
                Be specific. The more concrete the topic, the better the post.
              </p>
            </div>

            <div className="space-y-2">
              <Label htmlFor="mission-category">Category</Label>
              <Input
                id="mission-category"
                placeholder="AI & Automation"
                value={missionCategory}
                onChange={(e) => setMissionCategory(e.target.value)}
              />
            </div>

            <div className="flex items-center justify-end gap-2 pt-2">
              <Button variant="ghost" onClick={onClose} disabled={isMissionStarting}>
                Cancel
              </Button>
              <Button
                onClick={handleStartMission}
                disabled={!missionTopic.trim() || isMissionStarting}
              >
                {isMissionStarting ? (
                  <>
                    <Loader2 className="w-4 h-4 mr-1 animate-spin" /> Starting…
                  </>
                ) : (
                  <>
                    <Sparkles className="w-4 h-4 mr-1" /> Start Mission
                  </>
                )}
              </Button>
            </div>
          </div>
        )}

        {/* Manual mode — full editor (always shown for Edit, conditional for New) */}
        {(isEditMode || mode === 'manual') && (
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
              <label className="text-xs font-medium text-muted-foreground">Cover Image</label>
              <div className="mt-1 flex gap-2">
                <Input
                  value={coverImageUrl}
                  onChange={(e) => setCoverImageUrl(e.target.value)}
                  placeholder="https://... or upload your own →"
                  className="flex-1"
                />
                <input
                  ref={fileInputRef}
                  type="file"
                  accept="image/png,image/jpeg,image/webp,image/gif"
                  className="hidden"
                  onChange={handleCoverFileSelected}
                />
                <Button
                  type="button"
                  variant="outline"
                  onClick={() => fileInputRef.current?.click()}
                  disabled={isUploading}
                  title="Upload your own cover image (max 8 MB)"
                >
                  {isUploading ? (
                    <Loader2 className="w-4 h-4 animate-spin" />
                  ) : (
                    <>
                      <Upload className="w-4 h-4 mr-1" /> Upload
                    </>
                  )}
                </Button>
              </div>
              {coverImageUrl && coverImageUrl.startsWith('/api/generated-images/') && (
                <p className="text-xs text-muted-foreground mt-1">
                  Uploaded image saved. Saved with the post on Save.
                </p>
              )}
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

            {/* SEO Metadata Panel */}
            <div className="glass-card p-4 space-y-3">
              <h4 className="text-xs font-semibold text-muted-foreground uppercase tracking-wider">SEO Metadata</h4>
              <div>
                <label className="text-xs text-muted-foreground">Slug</label>
                <Input
                  value={slug}
                  onChange={(e) => setSlug(e.target.value)}
                  placeholder="my-blog-post-url"
                  className="mt-1"
                />
              </div>
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
                <label className="text-xs text-muted-foreground">Meta Description</label>
                <Textarea
                  value={seoDescription}
                  onChange={(e) => setSeoDescription(e.target.value)}
                  placeholder="Meta description for search engines"
                  className="mt-1"
                  rows={2}
                />
              </div>
            </div>

            {/* Draft / Publish toggle + Save */}
            <div className="flex items-center justify-between pt-2">
              <div className="flex items-center gap-3">
                {isEditMode ? (
                  <TooltipProvider>
                    <Tooltip>
                      <TooltipTrigger asChild>
                        <div className="flex items-center gap-2">
                          <Switch
                            id="publish-toggle"
                            checked={isPublished}
                            onCheckedChange={handleTogglePublish}
                            disabled={!isAdmin || isSaving}
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

              <Button
                onClick={handleSave}
                disabled={isSaving}
              >
                <Save className="w-4 h-4 mr-1" />
                Save
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
        )}

        {/* First-publish confirmation modal */}
        <AlertDialog open={showPublishConfirm} onOpenChange={setShowPublishConfirm}>
          <AlertDialogContent>
            <AlertDialogHeader>
              <AlertDialogTitle>Publish this post?</AlertDialogTitle>
              <AlertDialogDescription>
                This will make your blog visible at <span className="font-mono text-xs">/blog/{slug || existingPost?.slug || '...'}</span>. Publish?
              </AlertDialogDescription>
            </AlertDialogHeader>
            <AlertDialogFooter>
              <AlertDialogCancel>Cancel</AlertDialogCancel>
              <AlertDialogAction onClick={handleConfirmPublish}>
                Publish
              </AlertDialogAction>
            </AlertDialogFooter>
          </AlertDialogContent>
        </AlertDialog>
      </SheetContent>
    </Sheet>
  )
}
