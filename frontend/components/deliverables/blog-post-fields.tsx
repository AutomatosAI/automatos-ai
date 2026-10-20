'use client'

import { useRef } from 'react'
import { Upload, Loader2 } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Textarea } from '@/components/ui/textarea'
import { toast } from 'sonner'
import { useUploadCoverImage } from '@/hooks/use-blogs-api'
import type { BlogPostForm, SetBlogPostField } from './blog-post-form'

const EXCERPT_MAX_CHARS = 300
const UPLOADED_IMAGE_PREFIX = '/api/generated-images/'

interface FieldsProps {
  form: BlogPostForm
  setField: SetBlogPostField
}

function BlogCoverImageField({ form, setField }: FieldsProps) {
  const uploadMutation = useUploadCoverImage()
  const fileInputRef = useRef<HTMLInputElement | null>(null)
  const isUploading = uploadMutation.isLoading

  const handleCoverFileSelected = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    e.target.value = '' // allow re-selecting same file
    if (!file) return
    try {
      const result = await uploadMutation.mutateAsync(file)
      setField('coverImageUrl', result.cover_image_url)
      toast.success('Cover image uploaded')
    } catch {
      // error toast surfaced by hook
    }
  }

  return (
    <div>
      <label className="text-xs font-medium text-muted-foreground">Cover Image</label>
      <div className="mt-1 flex gap-2">
        <Input
          value={form.coverImageUrl}
          onChange={(e) => setField('coverImageUrl', e.target.value)}
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
      {form.coverImageUrl && form.coverImageUrl.startsWith(UPLOADED_IMAGE_PREFIX) && (
        <p className="text-xs text-muted-foreground mt-1">
          Uploaded image saved. Saved with the post on Save.
        </p>
      )}
    </div>
  )
}

function BlogSeoFields({ form, setField }: FieldsProps) {
  return (
    <div className="glass-card p-4 space-y-3">
      <h4 className="text-xs font-semibold text-muted-foreground uppercase tracking-wider">SEO Metadata</h4>
      <div>
        <label className="text-xs text-muted-foreground">Slug</label>
        <Input
          value={form.slug}
          onChange={(e) => setField('slug', e.target.value)}
          placeholder="my-blog-post-url"
          className="mt-1"
        />
      </div>
      <div>
        <label className="text-xs text-muted-foreground">SEO Title</label>
        <Input
          value={form.seoTitle}
          onChange={(e) => setField('seoTitle', e.target.value)}
          placeholder="Override title for search engines"
          className="mt-1"
        />
      </div>
      <div>
        <label className="text-xs text-muted-foreground">Meta Description</label>
        <Textarea
          value={form.seoDescription}
          onChange={(e) => setField('seoDescription', e.target.value)}
          placeholder="Meta description for search engines"
          className="mt-1"
          rows={2}
        />
      </div>
    </div>
  )
}

/** The manual editor's fields: text, cover image, category and tags, SEO. */
export function BlogPostFields({ form, setField }: FieldsProps) {
  return (
    <>
      <div>
        <label className="text-xs font-medium text-muted-foreground">Title</label>
        <Input
          value={form.title}
          onChange={(e) => setField('title', e.target.value)}
          placeholder="Post title"
          className="mt-1"
        />
      </div>

      <div>
        <label className="text-xs font-medium text-muted-foreground">Content (Markdown)</label>
        <Textarea
          value={form.content}
          onChange={(e) => setField('content', e.target.value)}
          placeholder="Write your post in markdown..."
          className="mt-1 min-h-[300px] font-mono text-sm"
        />
      </div>

      <div>
        <label className="text-xs font-medium text-muted-foreground">
          Excerpt ({form.excerpt.length}/{EXCERPT_MAX_CHARS})
        </label>
        <Textarea
          value={form.excerpt}
          onChange={(e) => setField('excerpt', e.target.value.slice(0, EXCERPT_MAX_CHARS))}
          placeholder="Short summary (auto-generated if empty)"
          className="mt-1"
          rows={2}
        />
      </div>

      <BlogCoverImageField form={form} setField={setField} />

      <div className="grid grid-cols-2 gap-3">
        <div>
          <label className="text-xs font-medium text-muted-foreground">Category</label>
          <Input
            value={form.category}
            onChange={(e) => setField('category', e.target.value)}
            placeholder="e.g. Research"
            className="mt-1"
          />
        </div>
        <div>
          <label className="text-xs font-medium text-muted-foreground">Tags (comma-separated)</label>
          <Input
            value={form.tagsInput}
            onChange={(e) => setField('tagsInput', e.target.value)}
            placeholder="ai, research"
            className="mt-1"
          />
        </div>
      </div>

      <BlogSeoFields form={form} setField={setField} />
    </>
  )
}
