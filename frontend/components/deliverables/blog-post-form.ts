/**
 * The blog editor's form: one state object for the post's editable fields, filled
 * from the post being edited, and the payload Save sends.
 */
import { useCallback, useEffect, useMemo, useState } from 'react'
import type { BlogPost } from '@/hooks/use-blogs-api'

export interface BlogPostForm {
  title: string
  slug: string
  content: string
  excerpt: string
  coverImageUrl: string
  category: string
  tagsInput: string
  seoTitle: string
  seoDescription: string
}

export type SetBlogPostField = <K extends keyof BlogPostForm>(key: K, value: BlogPostForm[K]) => void

export const EMPTY_BLOG_POST_FORM: BlogPostForm = {
  title: '',
  slug: '',
  content: '',
  excerpt: '',
  coverImageUrl: '',
  category: '',
  tagsInput: '',
  seoTitle: '',
  seoDescription: '',
}

export function formFromPost(post: BlogPost): BlogPostForm {
  return {
    title: post.title || '',
    slug: post.slug || '',
    content: post.content || '',
    excerpt: post.excerpt || '',
    coverImageUrl: post.cover_image_url || '',
    category: post.category || '',
    tagsInput: (post.tags || []).join(', '),
    seoTitle: post.seo_title || '',
    seoDescription: post.seo_description || '',
  }
}

/** True when the form still holds exactly what the post had: nothing is waiting to be saved. */
export function formMatchesPost(form: BlogPostForm, post: BlogPost): boolean {
  const saved = formFromPost(post)
  return (Object.keys(saved) as (keyof BlogPostForm)[]).every((key) => form[key] === saved[key])
}

export function parseTags(tagsInput: string): string[] {
  return tagsInput.split(',').map((t) => t.trim()).filter(Boolean)
}

/** What Save sends: trimmed text, and no key at all for an empty optional field. */
export function postDataFromForm(form: BlogPostForm) {
  return {
    title: form.title.trim(),
    slug: form.slug.trim() || undefined,
    content: form.content.trim(),
    excerpt: form.excerpt.trim() || undefined,
    cover_image_url: form.coverImageUrl.trim() || undefined,
    category: form.category.trim() || undefined,
    tags: parseTags(form.tagsInput),
    seo_title: form.seoTitle.trim() || undefined,
    seo_description: form.seoDescription.trim() || undefined,
  }
}

export function useBlogPostForm(existingPost: BlogPost | undefined, isEditMode: boolean) {
  const [form, setForm] = useState<BlogPostForm>(EMPTY_BLOG_POST_FORM)

  useEffect(() => {
    if (existingPost && isEditMode) setForm(formFromPost(existingPost))
  }, [existingPost, isEditMode])

  const setField = useCallback<SetBlogPostField>((key, value) => {
    setForm((prev) => ({ ...prev, [key]: value }))
  }, [])

  const tags = useMemo(() => parseTags(form.tagsInput), [form.tagsInput])

  return { form, setField, tags }
}
