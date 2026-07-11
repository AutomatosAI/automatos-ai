'use client'

import { useState, useCallback } from 'react'
import {
  Filter,
  Plus,
  Eye,
  Pencil,
  Trash2,
  ArrowUpFromLine,
  ArrowDownToLine,
  Clock,
  User,
  LayoutGrid,
  FileText,
  CheckCircle,
  Archive,
  CalendarClock,
} from 'lucide-react'
import { DeliverableIcon } from '@/components/icons/deliverable-icon'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Skeleton } from '@/components/ui/skeleton'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'
import { PageHeader, FilterTabs, TabsContent } from '@/components/shared'
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
  AlertDialogTrigger,
} from '@/components/ui/alert-dialog'
import {
  useBlogPosts,
  usePublishPost,
  useUnpublishPost,
  useDeletePost,
} from '@/hooks/use-blogs-api'
import type { BlogPost, BlogFilters } from '@/hooks/use-blogs-api'
import { BlogEditor } from './blog-editor'
import { useWorkspace } from '@/components/workspace-provider'

// ─── Status badge colours ───────────────────────────────

const STATUS_STYLES: Record<string, string> = {
  draft: 'bg-secondary text-secondary-foreground',
  published: 'bg-[hsl(var(--success))]/15 text-[hsl(var(--success))]',
  archived: 'bg-destructive/15 text-destructive',
  scheduled: 'bg-[hsl(var(--warning))]/15 text-[hsl(var(--warning))]',
}

// ─── Skeleton ───────────────────────────────────────────

function BlogCardSkeleton() {
  return (
    <div className="glass-card border-l-[3px] border-l-secondary p-4 space-y-3">
      <div className="flex items-start justify-between gap-2">
        <div className="space-y-2">
          <Skeleton className="h-5 w-64" />
          <Skeleton className="h-4 w-80" />
        </div>
        <Skeleton className="h-5 w-16 rounded-full" />
      </div>
      <div className="flex gap-2">
        <Skeleton className="h-5 w-14 rounded-full" />
        <Skeleton className="h-5 w-14 rounded-full" />
      </div>
      <div className="flex justify-between">
        <Skeleton className="h-4 w-36" />
        <div className="flex gap-1">
          <Skeleton className="h-7 w-7 rounded-md" />
          <Skeleton className="h-7 w-7 rounded-md" />
        </div>
      </div>
    </div>
  )
}

// ─── Empty State ────────────────────────────────────────

function BlogEmptyState({ onCreatePost }: { onCreatePost: () => void }) {
  return (
    <div className="glass-card p-6 sm:p-8 text-center text-muted-foreground">
      <div className="mx-auto mb-3 flex h-24 w-24 items-center justify-center overflow-hidden rounded-xl opacity-90">
        <DeliverableIcon type="blog_post" size="hero" width={96} height={96} />
      </div>
      <p className="font-medium">No blog posts yet</p>
      <p className="text-sm mt-1 max-w-xs mx-auto">
        Create posts manually or let your agents publish via the platform_publish_blog_post tool.
      </p>
      <Button size="sm" className="mt-4" onClick={onCreatePost}>
        <Plus className="w-4 h-4 mr-1" /> Create Post
      </Button>
    </div>
  )
}

// ─── Post Card ──────────────────────────────────────────

function BlogPostCard({
  post,
  onEdit,
  onPublish,
  onUnpublish,
  onDelete,
}: {
  post: BlogPost
  onEdit: (post: BlogPost) => void
  onPublish: (post: BlogPost) => void
  onUnpublish: (post: BlogPost) => void
  onDelete: (post: BlogPost) => void
}) {
  const { canEdit } = useWorkspace()
  const borderColor =
    post.status === 'published'
      ? 'border-l-[hsl(var(--success))]'
      : post.status === 'draft'
        ? 'border-l-secondary'
        : 'border-l-destructive'

  const formattedDate = post.published_at
    ? new Date(post.published_at).toLocaleDateString()
    : new Date(post.created_at).toLocaleDateString()

  return (
    <div className={`glass-card border-l-[3px] ${borderColor} p-4 space-y-2`}>
      <div className="flex items-start justify-between gap-2">
        <div className="min-w-0">
          <h3 className="font-semibold text-sm truncate">{post.title}</h3>
          {post.excerpt && (
            <p className="text-xs text-muted-foreground line-clamp-2 mt-0.5">
              {post.excerpt}
            </p>
          )}
        </div>
        <Badge className={`shrink-0 text-[10px] ${STATUS_STYLES[post.status] || ''}`}>
          {post.status}
        </Badge>
      </div>

      {(post.tags?.length > 0 || post.category) && (
        <div className="flex flex-wrap gap-1">
          {post.category && (
            <Badge variant="outline" className="text-[10px]">
              {post.category}
            </Badge>
          )}
          {post.tags?.map((tag) => (
            <Badge key={tag} variant="secondary" className="text-[10px]">
              {tag}
            </Badge>
          ))}
        </div>
      )}

      <div className="flex items-center justify-between text-xs text-muted-foreground">
        <div className="flex items-center gap-3">
          <span className="flex items-center gap-1">
            <User className="w-3 h-3" /> {post.author_name}
          </span>
          <span>{formattedDate}</span>
          <span className="flex items-center gap-1">
            <Clock className="w-3 h-3" /> {post.reading_time_minutes}m
          </span>
          {post.status === 'published' && (
            <span className="flex items-center gap-1">
              <Eye className="w-3 h-3" /> {post.view_count}
            </span>
          )}
        </div>

        <div className="flex items-center gap-1">
          <Button
            variant="ghost"
            size="icon"
            className="h-7 w-7"
            onClick={() => onEdit(post)}
            title="Edit"
          >
            <Pencil className="w-3.5 h-3.5" />
          </Button>

          {post.status === 'draft' ? (
            <Button
              variant="ghost"
              size="icon"
              className="h-7 w-7 text-[hsl(var(--success))]"
              onClick={() => onPublish(post)}
              title="Publish"
            >
              <ArrowUpFromLine className="w-3.5 h-3.5" />
            </Button>
          ) : post.status === 'published' ? (
            <Button
              variant="ghost"
              size="icon"
              className="h-7 w-7 text-[hsl(var(--warning))]"
              onClick={() => onUnpublish(post)}
              title="Unpublish"
            >
              <ArrowDownToLine className="w-3.5 h-3.5" />
            </Button>
          ) : null}

          <AlertDialog>
            <AlertDialogTrigger asChild>
              <Button
                variant="ghost"
                size="icon"
                className="h-7 w-7 text-destructive"
                title={canEdit ? 'Delete' : 'Viewers have read-only access'}
                disabled={!canEdit}
              >
                <Trash2 className="w-3.5 h-3.5" />
              </Button>
            </AlertDialogTrigger>
            <AlertDialogContent>
              <AlertDialogHeader>
                <AlertDialogTitle>Archive this post?</AlertDialogTitle>
                <AlertDialogDescription>
                  &quot;{post.title}&quot; will be archived and hidden from the blog widget.
                </AlertDialogDescription>
              </AlertDialogHeader>
              <AlertDialogFooter>
                <AlertDialogCancel>Cancel</AlertDialogCancel>
                <AlertDialogAction onClick={() => onDelete(post)}>
                  Archive
                </AlertDialogAction>
              </AlertDialogFooter>
            </AlertDialogContent>
          </AlertDialog>
        </div>
      </div>
    </div>
  )
}

// ─── Main Component ─────────────────────────────────────

export function DeliverablesBlog() {
  const [filters, setFilters] = useState<BlogFilters>({ per_page: 20 })
  const [isEditorOpen, setIsEditorOpen] = useState(false)
  const [editingPostId, setEditingPostId] = useState<string | null>(null)
  const [activeStatusTab, setActiveStatusTab] = useState('all')

  const { data, isLoading, refetch } = useBlogPosts(filters)
  const posts = data?.posts || []
  const total = data?.total || 0

  const publishMutation = usePublishPost()
  const unpublishMutation = useUnpublishPost()
  const deleteMutation = useDeletePost()

  const handleFilterChange = useCallback(
    (key: keyof BlogFilters, value: string | undefined) => {
      setFilters((prev) => ({
        ...prev,
        [key]: value === 'all' ? undefined : value,
        page: 1,
      }))
    },
    []
  )

  const handleEdit = useCallback((post: BlogPost) => {
    setEditingPostId(post.id)
    setIsEditorOpen(true)
  }, [])

  const handleCreatePost = useCallback(() => {
    setEditingPostId(null)
    setIsEditorOpen(true)
  }, [])

  const handleEditorClose = useCallback(() => {
    setIsEditorOpen(false)
    setEditingPostId(null)
    refetch()
  }, [refetch])

  const handleLoadMore = useCallback(() => {
    setFilters((prev) => ({
      ...prev,
      per_page: (prev.per_page || 20) + 20,
    }))
  }, [])

  const handleStatusTabChange = useCallback((value: string) => {
    setActiveStatusTab(value)
    handleFilterChange('status', value === 'all' ? undefined : value)
  }, [handleFilterChange])

  return (
    <div className="space-y-4">
      {/* Header */}
      <PageHeader
        title="Blog"
        titleAccent="Posts"
        subtitle="Manage published and draft blog content"
        actions={
          <Button size="sm" onClick={handleCreatePost}>
            <Plus className="w-4 h-4 mr-1" /> Create Post
          </Button>
        }
      />

      <FilterTabs
        tabs={[
          { value: 'all', label: 'All', icon: LayoutGrid, count: total },
          { value: 'draft', label: 'Draft', icon: FileText },
          { value: 'published', label: 'Published', icon: CheckCircle },
          { value: 'archived', label: 'Archived', icon: Archive },
        ]}
        value={activeStatusTab}
        onValueChange={handleStatusTabChange}
      >
        <TabsContent value={activeStatusTab}>
          {/* Post list */}
          {isLoading ? (
            <div className="space-y-3">
              {Array.from({ length: 4 }).map((_, i) => (
                <BlogCardSkeleton key={i} />
              ))}
            </div>
          ) : posts.length === 0 ? (
            <BlogEmptyState onCreatePost={handleCreatePost} />
          ) : (
            <div className="space-y-3">
              {posts.map((post) => (
                <BlogPostCard
                  key={post.id}
                  post={post}
                  onEdit={handleEdit}
                  onPublish={(p) => publishMutation.mutate(p.id)}
                  onUnpublish={(p) => unpublishMutation.mutate(p.id)}
                  onDelete={(p) => deleteMutation.mutate(p.id)}
                />
              ))}
            </div>
          )}

          {/* Load more */}
          {posts.length < total && (
            <div className="text-center mt-4">
              <Button variant="outline" size="sm" onClick={handleLoadMore}>
                Load More ({total - posts.length} remaining)
              </Button>
            </div>
          )}
        </TabsContent>
      </FilterTabs>

      {/* Blog editor — single entry point, has Write Manually / Have Agents Write It modes */}
      {isEditorOpen && (
        <BlogEditor
          postId={editingPostId}
          onClose={handleEditorClose}
        />
      )}
    </div>
  )
}
