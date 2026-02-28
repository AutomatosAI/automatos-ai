'use client'

import { useState, useEffect, useCallback } from 'react'
import { useParams, useRouter } from 'next/navigation'
import { motion } from 'framer-motion'
import {
  ArrowLeft,
  Star,
  Download,
  Puzzle,
  Loader2,
  Shield,
  Calendar,
  Package,
  User,
  ChevronLeft,
  ChevronRight,
  ImageIcon,
  FileText,
  MessageSquare,
  History,
} from 'lucide-react'
import { MainLayout } from '@/components/layout/main-layout'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Separator } from '@/components/ui/separator'
import { Textarea } from '@/components/ui/textarea'
import { apiClient } from '@/lib/api-client'
import { useToast } from '@/hooks/use-toast'

// ===================================================================
// Types
// ===================================================================

interface WidgetDetail {
  id: string
  name: string
  display_name: string
  description: string
  long_description: string | null
  developer_name: string
  icon_url: string | null
  screenshots: string[]
  readme: string | null
  changelog: string | null
  categories: string[]
  version: string
  bundle_size: number | null
  permissions: string[]
  pricing_type: string
  price_cents: number | null
  install_count: number
  rating_average: number
  rating_count: number
  published_at: string
  min_plan: string | null
}

interface Review {
  id: string
  user_name: string
  rating: number
  comment: string
  created_at: string
}

// ===================================================================
// Helpers
// ===================================================================

function formatInstallCount(count: number): string {
  if (count >= 1_000_000) return `${(count / 1_000_000).toFixed(1)}M`
  if (count >= 1_000) return `${(count / 1_000).toFixed(1)}k`
  return count.toString()
}

function formatPrice(pricingType: string, priceCents: number | null): string {
  if (pricingType === 'free') return 'Free'
  const dollars = ((priceCents ?? 0) / 100).toFixed(2)
  if (pricingType === 'subscription') return `$${dollars}/mo`
  return `$${dollars}`
}

function formatBundleSize(bytes: number): string {
  if (bytes >= 1_048_576) return `${(bytes / 1_048_576).toFixed(1)} MB`
  if (bytes >= 1_024) return `${(bytes / 1_024).toFixed(1)} KB`
  return `${bytes} B`
}

function formatDate(iso: string): string {
  return new Date(iso).toLocaleDateString('en-US', {
    year: 'numeric',
    month: 'short',
    day: 'numeric',
  })
}

/**
 * Basic HTML sanitiser -- strips <script> tags and inline event handlers.
 * For production you'd want DOMPurify; this covers the obvious XSS vectors.
 */
function sanitizeHtml(html: string): string {
  let clean = html
  // Remove <script>...</script> blocks (incl. multi-line)
  clean = clean.replace(/<script\b[^<]*(?:(?!<\/script>)<[^<]*)*<\/script\s*>/gi, '')
  // Remove on* event attributes (double- and single-quoted, or unquoted)
  clean = clean.replace(/\s+on\w+\s*=\s*(?:"[^"]*"|'[^']*'|[^\s>]+)/gi, '')
  // Remove javascript: hrefs
  clean = clean.replace(/href\s*=\s*["']?\s*javascript\s*:/gi, 'href="')
  return clean
}

// ===================================================================
// Sub-components
// ===================================================================

function RatingStars({ average, count, size = 'sm' }: { average: number; count: number; size?: 'sm' | 'lg' }) {
  const sizeClass = size === 'lg' ? 'h-5 w-5' : 'h-3.5 w-3.5'
  const textClass = size === 'lg' ? 'text-sm' : 'text-xs'

  return (
    <div className="flex items-center gap-1.5">
      <div className="flex items-center">
        {[1, 2, 3, 4, 5].map((star) => (
          <Star
            key={star}
            className={`${sizeClass} ${
              star <= Math.round(average)
                ? 'fill-yellow-400 text-yellow-400'
                : 'text-muted-foreground/40'
            }`}
          />
        ))}
      </div>
      <span className={`${textClass} text-muted-foreground`}>
        {average.toFixed(1)} ({count})
      </span>
    </div>
  )
}

function ScreenshotGallery({ screenshots }: { screenshots: string[] }) {
  const [activeIndex, setActiveIndex] = useState(0)

  if (screenshots.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center py-16 text-muted-foreground">
        <ImageIcon className="h-12 w-12 mb-3 opacity-40" />
        <p className="text-sm">No screenshots available</p>
      </div>
    )
  }

  return (
    <div className="space-y-4">
      {/* Main image */}
      <div className="relative aspect-video rounded-lg overflow-hidden bg-secondary/40 border border-border/40">
        <img
          src={screenshots[activeIndex]}
          alt={`Screenshot ${activeIndex + 1}`}
          className="w-full h-full object-contain"
        />
        {screenshots.length > 1 && (
          <>
            <button
              onClick={() => setActiveIndex((i) => (i - 1 + screenshots.length) % screenshots.length)}
              className="absolute left-2 top-1/2 -translate-y-1/2 p-1.5 rounded-full bg-background/80 backdrop-blur border border-border/40 hover:bg-background transition-colors"
              aria-label="Previous screenshot"
            >
              <ChevronLeft className="h-4 w-4" />
            </button>
            <button
              onClick={() => setActiveIndex((i) => (i + 1) % screenshots.length)}
              className="absolute right-2 top-1/2 -translate-y-1/2 p-1.5 rounded-full bg-background/80 backdrop-blur border border-border/40 hover:bg-background transition-colors"
              aria-label="Next screenshot"
            >
              <ChevronRight className="h-4 w-4" />
            </button>
          </>
        )}
      </div>

      {/* Thumbnails */}
      {screenshots.length > 1 && (
        <div className="flex gap-2 overflow-x-auto pb-1">
          {screenshots.map((src, i) => (
            <button
              key={i}
              onClick={() => setActiveIndex(i)}
              className={`flex-shrink-0 w-20 h-14 rounded-md overflow-hidden border-2 transition-colors ${
                i === activeIndex
                  ? 'border-primary'
                  : 'border-border/40 hover:border-border'
              }`}
            >
              <img src={src} alt={`Thumbnail ${i + 1}`} className="w-full h-full object-cover" />
            </button>
          ))}
        </div>
      )}
    </div>
  )
}

function ReviewItem({ review }: { review: Review }) {
  return (
    <div className="py-4 first:pt-0">
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center gap-2">
          <div className="h-8 w-8 rounded-full bg-secondary flex items-center justify-center">
            <User className="h-4 w-4 text-muted-foreground" />
          </div>
          <div>
            <p className="text-sm font-medium text-foreground">{review.user_name}</p>
            <p className="text-xs text-muted-foreground">{formatDate(review.created_at)}</p>
          </div>
        </div>
        <div className="flex items-center">
          {[1, 2, 3, 4, 5].map((star) => (
            <Star
              key={star}
              className={`h-3.5 w-3.5 ${
                star <= review.rating
                  ? 'fill-yellow-400 text-yellow-400'
                  : 'text-muted-foreground/40'
              }`}
            />
          ))}
        </div>
      </div>
      <p className="text-sm text-muted-foreground leading-relaxed">{review.comment}</p>
    </div>
  )
}

function ReviewForm({ widgetId, onSubmitted }: { widgetId: string; onSubmitted: () => void }) {
  const [rating, setRating] = useState(0)
  const [hoverRating, setHoverRating] = useState(0)
  const [comment, setComment] = useState('')
  const [submitting, setSubmitting] = useState(false)
  const { toast } = useToast()

  const handleSubmit = async () => {
    if (rating === 0) {
      toast({ title: 'Please select a rating', variant: 'destructive' })
      return
    }
    if (!comment.trim()) {
      toast({ title: 'Please write a review', variant: 'destructive' })
      return
    }

    try {
      setSubmitting(true)
      await apiClient.post(`/api/widget-marketplace/widgets/${widgetId}/reviews`, {
        rating,
        comment: comment.trim(),
      })
      toast({ title: 'Review submitted' })
      setRating(0)
      setComment('')
      onSubmitted()
    } catch {
      toast({ title: 'Failed to submit review', variant: 'destructive' })
    } finally {
      setSubmitting(false)
    }
  }

  return (
    <Card className="glass-card">
      <CardHeader className="pb-3">
        <CardTitle className="text-base">Write a Review</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* Star picker */}
        <div className="flex items-center gap-1">
          {[1, 2, 3, 4, 5].map((star) => (
            <button
              key={star}
              type="button"
              onClick={() => setRating(star)}
              onMouseEnter={() => setHoverRating(star)}
              onMouseLeave={() => setHoverRating(0)}
              className="p-0.5"
              aria-label={`Rate ${star} star${star > 1 ? 's' : ''}`}
            >
              <Star
                className={`h-6 w-6 transition-colors ${
                  star <= (hoverRating || rating)
                    ? 'fill-yellow-400 text-yellow-400'
                    : 'text-muted-foreground/40 hover:text-muted-foreground/60'
                }`}
              />
            </button>
          ))}
          {rating > 0 && (
            <span className="text-sm text-muted-foreground ml-2">{rating}/5</span>
          )}
        </div>

        <Textarea
          placeholder="Share your experience with this widget..."
          value={comment}
          onChange={(e) => setComment(e.target.value)}
          rows={3}
          className="bg-secondary/30"
        />

        <Button
          onClick={handleSubmit}
          disabled={submitting || rating === 0 || !comment.trim()}
          size="sm"
        >
          {submitting && <Loader2 className="h-4 w-4 mr-2 animate-spin" />}
          Submit Review
        </Button>
      </CardContent>
    </Card>
  )
}

// ===================================================================
// Main Page Component
// ===================================================================

export default function WidgetDetailPage() {
  const params = useParams()
  const router = useRouter()
  const { toast } = useToast()
  const widgetId = params.id as string

  const [widget, setWidget] = useState<WidgetDetail | null>(null)
  const [reviews, setReviews] = useState<Review[]>([])
  const [installed, setInstalled] = useState(false)
  const [loading, setLoading] = useState(true)
  const [installing, setInstalling] = useState(false)
  const [activeTab, setActiveTab] = useState('overview')

  // ------- Data fetching -------

  const fetchWidget = useCallback(async () => {
    try {
      setLoading(true)
      const data = await apiClient.get<WidgetDetail>(
        `/api/widget-marketplace/widgets/${widgetId}`
      )
      setWidget(data)
    } catch {
      toast({ title: 'Failed to load widget details', variant: 'destructive' })
    } finally {
      setLoading(false)
    }
  }, [widgetId, toast])

  const fetchReviews = useCallback(async () => {
    try {
      const data = await apiClient.get<Review[]>(
        `/api/widget-marketplace/widgets/${widgetId}/reviews`
      )
      setReviews(Array.isArray(data) ? data : [])
    } catch {
      // Reviews are non-critical; fail silently
      setReviews([])
    }
  }, [widgetId])

  const checkInstallStatus = useCallback(async () => {
    try {
      const data = await apiClient.get<{ widget_ids: string[] }>(
        '/api/widget-marketplace/installed'
      )
      if (data?.widget_ids) {
        setInstalled(data.widget_ids.includes(widgetId))
      }
    } catch {
      // Fall back: assume not installed
    }
  }, [widgetId])

  useEffect(() => {
    if (!widgetId) return
    fetchWidget()
    fetchReviews()
    checkInstallStatus()
  }, [widgetId, fetchWidget, fetchReviews, checkInstallStatus])

  // ------- Actions -------

  const handleInstallToggle = async () => {
    if (!widget) return
    try {
      setInstalling(true)
      if (installed) {
        await apiClient.delete(`/api/widget-marketplace/widgets/${widgetId}/install`)
        toast({ title: `${widget.display_name} uninstalled` })
        setInstalled(false)
      } else {
        await apiClient.post(`/api/widget-marketplace/widgets/${widgetId}/install`)
        toast({ title: `${widget.display_name} installed` })
        setInstalled(true)
      }
    } catch {
      toast({
        title: `Failed to ${installed ? 'uninstall' : 'install'} widget`,
        variant: 'destructive',
      })
    } finally {
      setInstalling(false)
    }
  }

  // ------- Loading state -------

  if (loading) {
    return (
      <MainLayout>
        <div className="flex items-center justify-center min-h-[60vh]">
          <Loader2 className="h-8 w-8 animate-spin text-primary" />
        </div>
      </MainLayout>
    )
  }

  if (!widget) {
    return (
      <MainLayout>
        <div className="flex flex-col items-center justify-center min-h-[60vh] gap-4">
          <p className="text-muted-foreground">Widget not found</p>
          <Button variant="outline" onClick={() => router.push('/marketplace')}>
            <ArrowLeft className="h-4 w-4 mr-2" />
            Back to Marketplace
          </Button>
        </div>
      </MainLayout>
    )
  }

  // ------- Render -------

  return (
    <MainLayout>
      <div className="space-y-6">
        {/* Back button */}
        <Button
          variant="ghost"
          size="sm"
          onClick={() => router.push('/marketplace')}
          className="text-muted-foreground hover:text-foreground -ml-2"
        >
          <ArrowLeft className="h-4 w-4 mr-1" />
          Back to Marketplace
        </Button>

        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.4 }}
        >
          <Card className="glass-card">
            <CardContent className="pt-6">
              <div className="flex flex-col sm:flex-row gap-6">
                {/* Icon */}
                <div className="flex-shrink-0 h-20 w-20 rounded-xl bg-secondary flex items-center justify-center overflow-hidden border border-border/40">
                  {widget.icon_url ? (
                    <img
                      src={widget.icon_url}
                      alt={widget.display_name}
                      className="h-20 w-20 rounded-xl object-cover"
                    />
                  ) : (
                    <Puzzle className="h-10 w-10 text-muted-foreground" />
                  )}
                </div>

                {/* Details */}
                <div className="flex-1 min-w-0 space-y-2">
                  <div>
                    <h1 className="text-2xl md:text-3xl font-bold text-foreground">
                      {widget.display_name}
                    </h1>
                    <p className="text-sm text-muted-foreground">
                      by {widget.developer_name}
                    </p>
                  </div>

                  <p className="text-sm text-muted-foreground leading-relaxed">
                    {widget.description}
                  </p>

                  <div className="flex flex-wrap items-center gap-4 pt-1">
                    <RatingStars
                      average={widget.rating_average}
                      count={widget.rating_count}
                      size="lg"
                    />
                    <Separator orientation="vertical" className="h-5 hidden sm:block" />
                    <div className="flex items-center gap-1.5 text-sm text-muted-foreground">
                      <Download className="h-4 w-4 text-[hsl(var(--success))]" />
                      <span>{formatInstallCount(widget.install_count)} installs</span>
                    </div>
                    <Separator orientation="vertical" className="h-5 hidden sm:block" />
                    <Badge
                      className={`text-xs ${
                        widget.pricing_type === 'free'
                          ? 'bg-[hsl(var(--success))]/20 text-[hsl(var(--success))] border-[hsl(var(--success))]/30'
                          : 'bg-primary/20 text-primary border-primary/30'
                      }`}
                    >
                      {formatPrice(widget.pricing_type, widget.price_cents)}
                    </Badge>
                  </div>
                </div>

                {/* Install button */}
                <div className="flex-shrink-0 self-start">
                  <Button
                    onClick={handleInstallToggle}
                    disabled={installing}
                    variant={installed ? 'outline' : 'default'}
                    size="lg"
                    className={
                      installed
                        ? 'border-destructive/50 text-destructive hover:bg-destructive/10'
                        : ''
                    }
                  >
                    {installing ? (
                      <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                    ) : (
                      <Download className="h-4 w-4 mr-2" />
                    )}
                    {installed ? 'Uninstall' : 'Install'}
                  </Button>
                </div>
              </div>
            </CardContent>
          </Card>
        </motion.div>

        {/* Body: Tabs + Sidebar */}
        <div className="grid grid-cols-1 lg:grid-cols-[1fr_320px] gap-6">
          {/* Main content */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.4, delay: 0.1 }}
          >
            <Tabs value={activeTab} onValueChange={setActiveTab}>
              <TabsList>
                <TabsTrigger value="overview">
                  <FileText className="h-4 w-4 mr-1.5" />
                  Overview
                </TabsTrigger>
                <TabsTrigger value="screenshots">
                  <ImageIcon className="h-4 w-4 mr-1.5" />
                  Screenshots
                </TabsTrigger>
                <TabsTrigger value="reviews">
                  <MessageSquare className="h-4 w-4 mr-1.5" />
                  Reviews
                  {widget.rating_count > 0 && (
                    <span className="ml-1 text-xs text-muted-foreground">
                      ({widget.rating_count})
                    </span>
                  )}
                </TabsTrigger>
                <TabsTrigger value="changelog">
                  <History className="h-4 w-4 mr-1.5" />
                  Changelog
                </TabsTrigger>
              </TabsList>

              {/* Overview tab */}
              <TabsContent value="overview" className="mt-4">
                <Card className="glass-card">
                  <CardContent className="pt-6">
                    {widget.readme ? (
                      <div
                        className="prose prose-sm dark:prose-invert max-w-none prose-headings:text-foreground prose-p:text-muted-foreground prose-a:text-primary prose-strong:text-foreground prose-code:bg-secondary prose-code:px-1.5 prose-code:py-0.5 prose-code:rounded prose-pre:bg-secondary/60 prose-pre:border prose-pre:border-border/40"
                        dangerouslySetInnerHTML={{
                          __html: sanitizeHtml(widget.readme),
                        }}
                      />
                    ) : widget.long_description ? (
                      <pre className="whitespace-pre-wrap text-sm text-muted-foreground font-sans leading-relaxed">
                        {widget.long_description}
                      </pre>
                    ) : (
                      <p className="text-sm text-muted-foreground">
                        No detailed description available.
                      </p>
                    )}
                  </CardContent>
                </Card>
              </TabsContent>

              {/* Screenshots tab */}
              <TabsContent value="screenshots" className="mt-4">
                <Card className="glass-card">
                  <CardContent className="pt-6">
                    <ScreenshotGallery screenshots={widget.screenshots} />
                  </CardContent>
                </Card>
              </TabsContent>

              {/* Reviews tab */}
              <TabsContent value="reviews" className="mt-4 space-y-4">
                <ReviewForm widgetId={widgetId} onSubmitted={fetchReviews} />

                <Card className="glass-card">
                  <CardHeader className="pb-3">
                    <CardTitle className="text-base">
                      Reviews ({reviews.length})
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    {reviews.length > 0 ? (
                      <div className="divide-y divide-border">
                        {reviews.map((review) => (
                          <ReviewItem key={review.id} review={review} />
                        ))}
                      </div>
                    ) : (
                      <p className="text-sm text-muted-foreground py-4 text-center">
                        No reviews yet. Be the first to review this widget.
                      </p>
                    )}
                  </CardContent>
                </Card>
              </TabsContent>

              {/* Changelog tab */}
              <TabsContent value="changelog" className="mt-4">
                <Card className="glass-card">
                  <CardContent className="pt-6">
                    {widget.changelog ? (
                      <pre className="whitespace-pre-wrap text-sm text-muted-foreground font-sans leading-relaxed">
                        {widget.changelog}
                      </pre>
                    ) : (
                      <p className="text-sm text-muted-foreground">
                        No changelog available.
                      </p>
                    )}
                  </CardContent>
                </Card>
              </TabsContent>
            </Tabs>
          </motion.div>

          {/* Sidebar */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.4, delay: 0.2 }}
            className="space-y-4"
          >
            {/* Info card */}
            <Card className="glass-card">
              <CardHeader className="pb-3">
                <CardTitle className="text-base">Details</CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                {/* Developer */}
                <div className="flex items-center gap-2">
                  <User className="h-4 w-4 text-muted-foreground flex-shrink-0" />
                  <div>
                    <p className="text-xs text-muted-foreground">Developer</p>
                    <p className="text-sm font-medium text-foreground">
                      {widget.developer_name}
                    </p>
                  </div>
                </div>

                <Separator />

                {/* Version */}
                <div className="flex items-center gap-2">
                  <Package className="h-4 w-4 text-muted-foreground flex-shrink-0" />
                  <div>
                    <p className="text-xs text-muted-foreground">Version</p>
                    <p className="text-sm font-medium text-foreground">{widget.version}</p>
                  </div>
                </div>

                <Separator />

                {/* Bundle size */}
                {widget.bundle_size != null && (
                  <>
                    <div className="flex items-center gap-2">
                      <Package className="h-4 w-4 text-muted-foreground flex-shrink-0" />
                      <div>
                        <p className="text-xs text-muted-foreground">Bundle Size</p>
                        <p className="text-sm font-medium text-foreground">
                          {formatBundleSize(widget.bundle_size)}
                        </p>
                      </div>
                    </div>
                    <Separator />
                  </>
                )}

                {/* Published date */}
                <div className="flex items-center gap-2">
                  <Calendar className="h-4 w-4 text-muted-foreground flex-shrink-0" />
                  <div>
                    <p className="text-xs text-muted-foreground">Published</p>
                    <p className="text-sm font-medium text-foreground">
                      {formatDate(widget.published_at)}
                    </p>
                  </div>
                </div>

                <Separator />

                {/* Pricing */}
                <div>
                  <p className="text-xs text-muted-foreground mb-1.5">Pricing</p>
                  <Badge
                    className={`text-xs ${
                      widget.pricing_type === 'free'
                        ? 'bg-[hsl(var(--success))]/20 text-[hsl(var(--success))] border-[hsl(var(--success))]/30'
                        : 'bg-primary/20 text-primary border-primary/30'
                    }`}
                  >
                    {formatPrice(widget.pricing_type, widget.price_cents)}
                  </Badge>
                  {widget.min_plan && (
                    <p className="text-xs text-muted-foreground mt-1">
                      Requires {widget.min_plan} plan
                    </p>
                  )}
                </div>
              </CardContent>
            </Card>

            {/* Categories */}
            {widget.categories.length > 0 && (
              <Card className="glass-card">
                <CardHeader className="pb-3">
                  <CardTitle className="text-base">Categories</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="flex flex-wrap gap-2">
                    {widget.categories.map((cat) => (
                      <Badge
                        key={cat}
                        variant="outline"
                        className="text-xs border-border text-muted-foreground"
                      >
                        {cat}
                      </Badge>
                    ))}
                  </div>
                </CardContent>
              </Card>
            )}

            {/* Permissions */}
            {widget.permissions.length > 0 && (
              <Card className="glass-card">
                <CardHeader className="pb-3">
                  <CardTitle className="text-base flex items-center gap-2">
                    <Shield className="h-4 w-4" />
                    Permissions
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <ul className="space-y-2">
                    {widget.permissions.map((perm) => (
                      <li
                        key={perm}
                        className="flex items-start gap-2 text-sm text-muted-foreground"
                      >
                        <Shield className="h-3.5 w-3.5 mt-0.5 text-[hsl(var(--warning))] flex-shrink-0" />
                        <span>{perm}</span>
                      </li>
                    ))}
                  </ul>
                </CardContent>
              </Card>
            )}
          </motion.div>
        </div>
      </div>
    </MainLayout>
  )
}
