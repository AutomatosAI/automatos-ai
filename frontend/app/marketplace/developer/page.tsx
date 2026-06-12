'use client'

import { useState, useEffect } from 'react'
import Link from 'next/link'
import { useRouter } from 'next/navigation'
import {
  Package,
  Download,
  Star,
  MessageSquare,
  Plus,
  ExternalLink,
  Edit,
  Loader2,
} from 'lucide-react'
import { MainLayout } from '@/components/layout/main-layout'
import { PageHeader } from '@/components/shared/page-header'
import { ErrorState } from '@/components/shared'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Card, CardContent } from '@/components/ui/card'
import { Skeleton } from '@/components/ui/skeleton'
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table'

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface DeveloperWidget {
  id: string
  name: string
  display_name: string
  version: string
  status: 'draft' | 'review' | 'published' | 'suspended'
  install_count: number
  rating_average: number
  rating_count: number
  created_at: string
  updated_at: string
}

interface DeveloperAnalytics {
  total_widgets: number
  total_installs: number
  average_rating: number
  total_reviews: number
}

// ---------------------------------------------------------------------------
// Constants & helpers
// ---------------------------------------------------------------------------

const API_BASE = process.env.NEXT_PUBLIC_API_URL || ''

function getWorkspaceHeaders(): Record<string, string> {
  const workspaceId =
    typeof window !== 'undefined'
      ? localStorage.getItem('last_active_workspace') ||
        localStorage.getItem('last_active_org')
      : null
  const headers: Record<string, string> = { 'Content-Type': 'application/json' }
  if (workspaceId) headers['X-Workspace-ID'] = workspaceId
  return headers
}

async function apiFetch<T>(path: string): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    headers: getWorkspaceHeaders(),
    credentials: 'include',
  })
  if (!res.ok) throw new Error(`API ${res.status}: ${res.statusText}`)
  return res.json()
}

function formatCount(n: number): string {
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`
  if (n >= 1_000) return `${(n / 1_000).toFixed(1)}k`
  return n.toString()
}

const STATUS_STYLES: Record<
  DeveloperWidget['status'],
  { label: string; className: string }
> = {
  draft: {
    label: 'Draft',
    className:
      'bg-muted text-muted-foreground border-border',
  },
  review: {
    label: 'In Review',
    className:
      'bg-[hsl(var(--warning))]/15 text-[hsl(var(--warning))] border-[hsl(var(--warning))]/30',
  },
  published: {
    label: 'Published',
    className:
      'bg-[hsl(var(--success))]/15 text-[hsl(var(--success))] border-[hsl(var(--success))]/30',
  },
  suspended: {
    label: 'Suspended',
    className:
      'bg-destructive/15 text-destructive border-destructive/30',
  },
}

// ---------------------------------------------------------------------------
// Star rating display
// ---------------------------------------------------------------------------

function StarRating({ value, className }: { value: number; className?: string }) {
  return (
    <span className={`inline-flex items-center gap-0.5 ${className ?? ''}`}>
      {[1, 2, 3, 4, 5].map((i) => (
        <Star
          key={i}
          className={`h-3.5 w-3.5 ${
            i <= Math.round(value)
              ? 'text-[hsl(var(--warning))] fill-[hsl(var(--warning))]'
              : 'text-muted-foreground/30'
          }`}
        />
      ))}
      <span className="ml-1 text-xs text-muted-foreground tabular-nums">
        {value.toFixed(1)}
      </span>
    </span>
  )
}

// ---------------------------------------------------------------------------
// Stat card
// ---------------------------------------------------------------------------

interface StatCardProps {
  label: string
  value: string | number
  icon: React.ReactNode
  detail?: React.ReactNode
  loading?: boolean
}

function StatCard({ label, value, icon, detail, loading }: StatCardProps) {
  return (
    <Card className="glass-card card-glow">
      <CardContent className="p-5">
        {loading ? (
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <Skeleton className="h-4 w-24" />
              <Skeleton className="h-8 w-8 rounded-lg" />
            </div>
            <Skeleton className="h-8 w-20" />
          </div>
        ) : (
          <>
            <div className="flex items-center justify-between mb-3">
              <p className="text-sm text-muted-foreground">{label}</p>
              <div className="h-8 w-8 rounded-lg bg-primary/10 flex items-center justify-center">
                {icon}
              </div>
            </div>
            <p className="text-2xl font-bold tabular-nums">{value}</p>
            {detail && <div className="mt-1">{detail}</div>}
          </>
        )}
      </CardContent>
    </Card>
  )
}

// ---------------------------------------------------------------------------
// Page
// ---------------------------------------------------------------------------

export default function DeveloperDashboardPage() {
  const router = useRouter()

  const [widgets, setWidgets] = useState<DeveloperWidget[]>([])
  const [analytics, setAnalytics] = useState<DeveloperAnalytics | null>(null)
  const [loadingWidgets, setLoadingWidgets] = useState(true)
  const [loadingAnalytics, setLoadingAnalytics] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [refreshKey, setRefreshKey] = useState(0)

  // --- fetch data on mount --------------------------------------------------
  useEffect(() => {
    let cancelled = false
    setError(null)
    setLoadingWidgets(true)
    setLoadingAnalytics(true)

    async function fetchWidgets() {
      try {
        const data = await apiFetch<DeveloperWidget[]>(
          '/api/widget-marketplace/developer/widgets',
        )
        if (!cancelled) setWidgets(data)
      } catch (err) {
        console.error('Failed to fetch developer widgets:', err)
        if (!cancelled) setError('Failed to load your widgets.')
      } finally {
        if (!cancelled) setLoadingWidgets(false)
      }
    }

    async function fetchAnalytics() {
      try {
        const data = await apiFetch<DeveloperAnalytics>(
          '/api/widget-marketplace/developer/analytics',
        )
        if (!cancelled) setAnalytics(data)
      } catch (err) {
        console.error('Failed to fetch developer analytics:', err)
      } finally {
        if (!cancelled) setLoadingAnalytics(false)
      }
    }

    fetchWidgets()
    fetchAnalytics()

    return () => {
      cancelled = true
    }
  }, [refreshKey])

  // --- render ---------------------------------------------------------------
  return (
    <MainLayout>
      <div className="space-y-8">
        {/* Header */}
        <PageHeader
          title="Developer"
          titleAccent="Dashboard"
          eyebrow="Marketplace · your widgets"
          lede="Everything you've published. Installs, reviews, version history, and the queue of widgets still in review."
          actions={
            <Button
              onClick={() => router.push('/marketplace/publish')}
              className="gap-2"
            >
              <Plus className="h-4 w-4" />
              Create Widget
            </Button>
          }
        />

        {/* Stats Cards */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          <StatCard
            label="Published Widgets"
            value={analytics?.total_widgets ?? 0}
            icon={<Package className="h-4 w-4 text-primary" />}
            loading={loadingAnalytics}
          />
          <StatCard
            label="Total Installs"
            value={formatCount(analytics?.total_installs ?? 0)}
            icon={<Download className="h-4 w-4 text-[hsl(var(--success))]" />}
            loading={loadingAnalytics}
          />
          <StatCard
            label="Average Rating"
            value={analytics?.average_rating?.toFixed(1) ?? '0.0'}
            icon={<Star className="h-4 w-4 text-[hsl(var(--warning))]" />}
            detail={
              analytics ? (
                <StarRating value={analytics.average_rating} />
              ) : null
            }
            loading={loadingAnalytics}
          />
          <StatCard
            label="Total Reviews"
            value={formatCount(analytics?.total_reviews ?? 0)}
            icon={<MessageSquare className="h-4 w-4 text-info" />}
            loading={loadingAnalytics}
          />
        </div>

        {/* My Widgets Table */}
        <div className="space-y-4">
          <h2 className="text-lg font-semibold">My Widgets</h2>

          {loadingWidgets ? (
            <Card className="glass-card">
              <CardContent className="p-0">
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead>Name</TableHead>
                      <TableHead>Version</TableHead>
                      <TableHead>Status</TableHead>
                      <TableHead className="text-right">Installs</TableHead>
                      <TableHead>Rating</TableHead>
                      <TableHead className="text-right">Actions</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {[...Array(3)].map((_, i) => (
                      <TableRow key={i}>
                        <TableCell>
                          <Skeleton className="h-4 w-36" />
                        </TableCell>
                        <TableCell>
                          <Skeleton className="h-4 w-14" />
                        </TableCell>
                        <TableCell>
                          <Skeleton className="h-5 w-20 rounded-full" />
                        </TableCell>
                        <TableCell className="text-right">
                          <Skeleton className="h-4 w-10 ml-auto" />
                        </TableCell>
                        <TableCell>
                          <Skeleton className="h-4 w-24" />
                        </TableCell>
                        <TableCell className="text-right">
                          <Skeleton className="h-8 w-20 ml-auto" />
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </CardContent>
            </Card>
          ) : error ? (
            <Card className="glass-card">
              <CardContent>
                <ErrorState description={error} onRetry={() => setRefreshKey((k) => k + 1)} />
              </CardContent>
            </Card>
          ) : widgets.length === 0 ? (
            /* Empty state */
            <Card className="glass-card">
              <CardContent className="py-16 text-center">
                <div className="w-16 h-16 rounded-lg bg-secondary/30 flex items-center justify-center mx-auto mb-4">
                  <Package className="w-8 h-8 text-muted-foreground" />
                </div>
                <h3 className="text-lg font-semibold mb-2">
                  No widgets yet
                </h3>
                <p className="text-sm text-muted-foreground mb-6 max-w-md mx-auto">
                  You haven&apos;t published any widgets. Create your first widget to
                  share it with the marketplace.
                </p>
                <Button
                  onClick={() => router.push('/marketplace/publish')}
                  className="gap-2"
                >
                  <Plus className="h-4 w-4" />
                  Create your first widget
                </Button>
              </CardContent>
            </Card>
          ) : (
            /* Widgets table */
            <Card className="glass-card">
              <CardContent className="p-0">
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead>Name</TableHead>
                      <TableHead>Version</TableHead>
                      <TableHead>Status</TableHead>
                      <TableHead className="text-right">Installs</TableHead>
                      <TableHead>Rating</TableHead>
                      <TableHead className="text-right">Actions</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {widgets.map((widget) => {
                      const statusStyle = STATUS_STYLES[widget.status]
                      return (
                        <TableRow key={widget.id}>
                          <TableCell className="font-medium">
                            {widget.display_name}
                          </TableCell>
                          <TableCell>
                            <code className="text-xs bg-secondary/50 px-1.5 py-0.5 rounded">
                              {widget.version}
                            </code>
                          </TableCell>
                          <TableCell>
                            <Badge
                              variant="outline"
                              className={statusStyle.className}
                            >
                              {statusStyle.label}
                            </Badge>
                          </TableCell>
                          <TableCell className="text-right tabular-nums">
                            {formatCount(widget.install_count)}
                          </TableCell>
                          <TableCell>
                            {widget.rating_count > 0 ? (
                              <StarRating value={widget.rating_average} />
                            ) : (
                              <span className="text-xs text-muted-foreground">
                                No reviews
                              </span>
                            )}
                          </TableCell>
                          <TableCell className="text-right">
                            <div className="flex items-center justify-end gap-1">
                              <Button
                                variant="ghost"
                                size="sm"
                                className="h-8 w-8 p-0"
                                asChild
                              >
                                <Link
                                  href={`/marketplace/publish?edit=${widget.id}`}
                                  title="Edit widget"
                                >
                                  <Edit className="h-4 w-4" />
                                </Link>
                              </Button>
                              <Button
                                variant="ghost"
                                size="sm"
                                className="h-8 w-8 p-0"
                                asChild
                              >
                                <Link
                                  href={`/marketplace/widgets/${widget.id}`}
                                  title="View in marketplace"
                                >
                                  <ExternalLink className="h-4 w-4" />
                                </Link>
                              </Button>
                            </div>
                          </TableCell>
                        </TableRow>
                      )
                    })}
                  </TableBody>
                </Table>
              </CardContent>
            </Card>
          )}
        </div>
      </div>
    </MainLayout>
  )
}
