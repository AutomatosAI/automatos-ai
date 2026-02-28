'use client'

import { useState, useEffect, useCallback, useRef } from 'react'
import { useRouter } from 'next/navigation'
import { Search, Star, Download, ChevronLeft, ChevronRight, Package } from 'lucide-react'
import { MainLayout } from '@/components/layout/main-layout'
import { PageHeader } from '@/components/shared/page-header'
import { Input } from '@/components/ui/input'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Skeleton } from '@/components/ui/skeleton'
import { Card, CardContent, CardHeader, CardFooter } from '@/components/ui/card'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'
import { WidgetCard } from '@/components/marketplace/WidgetCard'
import { WidgetGrid } from '@/components/marketplace/WidgetGrid'

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface WidgetSummary {
  id: string
  name: string
  display_name: string
  description: string
  developer_name: string
  icon_url: string | null
  categories: string[]
  pricing_type: 'free' | 'one_time' | 'subscription'
  price_cents: number | null
  install_count: number
  rating_average: number
  rating_count: number
  version: string
}

interface CategoryInfo {
  name: string
  count: number
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const API_BASE = process.env.NEXT_PUBLIC_API_URL || ''
const PAGE_SIZE = 24

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function useDebounce<T>(value: T, delay: number): T {
  const [debounced, setDebounced] = useState(value)

  useEffect(() => {
    const timer = setTimeout(() => setDebounced(value), delay)
    return () => clearTimeout(timer)
  }, [value, delay])

  return debounced
}

async function apiFetch<T>(path: string): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    headers: { 'Content-Type': 'application/json' },
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

function formatPrice(type: WidgetSummary['pricing_type'], cents: number | null): string {
  if (type === 'free') return 'Free'
  if (cents == null) return type === 'subscription' ? 'Subscription' : 'Paid'
  const dollars = (cents / 100).toFixed(2)
  return type === 'subscription' ? `$${dollars}/mo` : `$${dollars}`
}

// ---------------------------------------------------------------------------
// Page Component
// ---------------------------------------------------------------------------

export default function WidgetMarketplacePage() {
  const router = useRouter()

  // --- state ---------------------------------------------------------------
  const [widgets, setWidgets] = useState<WidgetSummary[]>([])
  const [featured, setFeatured] = useState<WidgetSummary[]>([])
  const [categories, setCategories] = useState<CategoryInfo[]>([])
  const [search, setSearch] = useState('')
  const [selectedCategory, setSelectedCategory] = useState<string | null>(null)
  const [sort, setSort] = useState('popular')
  const [page, setPage] = useState(1)
  const [total, setTotal] = useState(0)
  const [loading, setLoading] = useState(true)

  const debouncedSearch = useDebounce(search, 300)
  const featuredScrollRef = useRef<HTMLDivElement>(null)

  // --- fetch featured + categories on mount --------------------------------
  useEffect(() => {
    Promise.allSettled([
      apiFetch<WidgetSummary[]>('/api/widget-marketplace/featured'),
      apiFetch<CategoryInfo[]>('/api/widget-marketplace/categories'),
    ]).then(([featuredResult, categoriesResult]) => {
      if (featuredResult.status === 'fulfilled') setFeatured(featuredResult.value)
      if (categoriesResult.status === 'fulfilled') setCategories(categoriesResult.value)
    })
  }, [])

  // --- fetch widgets when filters change -----------------------------------
  useEffect(() => {
    let cancelled = false

    async function fetchWidgets() {
      setLoading(true)
      try {
        const params = new URLSearchParams()
        if (debouncedSearch) params.set('search', debouncedSearch)
        if (selectedCategory) params.set('category', selectedCategory)
        params.set('sort', sort)
        params.set('page', String(page))
        params.set('limit', String(PAGE_SIZE))

        const data = await apiFetch<{ items: WidgetSummary[]; total: number }>(
          `/api/widget-marketplace/widgets?${params}`,
        )

        if (!cancelled) {
          setWidgets(data.items)
          setTotal(data.total)
        }
      } catch (err) {
        console.error('Failed to fetch widgets:', err)
        if (!cancelled) {
          setWidgets([])
          setTotal(0)
        }
      } finally {
        if (!cancelled) setLoading(false)
      }
    }

    fetchWidgets()
    return () => { cancelled = true }
  }, [debouncedSearch, selectedCategory, sort, page])

  // Reset page when filters change
  useEffect(() => {
    setPage(1)
  }, [debouncedSearch, selectedCategory, sort])

  // --- handlers ------------------------------------------------------------
  const handleWidgetClick = useCallback(
    (id: string) => router.push(`/marketplace/widgets/${id}`),
    [router],
  )

  const totalPages = Math.max(1, Math.ceil(total / PAGE_SIZE))

  // --- render --------------------------------------------------------------
  return (
    <MainLayout>
      <div className="space-y-8">
        {/* Header */}
        <PageHeader
          title="Widget"
          titleAccent="Marketplace"
          subtitle="Browse, discover, and install widgets to extend your workspace"
        />

        {/* Search Bar */}
        <div className="relative max-w-2xl">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 text-muted-foreground w-5 h-5" />
          <Input
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            placeholder="Search widgets by name, description, or developer..."
            className="pl-11 pr-4 h-12 rounded-full bg-secondary/50 border-secondary focus-visible:border-primary/50 focus-visible:shadow-[0_0_12px_hsla(var(--primary)/0.15)] text-base"
          />
        </div>

        {/* Featured Widgets */}
        {featured.length > 0 && (
          <section className="space-y-3">
            <div className="flex items-center justify-between">
              <h2 className="text-lg font-semibold flex items-center gap-2">
                <Star className="w-5 h-5 text-[hsl(var(--warning))]" />
                Featured Widgets
              </h2>
              <div className="flex gap-1">
                <Button
                  variant="ghost"
                  size="sm"
                  className="h-8 w-8 p-0"
                  onClick={() => featuredScrollRef.current?.scrollBy({ left: -320, behavior: 'smooth' })}
                >
                  <ChevronLeft className="h-4 w-4" />
                </Button>
                <Button
                  variant="ghost"
                  size="sm"
                  className="h-8 w-8 p-0"
                  onClick={() => featuredScrollRef.current?.scrollBy({ left: 320, behavior: 'smooth' })}
                >
                  <ChevronRight className="h-4 w-4" />
                </Button>
              </div>
            </div>

            <div
              ref={featuredScrollRef}
              className="flex gap-4 overflow-x-auto pb-2 scrollbar-thin scrollbar-thumb-secondary scrollbar-track-transparent snap-x snap-mandatory"
            >
              {featured.map((widget) => (
                <Card
                  key={widget.id}
                  role="button"
                  tabIndex={0}
                  className="min-w-[300px] max-w-[300px] snap-start cursor-pointer glass-card card-glow hover:border-primary/50 transition-all duration-200 hover:shadow-lg hover:shadow-primary/10 focus:outline-none focus:ring-2 focus:ring-primary/50 shrink-0"
                  onClick={() => handleWidgetClick(widget.id)}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter' || e.key === ' ') {
                      e.preventDefault()
                      handleWidgetClick(widget.id)
                    }
                  }}
                >
                  <CardHeader className="pb-2">
                    <div className="flex items-center gap-3">
                      {widget.icon_url ? (
                        <img
                          src={widget.icon_url}
                          alt=""
                          className="w-10 h-10 rounded-lg object-cover"
                        />
                      ) : (
                        <div className="w-10 h-10 rounded-lg bg-primary/10 flex items-center justify-center">
                          <Package className="w-5 h-5 text-primary" />
                        </div>
                      )}
                      <div className="min-w-0 flex-1">
                        <h3 className="font-semibold text-sm truncate">{widget.display_name}</h3>
                        <p className="text-xs text-muted-foreground truncate">{widget.developer_name}</p>
                      </div>
                      <Badge className="bg-[hsl(var(--warning))]/20 text-[hsl(var(--warning))] border-[hsl(var(--warning))]/30 shrink-0">
                        <Star className="w-3 h-3 mr-1" />
                        Featured
                      </Badge>
                    </div>
                  </CardHeader>
                  <CardContent className="pb-2">
                    <p className="text-sm text-muted-foreground line-clamp-2">{widget.description}</p>
                  </CardContent>
                  <CardFooter className="pt-2 border-t border-border flex items-center justify-between text-xs text-muted-foreground">
                    <div className="flex items-center gap-3">
                      <span className="flex items-center gap-1">
                        <Download className="h-3.5 w-3.5 text-[hsl(var(--success))]" />
                        {formatCount(widget.install_count)}
                      </span>
                      {widget.rating_count > 0 && (
                        <span className="flex items-center gap-1">
                          <Star className="h-3.5 w-3.5 text-[hsl(var(--warning))] fill-[hsl(var(--warning))]" />
                          {widget.rating_average.toFixed(1)}
                        </span>
                      )}
                    </div>
                    <Badge variant="outline" className="text-[10px] h-5">
                      {formatPrice(widget.pricing_type, widget.price_cents)}
                    </Badge>
                  </CardFooter>
                </Card>
              ))}
            </div>
          </section>
        )}

        {/* Category Filter Chips */}
        {categories.length > 0 && (
          <div className="flex gap-2 overflow-x-auto pb-1 scrollbar-thin scrollbar-thumb-secondary scrollbar-track-transparent">
            <Badge
              role="button"
              tabIndex={0}
              className={`cursor-pointer whitespace-nowrap transition-colors ${
                selectedCategory === null
                  ? 'bg-primary text-primary-foreground hover:bg-primary/90'
                  : 'bg-secondary/50 text-muted-foreground hover:bg-secondary border border-border'
              }`}
              onClick={() => setSelectedCategory(null)}
              onKeyDown={(e) => {
                if (e.key === 'Enter' || e.key === ' ') {
                  e.preventDefault()
                  setSelectedCategory(null)
                }
              }}
            >
              All
            </Badge>
            {categories.map((cat) => (
              <Badge
                key={cat.name}
                role="button"
                tabIndex={0}
                className={`cursor-pointer whitespace-nowrap transition-colors ${
                  selectedCategory === cat.name
                    ? 'bg-primary text-primary-foreground hover:bg-primary/90'
                    : 'bg-secondary/50 text-muted-foreground hover:bg-secondary border border-border'
                }`}
                onClick={() => setSelectedCategory(cat.name)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' || e.key === ' ') {
                    e.preventDefault()
                    setSelectedCategory(cat.name)
                  }
                }}
              >
                {cat.name}
                <span className="ml-1.5 text-[10px] opacity-70">{cat.count}</span>
              </Badge>
            ))}
          </div>
        )}

        {/* Sort Controls + Result Count */}
        <div className="flex items-center justify-between">
          <p className="text-sm text-muted-foreground">
            {loading ? (
              <Skeleton className="h-4 w-32 inline-block" />
            ) : (
              <>
                {total} widget{total !== 1 ? 's' : ''} found
                {selectedCategory && (
                  <> in <span className="font-medium text-foreground">{selectedCategory}</span></>
                )}
              </>
            )}
          </p>

          <Select value={sort} onValueChange={setSort}>
            <SelectTrigger className="w-[180px]">
              <SelectValue placeholder="Sort by" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="popular">Popular</SelectItem>
              <SelectItem value="newest">Newest</SelectItem>
              <SelectItem value="highest_rated">Highest Rated</SelectItem>
            </SelectContent>
          </Select>
        </div>

        {/* Widget Grid */}
        {loading ? (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
            {[...Array(PAGE_SIZE > 8 ? 8 : PAGE_SIZE)].map((_, i) => (
              <Card key={i} className="glass-card">
                <CardHeader className="pb-2">
                  <div className="flex items-center gap-3">
                    <Skeleton className="w-10 h-10 rounded-lg" />
                    <div className="space-y-2 flex-1">
                      <Skeleton className="h-4 w-3/4" />
                      <Skeleton className="h-3 w-1/2" />
                    </div>
                  </div>
                </CardHeader>
                <CardContent>
                  <Skeleton className="h-4 w-full mb-2" />
                  <Skeleton className="h-4 w-2/3" />
                </CardContent>
                <CardFooter className="pt-3 border-t border-border">
                  <Skeleton className="h-4 w-24" />
                </CardFooter>
              </Card>
            ))}
          </div>
        ) : widgets.length === 0 ? (
          <div className="text-center py-16">
            <div className="w-16 h-16 rounded-lg bg-secondary/30 flex items-center justify-center mx-auto mb-4">
              <Package className="w-8 h-8 text-muted-foreground" />
            </div>
            <h3 className="text-lg font-semibold mb-2">No widgets found</h3>
            <p className="text-muted-foreground mb-4">
              {search
                ? `No widgets match "${search}"`
                : 'No widgets available in this category yet'}
            </p>
            <Button
              variant="outline"
              onClick={() => {
                setSearch('')
                setSelectedCategory(null)
                setSort('popular')
              }}
            >
              Clear Filters
            </Button>
          </div>
        ) : (
          <WidgetGrid>
            {widgets.map((widget) => (
              <WidgetCard
                key={widget.id}
                widget={widget}
                onClick={() => handleWidgetClick(widget.id)}
              />
            ))}
          </WidgetGrid>
        )}

        {/* Pagination */}
        {!loading && totalPages > 1 && (
          <div className="flex items-center justify-center gap-4 pt-4">
            <Button
              variant="outline"
              size="sm"
              disabled={page <= 1}
              onClick={() => setPage((p) => Math.max(1, p - 1))}
              className="gap-1"
            >
              <ChevronLeft className="h-4 w-4" />
              Previous
            </Button>

            <span className="text-sm text-muted-foreground tabular-nums">
              Page {page} of {totalPages}
            </span>

            <Button
              variant="outline"
              size="sm"
              disabled={page >= totalPages}
              onClick={() => setPage((p) => Math.min(totalPages, p + 1))}
              className="gap-1"
            >
              Next
              <ChevronRight className="h-4 w-4" />
            </Button>
          </div>
        )}
      </div>
    </MainLayout>
  )
}
