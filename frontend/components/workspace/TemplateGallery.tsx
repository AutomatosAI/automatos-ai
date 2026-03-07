'use client'

/**
 * US-009: TemplateGallery — browse and clone workspace templates.
 *
 * Fetches GET /api/workspaces/templates, displays cards in a responsive
 * grid, supports category filter chips, and clones via
 * POST /api/workspaces/from-template/{id}.
 */

import { useCallback, useEffect, useMemo, useState } from 'react'
import apiClient from '@/lib/api-client'
import { useWorkspaceStore } from '@/stores/workspace-store'
import { toast } from 'sonner'
import { cn } from '@/lib/utils'
import { Card, CardContent, CardFooter, CardHeader } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Skeleton } from '@/components/ui/skeleton'
import { LayoutGrid, Loader2 } from 'lucide-react'

// ── Types ──────────────────────────────────────────────────────────────

interface WorkspaceTemplate {
  id: string
  name: string
  description: string | null
  icon: string | null
  category: string | null
  layout_mode: string
  widgets: Array<Record<string, unknown>> | null
}

// ── Constants ──────────────────────────────────────────────────────────

const CATEGORIES = [
  { key: 'all', label: 'All' },
  { key: 'analytics', label: 'Analytics' },
  { key: 'business', label: 'Business' },
  { key: 'engineering', label: 'Engineering' },
  { key: 'creative', label: 'Creative' },
  { key: 'research', label: 'Research' },
] as const

type CategoryKey = (typeof CATEGORIES)[number]['key']

// ── Component ──────────────────────────────────────────────────────────

interface TemplateGalleryProps {
  className?: string
  /** Called after a template is successfully cloned and the workspace loaded */
  onWorkspaceCreated?: (workspaceId: string) => void
}

export function TemplateGallery({ className, onWorkspaceCreated }: TemplateGalleryProps) {
  const [templates, setTemplates] = useState<WorkspaceTemplate[]>([])
  const [isLoading, setIsLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [activeCategory, setActiveCategory] = useState<CategoryKey>('all')
  const [cloningId, setCloningId] = useState<string | null>(null)

  const loadWorkspace = useWorkspaceStore((s) => s.loadWorkspace)

  // ── Fetch templates on mount ──────────────────────────────────────

  useEffect(() => {
    let cancelled = false

    async function fetchTemplates() {
      setIsLoading(true)
      setError(null)
      try {
        const data = await apiClient.get<WorkspaceTemplate[]>('/api/workspaces/templates')
        if (!cancelled) {
          setTemplates(Array.isArray(data) ? data : [])
        }
      } catch (err) {
        console.error('[TemplateGallery] Failed to fetch templates:', err)
        if (!cancelled) {
          setError('Failed to load templates. Please try again.')
        }
      } finally {
        if (!cancelled) setIsLoading(false)
      }
    }

    fetchTemplates()
    return () => { cancelled = true }
  }, [])

  // ── Category filtering ────────────────────────────────────────────

  const filteredTemplates = useMemo(() => {
    if (activeCategory === 'all') return templates
    return templates.filter(
      (t) => t.category?.toLowerCase() === activeCategory
    )
  }, [templates, activeCategory])

  // ── Clone handler ─────────────────────────────────────────────────

  const handleUseTemplate = useCallback(
    async (template: WorkspaceTemplate) => {
      setCloningId(template.id)
      try {
        const created = await apiClient.post<{ id: string }>(
          `/api/workspaces/from-template/${template.id}`
        )
        const newId = created.id
        await loadWorkspace(newId)
        toast.success(`Workspace created from "${template.name}"`)
        onWorkspaceCreated?.(newId)
      } catch (err) {
        console.error('[TemplateGallery] Clone failed:', err)
        toast.error('Failed to create workspace from template')
      } finally {
        setCloningId(null)
      }
    },
    [loadWorkspace, onWorkspaceCreated]
  )

  // ── Render ────────────────────────────────────────────────────────

  return (
    <div className={cn('flex flex-col gap-6', className)}>
      {/* Category filter chips */}
      <div className="flex flex-wrap gap-2">
        {CATEGORIES.map((cat) => (
          <Button
            key={cat.key}
            variant={activeCategory === cat.key ? 'default' : 'outline'}
            size="sm"
            onClick={() => setActiveCategory(cat.key)}
            className="rounded-full"
          >
            {cat.label}
          </Button>
        ))}
      </div>

      {/* Error state */}
      {error && (
        <div className="rounded-xl border border-destructive/50 bg-destructive/10 p-4 text-sm text-destructive">
          {error}
          <Button
            variant="ghost"
            size="sm"
            className="ml-2"
            onClick={() => {
              setIsLoading(true)
              setError(null)
              apiClient
                .get<WorkspaceTemplate[]>('/api/workspaces/templates')
                .then((data) => setTemplates(Array.isArray(data) ? data : []))
                .catch(() => setError('Failed to load templates. Please try again.'))
                .finally(() => setIsLoading(false))
            }}
          >
            Retry
          </Button>
        </div>
      )}

      {/* Loading skeleton grid */}
      {isLoading && (
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {Array.from({ length: 6 }).map((_, i) => (
            <Card key={i} className="flex flex-col">
              <CardHeader className="gap-3">
                <Skeleton className="h-10 w-10 rounded-xl" />
                <Skeleton className="h-5 w-3/4" />
                <Skeleton className="h-4 w-full" />
                <Skeleton className="h-4 w-2/3" />
              </CardHeader>
              <CardFooter className="mt-auto">
                <Skeleton className="h-9 w-full rounded-xl" />
              </CardFooter>
            </Card>
          ))}
        </div>
      )}

      {/* Empty state */}
      {!isLoading && !error && filteredTemplates.length === 0 && (
        <div className="flex flex-col items-center justify-center gap-3 rounded-xl border border-dashed border-border/60 py-16 text-muted-foreground">
          <LayoutGrid className="h-10 w-10 opacity-40" />
          <p className="text-sm">
            {activeCategory === 'all'
              ? 'No templates available yet.'
              : `No templates in the "${CATEGORIES.find((c) => c.key === activeCategory)?.label}" category.`}
          </p>
        </div>
      )}

      {/* Template cards grid */}
      {!isLoading && filteredTemplates.length > 0 && (
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {filteredTemplates.map((template) => {
            const widgetCount = template.widgets?.length ?? 0
            const isCloning = cloningId === template.id

            return (
              <Card
                key={template.id}
                className="flex flex-col transition-all duration-300 hover:border-primary/20"
              >
                <CardHeader className="gap-1">
                  <div className="flex items-start justify-between">
                    <span className="text-3xl" role="img" aria-label={template.name}>
                      {template.icon || '\uD83D\uDCCB'}
                    </span>
                    {template.category && (
                      <Badge variant="secondary" className="text-[10px] capitalize">
                        {template.category}
                      </Badge>
                    )}
                  </div>
                  <h3 className="text-base font-semibold leading-snug">
                    {template.name}
                  </h3>
                  {template.description && (
                    <p className="line-clamp-2 text-sm text-muted-foreground">
                      {template.description}
                    </p>
                  )}
                </CardHeader>

                <CardContent className="mt-auto flex items-center gap-2 pb-3">
                  <Badge variant="outline" className="text-[10px]">
                    {widgetCount} {widgetCount === 1 ? 'widget' : 'widgets'}
                  </Badge>
                </CardContent>

                <CardFooter>
                  <Button
                    className="w-full"
                    size="sm"
                    disabled={isCloning || cloningId !== null}
                    onClick={() => handleUseTemplate(template)}
                  >
                    {isCloning ? (
                      <>
                        <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                        Creating...
                      </>
                    ) : (
                      'Use Template'
                    )}
                  </Button>
                </CardFooter>
              </Card>
            )
          })}
        </div>
      )}
    </div>
  )
}
