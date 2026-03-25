'use client'

import { useState, useMemo, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import {
  Loader2, Download, Bot, Zap, Wrench
} from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Card, CardContent, CardHeader } from '@/components/ui/card'
import { Separator } from '@/components/ui/separator'
import { PremiumIcon } from '@/components/shared'
import { ViewToggle } from '@/components/shared/view-toggle'
import { useViewMode } from '@/hooks/use-view-mode'
import { useSystemIcons } from '@/hooks/use-system-config-api'
import { AGENT_CATEGORIES as UNIFIED_CATEGORIES } from '@/lib/agent-constants'
import {
  useAgentCatalogTemplates,
  useAgentCatalogCategories,
  useDeployAgentCatalogTemplate,
  type AgentCatalogTemplate,
  type AgentCatalogCategory,
} from '@/hooks/use-marketplace-api'
import { AgentTemplateDetail } from './agent-template-detail'

// Map catalog categories → unified category IDs for icon lookup
const CATALOG_TO_UNIFIED: Record<string, string> = {
  engineering: 'development',
  design: 'design',
  marketing: 'marketing',
  sales: 'sales',
  product: 'business',
  'project-management': 'productivity',
  testing: 'development',
  support: 'support',
  'paid-media': 'marketing',
  specialized: 'custom',
}

function modelShortName(model?: string): string {
  if (!model) return ''
  if (model.includes('opus')) return 'Opus'
  if (model.includes('sonnet')) return 'Sonnet'
  if (model.includes('haiku')) return 'Haiku'
  return model.split('/').pop()?.substring(0, 20) || model
}

interface MarketplaceAgentsTabProps {
  searchQuery: string
}

export function MarketplaceAgentsTab({ searchQuery }: MarketplaceAgentsTabProps) {
  const [viewMode, setViewMode] = useViewMode('mp-agents')
  const [selectedCategory, setSelectedCategory] = useState('all')
  const [selectedSlug, setSelectedSlug] = useState<string | null>(null)
  const [deployingSlug, setDeployingSlug] = useState<string | null>(null)

  const { data: iconMappings = {} } = useSystemIcons()

  // Fetch categories with counts from API
  const { data: categories = [] } = useAgentCatalogCategories()

  // Fetch templates (category filter via API)
  const { data: rawTemplates = [], isLoading } = useAgentCatalogTemplates({
    category: selectedCategory !== 'all' ? selectedCategory : undefined,
    search: searchQuery || undefined,
    limit: 200,
  })

  const deployMutation = useDeployAgentCatalogTemplate()

  // Reset to "all" when search changes
  useEffect(() => {
    setSelectedCategory('all')
  }, [searchQuery])

  // Client-side search filter (API also filters, but this gives instant feedback)
  const templates = useMemo(() => {
    if (!searchQuery.trim()) return rawTemplates
    const q = searchQuery.toLowerCase()
    return (rawTemplates as AgentCatalogTemplate[]).filter(
      (t) =>
        t.name.toLowerCase().includes(q) ||
        t.description?.toLowerCase().includes(q) ||
        t.tags?.some((tag) => tag.toLowerCase().includes(q))
    )
  }, [rawTemplates, searchQuery])

  const totalCount = useMemo(
    () => categories.reduce((sum: number, c: AgentCatalogCategory) => sum + c.count, 0),
    [categories]
  )

  const handleDeploy = async (e: React.MouseEvent, slug: string) => {
    e.stopPropagation()
    setDeployingSlug(slug)
    try {
      await deployMutation.mutateAsync(slug)
    } finally {
      setDeployingSlug(null)
    }
  }

  // Icon helpers — match unified system icons, fall back to UNIFIED_CATEGORIES lucide icons
  const getIcon = (category: string, size: number) => {
    const unifiedId = CATALOG_TO_UNIFIED[category] || category
    const premiumName = iconMappings[unifiedId] || iconMappings[category]
    if (premiumName) return <PremiumIcon name={premiumName} size={size} className="shrink-0" />
    const catDef = UNIFIED_CATEGORIES.find((c) => c.id === unifiedId)
    if (catDef) {
      const Icon = catDef.icon
      return <Icon className="shrink-0 text-primary" style={{ width: size, height: size }} />
    }
    return <Bot className="shrink-0 text-primary" style={{ width: size, height: size }} />
  }

  const getCategoryName = (category: string) => {
    const unifiedId = CATALOG_TO_UNIFIED[category] || category
    return (
      UNIFIED_CATEGORIES.find((c) => c.id === unifiedId)?.name ||
      category.charAt(0).toUpperCase() + category.slice(1).replace(/-/g, ' ')
    )
  }

  // Loading skeleton
  if (isLoading) {
    return (
      <div className="space-y-6">
        {viewMode === 'list' ? (
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
            {[...Array(6)].map((_, i) => (
              <div key={i} className="h-16 glass-card animate-pulse rounded-xl" />
            ))}
          </div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
            {[...Array(8)].map((_, i) => (
              <Card key={i} className="glass-card animate-pulse">
                <CardHeader className="pb-2">
                  <div className="flex items-center gap-3">
                    <div className="w-10 h-10 bg-secondary/50 rounded-lg" />
                    <div className="space-y-2">
                      <div className="h-4 w-24 bg-secondary/50 rounded" />
                      <div className="h-3 w-32 bg-secondary/50 rounded" />
                    </div>
                  </div>
                </CardHeader>
                <CardContent>
                  <div className="h-8 w-full bg-secondary/50 rounded mt-2" />
                  <div className="h-6 w-20 bg-secondary/50 rounded mt-3" />
                </CardContent>
              </Card>
            ))}
          </div>
        )}
      </div>
    )
  }

  return (
    <div className="space-y-6">
      {/* Header Stats */}
      <div className="flex items-center justify-between">
        <div>
          <h3 className="text-xl font-semibold">Agent Templates</h3>
          <p className="text-sm text-muted-foreground">
            Pre-built agents ready to deploy to your workspace
          </p>
        </div>
        <div className="flex items-center gap-3">
          <Badge variant="outline" className="text-[hsl(var(--info))] border-[hsl(var(--info))]/30">
            {totalCount} Available
          </Badge>
          <ViewToggle value={viewMode} onChange={setViewMode} />
        </div>
      </div>

      {/* Category Filters — Horizontal Scroll */}
      <div className="relative">
        <div className="flex gap-2 overflow-x-auto pb-2 scrollbar-thin scrollbar-thumb-secondary scrollbar-track-transparent">
          <Button
            variant={selectedCategory === 'all' ? 'default' : 'outline'}
            size="sm"
            onClick={() => setSelectedCategory('all')}
            className={`whitespace-nowrap flex-shrink-0 ${
              selectedCategory === 'all'
                ? 'bg-secondary border-primary/50 text-foreground font-semibold'
                : 'border-secondary text-muted-foreground hover:bg-secondary'
            }`}
          >
            All Agents
            {totalCount > 0 && (
              <Badge variant="secondary" className="ml-2 bg-secondary/50 text-xs">
                {totalCount}
              </Badge>
            )}
          </Button>
          {categories.map((cat: AgentCatalogCategory) => {
            const unifiedId = CATALOG_TO_UNIFIED[cat.category] || cat.category
            const premiumName = iconMappings[unifiedId] || iconMappings[cat.category]
            return (
              <Button
                key={cat.category}
                variant={selectedCategory === cat.category ? 'default' : 'outline'}
                size="sm"
                onClick={() => setSelectedCategory(cat.category)}
                className={`whitespace-nowrap flex-shrink-0 ${
                  selectedCategory === cat.category
                    ? 'bg-secondary border-primary/50 text-foreground font-semibold'
                    : 'border-secondary text-muted-foreground hover:bg-secondary'
                }`}
              >
                {premiumName && <PremiumIcon name={premiumName} size={14} />}
                {getCategoryName(cat.category)}
                {cat.count > 0 && (
                  <Badge variant="secondary" className="ml-2 bg-secondary/50 text-xs">
                    {cat.count}
                  </Badge>
                )}
              </Button>
            )
          })}
        </div>
      </div>

      {/* Empty State */}
      {templates.length === 0 ? (
        <div className="text-center py-12">
          <div className="w-16 h-16 rounded-lg bg-secondary/30 flex items-center justify-center mx-auto mb-4">
            <Bot className="w-8 h-8 text-muted-foreground" />
          </div>
          <h3 className="text-lg font-semibold mb-2">No agents found</h3>
          <p className="text-muted-foreground mb-4">
            {searchQuery
              ? `No agents match "${searchQuery}"`
              : 'No agents available in this category'}
          </p>
          <Button variant="outline" onClick={() => setSelectedCategory('all')}>
            Clear Filters
          </Button>
        </div>
      ) : viewMode === 'list' ? (
        /* List View */
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
          {templates.map((template: AgentCatalogTemplate) => (
            <Card
              key={template.slug}
              className="glass-card hover:border-primary/20 transition-all cursor-pointer"
              onClick={() => setSelectedSlug(template.slug)}
            >
              <CardContent className="p-3">
                <div className="flex items-center gap-3">
                  {getIcon(template.category, 36)}
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2">
                      <span className="font-semibold text-sm truncate">{template.name}</span>
                      <Badge variant="outline" className="text-[10px] h-5 shrink-0">
                        {getCategoryName(template.category)}
                      </Badge>
                    </div>
                    <div className="flex items-center gap-2 text-xs text-muted-foreground mt-0.5">
                      {template.recommended_model && (
                        <span>{modelShortName(template.recommended_model)}</span>
                      )}
                      {template.recommended_tools.length > 0 && (
                        <>
                          <span>&middot;</span>
                          <span>{template.recommended_tools.length} tools</span>
                        </>
                      )}
                    </div>
                  </div>
                  <Button
                    variant="ghost"
                    size="sm"
                    className="h-8 w-8 p-0 shrink-0"
                    disabled={deployingSlug === template.slug}
                    onClick={(e) => handleDeploy(e, template.slug)}
                  >
                    {deployingSlug === template.slug ? (
                      <Loader2 className="w-4 h-4 animate-spin" />
                    ) : (
                      <Download className="w-4 h-4" />
                    )}
                  </Button>
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
      ) : (
        /* Grid View */
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
          <AnimatePresence>
            {templates.map((template: AgentCatalogTemplate, index: number) => (
              <motion.div
                key={template.slug}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: 20 }}
                transition={{ delay: Math.min(index * 0.05, 0.5) }}
              >
                <Card
                  className="glass-card card-glow hover:border-primary/20 transition-all duration-300 cursor-pointer"
                  onClick={() => setSelectedSlug(template.slug)}
                >
                  <CardHeader className="pb-3">
                    <div className="flex items-start justify-between">
                      <div className="flex items-center space-x-3">
                        {getIcon(template.category, 40)}
                        <div>
                          <h3 className="font-semibold line-clamp-1">{template.name}</h3>
                          <p className="text-xs text-muted-foreground">
                            {getCategoryName(template.category)}
                          </p>
                        </div>
                      </div>
                    </div>
                  </CardHeader>

                  <CardContent className="space-y-4">
                    <p className="text-sm text-muted-foreground line-clamp-2 min-h-[40px]">
                      {template.description}
                    </p>

                    {/* Tags */}
                    <div className="flex items-center justify-between text-xs">
                      <div />
                      <div className="flex gap-1">
                        {template.tags?.slice(0, 2).map((tag) => (
                          <Badge key={tag} variant="outline" className="text-[10px] h-5">
                            {tag}
                          </Badge>
                        ))}
                      </div>
                    </div>

                    {/* Model + Tools Count */}
                    <div className="flex items-center gap-4 text-xs text-muted-foreground">
                      {template.recommended_model && (
                        <div className="flex items-center gap-1">
                          <Zap className="w-3 h-3" />
                          <span>{modelShortName(template.recommended_model)}</span>
                        </div>
                      )}
                      {template.recommended_tools.length > 0 && (
                        <div className="flex items-center gap-1">
                          <Wrench className="w-3 h-3" />
                          <span>{template.recommended_tools.length} Tools</span>
                        </div>
                      )}
                    </div>
                  </CardContent>

                  <Separator />
                  <div className="flex items-center justify-between px-6 py-3">
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={(e) => { e.stopPropagation(); setSelectedSlug(template.slug) }}
                      className="text-muted-foreground hover:text-foreground p-0 h-auto"
                    >
                      Details
                    </Button>
                    <Button
                      size="sm"
                      variant="outline"
                      disabled={deployingSlug === template.slug}
                      onClick={(e) => handleDeploy(e, template.slug)}
                    >
                      {deployingSlug === template.slug ? 'Adding...' : 'Add to Workspace'}
                    </Button>
                  </div>
                </Card>
              </motion.div>
            ))}
          </AnimatePresence>
        </div>
      )}

      {/* Detail Modal */}
      <AgentTemplateDetail
        open={selectedSlug !== null}
        slug={selectedSlug}
        onClose={() => setSelectedSlug(null)}
      />
    </div>
  )
}
