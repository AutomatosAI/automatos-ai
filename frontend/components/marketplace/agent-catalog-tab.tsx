'use client'

import { useState, useEffect } from 'react'
import { Bot, Loader2, Download, Zap, MoreVertical, Eye } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Card, CardContent, CardHeader } from '@/components/ui/card'
import { Separator } from '@/components/ui/separator'
import { StatusBadge } from '@/components/shared'
import { PremiumIcon } from '@/components/shared'
import { ViewToggle } from '@/components/shared/view-toggle'
import { useViewMode } from '@/hooks/use-view-mode'
import { useSystemIcons } from '@/hooks/use-system-config-api'
import { AGENT_CATEGORIES as UNIFIED_CATEGORIES } from '@/lib/agent-constants'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu'
import {
  useAgentCatalogTemplates,
  useAgentCatalogCategories,
  useDeployAgentCatalogTemplate,
  type AgentCatalogTemplate,
  type AgentCatalogCategory,
} from '@/hooks/use-marketplace-api'
import { AgentTemplateDetail } from './agent-template-detail'

// Map catalog categories to unified category IDs for icon lookup
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

interface AgentCatalogTabProps {
  searchQuery: string
}

export function AgentCatalogTab({ searchQuery }: AgentCatalogTabProps) {
  const [viewMode, setViewMode] = useViewMode('mp-catalog')
  const [selectedCategory, setSelectedCategory] = useState<string | null>(null)
  const [selectedSlug, setSelectedSlug] = useState<string | null>(null)
  const [deployingSlug, setDeployingSlug] = useState<string | null>(null)

  const { data: iconMappings = {} } = useSystemIcons()

  // Reset category when search changes
  useEffect(() => {
    setSelectedCategory(null)
  }, [searchQuery])

  const { data: categories = [], isLoading: categoriesLoading } = useAgentCatalogCategories()

  const { data: templates = [], isLoading: templatesLoading } = useAgentCatalogTemplates({
    category: selectedCategory || undefined,
    search: searchQuery || undefined,
    limit: 100,
  })

  const deployMutation = useDeployAgentCatalogTemplate()

  const handleDeploy = async (e: React.MouseEvent, slug: string) => {
    e.stopPropagation()
    setDeployingSlug(slug)
    try {
      await deployMutation.mutateAsync(slug)
    } finally {
      setDeployingSlug(null)
    }
  }

  const isLoading = categoriesLoading || templatesLoading

  const getIconForCategory = (category: string) => {
    const unifiedId = CATALOG_TO_UNIFIED[category] || category
    const premiumName = iconMappings[unifiedId] || iconMappings[category]
    if (premiumName) return <PremiumIcon name={premiumName} size={40} className="shrink-0" />
    const catDef = UNIFIED_CATEGORIES.find(c => c.id === unifiedId)
    if (catDef) {
      const Icon = catDef.icon
      return <Icon className="w-10 h-10 text-primary shrink-0" />
    }
    return <Bot className="w-10 h-10 text-primary shrink-0" />
  }

  const getSmallIconForCategory = (category: string) => {
    const unifiedId = CATALOG_TO_UNIFIED[category] || category
    const premiumName = iconMappings[unifiedId] || iconMappings[category]
    if (premiumName) return <PremiumIcon name={premiumName} size={36} className="shrink-0" />
    const catDef = UNIFIED_CATEGORIES.find(c => c.id === unifiedId)
    if (catDef) {
      const Icon = catDef.icon
      return <Icon className="w-9 h-9 text-primary shrink-0" />
    }
    return <Bot className="w-9 h-9 text-primary shrink-0" />
  }

  const getCategoryName = (category: string) => {
    const unifiedId = CATALOG_TO_UNIFIED[category] || category
    const catDef = UNIFIED_CATEGORIES.find(c => c.id === unifiedId)
    return catDef?.name || category.charAt(0).toUpperCase() + category.slice(1).replace('-', ' ')
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h3 className="text-xl font-semibold">Agent Catalog</h3>
          <p className="text-sm text-muted-foreground">
            Pre-built agent templates ready to deploy to your workspace
          </p>
        </div>
        <div className="flex items-center gap-3">
          <StatusBadge size="sm" status="info">
            {templates.length} Available
          </StatusBadge>
        </div>
      </div>

      {/* Category Filter + View Toggle */}
      <div className="flex items-center justify-between gap-4">
        <div className="flex gap-2 overflow-x-auto pb-1 scrollbar-thin scrollbar-thumb-secondary scrollbar-track-transparent flex-1">
          <Button
            variant={selectedCategory === null ? 'default' : 'outline'}
            size="sm"
            onClick={() => setSelectedCategory(null)}
            className={`whitespace-nowrap flex-shrink-0 ${
              selectedCategory === null
                ? 'bg-secondary border-primary/50 text-foreground font-semibold'
                : 'border-secondary text-muted-foreground hover:bg-secondary'
            }`}
          >
            All Categories
            {!categoriesLoading && (
              <Badge variant="secondary" className="ml-1.5 text-[10px] px-1.5 py-0">
                {categories.reduce((sum: number, c: AgentCatalogCategory) => sum + c.count, 0)}
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
                className={`whitespace-nowrap flex-shrink-0 gap-1.5 ${
                  selectedCategory === cat.category
                    ? 'bg-secondary border-primary/50 text-foreground font-semibold'
                    : 'border-secondary text-muted-foreground hover:bg-secondary'
                }`}
              >
                {premiumName && <PremiumIcon name={premiumName} size={14} />}
                {getCategoryName(cat.category)}
                <Badge variant="secondary" className="ml-1 text-[10px] px-1.5 py-0">
                  {cat.count}
                </Badge>
              </Button>
            )
          })}
        </div>
        <ViewToggle value={viewMode} onChange={setViewMode} />
      </div>

      {/* Templates Grid */}
      {isLoading ? (
        viewMode === 'list' ? (
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
            {[...Array(6)].map((_, i) => (
              <div key={i} className="h-16 glass-card animate-pulse rounded-xl" />
            ))}
          </div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
            {[...Array(8)].map((_, i) => (
              <div key={i} className="h-64 glass-card animate-pulse" />
            ))}
          </div>
        )
      ) : templates.length === 0 ? (
        <div className="text-center py-12 text-muted-foreground">
          <Bot className="w-12 h-12 mx-auto mb-4 opacity-50" />
          <p>No agent templates found. Try adjusting your search or filters.</p>
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
                  {getSmallIconForCategory(template.category)}
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2">
                      <span className="font-semibold text-sm truncate">{template.name}</span>
                    </div>
                    <div className="flex items-center gap-2 text-xs text-muted-foreground mt-0.5">
                      <span>{getCategoryName(template.category)}</span>
                      {template.recommended_model && (
                        <>
                          <span>&middot;</span>
                          <span>{modelShortName(template.recommended_model)}</span>
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
          {templates.map((template: AgentCatalogTemplate) => (
            <Card
              key={template.slug}
              className="glass-card card-glow hover:border-primary/20 transition-all duration-300 cursor-pointer"
              onClick={() => setSelectedSlug(template.slug)}
            >
              <CardHeader className="pb-3">
                <div className="flex items-start justify-between gap-3">
                  <div className="flex items-center gap-3 flex-1 min-w-0">
                    {getIconForCategory(template.category)}
                    <div className="flex-1 min-w-0">
                      <h3 className="font-semibold text-foreground line-clamp-1">
                        {template.name}
                      </h3>
                      <p className="text-xs text-muted-foreground">
                        {getCategoryName(template.category)}
                      </p>
                    </div>
                  </div>
                  <DropdownMenu>
                    <DropdownMenuTrigger asChild>
                      <Button variant="ghost" size="sm" className="h-8 w-8 p-0" onClick={(e) => e.stopPropagation()}>
                        <MoreVertical className="h-4 w-4" />
                      </Button>
                    </DropdownMenuTrigger>
                    <DropdownMenuContent align="end">
                      <DropdownMenuItem onClick={(e) => { e.stopPropagation(); setSelectedSlug(template.slug) }}>
                        <Eye className="w-4 h-4 mr-2" />
                        View Details
                      </DropdownMenuItem>
                      <DropdownMenuItem
                        onClick={(e) => { e.stopPropagation(); handleDeploy(e as any, template.slug) }}
                        disabled={deployingSlug === template.slug}
                      >
                        <Download className="w-4 h-4 mr-2" />
                        {deployingSlug === template.slug ? 'Deploying...' : 'Add to Workspace'}
                      </DropdownMenuItem>
                    </DropdownMenuContent>
                  </DropdownMenu>
                </div>
              </CardHeader>

              <CardContent className="space-y-3">
                <p className="text-sm text-muted-foreground line-clamp-2">
                  {template.description}
                </p>

                {/* Category + Model */}
                <div className="flex flex-col gap-2">
                  <Badge variant="outline" className="text-xs border-border text-muted-foreground w-fit">
                    {getCategoryName(template.category)}
                  </Badge>
                  {template.recommended_model && (
                    <Badge className="text-xs bg-[hsl(var(--agent))]/10 text-[hsl(var(--agent))] border-[hsl(var(--agent))]/30 w-fit">
                      <Zap className="w-3 h-3 mr-1" />
                      {modelShortName(template.recommended_model)}
                    </Badge>
                  )}
                </div>

                {/* Tags */}
                {template.tags.length > 0 && (
                  <div className="flex flex-wrap gap-1.5">
                    {template.tags.slice(0, 2).map((tag) => (
                      <StatusBadge key={tag} size="sm" status="neutral">
                        {tag}
                      </StatusBadge>
                    ))}
                    {template.tags.length > 2 && (
                      <StatusBadge size="sm" status="neutral">
                        +{template.tags.length - 2}
                      </StatusBadge>
                    )}
                  </div>
                )}

                {/* Tools preview */}
                {template.recommended_tools.length > 0 && (
                  <div className="flex flex-wrap gap-2 pt-3 border-t border-border/30">
                    {template.recommended_tools.slice(0, 5).map((tool) => (
                      <div
                        key={tool}
                        className="bg-secondary/30 px-1.5 h-[24px] flex items-center justify-center rounded-md border border-border/30 text-[10px] text-muted-foreground"
                      >
                        {tool}
                      </div>
                    ))}
                    {template.recommended_tools.length > 5 && (
                      <div className="bg-secondary/30 px-1.5 h-[24px] flex items-center justify-center rounded-md border border-border/30 text-[10px] text-muted-foreground">
                        +{template.recommended_tools.length - 5}
                      </div>
                    )}
                  </div>
                )}
              </CardContent>

              <Separator />
              <div className="flex items-center justify-between px-6 py-3">
                <Button variant="ghost" size="sm"
                  onClick={(e) => { e.stopPropagation(); setSelectedSlug(template.slug) }}
                  className="text-muted-foreground hover:text-foreground p-0 h-auto">
                  Details
                </Button>
                <Button
                  size="sm"
                  variant="outline"
                  disabled={deployingSlug === template.slug}
                  onClick={(e) => handleDeploy(e, template.slug)}
                >
                  {deployingSlug === template.slug ? (
                    <>
                      <Loader2 className="w-3 h-3 mr-1 animate-spin" />
                      Deploying...
                    </>
                  ) : (
                    'Add to Workspace'
                  )}
                </Button>
              </div>
            </Card>
          ))}
        </div>
      )}

      {/* Template Detail Modal */}
      <AgentTemplateDetail
        open={selectedSlug !== null}
        slug={selectedSlug}
        onClose={() => setSelectedSlug(null)}
      />
    </div>
  )
}
