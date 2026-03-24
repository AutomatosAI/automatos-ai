'use client'

import { useState, useEffect } from 'react'
import { Bot, Loader2, Sparkles, Cpu } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Card, CardContent, CardHeader } from '@/components/ui/card'
import { Separator } from '@/components/ui/separator'
import {
  useAgentCatalogTemplates,
  useAgentCatalogCategories,
  useDeployAgentCatalogTemplate,
  type AgentCatalogTemplate,
  type AgentCatalogCategory,
} from '@/hooks/use-marketplace-api'
import { AgentTemplateDetail } from './agent-template-detail'

// Map category to display-friendly label
const CATEGORY_LABELS: Record<string, string> = {
  engineering: 'Engineering',
  design: 'Design',
  marketing: 'Marketing',
  sales: 'Sales',
  product: 'Product',
  'project-management': 'Project Management',
  testing: 'Testing',
  support: 'Support',
  'paid-media': 'Paid Media',
  specialized: 'Specialized',
}

// Map recommended_model to short display name
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
  const [selectedCategory, setSelectedCategory] = useState<string | null>(null)
  const [selectedSlug, setSelectedSlug] = useState<string | null>(null)
  const [deployingSlug, setDeployingSlug] = useState<string | null>(null)

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

  return (
    <div className="space-y-6">
      {/* Category Grid */}
      <div className="flex gap-2 overflow-x-auto pb-1 scrollbar-thin scrollbar-thumb-secondary scrollbar-track-transparent">
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
        {categories.map((cat: AgentCatalogCategory) => (
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
            {cat.icon && <span className="text-sm">{cat.icon}</span>}
            {CATEGORY_LABELS[cat.category] || cat.category}
            <Badge variant="secondary" className="ml-1 text-[10px] px-1.5 py-0">
              {cat.count}
            </Badge>
          </Button>
        ))}
      </div>

      {/* Templates Grid */}
      {isLoading ? (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
          {[...Array(8)].map((_, i) => (
            <div key={i} className="h-64 glass-card animate-pulse rounded-xl" />
          ))}
        </div>
      ) : templates.length === 0 ? (
        <div className="text-center py-12 text-muted-foreground">
          <Bot className="w-12 h-12 mx-auto mb-4 opacity-50" />
          <p>No agent templates found. Try adjusting your search or filters.</p>
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
          {templates.map((template: AgentCatalogTemplate) => (
            <Card
              key={template.slug}
              className="glass-card card-glow hover:border-primary/20 transition-all duration-300 flex flex-col cursor-pointer"
              onClick={() => setSelectedSlug(template.slug)}
            >
              <CardHeader className="pb-3">
                <div className="flex items-start gap-3">
                  <div className="text-2xl shrink-0">
                    {template.icon || <Bot className="w-8 h-8 text-primary" />}
                  </div>
                  <div className="flex-1 min-w-0">
                    <h3 className="font-semibold text-foreground line-clamp-1">
                      {template.name}
                    </h3>
                    <p className="text-xs text-muted-foreground">
                      {CATEGORY_LABELS[template.category] || template.category}
                    </p>
                  </div>
                  {template.tier !== 'free' && (
                    <Badge className="bg-[hsl(var(--warning))]/10 text-[hsl(var(--warning))] border-[hsl(var(--warning))]/30 shrink-0">
                      <Sparkles className="w-3 h-3 mr-1" />
                      {template.tier}
                    </Badge>
                  )}
                </div>
              </CardHeader>

              <CardContent className="space-y-3 flex-1 flex flex-col">
                <p className="text-sm text-muted-foreground line-clamp-2">
                  {template.description}
                </p>

                {/* Model badge */}
                {template.recommended_model && (
                  <Badge className="text-xs bg-[hsl(var(--agent))]/10 text-[hsl(var(--agent))] border-[hsl(var(--agent))]/30 w-fit">
                    <Cpu className="w-3 h-3 mr-1" />
                    {modelShortName(template.recommended_model)}
                  </Badge>
                )}

                {/* Tags */}
                {template.tags.length > 0 && (
                  <div className="flex flex-wrap gap-1">
                    {template.tags.slice(0, 3).map((tag) => (
                      <Badge
                        key={tag}
                        variant="outline"
                        className="text-[10px] border-border text-muted-foreground"
                      >
                        {tag}
                      </Badge>
                    ))}
                    {template.tags.length > 3 && (
                      <Badge variant="outline" className="text-[10px] border-border text-muted-foreground">
                        +{template.tags.length - 3}
                      </Badge>
                    )}
                  </div>
                )}

                {/* Tools preview */}
                {template.recommended_tools.length > 0 && (
                  <div className="flex flex-wrap gap-1 pt-2 border-t border-border/30 mt-auto">
                    {template.recommended_tools.slice(0, 4).map((tool) => (
                      <Badge
                        key={tool}
                        variant="secondary"
                        className="text-[10px] bg-secondary/50"
                      >
                        {tool}
                      </Badge>
                    ))}
                    {template.recommended_tools.length > 4 && (
                      <Badge variant="secondary" className="text-[10px] bg-secondary/50">
                        +{template.recommended_tools.length - 4}
                      </Badge>
                    )}
                  </div>
                )}
              </CardContent>

              <Separator />
              <div className="flex items-center justify-between px-6 py-3">
                <Badge variant="outline" className="text-xs border-border text-muted-foreground">
                  {CATEGORY_LABELS[template.category] || template.category}
                </Badge>
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
