'use client'

/**
 * Agent Catalog Template Detail Modal (PRD-120: US-009)
 * =====================================================
 * Shows full agent template information before deploying:
 * - Icon, name, category, tier
 * - Full description and persona preview (collapsible)
 * - Recommended model and tools
 * - Tags
 * - One-click deploy with optional name override
 */

import { useState } from 'react'
import { AnimatePresence, motion } from 'framer-motion'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Input } from '@/components/ui/input'
import { Separator } from '@/components/ui/separator'
import {
  X,
  Loader2,
  Bot,
  Cpu,
  Sparkles,
  Tag,
  Wrench,
  ChevronDown,
  ChevronUp,
  FileText,
  Rocket,
} from 'lucide-react'
import {
  useAgentCatalogTemplate,
  useDeployAgentCatalogTemplate,
  type AgentCatalogTemplateDetail as TemplateDetail,
} from '@/hooks/use-marketplace-api'

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

function modelShortName(model?: string): string {
  if (!model) return ''
  if (model.includes('opus')) return 'Opus'
  if (model.includes('sonnet')) return 'Sonnet'
  if (model.includes('haiku')) return 'Haiku'
  return model.split('/').pop()?.substring(0, 20) || model
}

interface AgentTemplateDetailProps {
  open: boolean
  slug: string | null
  onClose: () => void
}

export function AgentTemplateDetail({
  open,
  slug,
  onClose,
}: AgentTemplateDetailProps) {
  const [personaExpanded, setPersonaExpanded] = useState(false)
  const [customName, setCustomName] = useState('')
  const [deploying, setDeploying] = useState(false)

  const { data: template, isLoading, error } = useAgentCatalogTemplate(open ? slug : null)
  const deployMutation = useDeployAgentCatalogTemplate()

  const handleDeploy = async () => {
    if (!slug) return
    setDeploying(true)
    try {
      await deployMutation.mutateAsync(slug)
      onClose()
    } finally {
      setDeploying(false)
    }
  }

  const handleClose = () => {
    setPersonaExpanded(false)
    setCustomName('')
    onClose()
  }

  if (!open) return null

  return (
    <AnimatePresence>
      {open && (
        <>
          {/* Backdrop */}
          <motion.div
            className="fixed inset-0 bg-black/50 backdrop-blur-sm z-50"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={handleClose}
          />

          {/* Modal */}
          <motion.div
            className="fixed inset-0 z-50 flex items-center justify-center p-4"
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.95 }}
            transition={{ duration: 0.2 }}
          >
            <Card className="glass-card card-glow w-full max-w-2xl max-h-[85vh] overflow-hidden flex flex-col">
              {/* Loading */}
              {isLoading && (
                <div className="p-12 flex flex-col items-center justify-center">
                  <Loader2 className="w-8 h-8 animate-spin text-muted-foreground mb-4" />
                  <p className="text-sm text-muted-foreground">Loading template details...</p>
                </div>
              )}

              {/* Error */}
              {!!error && !isLoading && (
                <div className="p-12 flex flex-col items-center justify-center">
                  <p className="text-[hsl(var(--destructive))] mb-4">
                    {(error as any)?.message || 'Failed to load template details'}
                  </p>
                  <Button variant="outline" onClick={handleClose}>Close</Button>
                </div>
              )}

              {/* Content */}
              {template && !isLoading && !error && (
                <>
                  {/* Header */}
                  <CardHeader className="flex flex-row items-start justify-between pb-2">
                    <CardTitle className="flex items-start gap-3 text-xl flex-1 min-w-0">
                      <div className="w-12 h-12 rounded-lg bg-primary/10 border border-primary/20 flex items-center justify-center flex-shrink-0 text-2xl">
                        {template.icon || <Bot className="w-6 h-6 text-primary" />}
                      </div>
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center gap-2 flex-wrap">
                          <span className="truncate">{template.name}</span>
                          {template.tier !== 'free' && (
                            <Badge className="bg-[hsl(var(--warning))]/10 text-[hsl(var(--warning))] border-[hsl(var(--warning))]/30">
                              <Sparkles className="w-3 h-3 mr-1" />
                              {template.tier}
                            </Badge>
                          )}
                        </div>
                        <p className="text-sm text-muted-foreground mt-0.5">
                          {CATEGORY_LABELS[template.category] || template.category}
                        </p>
                      </div>
                    </CardTitle>
                    <Button variant="ghost" size="icon" onClick={handleClose} className="flex-shrink-0">
                      <X className="w-5 h-5" />
                    </Button>
                  </CardHeader>

                  {/* Scrollable Body */}
                  <CardContent className="flex-1 overflow-y-auto pt-4 space-y-5">
                    {/* Description */}
                    <div className="space-y-2">
                      <h3 className="text-sm font-semibold flex items-center gap-2">
                        <FileText className="w-4 h-4" />
                        Description
                      </h3>
                      <div className="rounded-lg border border-border/40 bg-secondary/20 p-4">
                        <p className="text-sm text-muted-foreground leading-relaxed">
                          {template.description || 'No description available'}
                        </p>
                      </div>
                    </div>

                    {/* Persona Preview */}
                    {template.persona && (
                      <div className="space-y-2">
                        <button
                          onClick={() => setPersonaExpanded(!personaExpanded)}
                          className="flex items-center gap-2 text-sm font-semibold hover:text-primary transition-colors w-full text-left"
                        >
                          <Bot className="w-4 h-4" />
                          Persona
                          {personaExpanded ? (
                            <ChevronUp className="w-4 h-4 ml-auto" />
                          ) : (
                            <ChevronDown className="w-4 h-4 ml-auto" />
                          )}
                        </button>
                        {personaExpanded && (
                          <div className="rounded-lg border border-border/40 bg-secondary/20 p-4 max-h-48 overflow-y-auto">
                            <p className="text-sm text-muted-foreground leading-relaxed whitespace-pre-wrap">
                              {template.persona}
                            </p>
                          </div>
                        )}
                      </div>
                    )}

                    {/* Recommended Model */}
                    {template.recommended_model && (
                      <div className="space-y-2">
                        <h3 className="text-sm font-semibold flex items-center gap-2">
                          <Cpu className="w-4 h-4" />
                          Recommended Model
                        </h3>
                        <Badge className="text-xs bg-[hsl(var(--agent))]/10 text-[hsl(var(--agent))] border-[hsl(var(--agent))]/30">
                          <Cpu className="w-3 h-3 mr-1" />
                          {modelShortName(template.recommended_model)}
                        </Badge>
                        <span className="text-xs text-muted-foreground ml-2">
                          {template.recommended_model}
                        </span>
                      </div>
                    )}

                    {/* Recommended Tools */}
                    {template.recommended_tools.length > 0 && (
                      <div className="space-y-2">
                        <h3 className="text-sm font-semibold flex items-center gap-2">
                          <Wrench className="w-4 h-4" />
                          Recommended Tools ({template.recommended_tools.length})
                        </h3>
                        <div className="flex flex-wrap gap-2">
                          {template.recommended_tools.map((tool) => (
                            <Badge
                              key={tool}
                              variant="outline"
                              className="text-xs border-[hsl(var(--info))]/30 text-[hsl(var(--info))]"
                            >
                              {tool}
                            </Badge>
                          ))}
                        </div>
                      </div>
                    )}

                    {/* Tags */}
                    {template.tags.length > 0 && (
                      <div className="space-y-2">
                        <h3 className="text-sm font-semibold flex items-center gap-2">
                          <Tag className="w-4 h-4" />
                          Tags
                        </h3>
                        <div className="flex flex-wrap gap-2">
                          {template.tags.map((tag) => (
                            <Badge
                              key={tag}
                              variant="outline"
                              className="text-xs border-border text-muted-foreground"
                            >
                              {tag}
                            </Badge>
                          ))}
                        </div>
                      </div>
                    )}
                  </CardContent>

                  {/* Footer — Deploy */}
                  <div className="p-6 border-t border-border/40 bg-background/50 backdrop-blur z-20 relative space-y-3">
                    {/* Optional name override */}
                    <div className="flex items-center gap-2">
                      <Input
                        placeholder={template.name}
                        value={customName}
                        onChange={(e) => setCustomName(e.target.value)}
                        className="flex-1 bg-secondary/30 border-border/40 text-sm"
                      />
                      <span className="text-xs text-muted-foreground whitespace-nowrap">
                        Custom name (optional)
                      </span>
                    </div>

                    <Button
                      onClick={handleDeploy}
                      disabled={deploying}
                      className="w-full"
                    >
                      {deploying ? (
                        <>
                          <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                          Deploying...
                        </>
                      ) : (
                        <>
                          <Rocket className="w-4 h-4 mr-2" />
                          Add to Workspace
                        </>
                      )}
                    </Button>
                  </div>
                </>
              )}
            </Card>
          </motion.div>
        </>
      )}
    </AnimatePresence>
  )
}
