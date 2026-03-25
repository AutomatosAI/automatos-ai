'use client'

import { useState } from 'react'
import { AnimatePresence, motion } from 'framer-motion'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Input } from '@/components/ui/input'
import { StatusBadge } from '@/components/shared'
import { PremiumIcon } from '@/components/shared'
import { useSystemIcons } from '@/hooks/use-system-config-api'
import { AGENT_CATEGORIES as UNIFIED_CATEGORIES } from '@/lib/agent-constants'
import {
  X,
  Loader2,
  Bot,
  Zap,
  Tag,
  Wrench,
  ChevronDown,
  ChevronUp,
  FileText,
  Download,
} from 'lucide-react'
import {
  useAgentCatalogTemplate,
  useDeployAgentCatalogTemplate,
  type AgentCatalogTemplateDetail as TemplateDetail,
} from '@/hooks/use-marketplace-api'

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

  const { data: iconMappings = {} } = useSystemIcons()
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

  const getIconForCategory = (category: string) => {
    const unifiedId = CATALOG_TO_UNIFIED[category] || category
    const premiumName = iconMappings[unifiedId] || iconMappings[category]
    if (premiumName) return <PremiumIcon name={premiumName} size={48} className="shrink-0" />
    const catDef = UNIFIED_CATEGORIES.find(c => c.id === unifiedId)
    if (catDef) {
      const Icon = catDef.icon
      return <Icon className="w-12 h-12 text-primary shrink-0" />
    }
    return <Bot className="w-12 h-12 text-primary shrink-0" />
  }

  const getCategoryName = (category: string) => {
    const unifiedId = CATALOG_TO_UNIFIED[category] || category
    const catDef = UNIFIED_CATEGORIES.find(c => c.id === unifiedId)
    return catDef?.name || category.charAt(0).toUpperCase() + category.slice(1).replace('-', ' ')
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
                      {getIconForCategory(template.category)}
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center gap-2 flex-wrap">
                          <span className="truncate">{template.name}</span>
                        </div>
                        <p className="text-sm text-muted-foreground mt-0.5">
                          {getCategoryName(template.category)}
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
                          <Zap className="w-4 h-4" />
                          Recommended Model
                        </h3>
                        <div className="flex items-center gap-2">
                          <Badge className="text-xs bg-[hsl(var(--agent))]/10 text-[hsl(var(--agent))] border-[hsl(var(--agent))]/30">
                            <Zap className="w-3 h-3 mr-1" />
                            {modelShortName(template.recommended_model)}
                          </Badge>
                          <span className="text-xs text-muted-foreground">
                            {template.recommended_model}
                          </span>
                        </div>
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
                            <div
                              key={tool}
                              className="bg-secondary/30 px-2 py-1 flex items-center rounded-md border border-border/30 text-xs text-muted-foreground"
                            >
                              {tool}
                            </div>
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
                        <div className="flex flex-wrap gap-1.5">
                          {template.tags.map((tag) => (
                            <StatusBadge key={tag} size="sm" status="neutral">
                              {tag}
                            </StatusBadge>
                          ))}
                        </div>
                      </div>
                    )}
                  </CardContent>

                  {/* Footer - Deploy */}
                  <div className="p-6 border-t border-border/40 bg-background/50 backdrop-blur z-20 relative space-y-3">
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
                          <Download className="w-4 h-4 mr-2" />
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
