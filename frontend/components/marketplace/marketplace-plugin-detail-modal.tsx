'use client'

/**
 * Marketplace Plugin Detail Modal (PRD-42: US-020)
 * =================================================
 * Shows full plugin information before enabling:
 * - Name, version, full description, long_description
 * - Stats: category, token estimate, skills/commands count, enable count
 * - Included skills/commands lists from manifest
 * - Recommended models, source info, security status
 * - Enable/Disable button for workspace
 */

import React, { useState, useEffect, useCallback, useRef } from 'react'
import { AnimatePresence, motion } from 'framer-motion'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Separator } from '@/components/ui/separator'
import {
  X,
  Loader2,
  Shield,
  ShieldCheck,
  ShieldAlert,
  AlertTriangle,
  Terminal,
  Zap,
  Coins,
  Download,
  ExternalLink,
  Brain,
  Tag,
  FileText,
  Cpu,
  User,
  Scale,
  CheckCircle,
  Star,
  Package,
} from 'lucide-react'
import { useToast } from '@/hooks/use-toast'
import { apiClient } from '@/lib/api-client'

// ===================================================================
// Types
// ===================================================================

interface PluginDetail {
  id: string
  slug: string
  name: string
  version: string
  description: string | null
  long_description: string | null
  category_id: string | null
  category_name: string | null
  tags: string[]
  skills_count: number
  commands_count: number
  agents_count: number
  hooks_count: number
  token_estimate: number
  recommended_models: string[]
  enable_count: number
  is_featured: boolean
  is_active: boolean
  security_status: string | null
  source_type: string | null
  source_url: string | null
  original_author: string | null
  license: string | null
  created_at: string | null
  updated_at: string | null
  manifest: {
    name?: string
    description?: string
    version?: string
    contents?: {
      skills?: Array<{ name: string; description?: string }>
      commands?: Array<{ name: string; description?: string }>
      agents?: Array<{ name: string; description?: string }>
      hooks?: Array<{ name: string; description?: string }>
    }
  } | null
}

interface MarketplacePluginDetailModalProps {
  open: boolean
  pluginId: string | null
  onClose: () => void
  isEnabled: boolean
  isEnabling: boolean
  onEnable: () => void
  onDisable?: () => void
}

// ===================================================================
// Component
// ===================================================================

export function MarketplacePluginDetailModal({
  open,
  pluginId,
  onClose,
  isEnabled,
  isEnabling,
  onEnable,
  onDisable,
}: MarketplacePluginDetailModalProps) {
  const { toast } = useToast()
  const [plugin, setPlugin] = useState<PluginDetail | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const requestIdRef = useRef(0)

  const fetchPluginDetail = useCallback(async () => {
    if (!pluginId) return

    const currentRequestId = ++requestIdRef.current
    setLoading(true)
    setError(null)
    try {
      const data = await apiClient.get(`/api/marketplace/plugins/${pluginId}`)
      if (requestIdRef.current !== currentRequestId) return // stale response
      setPlugin(data as PluginDetail)
    } catch (err: any) {
      if (requestIdRef.current !== currentRequestId) return // stale response
      const msg = err?.message || 'Failed to load plugin details'
      console.error('Failed to fetch plugin details:', err)
      setError(msg)
    } finally {
      if (requestIdRef.current === currentRequestId) {
        setLoading(false)
      }
    }
  }, [pluginId])

  useEffect(() => {
    if (open && pluginId) {
      fetchPluginDetail()
    }
    if (!open) {
      // Invalidate any in-flight requests when modal closes
      requestIdRef.current++
      setPlugin(null)
      setError(null)
    }
  }, [open, pluginId, fetchPluginDetail])

  if (!open) return null

  // Extract manifest contents
  const skills = plugin?.manifest?.contents?.skills || []
  const commands = plugin?.manifest?.contents?.commands || []

  const securityBadge = (status: string | null) => {
    switch (status) {
      case 'safe':
        return (
          <Badge variant="outline" className="border-green-500/30 text-green-400">
            <ShieldCheck className="w-3 h-3 mr-1" />
            Verified Safe
          </Badge>
        )
      case 'review_required':
        return (
          <Badge variant="outline" className="border-yellow-500/30 text-yellow-400">
            <ShieldAlert className="w-3 h-3 mr-1" />
            Review Required
          </Badge>
        )
      default:
        return (
          <Badge variant="outline" className="border-gray-500/30 text-gray-400">
            <Shield className="w-3 h-3 mr-1" />
            Pending
          </Badge>
        )
    }
  }

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
            onClick={onClose}
          />

          {/* Modal */}
          <motion.div
            className="fixed inset-0 z-50 flex items-center justify-center p-4"
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.95 }}
            transition={{ duration: 0.2 }}
          >
            <Card className="glass-card w-full max-w-4xl max-h-[90vh] overflow-hidden flex flex-col">
              {/* Loading State */}
              {loading && (
                <div className="p-12 flex flex-col items-center justify-center">
                  <Loader2 className="w-8 h-8 animate-spin text-muted-foreground mb-4" />
                  <p className="text-sm text-muted-foreground">Loading plugin details...</p>
                </div>
              )}

              {/* Error State */}
              {error && !loading && (
                <div className="p-12 flex flex-col items-center justify-center">
                  <p className="text-red-400 mb-4">{error}</p>
                  <Button variant="outline" onClick={onClose}>Close</Button>
                </div>
              )}

              {/* Content */}
              {plugin && !loading && !error && (
                <>
                  {/* Header */}
                  <CardHeader className="flex flex-row items-start justify-between pb-2">
                    <CardTitle className="flex items-start gap-3 text-xl flex-1 min-w-0">
                      <div className="w-12 h-12 rounded-lg bg-orange-500/10 border border-orange-500/20 flex items-center justify-center flex-shrink-0">
                        <Package className="w-6 h-6 text-orange-400" />
                      </div>
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center gap-2 flex-wrap">
                          <span className="truncate">{plugin.name}</span>
                          <Badge variant="outline" className="text-xs">
                            v{plugin.version}
                          </Badge>
                          {plugin.is_featured && (
                            <Badge className="bg-orange-500/20 text-orange-400 border-orange-500/30 text-[10px] h-5">
                              <Star className="w-2.5 h-2.5 mr-0.5" />
                              Featured
                            </Badge>
                          )}
                        </div>
                        <div className="flex items-center gap-2 mt-1 text-sm text-muted-foreground">
                          {plugin.original_author && (
                            <span className="flex items-center gap-1">
                              <User className="w-3 h-3" />
                              {plugin.original_author}
                            </span>
                          )}
                          {plugin.category_name && (
                            <>
                              {plugin.original_author && <span>&middot;</span>}
                              <span>{plugin.category_name}</span>
                            </>
                          )}
                        </div>
                      </div>
                    </CardTitle>
                    <Button variant="ghost" size="icon" onClick={onClose} className="flex-shrink-0">
                      <X className="w-5 h-5" />
                    </Button>
                  </CardHeader>

                  {/* Scrollable Body */}
                  <CardContent className="flex-1 overflow-y-auto pt-4 space-y-6">
                    {/* Security Status Badge */}
                    <div className="flex items-center gap-2">
                      {securityBadge(plugin.security_status)}
                    </div>

                    {/* Description */}
                    <div className="space-y-3">
                      <h3 className="text-sm font-semibold flex items-center gap-2">
                        <FileText className="w-4 h-4" />
                        Description
                      </h3>
                      <div className="rounded-lg border border-border/40 bg-secondary/20 p-4">
                        <p className="text-sm text-muted-foreground leading-relaxed">
                          {plugin.description || 'No description available'}
                        </p>
                        {plugin.long_description && (
                          <p className="text-sm text-muted-foreground leading-relaxed mt-3">
                            {plugin.long_description}
                          </p>
                        )}
                      </div>
                    </div>

                    {/* Stats Row */}
                    <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
                      <div className="rounded-lg border border-border/40 bg-secondary/20 p-3 text-center">
                        <div className="text-xl font-bold text-blue-400">{plugin.skills_count}</div>
                        <div className="text-xs text-muted-foreground flex items-center justify-center gap-1">
                          <Terminal className="w-3 h-3" />
                          Skills
                        </div>
                      </div>
                      <div className="rounded-lg border border-border/40 bg-secondary/20 p-3 text-center">
                        <div className="text-xl font-bold text-purple-400">{plugin.commands_count}</div>
                        <div className="text-xs text-muted-foreground flex items-center justify-center gap-1">
                          <Zap className="w-3 h-3" />
                          Commands
                        </div>
                      </div>
                      <div className="rounded-lg border border-border/40 bg-secondary/20 p-3 text-center">
                        <div className="text-xl font-bold text-orange-400">~{plugin.token_estimate}</div>
                        <div className="text-xs text-muted-foreground flex items-center justify-center gap-1">
                          <Coins className="w-3 h-3" />
                          Tokens
                        </div>
                      </div>
                      <div className="rounded-lg border border-border/40 bg-secondary/20 p-3 text-center">
                        <div className="text-xl font-bold text-green-400">{plugin.enable_count}</div>
                        <div className="text-xs text-muted-foreground flex items-center justify-center gap-1">
                          <Download className="w-3 h-3" />
                          Enabled
                        </div>
                      </div>
                    </div>

                    {/* Included Skills */}
                    {skills.length > 0 && (
                      <div className="space-y-3">
                        <h3 className="text-sm font-semibold flex items-center gap-2">
                          <Brain className="w-4 h-4" />
                          Included Skills ({skills.length})
                        </h3>
                        <div className="space-y-2">
                          {skills.map((skill, idx) => (
                            <div
                              key={idx}
                              className="flex items-start p-3 rounded-lg border border-white/5 bg-secondary/10"
                            >
                              <div className="min-w-0 flex-1">
                                <div className="font-medium text-sm flex items-center gap-2">
                                  <Terminal className="w-3 h-3 text-blue-400 shrink-0" />
                                  {skill.name}
                                </div>
                                {skill.description && (
                                  <p className="text-xs text-muted-foreground mt-1">
                                    {skill.description}
                                  </p>
                                )}
                              </div>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}

                    {/* Included Commands */}
                    {commands.length > 0 && (
                      <div className="space-y-3">
                        <h3 className="text-sm font-semibold flex items-center gap-2">
                          <Zap className="w-4 h-4" />
                          Included Commands ({commands.length})
                        </h3>
                        <div className="space-y-2">
                          {commands.map((cmd, idx) => (
                            <div
                              key={idx}
                              className="flex items-start p-3 rounded-lg border border-white/5 bg-secondary/10"
                            >
                              <div className="min-w-0 flex-1">
                                <div className="font-medium text-sm flex items-center gap-2">
                                  <Zap className="w-3 h-3 text-purple-400 shrink-0" />
                                  {cmd.name}
                                </div>
                                {cmd.description && (
                                  <p className="text-xs text-muted-foreground mt-1">
                                    {cmd.description}
                                  </p>
                                )}
                              </div>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}

                    {/* Recommended Models */}
                    {plugin.recommended_models && plugin.recommended_models.length > 0 && (
                      <div className="space-y-3">
                        <h3 className="text-sm font-semibold flex items-center gap-2">
                          <Cpu className="w-4 h-4" />
                          Recommended Models
                        </h3>
                        <div className="flex flex-wrap gap-2">
                          {plugin.recommended_models.map((model) => (
                            <Badge
                              key={model}
                              variant="outline"
                              className="text-xs border-blue-500/30 text-blue-400"
                            >
                              {model}
                            </Badge>
                          ))}
                        </div>
                      </div>
                    )}

                    {/* Tags */}
                    {plugin.tags && plugin.tags.length > 0 && (
                      <div className="space-y-3">
                        <h3 className="text-sm font-semibold flex items-center gap-2">
                          <Tag className="w-4 h-4" />
                          Tags
                        </h3>
                        <div className="flex flex-wrap gap-2">
                          {plugin.tags.map((tag) => (
                            <Badge
                              key={tag}
                              variant="outline"
                              className="text-xs border-gray-600 text-gray-400"
                            >
                              {tag}
                            </Badge>
                          ))}
                        </div>
                      </div>
                    )}

                    {/* Source Info */}
                    {(plugin.original_author || plugin.license || plugin.source_url) && (
                      <div className="space-y-3">
                        <Separator />
                        <h3 className="text-sm font-semibold">Source Information</h3>
                        <div className="rounded-lg border border-border/40 bg-secondary/20 p-4 space-y-2 text-sm">
                          {plugin.original_author && (
                            <div className="flex items-center gap-2">
                              <User className="w-4 h-4 text-muted-foreground" />
                              <span className="text-muted-foreground">Author:</span>
                              <span>{plugin.original_author}</span>
                            </div>
                          )}
                          {plugin.license && (
                            <div className="flex items-center gap-2">
                              <Scale className="w-4 h-4 text-muted-foreground" />
                              <span className="text-muted-foreground">License:</span>
                              <span>{plugin.license}</span>
                            </div>
                          )}
                          {plugin.source_url && (
                            <div className="flex items-center gap-2">
                              <ExternalLink className="w-4 h-4 text-muted-foreground" />
                              <span className="text-muted-foreground">Source:</span>
                              <a
                                href={plugin.source_url}
                                target="_blank"
                                rel="noopener noreferrer"
                                className="text-blue-400 hover:underline truncate"
                              >
                                {plugin.source_url}
                              </a>
                            </div>
                          )}
                          {plugin.source_type && (
                            <div className="flex items-center gap-2">
                              <Package className="w-4 h-4 text-muted-foreground" />
                              <span className="text-muted-foreground">Type:</span>
                              <span className="capitalize">{plugin.source_type}</span>
                            </div>
                          )}
                        </div>
                      </div>
                    )}
                  </CardContent>

                  {/* Footer — Enable/Disable or Deactivated notice */}
                  <div className="p-6 border-t border-border/40 bg-background/50 backdrop-blur z-20 relative">
                    {plugin.is_active === false ? (
                      <div className="flex items-center justify-center gap-2 p-3 rounded-lg border border-red-500/30 bg-red-500/10 text-red-400">
                        <AlertTriangle className="w-5 h-5 flex-shrink-0" />
                        <span className="text-sm font-medium">This plugin has been deactivated</span>
                      </div>
                    ) : (
                      <div className="flex gap-3">
                        {isEnabled ? (
                          onDisable ? (
                            <Button
                              onClick={onDisable}
                              variant="outline"
                              className="flex-1 border-red-500/30 text-red-400 hover:bg-red-500/10"
                            >
                              <X className="w-4 h-4 mr-2" />
                              Disable Plugin
                            </Button>
                          ) : (
                            <Button
                              disabled
                              variant="secondary"
                              className="flex-1 bg-secondary/50 border border-white/10"
                            >
                              <CheckCircle className="w-4 h-4 mr-2" />
                              Enabled for Workspace
                            </Button>
                          )
                        ) : (
                          <Button
                            onClick={onEnable}
                            disabled={isEnabling}
                            className="flex-1 bg-orange-600 hover:bg-orange-700 text-white"
                          >
                            {isEnabling ? (
                              <>
                                <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                                Enabling...
                              </>
                            ) : (
                              <>
                                <Download className="w-4 h-4 mr-2" />
                                Enable for Workspace
                              </>
                            )}
                          </Button>
                        )}
                      </div>
                    )}
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
