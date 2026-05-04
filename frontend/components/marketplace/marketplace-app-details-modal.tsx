'use client'

/**
 * Marketplace App Details Modal
 * =============================
 * Display-only modal for marketplace app features and triggers
 * No enable/disable toggles - info only
 */

import React from 'react'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog'
import { ToolLogo } from '@/components/ui/tool-logo'
import {
  ExternalLink,
  BookOpen,
  Settings,
  Shield,
  Zap,
  Database,
  Cloud,
  MessageSquare,
  Terminal,
  Container,
  CreditCard,
  Activity,
  HardDrive,
  Globe,
  X,
  Search,
  Loader2,
  CheckCircle
} from 'lucide-react'
import { useState, useEffect } from 'react'
import { Input } from '@/components/ui/input'
import { apiClient } from '@/lib/api-client'
import type { ComposioApp } from '@/hooks/use-composio-api'

interface MarketplaceAppDetailsModalProps {
  open: boolean
  onClose: () => void
  app: ComposioApp | null
  onConnect?: () => void
  loading?: boolean
  isConnected?: boolean
}

const categoryIcons: Record<string, any> = {
  'AI/ML': Zap,
  'Database': Database,
  'Cloud': Cloud,
  'Communication': MessageSquare,
  'Developer': Terminal,
  'Infrastructure': Container,
  'Business': CreditCard,
  'Monitoring': Activity,
  'Storage': HardDrive,
  'Integration': Globe,
  'Security': Shield
}

export function MarketplaceAppDetailsModal({
  open,
  onClose,
  app,
  onConnect,
  loading = false,
  isConnected = false
}: MarketplaceAppDetailsModalProps) {
  const [activeTab, setActiveTab] = useState<'features' | 'triggers'>('features')
  const [actions, setActions] = useState<any[]>([])
  const [actionsLoading, setActionsLoading] = useState(false)
  const [searchQuery, setSearchQuery] = useState('')
  const [triggerSearchQuery, setTriggerSearchQuery] = useState('')

  useEffect(() => {
    if (open && app) {
      fetchActions()
    }
  }, [open, app])

  const fetchActions = async () => {
    if (!app) return

    setActionsLoading(true)
    try {
      const actionsResponse = await (apiClient as any).get(`/api/composio/apps/${app.name}/actions`)
      const mapped = (actionsResponse.data || actionsResponse || []).map((a: any) => ({
        ...a,
        enabled: !!a.enabled
      }))
      setActions(mapped)
    } catch (error) {
      console.error('Failed to fetch actions:', error)
      setActions([])
    } finally {
      setActionsLoading(false)
    }
  }

  const filteredActions = actions.filter(a =>
    a.name?.toLowerCase().includes(searchQuery.toLowerCase()) ||
    (a.display_name || '').toLowerCase().includes(searchQuery.toLowerCase())
  )

  if (!app) return null

  const CategoryIcon = categoryIcons[app.categories?.[0]] || Settings
  const triggerList = Array.isArray(app.triggers) ? app.triggers : []
  const filteredTriggers = triggerList.filter((trigger: any) => {
    const name = typeof trigger === 'string' ? trigger : (trigger.name || trigger.trigger_name || trigger.display_name || '')
    return name.toLowerCase().includes(triggerSearchQuery.toLowerCase())
  })

  const formatAuthScheme = (scheme: string) => {
    const normalized = scheme.toLowerCase()
    if (normalized.includes('oauth')) return 'OAuth2'
    if (normalized.includes('bearer')) return 'Bearer Token'
    if (normalized.includes('api_key') || normalized.includes('apikey')) return 'API Key'
    if (normalized.includes('basic')) return 'Basic Auth'
    return scheme.replace(/_/g, ' ').toUpperCase()
  }

  const authSchemes = Array.isArray(app.auth_schemes) ? app.auth_schemes : []

  return (
    <Dialog open={open} onOpenChange={onClose}>
      <DialogContent size="lg">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-3 text-xl">
            <ToolLogo
              logo={app.logo_url}
              name={app.display_name}
              size={48}
              fallbackIcon={undefined}
              showBackground={true}
            />
            <div>
              <div className="flex items-center gap-2">
                {app.display_name}
                {app.categories?.[0] && (
                  <Badge variant="outline" className="text-xs">
                    {app.categories[0]}
                  </Badge>
                )}
              </div>
              <p className="text-sm text-muted-foreground font-normal">
                Composio
                {isConnected && (
                  <>
                    {' • '}
                    <span className="text-[hsl(var(--success))]">Connected</span>
                  </>
                )}
              </p>
            </div>
          </DialogTitle>
        </DialogHeader>

        <div className="flex-1 overflow-y-auto pt-4 space-y-6">
                <div className="space-y-6">
                  {/* Description */}
                  <div className="space-y-3">
                    <h3 className="text-sm font-semibold flex items-center gap-2">
                      <CategoryIcon className="w-4 h-4" />
                      Description
                    </h3>
                    <div className="rounded-lg border border-border/40 bg-secondary/20 p-4">
                      <p className="text-sm text-muted-foreground leading-relaxed">
                        {app.description || 'No description available'}
                      </p>
                    </div>
                  </div>

                  {/* Capabilities Overview */}
                  <div className="grid grid-cols-2 gap-4">
                    <div className="rounded-lg border border-border/40 bg-secondary/20 p-4">
                      <div className="text-2xl font-bold text-[hsl(var(--info))]">{app.action_count || 0}</div>
                      <div className="text-sm text-muted-foreground">Features Available</div>
                    </div>
                    <div className="rounded-lg border border-border/40 bg-secondary/20 p-4">
                      <div className="text-2xl font-bold text-[hsl(var(--agent))]">{app.trigger_count || 0}</div>
                      <div className="text-sm text-muted-foreground">Triggers Available</div>
                    </div>
                  </div>

                  {/* Auth Schemes */}
                  {authSchemes.length > 0 && (
                    <div className="space-y-3">
                      <h3 className="text-sm font-semibold flex items-center gap-2">
                        <Shield className="w-4 h-4" />
                        Authentication Methods
                      </h3>
                      <div className="flex flex-wrap gap-2">
                        {authSchemes.map((scheme: string) => (
                          <Badge
                            key={scheme}
                            variant="outline"
                            className="text-xs border-primary/40 text-primary"
                          >
                            {formatAuthScheme(scheme)}
                          </Badge>
                        ))}
                      </div>
                    </div>
                  )}

                  {/* Features & Triggers Tabs */}
                  <div className="space-y-4">
                    <div className="flex space-x-4 border-b border-border/40">
                      <button
                        onClick={() => setActiveTab('features')}
                        className={`text-sm font-medium pb-2 border-b-2 transition-colors ${
                          activeTab === 'features'
                            ? 'border-primary text-primary'
                            : 'border-transparent text-muted-foreground hover:text-foreground'
                        }`}
                      >
                        Features ({actions.length})
                      </button>
                      <button
                        onClick={() => setActiveTab('triggers')}
                        className={`text-sm font-medium pb-2 border-b-2 transition-colors ${
                          activeTab === 'triggers'
                            ? 'border-primary text-primary'
                            : 'border-transparent text-muted-foreground hover:text-foreground'
                        }`}
                      >
                        Triggers ({triggerList.length})
                      </button>
                    </div>

                    {/* Features Tab */}
                    {activeTab === 'features' && (
                      <div className="h-full flex flex-col">
                        <div className="flex items-center gap-4 mb-4">
                          <div className="relative flex-1">
                            <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" />
                            <Input
                              placeholder="Search features..."
                              value={searchQuery}
                              onChange={(e) => setSearchQuery(e.target.value)}
                              className="pl-9"
                            />
                          </div>
                        </div>

                        <div className="space-y-2 max-h-96 overflow-y-auto">
                          {actionsLoading ? (
                            <div className="flex justify-center p-8">
                              <Loader2 className="w-6 h-6 animate-spin text-muted-foreground" />
                            </div>
                          ) : filteredActions.length === 0 ? (
                            <div className="text-center p-8 text-muted-foreground">
                              {searchQuery ? 'No features found matching your search.' : 'No features found.'}
                            </div>
                          ) : (
                            filteredActions.map(action => (
                              <div
                                key={action.name}
                                className="flex items-start justify-between p-3 rounded-lg border border-border/30 bg-secondary/10"
                              >
                                <div className="min-w-0 flex-1">
                                  <div className="font-medium text-sm flex items-center gap-2">
                                    <CheckCircle className="w-3 h-3 text-[hsl(var(--info))] shrink-0" />
                                    {action.display_name || action.name}
                                  </div>
                                  {action.description && (
                                    <div className="text-xs text-muted-foreground mt-1 line-clamp-2">
                                      {action.description}
                                    </div>
                                  )}
                                </div>
                              </div>
                            ))
                          )}
                        </div>
                      </div>
                    )}

                    {/* Triggers Tab */}
                    {activeTab === 'triggers' && (
                      <div className="h-full flex flex-col">
                        <div className="flex items-center gap-4 mb-4">
                          <div className="relative flex-1">
                            <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" />
                            <Input
                              placeholder="Search triggers..."
                              value={triggerSearchQuery}
                              onChange={(e) => setTriggerSearchQuery(e.target.value)}
                              className="pl-9"
                            />
                          </div>
                        </div>

                        <div className="space-y-2 max-h-96 overflow-y-auto">
                          {filteredTriggers.length === 0 ? (
                            <div className="text-center p-8 text-muted-foreground">
                              {triggerSearchQuery ? 'No triggers found matching your search.' : 'No triggers available.'}
                            </div>
                          ) : (
                            filteredTriggers.map((trigger: any) => {
                              const triggerName = typeof trigger === 'string'
                                ? trigger
                                : (trigger.display_name || trigger.name || trigger.trigger_name || 'Trigger')
                              const triggerDescription = typeof trigger === 'string' ? '' : (trigger.description || '')

                              return (
                                <div
                                  key={triggerName}
                                  className="flex items-start justify-between p-3 rounded-lg border border-border/30 bg-secondary/10"
                                >
                                  <div className="min-w-0 flex-1">
                                    <div className="font-medium text-sm flex items-center gap-2">
                                      <Zap className="w-3 h-3 text-[hsl(var(--agent))] shrink-0" />
                                      {triggerName}
                                    </div>
                                    {triggerDescription && (
                                      <div className="text-xs text-muted-foreground mt-1 line-clamp-2">
                                        {triggerDescription}
                                      </div>
                                    )}
                                  </div>
                                </div>
                              )
                            })
                          )}
                        </div>
                      </div>
                    )}
                  </div>
                </div>
        </div>

        {/* Action Footer - Add to Workspace button */}
        <div className="border-t border-border/40 pt-4">
          <div className="flex gap-3">
            <Button
              onClick={onConnect}
              disabled={loading}
              variant="outline"
              className="flex-1"
            >
              <ExternalLink className="w-4 h-4 mr-2" />
              {loading ? 'Adding...' : 'Add to Workspace'}
            </Button>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  )
}
