'use client'

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
  Trash2
} from 'lucide-react'

interface ToolDetailsModalProps {
  open: boolean
  onClose: () => void
  tool: any
  onInstall?: () => void
  onConfigure?: () => void
  onUninstall?: () => void
  onRemoveFromWorkspace?: () => void
  loading?: boolean
  initialTab?: 'features' | 'triggers'
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

// ... (previous imports)
import { useState, useEffect } from 'react'
import { Switch } from '@/components/ui/switch'
import { Input } from '@/components/ui/input'
import { Search, Loader2 } from 'lucide-react'
import { apiClient } from '@/lib/api-client'
import { useToast } from '@/hooks/use-toast'

// ... (keep ToolDetailsModalProps)

export function ToolDetailsModal({
  open,
  onClose,
  tool,
  onInstall,
  onConfigure,
  onUninstall,
  onRemoveFromWorkspace,
  loading = false,
  initialTab
}: ToolDetailsModalProps) {
  const { toast } = useToast()
  const [activeTab, setActiveTab] = useState<'features' | 'triggers'>(initialTab || 'features')
  const [actions, setActions] = useState<any[]>([])
  const [actionsLoading, setActionsLoading] = useState(false)
  const [searchQuery, setSearchQuery] = useState('')
  const [triggerSearchQuery, setTriggerSearchQuery] = useState('')

  useEffect(() => {
    if (open && tool && tool.provider === 'Composio') {
      fetchActions()
    }
  }, [open, tool])

  const fetchActions = async () => {
    setActionsLoading(true)
    try {
      const actionsResponse = await (apiClient as any).get(`/api/tools/${tool.name}/actions`)
      // Backend now returns correct enabled state
      const mapped = (actionsResponse.data || actionsResponse || []).map((a: any) => ({
        ...a,
        enabled: !!a.enabled // Respect backend state
      }))
      setActions(mapped)
    } catch (error) {
      console.error('Failed to fetch actions:', error)
    } finally {
      setActionsLoading(false)
    }
  }

  const handleToggleAction = async (name: string) => {
    // 1. Optimistic update
    const newActions = actions.map(a => a.name === name ? { ...a, enabled: !a.enabled } : a)
    setActions(newActions)

    // 2. Persist to Backend
    try {
      const enabledList = newActions.filter(a => a.enabled).map(a => a.name)
      await (apiClient as any).post(`/api/tools/${tool.name}/actions`, {
        actions: enabledList
      })
      toast({ title: "Updated", description: "Permissions saved." })
    } catch (error) {
      console.error("Failed to save permissions", error)
      toast({ title: "Error", description: "Failed to save permissions", variant: "destructive" })
      // Revert on error
      setActions(actions)
    }
  }

  const filteredActions = actions.filter(a =>
    a.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
    (a.display_name || '').toLowerCase().includes(searchQuery.toLowerCase())
  )

  if (!tool) return null

  const CategoryIcon = categoryIcons[tool.category] || Settings
  const metadata = tool.metadata || {}
  const triggerList = Array.isArray(tool?.metadata?.triggers) ? tool.metadata.triggers : []
  const filteredTriggers = triggerList.filter((trigger: any) => {
    const name = typeof trigger === 'string' ? trigger : (trigger.name || trigger.trigger_name || '')
    return name.toLowerCase().includes(triggerSearchQuery.toLowerCase())
  })

  return (
    <Dialog open={open} onOpenChange={onClose}>
      <DialogContent size="md">
        <DialogHeader>
          <div className="flex items-center justify-between">
            <DialogTitle className="flex items-center gap-3 text-xl">
              <ToolLogo
                logo={tool.logo}
                name={tool.name}
                size={48}
                fallbackIcon={tool.icon}
                showBackground={true}
              />
              <div>
                <div className="flex items-center gap-2">
                  <span className="gradient-text">{tool.name}</span>
                  <Badge variant="outline" className="text-xs">
                    {tool.category}
                  </Badge>
                </div>
                <p className="text-sm text-muted-foreground font-normal">
                  {tool.provider} • v{tool.version}
                </p>
              </div>
            </DialogTitle>
            {/* Save button - only show if connected and on features tab */}
            {tool.isInstalled && activeTab === 'features' && (
              <Button
                size="sm"
                onClick={async () => {
                  try {
                    const enabledList = actions.filter(a => a.enabled).map(a => a.name)
                    await (apiClient as any).post(`/api/tools/${tool.name}/actions`, {
                      actions: enabledList
                    })
                    toast({ title: "Settings Saved", description: `${enabledList.length} features enabled.` })
                    onClose()
                  } catch (e: any) {
                    console.error("Save failed:", e)
                    toast({ title: "Error", description: `Failed to save: ${e.message || "Unknown error"}`, variant: "destructive" })
                  }
                }}
                variant="outline"
                className="mr-8"
              >
                Save Changes
              </Button>
            )}
          </div>
        </DialogHeader>

        <div className="flex-1 overflow-y-auto pt-4 space-y-6">

                <div className="space-y-6">
                  <div className="space-y-3">
                    <h3 className="text-sm font-semibold flex items-center gap-2">
                      <CategoryIcon className="w-4 h-4" />
                      Description
                    </h3>
                    <div className="rounded-lg border border-border/40 bg-secondary/20 p-4">
                      <p className="text-sm text-muted-foreground leading-relaxed">
                        {tool.description}
                      </p>
                    </div>
                  </div>

                  {tool.provider === 'Composio' && (
                    <div className="space-y-4">
                      <div className="flex space-x-4 border-b border-border/40">
                        <button
                          onClick={() => setActiveTab('features')}
                          className={`text-sm font-medium pb-2 border-b-2 transition-colors ${activeTab === 'features' ? 'border-primary text-primary' : 'border-transparent text-muted-foreground hover:text-foreground'}`}
                        >
                          Features ({actions.length})
                        </button>
                        <button
                          onClick={() => setActiveTab('triggers')}
                          className={`text-sm font-medium pb-2 border-b-2 transition-colors ${activeTab === 'triggers' ? 'border-primary text-primary' : 'border-transparent text-muted-foreground hover:text-foreground'}`}
                        >
                          Trigger ({triggerList.length})
                        </button>
                      </div>

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
                            {actions.length > 0 && (() => {
                              const allEnabled = actions.every(a => a.enabled)
                              const noneEnabled = actions.every(a => !a.enabled)
                              return (
                                <Button
                                  size="sm"
                                  variant="outline"
                                  className="shrink-0 text-xs"
                                  onClick={() => {
                                    const enableAll = !allEnabled
                                    setActions(prev => prev.map(a => ({ ...a, enabled: enableAll })))
                                  }}
                                >
                                  {allEnabled ? 'Disable All' : noneEnabled ? 'Enable All' : 'Enable All'}
                                </Button>
                              )
                            })()}
                          </div>

                          <div className="space-y-2 mb-4">
                            {actionsLoading ? (
                              <div className="flex justify-center p-8"><Loader2 className="w-6 h-6 animate-spin text-muted-foreground" /></div>
                            ) : filteredActions.length === 0 ? (
                              <div className="text-center p-8 text-muted-foreground">No features found.</div>
                            ) : (
                              filteredActions.map(action => (
                                <div key={action.name} className="flex items-center justify-between p-3 rounded-lg border border-border/50 bg-secondary/10">
                                  <div className="min-w-0 mr-4">
                                    <div className="font-medium text-sm">{action.display_name || action.name}</div>
                                    <div className="text-xs text-muted-foreground truncate">{action.description}</div>
                                  </div>
                                  <Switch
                                    checked={action.enabled}
                                    onCheckedChange={() => {
                                      // Update local state ONLY
                                      setActions(prev => prev.map(a => a.name === action.name ? { ...a, enabled: !a.enabled } : a))
                                    }}
                                  />
                                </div>
                              ))
                            )}
                          </div>


                        </div>
                      )}

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

                          <div className="space-y-2">
                            {filteredTriggers.length === 0 ? (
                              <div className="text-center p-8 text-muted-foreground">No triggers available.</div>
                            ) : (
                              filteredTriggers.map((trigger: any) => {
                                const triggerName = typeof trigger === 'string'
                                  ? trigger
                                  : (trigger.display_name || trigger.name || trigger.trigger_name || 'Trigger')
                                const triggerDescription = typeof trigger === 'string' ? '' : (trigger.description || '')
                                return (
                                  <div key={triggerName} className="flex items-center justify-between p-3 rounded-lg border border-border/50 bg-secondary/10">
                                    <div className="min-w-0 mr-4">
                                      <div className="font-medium text-sm">{triggerName}</div>
                                      {triggerDescription && (
                                        <div className="text-xs text-muted-foreground truncate">{triggerDescription}</div>
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
                  )}

                  {metadata.documentation && (
                    <div className="mt-4">
                      <Button variant="outline" size="sm" onClick={() => window.open(metadata.documentation, '_blank')} className="w-full">
                        <BookOpen className="w-4 h-4 mr-2" /> View Documentation <ExternalLink className="w-4 h-4 ml-2" />
                      </Button>
                    </div>
                  )}
                </div>

              </div>

        <div className="border-t border-border/40 pt-4">
          <div className="flex flex-col gap-3">
            <div className="flex gap-3">
              {tool.isInstalled ? (
                <>
                  {/* Disconnect OAuth button - only for connected apps */}
                  <Button
                    variant="outline"
                    onClick={async () => {
                      if (onUninstall) {
                        await onUninstall()
                        onClose()
                      }
                    }}
                    className="flex-1 hover:border-destructive/50 text-destructive"
                    disabled={loading}
                  >
                    Disconnect OAuth
                  </Button>
                </>
              ) : (
                <Button
                  variant="outline"
                  onClick={onInstall}
                  disabled={loading}
                  className="flex-1"
                >
                  <ExternalLink className="w-4 h-4 mr-2" />
                  Connect with Composio
                </Button>
              )}
            </div>

            {/* Remove from Workspace - show for ALL workspace apps (connected or not) */}
            {onRemoveFromWorkspace && (
              <Button
                variant="outline"
                onClick={onRemoveFromWorkspace}
                className="w-full hover:border-destructive/50 text-destructive"
                disabled={loading}
              >
                <Trash2 className="w-4 h-4 mr-2" />
                Remove from Workspace
              </Button>
            )}
          </div>
        </div>
      </DialogContent>
    </Dialog>
  )
}