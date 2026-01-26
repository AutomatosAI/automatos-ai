'use client'

import React from 'react'
import { AnimatePresence, motion } from 'framer-motion'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
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
  X
} from 'lucide-react'

interface ToolDetailsModalProps {
  open: boolean
  onClose: () => void
  tool: any
  onInstall?: () => void
  onConfigure?: () => void
  onUninstall?: () => void
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
    <AnimatePresence>
      {open && (
        <>
          <motion.div
            className="fixed inset-0 bg-black/50 backdrop-blur-sm z-50"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={onClose}
          />

          <motion.div
            className="fixed inset-0 z-50 flex items-center justify-center p-4"
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.95 }}
            transition={{ duration: 0.2 }}
          >
            <Card className="glass-card w-full max-w-4xl max-h-[90vh] overflow-hidden flex flex-col">
              <CardHeader className="flex flex-row items-center justify-between pb-2">
                <CardTitle className="flex items-center gap-3 text-xl">
                  <ToolLogo
                    logo={tool.logo}
                    name={tool.name}
                    size={48}
                    fallbackIcon={tool.icon}
                    showBackground={true}
                  />
                  <div>
                    <div className="flex items-center gap-2">
                      {tool.name}
                      <Badge variant="outline" className="text-xs">
                        {tool.category}
                      </Badge>
                    </div>
                    <p className="text-sm text-muted-foreground font-normal">
                      {tool.provider} • v{tool.version}
                    </p>
                  </div>
                </CardTitle>
                <Button variant="ghost" size="icon" onClick={onClose}>
                  <X className="w-5 h-5" />
                </Button>
              </CardHeader>

              <CardContent className="flex-1 overflow-y-auto pt-4 space-y-6">

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
                          </div>

                          <div className="space-y-2 mb-4">
                            {actionsLoading ? (
                              <div className="flex justify-center p-8"><Loader2 className="w-6 h-6 animate-spin text-muted-foreground" /></div>
                            ) : filteredActions.length === 0 ? (
                              <div className="text-center p-8 text-muted-foreground">No features found.</div>
                            ) : (
                              filteredActions.map(action => (
                                <div key={action.name} className="flex items-center justify-between p-3 rounded-lg border border-white/5 bg-secondary/10">
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
                                  <div key={triggerName} className="flex items-center justify-between p-3 rounded-lg border border-white/5 bg-secondary/10">
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

              </CardContent>

              <div className="p-6 border-t border-border/40 bg-background/50 backdrop-blur z-20 relative">
                <div className="flex gap-3">
                  {tool.isInstalled ? (
                    <>
                      {/* Show Save button if on features tab */}
                      {activeTab === 'features' ? (
                        <Button
                          type="button"
                          onClick={async () => {
                            try {
                              console.log("DEBUG: Save clicked")
                              const enabledList = actions.filter(a => a.enabled).map(a => a.name)
                              console.log("DEBUG: Saving actions:", enabledList)

                              // Visual confirmation for user debugging
                              // alert(`DEBUG: Attempting to save ${enabledList.length} actions for tool ${tool.id}`)

                              await (apiClient as any).post(`/api/tools/${tool.name}/actions`, {
                                actions: enabledList
                              })

                              toast({ title: "Settings Saved", description: `${enabledList.length} features enabled.` })
                            } catch (e: any) {
                              console.error("Save failed:", e)
                              alert(`DEBUG ERROR: ${e.message || JSON.stringify(e)}`)
                              toast({ title: "Error", description: `Failed to save: ${e.message || "Unknown error"}`, variant: "destructive" })
                            }
                          }}
                          className="flex-1 bg-green-600 hover:bg-green-700 text-white cursor-pointer"
                        >
                          Save Changes
                        </Button>
                      ) : (
                        <Button
                          variant="outline"
                          className="flex-1 hover:border-blue-500/50"
                          onClick={() => setActiveTab('features')}
                        >
                          <Settings className="w-4 h-4 mr-2" />
                          Manage Features
                        </Button>
                      )}

                      <Button
                        variant="outline"
                        onClick={onUninstall}
                        className="hover:border-red-500/50 text-red-400"
                        disabled={loading}
                      >
                        Disconnect
                      </Button>
                    </>
                  ) : (
                    <Button
                      onClick={onInstall}
                      disabled={loading}
                      className="flex-1 bg-brand-primary hover:bg-brand-primary/90"
                    >
                      <ExternalLink className="w-4 h-4 mr-2" />
                      Connect with Composio
                    </Button>
                  )}
                </div>
              </div>
            </Card>
          </motion.div>
        </>
      )}
    </AnimatePresence>
  )
}