'use client'

import React, { useState, useEffect } from 'react'
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription, DialogFooter } from '@/components/ui/dialog'
import { Button } from '@/components/ui/button'
import { Switch } from '@/components/ui/switch'
import { Badge } from '@/components/ui/badge'
import { Loader2, Search, CheckCircle, AlertCircle } from 'lucide-react'
import { Input } from '@/components/ui/input'
import { Separator } from '@/components/ui/separator'
import { ToolLogo } from '@/components/ui/tool-logo'
import { ScrollArea } from '@/components/ui/scroll-area'
import { apiClient } from '@/lib/api-client'
import { useToast } from '@/hooks/use-toast'

interface ToolActionsModalProps {
    open: boolean
    onClose: () => void
    tool: any // Using any for flexibility with partial tool objects
}

interface Action {
    name: string
    display_name: string
    description: string
    enabled: boolean
}

export function ToolActionsModal({ open, onClose, tool }: ToolActionsModalProps) {
    const { toast } = useToast()
    const [loading, setLoading] = useState(true)
    const [saving, setSaving] = useState(false)
    const [actions, setActions] = useState<Action[]>([])
    const [searchQuery, setSearchQuery] = useState('')

    useEffect(() => {
        if (open && tool?.id) {
            fetchActions()
        }
    }, [open, tool])

    const fetchActions = async () => {
        setLoading(true)
        try {
            // 1. Fetch available actions from Composio (via backend)
            // Backend now merges this with enabled state from workspace config
            const actionsResponse = await apiClient.get(`/api/mcp-tools/${tool.id}/actions`)
            const availableActions = actionsResponse || []

            const mappedActions = availableActions.map((a: any) => ({
                name: a.name,
                display_name: a.display_name || a.name,
                description: a.description,
                // Backend provides 'enabled' state based on WorkspaceToolConfig
                // If no config exists, backend currently returns FALSE (implicit disable by default for safety)
                // OR we can force it here. Let's trust the backend.
                enabled: !!a.enabled
            }))

            setActions(mappedActions)
        } catch (error) {
            console.error('Failed to fetch actions:', error)
            toast({
                title: 'Error',
                description: 'Failed to load app actions',
                variant: 'destructive',
            })
        } finally {
            setLoading(false)
        }
    }

    const handleToggleAll = (enabled: boolean) => {
        setActions(prev => prev.map(a => ({ ...a, enabled })))
    }

    const handleToggleAction = (actionName: string) => {
        setActions(prev => prev.map(a =>
            a.name === actionName ? { ...a, enabled: !a.enabled } : a
        ))
    }

    const handleSave = async () => {
        setSaving(true)
        try {
            // Filter only enabled actions to save
            const enabledActions = actions.filter(a => a.enabled).map(a => a.name)

            // Call the NEW backend endpoint to save permissions
            await apiClient.post(`/api/mcp-tools/${tool.id}/actions`, {
                actions: enabledActions
            })

            toast({
                title: 'Settings Saved',
                description: `Updated permissions for ${tool.name}. ${enabledActions.length} actions enabled.`,
            })
            onClose()
        } catch (error) {
            console.error(error)
            toast({
                title: 'Save Failed',
                description: 'Could not save permission changes.',
                variant: 'destructive',
            })
        } finally {
            setSaving(false)
        }
    }

    const filteredActions = actions.filter(action =>
        action.display_name.toLowerCase().includes(searchQuery.toLowerCase()) ||
        action.name.toLowerCase().includes(searchQuery.toLowerCase())
    )

    const enabledCount = actions.filter(a => a.enabled).length

    return (
        <Dialog open={open} onOpenChange={onClose}>
            <DialogContent className="max-w-2xl max-h-[85vh] flex flex-col glass-card border-none bg-black/95">
                <DialogHeader>
                    <div className="flex items-center gap-4">
                        <ToolLogo
                            logo={tool?.logo}
                            name={tool?.name}
                            size={48}
                            fallbackIcon={tool?.icon}
                            showBackground={true}
                        />
                        <div>
                            <DialogTitle className="text-xl">Manage {tool?.name}</DialogTitle>
                            <DialogDescription>
                                Enable or disable which features your agents can use for this app.
                            </DialogDescription>
                        </div>
                    </div>
                </DialogHeader>

                <div className="space-y-4 my-2">
                    {/* Controls */}
                    <div className="flex items-center justify-between gap-4">
                        <div className="relative flex-1">
                            <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" />
                            <Input
                                placeholder="Search actions..."
                                value={searchQuery}
                                onChange={(e) => setSearchQuery(e.target.value)}
                                className="pl-9 bg-secondary/20 border-white/10"
                            />
                        </div>
                        <div className="flex items-center gap-2">
                            <Button variant="outline" size="sm" onClick={() => handleToggleAll(true)}>
                                Enable all
                            </Button>
                            <Button variant="outline" size="sm" onClick={() => handleToggleAll(false)}>
                                Disable all
                            </Button>
                        </div>
                    </div>

                    {/* Stats */}
                    <div className="flex items-center gap-2 text-sm text-muted-foreground">
                        <span className={`flex items-center gap-1 ${enabledCount > 0 ? 'text-green-400' : 'text-gray-400'}`}>
                            {enabledCount > 0 ? <CheckCircle className="w-3 h-3" /> : <AlertCircle className="w-3 h-3" />}
                            {enabledCount}/{actions.length} Enabled
                        </span>
                    </div>
                </div>

                {/* Actions List */}
                <ScrollArea className="flex-1 pr-4 -mr-4 h-[400px]">
                    {loading ? (
                        <div className="flex items-center justify-center h-40">
                            <Loader2 className="w-8 h-8 animate-spin text-orange-500" />
                        </div>
                    ) : filteredActions.length === 0 ? (
                        <div className="text-center py-12 text-muted-foreground">
                            No actions found.
                        </div>
                    ) : (
                        <div className="grid grid-cols-1 gap-2">
                            {filteredActions.map((action) => (
                                <div
                                    key={action.name}
                                    className="flex items-center justify-between p-3 rounded-lg border border-white/5 bg-secondary/10 hover:bg-secondary/20 transition-colors"
                                >
                                    <div className="flex-1 min-w-0 mr-4">
                                        <div className="font-medium text-sm truncate">{action.display_name}</div>
                                        <div className="text-xs text-muted-foreground truncate">{action.description}</div>
                                    </div>
                                    <Switch
                                        checked={action.enabled}
                                        onCheckedChange={() => handleToggleAction(action.name)}
                                        className="data-[state=checked]:bg-green-500"
                                    />
                                </div>
                            ))}
                        </div>
                    )}
                </ScrollArea>

                <DialogFooter className="mt-4 pt-4 border-t border-white/10">
                    <Button variant="ghost" onClick={onClose} disabled={saving}>Cancel</Button>
                    <Button onClick={handleSave} disabled={saving || loading}>
                        {saving ? (
                            <>
                                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                                Saving...
                            </>
                        ) : (
                            'Save Changes'
                        )}
                    </Button>
                </DialogFooter>
            </DialogContent>
        </Dialog>
    )
}
