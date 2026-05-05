/**
 * Manage Apps Modal Component (PRD-36)
 * =====================================
 * 
 * Main modal for managing Composio app features per agent.
 * Implements the "Manage Apps" UI pattern from the PRD.
 */

'use client'

import { useState, useMemo } from 'react'
import { Check, X, Trash2, ExternalLink } from 'lucide-react'
import {
    Dialog,
    DialogContent,
    DialogDescription,
    DialogFooter,
    DialogHeader,
    DialogTitle,
} from '@/components/ui/dialog'
import { Button } from '@/components/ui/button'
import { useToast } from '@/hooks/use-toast'
import { AppTabList } from './app-tab-list'
import { FeatureGrid } from './feature-grid'
import {
    useConnectedApps,
    useAgentAppFeatures,
    useUpdateAgentFeatures,
    useEnableAllFeatures,
    useDisableAllFeatures,
    useDisconnectApp,
    type ComposioAction,
} from '@/hooks/use-composio-api'

interface ManageAppsModalProps {
    agentId: number
    open: boolean
    onClose: () => void
}

export function ManageAppsModal({ agentId, open, onClose }: ManageAppsModalProps) {
    const { toast } = useToast()
    const [selectedApp, setSelectedApp] = useState<string | null>(null)
    const [pendingToggles, setPendingToggles] = useState<Map<string, boolean>>(new Map())

    // Fetch data
    const { data: connections, isLoading: loadingConnections } = useConnectedApps()
    const { data: features, isLoading: loadingFeatures } = useAgentAppFeatures(
        agentId,
        selectedApp || ''
    )

    // Mutations
    const updateFeatures = useUpdateAgentFeatures()
    const enableAllFeatures = useEnableAllFeatures()
    const disableAllFeatures = useDisableAllFeatures()
    const disconnectApp = useDisconnectApp()

    // Prepare app tabs with feature counts
    const appTabs = useMemo(() => {
        if (!connections) return []

        return connections
            .filter((c) => c.status === 'active')
            .map((conn) => ({
                name: conn.app_name,
                display_name: conn.app_name.charAt(0) + conn.app_name.slice(1).toLowerCase(),
                logo_url: undefined, // TODO: Get from Composio app metadata
                enabled_count: features?.filter((f) => f.enabled).length || 0,
                total_count: features?.length || 0,
            }))
    }, [connections, features])

    // Set default selected app
    if (!selectedApp && appTabs.length > 0) {
        setSelectedApp(appTabs[0].name)
    }

    // Merge pending toggles with server state
    const mergedFeatures = useMemo(() => {
        if (!features) return []

        return features.map((f) => ({
            ...f,
            enabled: pendingToggles.has(f.name) ? pendingToggles.get(f.name)! : f.enabled,
        }))
    }, [features, pendingToggles])

    const handleToggle = (actionName: string, enabled: boolean) => {
        setPendingToggles((prev) => {
            const next = new Map(prev)
            next.set(actionName, enabled)
            return next
        })
    }

    const handleSave = async () => {
        if (!selectedApp || pendingToggles.size === 0) {
            onClose()
            return
        }

        const featureUpdates = Array.from(pendingToggles.entries()).map(([name, enabled]) => ({
            action_name: name,
            enabled,
        }))

        try {
            await updateFeatures.mutateAsync({
                agentId,
                appName: selectedApp,
                features: featureUpdates,
            })

            toast({
                title: 'Features updated',
                description: `Updated ${featureUpdates.length} feature(s) for ${selectedApp}`,
            })

            setPendingToggles(new Map())
            onClose()
        } catch (error) {
            toast({
                title: 'Error',
                description: 'Failed to update features',
                variant: 'destructive',
            })
        }
    }

    const handleEnableAll = async () => {
        if (!selectedApp) return

        try {
            await enableAllFeatures.mutateAsync({ agentId, appName: selectedApp })
            setPendingToggles(new Map())
            toast({
                title: 'All features enabled',
                description: `Enabled all features for ${selectedApp}`,
            })
        } catch (error) {
            toast({
                title: 'Error',
                description: 'Failed to enable features',
                variant: 'destructive',
            })
        }
    }

    const handleDisableAll = async () => {
        if (!selectedApp) return

        try {
            await disableAllFeatures.mutateAsync({ agentId, appName: selectedApp })
            setPendingToggles(new Map())
            toast({
                title: 'All features disabled',
                description: `Disabled all features for ${selectedApp}`,
            })
        } catch (error) {
            toast({
                title: 'Error',
                description: 'Failed to disable features',
                variant: 'destructive',
            })
        }
    }

    const handleDisconnect = async () => {
        if (!selectedApp) return

        try {
            await disconnectApp.mutateAsync(selectedApp)
            setSelectedApp(null)
            toast({
                title: 'App disconnected',
                description: `${selectedApp} has been disconnected`,
            })
        } catch (error) {
            toast({
                title: 'Error',
                description: 'Failed to disconnect app',
                variant: 'destructive',
            })
        }
    }

    const hasChanges = pendingToggles.size > 0

    return (
        <Dialog open={open} onOpenChange={(isOpen) => !isOpen && onClose()}>
            <DialogContent className="max-w-4xl bg-slate-900 border-slate-700">
                <DialogHeader>
                    <DialogTitle className="text-xl text-slate-100">Manage Apps</DialogTitle>
                    <DialogDescription className="text-slate-400">
                        Enable or disable which features your agent can use for each app
                    </DialogDescription>
                </DialogHeader>

                {loadingConnections ? (
                    <div className="py-8 text-center text-slate-400">Loading connected apps...</div>
                ) : appTabs.length === 0 ? (
                    <div className="py-12 text-center">
                        <p className="text-slate-400 mb-4">No apps connected yet</p>
                        <Button variant="outline" onClick={onClose}>
                            <ExternalLink className="h-4 w-4 mr-2" />
                            Connect Apps in Settings
                        </Button>
                    </div>
                ) : (
                    <div className="space-y-6">
                        {/* App Tabs */}
                        <AppTabList
                            apps={appTabs}
                            selectedApp={selectedApp}
                            onSelectApp={setSelectedApp}
                        />

                        {/* Bulk Actions */}
                        <div className="flex items-center gap-3 border-b border-slate-700/50 pb-4">
                            <Button
                                variant="ghost"
                                size="sm"
                                onClick={handleEnableAll}
                                disabled={!selectedApp || enableAllFeatures.isPending}
                                className="text-success hover:text-green-300 hover:bg-success/10"
                            >
                                <Check className="h-4 w-4 mr-1.5" />
                                Enable all
                            </Button>
                            <Button
                                variant="ghost"
                                size="sm"
                                onClick={handleDisableAll}
                                disabled={!selectedApp || disableAllFeatures.isPending}
                                className="text-slate-400 hover:text-slate-300 hover:bg-slate-700/50"
                            >
                                <X className="h-4 w-4 mr-1.5" />
                                Disable all
                            </Button>
                            <div className="flex-1" />
                            <Button
                                variant="ghost"
                                size="sm"
                                onClick={handleDisconnect}
                                disabled={!selectedApp || disconnectApp.isPending}
                                className="text-destructive hover:text-destructive/80 hover:bg-destructive/10"
                            >
                                <Trash2 className="h-4 w-4 mr-1.5" />
                                Remove this app
                            </Button>
                        </div>

                        {/* Feature Grid */}
                        <FeatureGrid
                            features={mergedFeatures}
                            onToggle={handleToggle}
                            loading={loadingFeatures}
                        />
                    </div>
                )}

                <DialogFooter className="gap-2">
                    <Button variant="outline" onClick={onClose}>
                        Cancel
                    </Button>
                    <Button
                        onClick={handleSave}
                        disabled={updateFeatures.isPending}
                        className={hasChanges ? 'bg-info hover:bg-info/80' : ''}
                    >
                        {hasChanges ? `Save Changes (${pendingToggles.size})` : 'Done'}
                    </Button>
                </DialogFooter>
            </DialogContent>
        </Dialog>
    )
}
