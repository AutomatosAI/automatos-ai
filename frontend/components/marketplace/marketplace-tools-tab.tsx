'use client'

import { useState, useMemo } from 'react'
import { Search, CheckCircle, Loader2, ExternalLink, Download } from 'lucide-react'
import { Input } from '@/components/ui/input'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Card, CardContent, CardHeader } from '@/components/ui/card'
import { useToast } from '@/hooks/use-toast'
import { useMutation, useQueryClient } from '@tanstack/react-query'
import {
    useAvailableApps,
    useConnectedApps,
    useInitiateConnection,
    type ComposioApp,
} from '@/hooks/use-composio-api'

// App categories for filtering
const APP_CATEGORIES = [
    { id: 'all', name: 'All Apps' },
    { id: 'developer', name: 'Developer' },
    { id: 'productivity', name: 'Productivity' },
    { id: 'communication', name: 'Communication' },
    { id: 'crm', name: 'CRM' },
    { id: 'analytics', name: 'Analytics' },
]

interface MarketplaceToolsTabProps {
    searchQuery: string
}

export function MarketplaceToolsTab({ searchQuery }: MarketplaceToolsTabProps) {
    const { toast } = useToast()
    const queryClient = useQueryClient()
    const [selectedCategory, setSelectedCategory] = useState('all')
    const [connectingApp, setConnectingApp] = useState<string | null>(null)

    // Fetch data
    const { data: apps = [], isLoading: appsLoading } = useAvailableApps()
    const { data: connections = [], refetch: refetchConnections } = useConnectedApps()

    // Mutations
    const initiateConnection = useInitiateConnection()

    // Sync marketplace mutation
    const syncCacheMutation = useMutation({
        mutationFn: async () => {
            const response = await fetch('/api/tools/sync-cache', { method: 'POST' })
            if (!response.ok) throw new Error('Sync failed')
            return response.json()
        },
        onSuccess: () => {
            toast({
                title: 'Marketplace Synced',
                description: 'Tool marketplace has been updated',
            })
            queryClient.invalidateQueries(['availableApps'])
        },
        onError: (error: any) => {
            toast({
                title: 'Sync Failed',
                description: error.message || 'Failed to sync marketplace',
                variant: 'destructive',
            })
        },
    })

    // Get connected app names for quick lookup
    const connectedApps = useMemo(() => {
        return new Set(
            connections
                .filter((c) => c.status === 'active')
                .map((c) => c.app_name.toUpperCase())
        )
    }, [connections])

    // Filter apps
    const filteredApps = useMemo(() => {
        let filtered = [...apps]

        // Search filter
        if (searchQuery.trim()) {
            const query = searchQuery.toLowerCase()
            filtered = filtered.filter(
                (app) =>
                    app.name.toLowerCase().includes(query) ||
                    app.display_name.toLowerCase().includes(query) ||
                    app.description?.toLowerCase().includes(query)
            )
        }

        // Category filter
        if (selectedCategory !== 'all') {
            filtered = filtered.filter((app) =>
                app.categories.some((c) => c.toLowerCase() === selectedCategory.toLowerCase())
            )
        }

        return filtered
    }, [apps, searchQuery, selectedCategory])

    const handleConnect = async (app: ComposioApp) => {
        setConnectingApp(app.name)
        try {
            const result = await initiateConnection.mutateAsync({
                appName: app.name,
                callbackUrl: `${window.location.origin}/marketplace?tab=tools&connected=${app.name}`,
            })

            // Open OAuth popup
            const width = 600
            const height = 700
            const left = window.screenX + (window.outerWidth - width) / 2
            const top = window.screenY + (window.outerHeight - height) / 2

            const popup = window.open(
                result.redirect_url,
                'ConnectApp',
                `width=${width},height=${height},left=${left},top=${top}`
            )

            // Poll for popup close
            const checkPopup = setInterval(() => {
                if (!popup || popup.closed) {
                    clearInterval(checkPopup)
                    setConnectingApp(null)
                    refetchConnections()
                }
            }, 500)

            // Timeout after 5 minutes
            setTimeout(() => {
                clearInterval(checkPopup)
                if (popup && !popup.closed) {
                    popup.close()
                }
                setConnectingApp(null)
            }, 300000)
        } catch (error: any) {
            console.error('Connection failed:', error)
            toast({
                title: 'Connection Failed',
                description: error.message || 'Failed to initiate connection',
                variant: 'destructive',
            })
            setConnectingApp(null)
        }
    }

    return (
        <div className="space-y-6">
            {/* Sync Button and Category Filters */}
            <div className="flex flex-col sm:flex-row gap-4 items-start sm:items-center justify-between">
                <div className="flex flex-wrap gap-2">
                    {APP_CATEGORIES.map((category) => (
                        <Button
                            key={category.id}
                            variant={selectedCategory === category.id ? 'default' : 'outline'}
                            size="sm"
                            onClick={() => setSelectedCategory(category.id)}
                            className={
                                selectedCategory === category.id
                                    ? 'bg-primary hover:bg-primary/90 text-primary-foreground'
                                    : 'border-secondary text-muted-foreground hover:bg-secondary'
                            }
                        >
                            {category.name}
                        </Button>
                    ))}
                </div>
                <Button
                    variant="outline"
                    disabled={syncCacheMutation.isPending}
                    onClick={() => syncCacheMutation.mutate()}
                    className="shrink-0"
                >
                    {syncCacheMutation.isPending ? (
                        <>
                            <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                            Syncing…
                        </>
                    ) : (
                        <>
                            <Download className="w-4 h-4 mr-2" />
                            Sync marketplace
                        </>
                    )}
                </Button>
            </div>

            {/* Tools Grid - Removed category buttons since they're above now */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
                {appsLoading ? (
                    [...Array(8)].map((_, i) => (
                        <div key={i} className="h-48 glass-card animate-pulse" />
                    ))
                ) : filteredApps.length === 0 ? (
                    <div className="col-span-full text-center py-12 text-muted-foreground">
                        <Search className="w-12 h-12 mx-auto mb-4 opacity-50" />
                        <p>No tools found. Try adjusting your search or filters.</p>
                    </div>
                ) : (
                    filteredApps.map((app) => (
                        <Card key={app.id} className="glass-card card-glow hover:border-primary/20 transition-all duration-300">
                            <CardHeader className="pb-3">
                                <div className="flex items-center gap-3">
                                    {app.logo_url && (
                                        <img src={app.logo_url} alt={app.display_name} className="w-10 h-10 rounded-lg" />
                                    )}
                                    <div className="flex-1 min-w-0">
                                        <h3 className="font-semibold line-clamp-1">{app.display_name}</h3>
                                        <p className="text-xs text-muted-foreground">{app.categories[0] || 'Tool'}</p>
                                    </div>
                                    {connectedApps.has(app.name) && (
                                        <Badge className="bg-green-500/20 text-green-300 border-green-500/30 text-xs">
                                            <CheckCircle className="w-3 h-3 mr-1" />
                                            Connected
                                        </Badge>
                                    )}
                                </div>
                            </CardHeader>
                            <CardContent className="space-y-3">
                                <p className="text-sm text-muted-foreground line-clamp-2">{app.description || 'No description available'}</p>

                                <div className="flex items-center justify-between pt-2 border-t border-secondary">
                                    <span className="text-xs text-muted-foreground">
                                        {app.action_count || 0} actions
                                    </span>
                                    {connectedApps.has(app.name) ? (
                                        <Badge variant="outline" className="text-xs">
                                            Active
                                        </Badge>
                                    ) : (
                                        <Button
                                            size="sm"
                                            variant="outline"
                                            disabled={connectingApp === app.name}
                                            onClick={() => handleConnect(app)}
                                        >
                                            {connectingApp === app.name ? (
                                                <>
                                                    <Loader2 className="w-3 h-3 mr-1 animate-spin" />
                                                    Connecting...
                                                </>
                                            ) : (
                                                'Connect'
                                            )}
                                        </Button>
                                    )}
                                </div>
                            </CardContent>
                        </Card>
                ))}
            </div>
        </div>
    )
}
