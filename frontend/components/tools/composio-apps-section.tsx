/**
 * Composio Apps Section Component (PRD-36)
 * =========================================
 * 
 * Self-contained component for the "App Integrations" tab in tools-dashboard.
 * Displays Composio apps in grid format with connect/disconnect functionality.
 */

'use client'

import { useState, useMemo } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Search, Zap, CheckCircle, ExternalLink, Loader2, Wrench } from 'lucide-react'
import { Input } from '@/components/ui/input'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Card, CardContent, CardHeader } from '@/components/ui/card'
import { useToast } from '@/hooks/use-toast'
import {
    useAvailableApps,
    useConnectedApps,
    useInitiateConnection,
    useDisconnectApp,
    type ComposioApp,
} from '@/hooks/use-composio-api'

// App categories for filtering
const APP_CATEGORIES = [
    { id: 'all', name: 'All Apps', color: 'text-muted-foreground' },
    { id: 'developer', name: 'Developer', color: 'text-[hsl(var(--info))]' },
    { id: 'productivity', name: 'Productivity', color: 'text-[hsl(var(--success))]' },
    { id: 'communication', name: 'Communication', color: 'text-[hsl(var(--agent))]' },
    { id: 'crm', name: 'CRM', color: 'text-primary' },
    { id: 'analytics', name: 'Analytics', color: 'text-[hsl(var(--info))]' },
]

export function ComposioAppsSection() {
    const { toast } = useToast()
    const [searchQuery, setSearchQuery] = useState('')
    const [selectedCategory, setSelectedCategory] = useState('all')
    const [connectingApp, setConnectingApp] = useState<string | null>(null)

    // Fetch data
    const { data: apps = [], isLoading: appsLoading, refetch: refetchApps } = useAvailableApps()
    const { data: connections = [], refetch: refetchConnections } = useConnectedApps()

    // Mutations
    const initiateConnection = useInitiateConnection()
    const disconnectApp = useDisconnectApp()

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
                callbackUrl: `${window.location.origin}/tools?connected=${app.name}`,
            })

            // Open OAuth popup
            const width = 600
            const height = 700
            const left = window.screenX + (window.outerWidth - width) / 2
            const top = window.screenY + (window.outerHeight - height) / 2

            const popup = window.open(
                result.redirect_url,
                `Connect ${app.display_name}`,
                `width=${width},height=${height},left=${left},top=${top}`
            )

            // Poll for popup close
            const checkClosed = setInterval(() => {
                if (popup?.closed) {
                    clearInterval(checkClosed)
                    setConnectingApp(null)
                    refetchApps()
                    refetchConnections()
                    toast({
                        title: 'Connection initiated',
                        description: `Check if ${app.display_name} is now connected.`,
                    })
                }
            }, 1000)

            // Timeout after 5 minutes
            setTimeout(() => {
                clearInterval(checkClosed)
                if (!popup?.closed) popup?.close()
                setConnectingApp(null)
            }, 5 * 60 * 1000)
        } catch (error) {
            setConnectingApp(null)
            toast({
                title: 'Connection failed',
                description: `Failed to connect to ${app.display_name}`,
                variant: 'destructive',
            })
        }
    }

    const handleDisconnect = async (appName: string) => {
        try {
            await disconnectApp.mutateAsync(appName)
            refetchApps()
            refetchConnections()
            toast({
                title: 'Disconnected',
                description: `${appName} has been disconnected.`,
            })
        } catch (error) {
            toast({
                title: 'Error',
                description: 'Failed to disconnect app',
                variant: 'destructive',
            })
        }
    }

    // Loading state
    if (appsLoading) {
        return (
            <div className="space-y-6">
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
                    {[...Array(6)].map((_, i) => (
                        <Card key={i} className="glass-card animate-pulse">
                            <CardHeader className="pb-2">
                                <div className="flex items-center gap-3">
                                    <div className="w-10 h-10 bg-secondary/50 rounded-lg" />
                                    <div className="space-y-2">
                                        <div className="h-4 w-24 bg-secondary/50 rounded" />
                                        <div className="h-3 w-32 bg-secondary/50 rounded" />
                                    </div>
                                </div>
                            </CardHeader>
                            <CardContent>
                                <div className="h-8 w-20 bg-secondary/50 rounded mt-2" />
                            </CardContent>
                        </Card>
                    ))}
                </div>
            </div>
        )
    }

    const connectedCount = connections.filter((c) => c.status === 'active').length

    return (
        <div className="space-y-6">
            {/* Header Stats */}
            <div className="flex items-center justify-between">
                <div>
                    <h3 className="text-xl font-semibold">App Integrations</h3>
                    <p className="text-sm text-muted-foreground">
                        Connect external apps to give your agents 500+ capabilities
                    </p>
                </div>
                <div className="flex items-center gap-3">
                    <Badge variant="outline" className="text-[hsl(var(--success))] border-[hsl(var(--success))]/30">
                        <div className="w-2 h-2 bg-[hsl(var(--success))] rounded-full animate-pulse mr-2" />
                        {connectedCount} Connected
                    </Badge>
                    <Badge variant="outline" className="text-[hsl(var(--info))] border-[hsl(var(--info))]/30">
                        {apps.length} Available
                    </Badge>
                </div>
            </div>

            {/* Search and Filters */}
            <div className="flex flex-col sm:flex-row gap-4">
                <div className="relative flex-1">
                    <Search className="absolute left-3 top-1/2 -translate-y-1/2 text-muted-foreground w-4 h-4" />
                    <Input
                        placeholder="Search apps..."
                        value={searchQuery}
                        onChange={(e) => setSearchQuery(e.target.value)}
                        className="pl-10 bg-secondary/50 border-secondary focus:border-primary/50"
                    />
                </div>
            </div>

            {/* Category Filters */}
            <div className="flex flex-wrap gap-2">
                {APP_CATEGORIES.map((category) => (
                    <Button
                        key={category.id}
                        variant={selectedCategory === category.id ? 'default' : 'outline'}
                        size="sm"
                        onClick={() => setSelectedCategory(category.id)}
                        className={
                            selectedCategory === category.id
                                ? 'bg-secondary border-primary/50 text-foreground'
                                : 'hover:border-primary/50'
                        }
                    >
                        {category.name}
                    </Button>
                ))}
            </div>

            {/* Apps Grid */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
                <AnimatePresence>
                    {filteredApps.map((app, index) => {
                        const isConnected = connectedApps.has(app.name.toUpperCase())
                        const isConnecting = connectingApp === app.name

                        return (
                            <motion.div
                                key={app.name}
                                initial={{ opacity: 0, y: 20 }}
                                animate={{ opacity: 1, y: 0 }}
                                exit={{ opacity: 0, scale: 0.95 }}
                                transition={{ duration: 0.3, delay: index * 0.05 }}
                            >
                                <Card
                                    className={`glass-card hover:border-primary/20 transition-all duration-300 ${isConnected ? 'border-[hsl(var(--success))]/30' : ''
                                        }`}
                                >
                                    <CardHeader className="pb-2">
                                        <div className="flex items-start justify-between">
                                            <div className="flex items-center gap-3">
                                                {app.logo_url ? (
                                                    <img
                                                        src={app.logo_url}
                                                        alt={app.display_name}
                                                        className="w-10 h-10 rounded-lg"
                                                    />
                                                ) : (
                                                    <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-[hsl(var(--info))] to-[hsl(var(--agent))] flex items-center justify-center text-foreground font-semibold">
                                                        {app.display_name.charAt(0)}
                                                    </div>
                                                )}
                                                <div>
                                                    <h4 className="font-medium text-sm">{app.display_name}</h4>
                                                    {app.description && (
                                                        <p className="text-xs text-muted-foreground line-clamp-1">
                                                            {app.description}
                                                        </p>
                                                    )}
                                                </div>
                                            </div>
                                            {isConnected && (
                                                <Badge className="bg-[hsl(var(--success))]/10 text-[hsl(var(--success))] border-[hsl(var(--success))]/20">
                                                    <CheckCircle className="w-3 h-3 mr-1" />
                                                    Connected
                                                </Badge>
                                            )}
                                        </div>
                                    </CardHeader>
                                    <CardContent>
                                        <div className="flex items-center justify-between mt-2">
                                            <div className="flex gap-1">
                                                {app.categories.slice(0, 2).map((cat) => (
                                                    <Badge
                                                        key={cat}
                                                        variant="outline"
                                                        className="text-xs border-border text-muted-foreground"
                                                    >
                                                        {cat}
                                                    </Badge>
                                                ))}
                                            </div>
                                            {isConnected ? (
                                                <Button
                                                    variant="ghost"
                                                    size="sm"
                                                    onClick={() => handleDisconnect(app.name)}
                                                    className="text-[hsl(var(--destructive))] hover:text-[hsl(var(--destructive))] hover:bg-[hsl(var(--destructive))]/10 h-8"
                                                >
                                                    Disconnect
                                                </Button>
                                            ) : (
                                                <Button
                                                    variant="outline"
                                                    size="sm"
                                                    onClick={() => handleConnect(app)}
                                                    disabled={isConnecting}
                                                    className="bg-secondary hover:bg-secondary/80 border-border h-8"
                                                >
                                                    {isConnecting ? (
                                                        <>
                                                            <Loader2 className="w-3 h-3 mr-1 animate-spin" />
                                                            Connecting...
                                                        </>
                                                    ) : (
                                                        <>
                                                            <ExternalLink className="w-3 h-3 mr-1" />
                                                            Connect
                                                        </>
                                                    )}
                                                </Button>
                                            )}
                                        </div>
                                        <div className="mt-4 flex items-center gap-4 text-xs text-muted-foreground">
                                            <div className="flex items-center gap-1">
                                                <Wrench className="w-3 h-3" />
                                                <span>{app.action_count ?? 0} Tools</span>
                                            </div>
                                            <div className="flex items-center gap-1">
                                                <Zap className="w-3 h-3" />
                                                <span>{(app as any).trigger_count ?? 0} Triggers</span>
                                            </div>
                                        </div>

                                        {Array.isArray((app as any).auth_schemes) && (app as any).auth_schemes.length > 0 && (
                                            <div className="mt-3 flex flex-wrap gap-2">
                                                {(app as any).auth_schemes.slice(0, 3).map((scheme: string) => {
                                                    const normalized = scheme.toLowerCase()
                                                    const label = normalized.includes('oauth')
                                                        ? 'OAuth2'
                                                        : normalized.includes('bearer')
                                                            ? 'Bearer Token'
                                                            : normalized.includes('api_key') || normalized.includes('apikey')
                                                                ? 'API Key'
                                                                : normalized.includes('basic')
                                                                    ? 'Basic Auth'
                                                                    : scheme.replace(/_/g, ' ').toUpperCase()
                                                    return (
                                                        <Badge key={scheme} variant="outline" className="text-[10px] h-5 border-primary/30 text-primary">
                                                            {label}
                                                        </Badge>
                                                    )
                                                })}
                                            </div>
                                        )}
                                    </CardContent>
                                </Card>
                            </motion.div>
                        )
                    })}
                </AnimatePresence>
            </div>

            {/* Empty State */}
            {filteredApps.length === 0 && (
                <div className="text-center py-12">
                    <div className="w-16 h-16 rounded-lg bg-secondary/30 flex items-center justify-center mx-auto mb-4">
                        <Zap className="w-8 h-8 text-muted-foreground" />
                    </div>
                    <h3 className="text-lg font-semibold mb-2">No apps found</h3>
                    <p className="text-muted-foreground mb-4">
                        {searchQuery ? `No apps match "${searchQuery}"` : 'No apps available in this category'}
                    </p>
                    <Button
                        variant="outline"
                        onClick={() => {
                            setSearchQuery('')
                            setSelectedCategory('all')
                        }}
                    >
                        Clear Filters
                    </Button>
                </div>
            )}
        </div>
    )
}
