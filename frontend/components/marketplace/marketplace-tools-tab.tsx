'use client'

import { useState, useMemo } from 'react'
import { Search, CheckCircle, Loader2, ExternalLink } from 'lucide-react'
import { Input } from '@/components/ui/input'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Card, CardContent, CardHeader } from '@/components/ui/card'
import { useToast } from '@/hooks/use-toast'
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

export function MarketplaceToolsTab() {
    const { toast } = useToast()
    const [searchQuery, setSearchQuery] = useState('')
    const [selectedCategory, setSelectedCategory] = useState('all')
    const [connectingApp, setConnectingApp] = useState<string | null>(null)

    // Fetch data
    const { data: apps = [], isLoading: appsLoading } = useAvailableApps()
    const { data: connections = [], refetch: refetchConnections } = useConnectedApps()

    // Mutations
    const initiateConnection = useInitiateConnection()

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
            {/* Search and Filters */}
            <div className="flex flex-col sm:flex-row gap-4">
                <div className="relative flex-1">
                    <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 h-4 w-4" />
                    <Input
                        type="text"
                        placeholder="Search tools..."
                        value={searchQuery}
                        onChange={(e) => setSearchQuery(e.target.value)}
                        className="pl-10 bg-[#1a1a1a] border-gray-800 text-white placeholder:text-gray-500"
                    />
                </div>
            </div>

            {/* Category Filter Buttons */}
            <div className="flex flex-wrap gap-2">
                {APP_CATEGORIES.map((category) => (
                    <Button
                        key={category.id}
                        variant={selectedCategory === category.id ? 'default' : 'outline'}
                        size="sm"
                        onClick={() => setSelectedCategory(category.id)}
                        className={
                            selectedCategory === category.id
                                ? 'bg-orange-500 hover:bg-orange-600 text-white'
                                : 'border-gray-700 text-gray-300 hover:bg-gray-800'
                        }
                    >
                        {category.name}
                    </Button>
                ))}
            </div>

            {/* Apps Grid */}
            {appsLoading ? (
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                    {[...Array(6)].map((_, i) => (
                        <div key={i} className="h-48 bg-gray-800 animate-pulse rounded-lg" />
                    ))}
                </div>
            ) : filteredApps.length === 0 ? (
                <div className="text-center py-12 text-gray-400">
                    No tools found. Try adjusting your search or filters.
                </div>
            ) : (
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                    {filteredApps.map((app) => {
                        const isConnected = connectedApps.has(app.name.toUpperCase())
                        const isConnecting = connectingApp === app.name

                        return (
                            <Card
                                key={app.name}
                                className="bg-[#1a1a1a] border-gray-800 hover:border-orange-500/50 transition-all duration-200"
                            >
                                <CardHeader className="pb-3">
                                    <div className="flex items-start justify-between gap-3">
                                        <div className="flex items-center gap-3 flex-1 min-w-0">
                                            {app.logo && (
                                                <img
                                                    src={app.logo}
                                                    alt={app.display_name}
                                                    className="w-10 h-10 rounded object-contain bg-white p-1"
                                                />
                                            )}
                                            <div className="flex-1 min-w-0">
                                                <h3 className="font-semibold text-white line-clamp-1">
                                                    {app.display_name}
                                                </h3>
                                                {app.categories.length > 0 && (
                                                    <p className="text-xs text-gray-500">
                                                        {app.categories[0]}
                                                    </p>
                                                )}
                                            </div>
                                        </div>
                                        {isConnected && (
                                            <Badge className="bg-green-500/20 text-green-500 border-green-500/30 shrink-0">
                                                <CheckCircle className="h-3 w-3 mr-1" />
                                                Connected
                                            </Badge>
                                        )}
                                    </div>
                                </CardHeader>

                                <CardContent className="space-y-4">
                                    <p className="text-sm text-gray-400 line-clamp-2">
                                        {app.description || 'No description available'}
                                    </p>

                                    <div className="flex items-center gap-2 text-xs text-gray-500">
                                        <span>{app.actions?.length || 0} Tools</span>
                                        {app.auth_schemes && app.auth_schemes.length > 0 && (
                                            <>
                                                <span>•</span>
                                                <span className="capitalize">
                                                    {app.auth_schemes[0].replace('_', ' ')}
                                                </span>
                                            </>
                                        )}
                                    </div>

                                    <Button
                                        onClick={() => !isConnected && handleConnect(app)}
                                        disabled={isConnected || isConnecting}
                                        className={
                                            isConnected
                                                ? 'w-full bg-gray-700 text-gray-400 cursor-not-allowed'
                                                : 'w-full bg-orange-500 hover:bg-orange-600 text-white'
                                        }
                                    >
                                        {isConnecting ? (
                                            <>
                                                <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                                                Connecting...
                                            </>
                                        ) : isConnected ? (
                                            'Connected'
                                        ) : (
                                            'Connect'
                                        )}
                                    </Button>
                                </CardContent>
                            </Card>
                        )
                    })}
                </div>
            )}
        </div>
    )
}
