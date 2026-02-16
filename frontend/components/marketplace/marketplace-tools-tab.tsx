/**
 * Marketplace Tools Tab - Matching tools-dashboard.tsx Design
 * ============================================================
 * Uses the exact ToolCard component from tools-dashboard.tsx
 * Adapted to work with Composio apps from useAvailableApps() hook
 */

'use client'

import { useState, useMemo, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import {
    Search,
    Zap,
    CheckCircle,
    ExternalLink,
    Loader2,
    Wrench,
    Settings,
    Star,
    AlertTriangle,
    Clock,
    RefreshCw,
    MoreVertical,
    Eye,
    Download,
} from 'lucide-react'
import { Input } from '@/components/ui/input'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import {
    DropdownMenu,
    DropdownMenuContent,
    DropdownMenuItem,
    DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu'
import { Separator } from '@/components/ui/separator'
import { useToast } from '@/hooks/use-toast'
import { ViewToggle } from '@/components/shared/view-toggle'
import { useViewMode } from '@/hooks/use-view-mode'
import { ToolLogo } from '@/components/ui/tool-logo'
import {
    useAvailableApps,
    useInitiateConnection,
    useDisconnectApp,
    type ComposioApp,
} from '@/hooks/use-composio-api'
import { MarketplaceAppDetailsModal } from './marketplace-app-details-modal'
import { apiClient } from '@/lib/api-client'
import { useQuery, useQueryClient } from '@tanstack/react-query'
import { useSyncToolsCache } from '@/hooks/use-tools-api'
import { useSystemRole } from '@/contexts/role-context'
import { EnhancedPagination } from '@/components/ui/pagination'

const statusIcons = {
    available: CheckCircle,
    deprecated: AlertTriangle,
    maintenance: Clock,
    beta: Zap
}

const statusColors = {
    available: 'text-[hsl(var(--success))]',
    deprecated: 'text-[hsl(var(--destructive))]',
    maintenance: 'text-[hsl(var(--warning))]',
    beta: 'text-[hsl(var(--info))]'
}

interface MarketplaceToolsTabProps {
    searchQuery: string
}

export function MarketplaceToolsTab({ searchQuery }: MarketplaceToolsTabProps) {
    const { toast } = useToast()
    const { isAdmin } = useSystemRole()
    const syncCacheMutation = useSyncToolsCache()
    const [viewMode, setViewMode] = useViewMode('mp-tools')
    const [selectedCategory, setSelectedCategory] = useState('all')
    const [connectingApp, setConnectingApp] = useState<string | null>(null)
    const [detailsModalOpen, setDetailsModalOpen] = useState(false)
    const [selectedApp, setSelectedApp] = useState<ComposioApp | null>(null)
    const [directApps, setDirectApps] = useState<ComposioApp[]>([])

    // Pagination state - 40 items per page
    const [currentPage, setCurrentPage] = useState(1)
    const pageSize = 40

    console.log('🎬 [MARKETPLACE] MarketplaceToolsTab component rendering...')

    // Fetch categories from API (top 15 by count)
    const { data: categoriesData } = useQuery({
        queryKey: ['tool-categories'],
        queryFn: async () => {
            const categories = await apiClient.getToolCategories()
            // Add "All Apps" at the beginning
            return [
                { id: 'all', name: 'All Apps', count: 0 },
                ...categories
            ]
        },
        staleTime: 5 * 60 * 1000, // 5 minutes
    })

    const categories = categoriesData || [{ id: 'all', name: 'All Apps', count: 0 }]

    // Fetch from /api/tools/marketplace (DB cache - FAST)
    const [isLoadingApps, setIsLoadingApps] = useState(true)

    useEffect(() => {
        setIsLoadingApps(true)

        // Build query string - pass category filter and limit to backend
        const queryParts: string[] = ['limit=1000']
        if (selectedCategory !== 'all') {
            queryParts.push(`category=${encodeURIComponent(selectedCategory)}`)
        }

        // Use the faster DB-cached endpoint with category filter
        apiClient.get(`/api/tools/marketplace?${queryParts.join('&')}`)
            .then((data: any) => {
                console.log('📦 [MARKETPLACE] Fetched from DB cache:', data)
                // Transform the response to match ComposioApp interface
                const apps = (data.apps || []).map((app: any) => ({
                    name: app.app_name,
                    display_name: app.display_name,
                    description: app.description,
                    logo_url: app.logo_url,
                    categories: app.categories || [],
                    is_connected: app.is_connected || false,
                    action_count: app.action_count || 0,
                    trigger_count: app.trigger_count || 0,
                    auth_schemes: app.auth_schemes || [],
                    triggers: app.triggers || []
                }))
                setDirectApps(apps)
                setIsLoadingApps(false)
            })
            .catch(err => {
                console.error('Failed to fetch apps from marketplace:', err)
                setIsLoadingApps(false)
            })
    }, [selectedCategory])

    const apps = directApps
    const appsLoading = isLoadingApps
    const appsError = null
    const refetchApps = () => {}

    console.log('📊 [MARKETPLACE] useAvailableApps hook result:', {
        apps,
        appsType: Array.isArray(apps) ? 'array' : typeof apps,
        appsCount: apps?.length || 0,
        appsLoading,
        appsError: appsError ? String(appsError) : null
    })

    // Fetch workspace tools from YOUR database (not Composio API)
    const { data: workspaceData, refetch: refetchWorkspace } = useQuery({
        queryKey: ['workspace-tools'],
        queryFn: async () => {
            console.log('📡 [MARKETPLACE] Fetching workspace tools...')
            const response = await apiClient.get('/api/tools/connected')
            console.log('📡 [MARKETPLACE] Workspace tools response:', response)
            return (response as any)?.apps || []
        },
        staleTime: 30 * 1000,
    })

    // Mutations
    const initiateConnection = useInitiateConnection()
    const disconnectApp = useDisconnectApp()

    // Get workspace tools from YOUR database
    const workspaceTools = workspaceData || []

    // Debug logging
    console.log('🔧 [MARKETPLACE] Component state:', {
        apps,
        appsCount: apps.length,
        appsLoading,
        appsError,
        workspaceTools,
        workspaceCount: workspaceTools.length,
        searchQuery,
        selectedCategory
    })

    // Get connected app names (status === 'active')
    const connectedApps = useMemo(() => {
        return new Set(
            workspaceTools
                .filter((tool: any) => tool.status === 'active')
                .map((tool: any) => (tool.app_name || '').toUpperCase())
        )
    }, [workspaceTools])

    // Get ALL workspace apps (active + added + pending)
    // All represent apps the user has interacted with in their workspace
    const workspaceApps = useMemo(() => {
        return new Set(
            workspaceTools
                .filter((tool: any) => {
                    const status = (tool.status || '').toLowerCase()
                    return status === 'active' || status === 'added' || status === 'pending'
                })
                .map((tool: any) => (tool.app_name || '').toUpperCase())
        )
    }, [workspaceTools])

    // Filter apps (category filter is done on backend, only search filter here)
    const filteredApps = useMemo(() => {
        let filtered = [...apps]

        // Search filter (client-side for instant feedback)
        if (searchQuery.trim()) {
            const query = searchQuery.toLowerCase()
            filtered = filtered.filter(
                (app) =>
                    app.name.toLowerCase().includes(query) ||
                    app.display_name.toLowerCase().includes(query) ||
                    app.description?.toLowerCase().includes(query)
            )
        }

        // Category filter is handled by backend API call
        // (see useEffect dependency on selectedCategory)

        return filtered
    }, [apps, searchQuery])

    // Paginated apps - show only current page
    const paginatedApps = useMemo(() => {
        const startIndex = (currentPage - 1) * pageSize
        const endIndex = startIndex + pageSize
        return filteredApps.slice(startIndex, endIndex)
    }, [filteredApps, currentPage, pageSize])

    // Pagination data for EnhancedPagination component
    const paginationData = useMemo(() => ({
        total: filteredApps.length,
        skip: (currentPage - 1) * pageSize,
        limit: pageSize,
        pages: Math.ceil(filteredApps.length / pageSize),
        current_page: currentPage
    }), [filteredApps.length, currentPage, pageSize])

    // Reset to page 1 when filters change
    useEffect(() => {
        setCurrentPage(1)
    }, [searchQuery, selectedCategory])

    const handleAddToWorkspace = async (app: ComposioApp) => {
        console.log('🚀 [ADD_TO_WORKSPACE] ==================== BUTTON CLICKED ====================')
        console.log('🚀 [ADD_TO_WORKSPACE] App:', JSON.stringify(app, null, 2))
        console.log('🚀 [ADD_TO_WORKSPACE] App name:', app.name)
        console.log('🚀 [ADD_TO_WORKSPACE] Setting connectingApp state...')
        setConnectingApp(app.name)

        try {
            console.log(`🚀 [ADD_TO_WORKSPACE] Making API call to /api/tools/add-to-workspace`)
            console.log(`🚀 [ADD_TO_WORKSPACE] Payload:`, { app_name: app.name })

            // Call the new add-to-workspace endpoint
            const result = await apiClient.post('/api/tools/add-to-workspace', {
                app_name: app.name,
            })

            console.log('✅ [ADD_TO_WORKSPACE] API Response received:', JSON.stringify(result, null, 2))

            // Refresh workspace tools to update UI
            await refetchWorkspace()

            setConnectingApp(null)

            const statusMsg = (result as any)?.status
            if (statusMsg === 'already_added') {
                toast({
                    title: 'Already in Workspace',
                    description: `${app.display_name} is already in your workspace.`,
                })
            } else {
                toast({
                    title: 'Added to Workspace',
                    description: `${app.display_name} has been added. Go to Tools > Applications to connect it.`,
                })
            }
        } catch (error) {
            console.error('[ADD_TO_WORKSPACE] Failed:', error)
            setConnectingApp(null)

            // Better error handling
            let errorMsg = 'Unknown error'
            if (error instanceof Error) {
                errorMsg = error.message
            } else if (typeof error === 'object' && error !== null) {
                errorMsg = JSON.stringify(error)
            }

            toast({
                title: 'Failed to add',
                description: `Failed to add ${app.display_name}: ${errorMsg}`,
                variant: 'destructive',
            })
        }
    }

    const handleDisconnect = async (appName: string) => {
        try {
            await disconnectApp.mutateAsync(appName)
            refetchApps()
            refetchWorkspace()
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

    const handleOpenDetails = (app: ComposioApp) => {
        setSelectedApp(app)
        setDetailsModalOpen(true)
    }

    // Loading state
    if (appsLoading) {
        return (
            <div className="space-y-6">
                {viewMode === 'list' ? (
                    <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
                        {[...Array(6)].map((_, i) => (
                            <div key={i} className="h-16 glass-card animate-pulse rounded-xl" />
                        ))}
                    </div>
                ) : (
                    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
                        {[...Array(8)].map((_, i) => (
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
                )}
            </div>
        )
    }

    // Error state
    if (appsError) {
        return (
            <div className="text-center py-12">
                <div className="text-[hsl(var(--destructive))] mb-4">
                    <p className="text-lg font-semibold">Error loading tools</p>
                    <p className="text-sm mt-2">{(appsError as any)?.message || 'Unknown error'}</p>
                </div>
            </div>
        )
    }

    const connectedCount = workspaceTools.filter((tool: any) => tool.status === 'active').length

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
                    <ViewToggle value={viewMode} onChange={setViewMode} />
                    {isAdmin && (
                        <Button
                            variant="outline"
                            size="sm"
                            onClick={() => syncCacheMutation.mutate('full')}
                            disabled={syncCacheMutation.isLoading}
                            className="gap-2 text-muted-foreground hover:text-foreground"
                        >
                            <RefreshCw className={`w-3.5 h-3.5 ${syncCacheMutation.isLoading ? 'animate-spin' : ''}`} />
                            {syncCacheMutation.isLoading ? 'Syncing...' : 'Sync'}
                        </Button>
                    )}
                </div>
            </div>

            {/* Category Filters - Horizontal Scroll */}
            <div className="relative">
                <div className="flex gap-2 overflow-x-auto pb-2 scrollbar-thin scrollbar-thumb-secondary scrollbar-track-transparent">
                    {categories.map((category) => (
                        <Button
                            key={category.id}
                            variant={selectedCategory === category.id ? 'default' : 'outline'}
                            size="sm"
                            onClick={() => setSelectedCategory(category.id)}
                            className={`whitespace-nowrap flex-shrink-0 ${
                                selectedCategory === category.id
                                    ? 'bg-secondary border-primary/50 text-foreground font-semibold'
                                    : 'border-secondary text-muted-foreground hover:bg-secondary'
                            }`}
                        >
                            {category.name}
                            {category.count > 0 && (
                                <Badge variant="secondary" className="ml-2 bg-secondary/50 text-xs">
                                    {category.count}
                                </Badge>
                            )}
                        </Button>
                    ))}
                </div>
            </div>

            {/* Apps Grid - Using ToolCard component */}
            {viewMode === 'list' ? (
                <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
                    {paginatedApps.map((app) => {
                        const isConnected = connectedApps.has(app.name.toUpperCase())
                        const isInWorkspace = workspaceApps.has(app.name.toUpperCase())
                        const isConnecting = connectingApp === app.name

                        return (
                            <Card
                                key={app.name}
                                className="glass-card hover:border-primary/20 transition-all cursor-pointer"
                                onClick={() => handleOpenDetails(app)}
                            >
                                <CardContent className="p-3">
                                    <div className="flex items-center gap-3">
                                        <ToolLogo
                                            logo={app.logo_url}
                                            name={app.display_name}
                                            size={36}
                                            fallbackIcon={undefined}
                                            showBackground={true}
                                        />
                                        <div className="flex-1 min-w-0">
                                            <div className="flex items-center gap-2">
                                                <span className="font-semibold text-sm truncate">{app.display_name}</span>
                                                {app.categories?.slice(0, 1).map((tag) => (
                                                    <Badge key={tag} variant="outline" className="text-[10px] h-5 shrink-0">{tag}</Badge>
                                                ))}
                                            </div>
                                            <div className="flex items-center gap-2 text-xs text-muted-foreground mt-0.5">
                                                <span>{app.action_count ?? 0} Tools</span>
                                                <span>&middot;</span>
                                                <span>{app.trigger_count ?? 0} Triggers</span>
                                            </div>
                                        </div>
                                        {isInWorkspace ? (
                                            <Badge variant="secondary" className="text-xs shrink-0">
                                                <CheckCircle className="w-3 h-3 mr-1" />
                                                Added
                                            </Badge>
                                        ) : (
                                            <Button
                                                variant="ghost"
                                                size="sm"
                                                className="h-8 w-8 p-0 shrink-0"
                                                onClick={(e) => { e.stopPropagation(); handleAddToWorkspace(app) }}
                                                disabled={isConnecting}
                                            >
                                                {isConnecting ? <Loader2 className="w-4 h-4 animate-spin" /> : <Download className="w-4 h-4" />}
                                            </Button>
                                        )}
                                    </div>
                                </CardContent>
                            </Card>
                        )
                    })}
                </div>
            ) : (
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
                    <AnimatePresence>
                        {paginatedApps.map((app, index) => {
                            const isConnected = connectedApps.has(app.name.toUpperCase())
                            const isInWorkspace = workspaceApps.has(app.name.toUpperCase())
                            const isConnecting = connectingApp === app.name

                            return (
                                <ToolCard
                                    key={app.name}
                                    app={app}
                                    index={index}
                                    isConnected={isConnected}
                                    isInWorkspace={isInWorkspace}
                                    isConnecting={isConnecting}
                                    onConnect={() => handleAddToWorkspace(app)}
                                    onDisconnect={() => handleDisconnect(app.name)}
                                    onDetails={() => handleOpenDetails(app)}
                                />
                            )
                        })}
                    </AnimatePresence>
                </div>
            )}

            {/* Pagination Controls */}
            {filteredApps.length > pageSize && (
                <EnhancedPagination
                    data={paginationData}
                    onPageChange={setCurrentPage}
                    className="mt-8"
                />
            )}

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
                    <p className="text-xs text-muted-foreground">Total apps loaded: {apps.length}</p>
                    <Button
                        variant="outline"
                        className="mt-4"
                        onClick={() => {
                            setSelectedCategory('all')
                        }}
                    >
                        Clear Filters
                    </Button>
                </div>
            )}

            {/* Details Modal */}
            <MarketplaceAppDetailsModal
                open={detailsModalOpen}
                onClose={() => setDetailsModalOpen(false)}
                app={selectedApp}
                onConnect={() => selectedApp && handleAddToWorkspace(selectedApp)}
                loading={connectingApp === selectedApp?.name}
                isConnected={selectedApp ? connectedApps.has(selectedApp.name.toUpperCase()) : false}
            />
        </div>
    )
}

// ============================================================================
// ToolCard Component - Exact match from tools-dashboard.tsx
// ============================================================================

interface ToolCardProps {
    app: ComposioApp
    index: number
    isConnected: boolean
    isInWorkspace: boolean
    isConnecting: boolean
    onConnect: () => void
    onDisconnect: () => void
    onDetails: () => void
}

function ToolCard({
    app,
    index,
    isConnected,
    isInWorkspace,
    isConnecting,
    onConnect,
    onDisconnect,
    onDetails
}: ToolCardProps) {
    // Format auth scheme helper
    const formatAuthScheme = (scheme: string) => {
        const normalized = scheme.toLowerCase()
        if (normalized.includes('oauth')) return 'OAuth2'
        if (normalized.includes('bearer')) return 'Bearer Token'
        if (normalized.includes('api_key') || normalized.includes('apikey')) return 'API Key'
        if (normalized.includes('basic')) return 'Basic Auth'
        return scheme.replace(/_/g, ' ').toUpperCase()
    }

    const toolsCount = app.action_count ?? 0
    const triggersCount = app.trigger_count ?? 0
    const authSchemes = Array.isArray(app.auth_schemes) ? app.auth_schemes : []

    return (
        <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: 20 }}
            transition={{ delay: index * 0.1 }}
        >
            <Card
                className="glass-card card-glow hover:border-primary/30 transition-all duration-200 group hover:shadow-lg hover:shadow-primary/20"
            >
                <CardHeader className="pb-3">
                    <div className="flex items-start justify-between">
                        <div className="flex items-center space-x-3">
                            <ToolLogo
                                logo={app.logo_url}
                                name={app.display_name}
                                size={40}
                                fallbackIcon={undefined}
                                showBackground={true}
                            />
                            <div>
                                <h3 className="font-semibold">{app.display_name}</h3>
                                <p className="text-xs text-muted-foreground">Composio</p>
                            </div>
                        </div>
                        <div className="flex items-center space-x-2">
                            {isInWorkspace && (
                                <Badge variant="secondary" className="text-xs bg-secondary/50 border border-border/50">
                                    <CheckCircle className="w-3 h-3 mr-1" />
                                    Added
                                </Badge>
                            )}
                            <DropdownMenu>
                                <DropdownMenuTrigger asChild>
                                    <Button variant="ghost" size="sm" className="h-8 w-8 p-0">
                                        <MoreVertical className="h-4 w-4" />
                                    </Button>
                                </DropdownMenuTrigger>
                                <DropdownMenuContent align="end">
                                    <DropdownMenuItem onClick={onDetails}>
                                        <Eye className="w-4 h-4 mr-2" />
                                        View Details
                                    </DropdownMenuItem>
                                    {!isInWorkspace && (
                                        <DropdownMenuItem onClick={onConnect} disabled={isConnecting}>
                                            <Download className="w-4 h-4 mr-2" />
                                            {isConnecting ? 'Adding...' : 'Add to Workspace'}
                                        </DropdownMenuItem>
                                    )}
                                </DropdownMenuContent>
                            </DropdownMenu>
                        </div>
                    </div>
                </CardHeader>
                <CardContent className="space-y-4">
                    <p className="text-sm text-muted-foreground line-clamp-2 min-h-[40px]">
                        {app.description || 'No description available'}
                    </p>

                    {/* Category Badges - Show first 2 categories */}
                    <div className="flex items-center justify-between text-xs">
                        <div />
                        <div className="flex gap-1">
                            {app.categories?.slice(0, 2)?.map((tag) => (
                                <Badge key={tag} variant="outline" className="text-[10px] h-5">
                                    {tag}
                                </Badge>
                            )) || []}
                        </div>
                    </div>

                    {/* Tools and Triggers Count */}
                    <div className="flex items-center gap-4 text-xs text-muted-foreground">
                        <div className="flex items-center gap-1">
                            <Wrench className="w-3 h-3" />
                            <span>{toolsCount} Tools</span>
                        </div>
                        <div className="flex items-center gap-1">
                            <Zap className="w-3 h-3" />
                            <span>{triggersCount} Triggers</span>
                        </div>
                    </div>

                    {/* Auth Schemes Badges */}
                    {authSchemes.length > 0 && (
                        <div className="flex flex-wrap gap-2">
                            {authSchemes.slice(0, 2).map((scheme: string) => (
                                <Badge
                                    key={scheme}
                                    variant="outline"
                                    className="text-[10px] h-5 border-primary/30 text-primary"
                                >
                                    {formatAuthScheme(scheme)}
                                </Badge>
                            ))}
                        </div>
                    )}

                </CardContent>
            </Card>
        </motion.div>
    )
}
