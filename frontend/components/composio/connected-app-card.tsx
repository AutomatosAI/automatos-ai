/**
 * Connected App Card Component (PRD-36)
 * ======================================
 * 
 * Card displaying a connected Composio app with Manage and Disconnect buttons.
 */

'use client'

import { Settings, Unplug, ExternalLink } from 'lucide-react'
import { Card, CardContent } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { cn } from '@/lib/utils'

interface ConnectedAppCardProps {
    app: {
        name: string
        display_name: string
        logo_url?: string
        status: string
        connected_at?: string
        action_count?: number
    }
    onManage: () => void
    onDisconnect: () => void
}

export function ConnectedAppCard({ app, onManage, onDisconnect }: ConnectedAppCardProps) {
    const isActive = app.status === 'active'

    return (
        <Card className="bg-slate-800/50 border-slate-700/50 hover:border-slate-600/50 transition-all group">
            <CardContent className="p-4">
                <div className="flex items-start gap-3">
                    {/* App Icon */}
                    <div className="shrink-0">
                        {app.logo_url ? (
                            <img
                                src={app.logo_url}
                                alt={app.display_name}
                                className="h-10 w-10 rounded-lg"
                            />
                        ) : (
                            <div className="h-10 w-10 rounded-lg bg-gradient-to-br from-blue-500 to-purple-600 flex items-center justify-center text-white font-semibold">
                                {app.display_name.charAt(0)}
                            </div>
                        )}
                    </div>

                    {/* App Info */}
                    <div className="flex-1 min-w-0">
                        <div className="flex items-center gap-2">
                            <h3 className="font-medium text-slate-100 truncate">
                                {app.display_name}
                            </h3>
                            <Badge
                                variant="outline"
                                className={cn(
                                    "text-xs",
                                    isActive
                                        ? "border-green-500/50 text-green-400 bg-green-500/10"
                                        : "border-yellow-500/50 text-yellow-400 bg-yellow-500/10"
                                )}
                            >
                                {isActive ? 'Connected' : app.status}
                            </Badge>
                        </div>
                        {app.action_count !== undefined && (
                            <p className="text-xs text-slate-400 mt-1">
                                {app.action_count} features available
                            </p>
                        )}
                    </div>

                    {/* Actions */}
                    <div className="flex items-center gap-1 opacity-0 group-hover:opacity-100 transition-opacity">
                        <Button
                            variant="ghost"
                            size="icon"
                            onClick={onManage}
                            className="h-8 w-8 text-slate-400 hover:text-blue-400 hover:bg-blue-500/10"
                            title="Manage features"
                        >
                            <Settings className="h-4 w-4" />
                        </Button>
                        <Button
                            variant="ghost"
                            size="icon"
                            onClick={onDisconnect}
                            className="h-8 w-8 text-slate-400 hover:text-red-400 hover:bg-red-500/10"
                            title="Disconnect"
                        >
                            <Unplug className="h-4 w-4" />
                        </Button>
                    </div>
                </div>
            </CardContent>
        </Card>
    )
}
