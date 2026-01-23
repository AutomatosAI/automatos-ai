/**
 * Available App Card Component (PRD-36)
 * ======================================
 * 
 * Card displaying an available Composio app with Connect button.
 */

'use client'

import { Card, CardContent } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { AppConnectionButton } from './app-connection-button'

interface AvailableAppCardProps {
    app: {
        name: string
        display_name: string
        description?: string
        logo_url?: string
        categories: string[]
        is_connected: boolean
    }
    onConnected?: () => void
}

export function AvailableAppCard({ app, onConnected }: AvailableAppCardProps) {
    return (
        <Card className="bg-slate-800/50 border-slate-700/50 hover:border-slate-600/50 transition-all">
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
                            <div className="h-10 w-10 rounded-lg bg-gradient-to-br from-slate-600 to-slate-700 flex items-center justify-center text-slate-300 font-semibold">
                                {app.display_name.charAt(0)}
                            </div>
                        )}
                    </div>

                    {/* App Info */}
                    <div className="flex-1 min-w-0">
                        <h3 className="font-medium text-slate-100 truncate">
                            {app.display_name}
                        </h3>
                        {app.description && (
                            <p className="text-xs text-slate-400 mt-1 line-clamp-2">
                                {app.description}
                            </p>
                        )}
                        {app.categories.length > 0 && (
                            <div className="flex flex-wrap gap-1 mt-2">
                                {app.categories.slice(0, 2).map((cat) => (
                                    <Badge
                                        key={cat}
                                        variant="outline"
                                        className="text-xs border-slate-600 text-slate-400"
                                    >
                                        {cat}
                                    </Badge>
                                ))}
                            </div>
                        )}
                    </div>

                    {/* Connect Button */}
                    <div className="shrink-0">
                        <AppConnectionButton
                            appName={app.name}
                            displayName={app.display_name}
                            isConnected={app.is_connected}
                            onConnected={onConnected}
                        />
                    </div>
                </div>
            </CardContent>
        </Card>
    )
}
