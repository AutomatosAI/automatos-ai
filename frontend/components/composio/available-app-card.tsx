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
        <Card className="bg-secondary/50 border-border/50 hover:border-border/50 transition-all">
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
                            <div className="h-10 w-10 rounded-lg bg-gradient-to-br from-muted-foreground to-muted flex items-center justify-center text-foreground font-semibold">
                                {app.display_name.charAt(0)}
                            </div>
                        )}
                    </div>

                    {/* App Info */}
                    <div className="flex-1 min-w-0">
                        <h3 className="font-medium text-foreground truncate">
                            {app.display_name}
                        </h3>
                        {app.description && (
                            <p className="text-xs text-muted-foreground mt-1 line-clamp-2">
                                {app.description}
                            </p>
                        )}
                        {app.categories.length > 0 && (
                            <div className="flex flex-wrap gap-1 mt-2">
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
