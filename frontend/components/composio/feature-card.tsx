/**
 * Feature Card Component (PRD-36)
 * ================================
 * 
 * Individual feature toggle card for Composio app actions.
 * Shows action name, description, and toggle switch.
 */

'use client'

import { Switch } from '@/components/ui/switch'
import { cn } from '@/lib/utils'

interface FeatureCardProps {
    feature: {
        name: string
        display_name: string
        description?: string
        enabled: boolean
    }
    onToggle: (actionName: string, enabled: boolean) => void
    disabled?: boolean
}

export function FeatureCard({ feature, onToggle, disabled }: FeatureCardProps) {
    return (
        <div
            className={cn(
                "group relative p-4 rounded-xl border transition-all duration-220 cursor-pointer",
                "bg-secondary/50 border-border/50",
                "hover:bg-secondary/50 hover:border-border/50",
                feature.enabled && "border-blue-500/50 bg-blue-500/10",
                disabled && "opacity-50 cursor-not-allowed"
            )}
            onClick={() => !disabled && onToggle(feature.name, !feature.enabled)}
        >
            <div className="flex justify-between items-start gap-3">
                <div className="flex-1 min-w-0">
                    <h4 className="font-medium text-sm text-foreground truncate">
                        {feature.display_name}
                    </h4>
                    {feature.description && (
                        <p className="text-xs text-muted-foreground mt-1 line-clamp-2">
                            {feature.description}
                        </p>
                    )}
                </div>
                <Switch
                    checked={feature.enabled}
                    onCheckedChange={(checked) => !disabled && onToggle(feature.name, checked)}
                    disabled={disabled}
                    className="shrink-0"
                />
            </div>

            {/* Visual indicator */}
            <div
                className={cn(
                    "absolute left-0 top-0 bottom-0 w-1 rounded-l-xl transition-colors",
                    feature.enabled ? "bg-blue-500" : "bg-transparent"
                )}
            />
        </div>
    )
}
