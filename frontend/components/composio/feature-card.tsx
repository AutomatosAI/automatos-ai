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
                "group relative p-4 rounded-xl border transition-all duration-200 cursor-pointer",
                "bg-slate-800/50 border-slate-700/50",
                "hover:bg-slate-700/50 hover:border-slate-600/50",
                feature.enabled && "border-info/50 bg-info/10",
                disabled && "opacity-50 cursor-not-allowed"
            )}
            onClick={() => !disabled && onToggle(feature.name, !feature.enabled)}
        >
            <div className="flex justify-between items-start gap-3">
                <div className="flex-1 min-w-0">
                    <h4 className="font-medium text-sm text-slate-100 truncate">
                        {feature.display_name}
                    </h4>
                    {feature.description && (
                        <p className="text-xs text-slate-400 mt-1 line-clamp-2">
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
                    feature.enabled ? "bg-info" : "bg-transparent"
                )}
            />
        </div>
    )
}
