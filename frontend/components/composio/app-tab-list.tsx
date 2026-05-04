/**
 * App Tab List Component (PRD-36)
 * ================================
 * 
 * Horizontal tabs for connected apps showing enabled feature count.
 */

'use client'

import { cn } from '@/lib/utils'

interface AppTab {
    name: string
    display_name: string
    logo_url?: string
    enabled_count: number
    total_count: number
}

interface AppTabListProps {
    apps: AppTab[]
    selectedApp: string | null
    onSelectApp: (appName: string) => void
}

export function AppTabList({ apps, selectedApp, onSelectApp }: AppTabListProps) {
    if (apps.length === 0) {
        return (
            <div className="text-center py-4 text-slate-400">
                No apps connected. Connect an app to manage features.
            </div>
        )
    }

    return (
        <div className="flex gap-2 overflow-x-auto pb-2 scrollbar-thin scrollbar-thumb-slate-700 scrollbar-track-transparent">
            {apps.map((app) => (
                <button
                    key={app.name}
                    onClick={() => onSelectApp(app.name)}
                    className={cn(
                        "flex items-center gap-2 px-4 py-2 rounded-lg transition-all shrink-0",
                        "border text-sm font-medium",
                        selectedApp === app.name
                            ? "bg-info/20 border-info/50 text-info/80"
                            : "bg-slate-800/50 border-slate-700/50 text-slate-300 hover:bg-slate-700/50 hover:border-slate-600/50"
                    )}
                >
                    {app.logo_url ? (
                        <img
                            src={app.logo_url}
                            alt={app.display_name}
                            className="h-5 w-5 rounded"
                        />
                    ) : (
                        <div className="h-5 w-5 rounded bg-slate-600 flex items-center justify-center text-xs">
                            {app.display_name.charAt(0)}
                        </div>
                    )}
                    <span>{app.display_name}</span>
                    <span className="text-xs text-slate-400">
                        {app.enabled_count}/{app.total_count}
                    </span>
                </button>
            ))}
        </div>
    )
}
