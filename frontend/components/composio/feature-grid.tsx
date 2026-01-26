/**
 * Feature Grid Component (PRD-36)
 * ================================
 * 
 * Grid layout for app feature cards with search filtering.
 */

'use client'

import { useState, useMemo } from 'react'
import { Search } from 'lucide-react'
import { Input } from '@/components/ui/input'
import { FeatureCard } from './feature-card'

interface Feature {
    name: string
    display_name: string
    description?: string
    enabled: boolean
}

interface FeatureGridProps {
    features: Feature[]
    onToggle: (actionName: string, enabled: boolean) => void
    loading?: boolean
}

export function FeatureGrid({ features, onToggle, loading }: FeatureGridProps) {
    const [searchQuery, setSearchQuery] = useState('')

    const filteredFeatures = useMemo(() => {
        if (!searchQuery.trim()) return features

        const query = searchQuery.toLowerCase()
        return features.filter(
            (f) =>
                f.display_name.toLowerCase().includes(query) ||
                f.name.toLowerCase().includes(query) ||
                f.description?.toLowerCase().includes(query)
        )
    }, [features, searchQuery])

    const enabledCount = features.filter((f) => f.enabled).length

    return (
        <div className="space-y-4">
            {/* Search and stats */}
            <div className="flex items-center gap-4">
                <div className="relative flex-1">
                    <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-slate-400" />
                    <Input
                        placeholder="Search features..."
                        value={searchQuery}
                        onChange={(e) => setSearchQuery(e.target.value)}
                        className="pl-10 bg-slate-800/50 border-slate-700 focus:border-blue-500"
                    />
                </div>
                <div className="text-sm text-slate-400 whitespace-nowrap">
                    <span className="text-blue-400 font-medium">{enabledCount}</span>
                    <span className="mx-1">/</span>
                    <span>{features.length}</span>
                    <span className="ml-1">enabled</span>
                </div>
            </div>

            {/* Grid */}
            {loading ? (
                <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                    {[...Array(6)].map((_, i) => (
                        <div
                            key={i}
                            className="h-20 rounded-xl bg-slate-800/50 animate-pulse"
                        />
                    ))}
                </div>
            ) : filteredFeatures.length === 0 ? (
                <div className="text-center py-8 text-slate-400">
                    {searchQuery ? (
                        <p>No features match "{searchQuery}"</p>
                    ) : (
                        <p>No features available</p>
                    )}
                </div>
            ) : (
                <div className="grid grid-cols-1 md:grid-cols-2 gap-3 max-h-[400px] overflow-y-auto pr-2 scrollbar-thin scrollbar-thumb-slate-700 scrollbar-track-transparent">
                    {filteredFeatures.map((feature) => (
                        <FeatureCard
                            key={feature.name}
                            feature={feature}
                            onToggle={onToggle}
                            disabled={loading}
                        />
                    ))}
                </div>
            )}
        </div>
    )
}
