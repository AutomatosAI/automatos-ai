'use client'

import { Puzzle } from 'lucide-react'
import { PremiumIcon } from '@/components/shared'
import { useSystemIcons } from '@/hooks/use-system-config-api'
import { WidgetCard } from './WidgetCard'
import type { MarketplaceWidgetSummary } from './WidgetCard'

interface WidgetGridProps {
  widgets: MarketplaceWidgetSummary[]
  onWidgetClick: (widget: MarketplaceWidgetSummary) => void
  emptyMessage?: string
}

export function WidgetGrid({
  widgets,
  onWidgetClick,
  emptyMessage = 'No widgets found',
}: WidgetGridProps) {
  const { data: iconMappings = {} } = useSystemIcons()
  const globalPluginIcon = iconMappings['global_plugin'] || null

  if (widgets.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center py-16 text-muted-foreground">
        {globalPluginIcon ? (
          <div className="mb-4 opacity-40"><PremiumIcon name={globalPluginIcon} size={48} /></div>
        ) : (
          <Puzzle className="h-12 w-12 mb-4 opacity-40" />
        )}
        <p className="text-lg">{emptyMessage}</p>
      </div>
    )
  }

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
      {widgets.map((widget) => (
        <WidgetCard
          key={widget.id}
          widget={widget}
          onClick={() => onWidgetClick(widget)}
        />
      ))}
    </div>
  )
}
