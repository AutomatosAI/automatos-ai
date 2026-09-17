'use client'

import type { ReactNode } from 'react'
import { Tabs, TabsList, TabsTrigger, TabsContent } from '@/components/ui/tabs'
import { cn } from '@/lib/utils'
import { useIsStudio } from '@/hooks/use-studio-theme'
import type { LucideIcon } from 'lucide-react'

export interface FilterTab {
  value: string
  label: string
  icon?: LucideIcon
  count?: number
}

export interface FilterTabsProps {
  tabs: FilterTab[]
  value: string
  onValueChange: (value: string) => void
  trailing?: ReactNode
  children: ReactNode
  className?: string
}

export function FilterTabs({
  tabs,
  value,
  onValueChange,
  trailing,
  children,
  className,
}: FilterTabsProps) {
  const isStudio = useIsStudio()

  // Studio style (PRD-244 D8): the designed pages' cc-tabs strip. The Radix
  // root stays so the callers' <TabsContent> keeps showing the active panel.
  if (isStudio) {
    return (
      <Tabs value={value} onValueChange={onValueChange} className={cn('space-y-6', className)}>
        <div className="flex items-center gap-4">
          <nav className="cc-tabs" aria-label="Sections" style={{ flex: 1, minWidth: 0 }}>
            {tabs.map((tab) => (
              <button
                key={tab.value}
                type="button"
                className={`cc-tab${tab.value === value ? ' active' : ''}`}
                aria-current={tab.value === value ? 'page' : undefined}
                onClick={() => onValueChange(tab.value)}
              >
                <span>{tab.label}</span>
                {tab.count !== undefined && tab.count > 0 && <span className="cc-tab-ct">{tab.count}</span>}
              </button>
            ))}
          </nav>
          {trailing && <div className="shrink-0">{trailing}</div>}
        </div>
        {children}
      </Tabs>
    )
  }

  return (
    <Tabs value={value} onValueChange={onValueChange} className={cn('space-y-6', className)}>
      <div className="flex items-center gap-4">
        <TabsList className="bg-secondary/50 shrink-0">
          {tabs.map((tab) => (
            <TabsTrigger key={tab.value} value={tab.value} className="flex items-center gap-1.5 min-h-[44px] sm:min-h-0">
              {tab.icon && <tab.icon className="w-4 h-4" />}
              <span className="hidden sm:inline">{tab.label}</span>
              {tab.count !== undefined && (
                <span className="text-[10px] opacity-60">({tab.count})</span>
              )}
            </TabsTrigger>
          ))}
        </TabsList>
        {trailing && <div className="flex-1">{trailing}</div>}
      </div>
      {children}
    </Tabs>
  )
}

// Re-export TabsContent for convenience
export { TabsContent }
