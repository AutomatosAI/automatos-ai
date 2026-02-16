'use client'
import { Grid3X3, List } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { cn } from '@/lib/utils'

export type ViewMode = 'grid' | 'list'

interface ViewToggleProps {
  value: ViewMode
  onChange: (mode: ViewMode) => void
  className?: string
}

export function ViewToggle({ value, onChange, className }: ViewToggleProps) {
  return (
    <div className={cn('flex items-center space-x-1 bg-secondary/30 rounded-lg p-1', className)}>
      <Button
        variant={value === 'grid' ? 'default' : 'ghost'}
        size="sm"
        onClick={() => onChange('grid')}
        className="h-9 w-9 md:h-7 md:w-7 p-0"
        aria-label="Grid view"
      >
        <Grid3X3 className="w-4 h-4" />
      </Button>
      <Button
        variant={value === 'list' ? 'default' : 'ghost'}
        size="sm"
        onClick={() => onChange('list')}
        className="h-9 w-9 md:h-7 md:w-7 p-0"
        aria-label="List view"
      >
        <List className="w-4 h-4" />
      </Button>
    </div>
  )
}
