'use client'

import { Download, Star } from 'lucide-react'
import { Card, CardHeader, CardContent, CardFooter } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import type { MarketplaceItem } from './marketplace-homepage'

interface MarketplaceCardProps {
  item: MarketplaceItem
  onClick: () => void
}

export function MarketplaceCard({ item, onClick }: MarketplaceCardProps) {
  const formatInstallCount = (count: number) => {
    if (count >= 1000000) return `${(count / 1000000).toFixed(1)}M`
    if (count >= 1000) return `${(count / 1000).toFixed(1)}k`
    return count.toString()
  }

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' || e.key === ' ') {
      e.preventDefault()
      onClick()
    }
  }

  return (
    <Card
      role="button"
      tabIndex={0}
      className="cursor-pointer glass-card card-glow hover:border-primary/20 transition-all duration-300 focus:outline-none focus:ring-2 focus:ring-primary/50"
      onClick={onClick}
      onKeyDown={handleKeyDown}
    >
      <CardHeader className="pb-3">
        <div className="flex items-start justify-between">
          <div className="flex items-center gap-3 flex-1">
            {item.icon && (
              <div className="text-3xl">{item.icon}</div>
            )}
            <div className="flex-1 min-w-0">
              <h3 className="font-semibold text-lg text-foreground line-clamp-1">{item.name}</h3>
              <p className="text-sm text-muted-foreground">{item.creator_name}</p>
            </div>
          </div>
          {item.is_featured && (
            <Badge className="bg-primary/20 text-primary border-primary/30 flex items-center gap-1 shrink-0">
              <Star className="h-3 w-3" />
              Featured
            </Badge>
          )}
        </div>
      </CardHeader>

      <CardContent>
        <p className="text-sm text-muted-foreground line-clamp-2 mb-3">
          {item.description}
        </p>

        {item.category && (
          <Badge variant="outline" className="text-xs border-border text-muted-foreground">
            {item.category}
          </Badge>
        )}
      </CardContent>

      <CardFooter className="pt-3 border-t border-border flex items-center justify-between text-sm text-muted-foreground">
        <div className="flex items-center gap-1">
          <Download className="h-4 w-4 text-[hsl(var(--success))]" />
          <span className="text-muted-foreground">{formatInstallCount(item.install_count)}</span> installs
        </div>
        <Badge variant="outline" className="text-xs border-border text-muted-foreground">
          v{item.version}
        </Badge>
      </CardFooter>
    </Card>
  )
}
