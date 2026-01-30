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

  return (
    <Card
      className="cursor-pointer hover:shadow-lg transition-shadow duration-200"
      onClick={onClick}
    >
      <CardHeader className="pb-3">
        <div className="flex items-start justify-between">
          <div className="flex items-center gap-3">
            {item.icon && (
              <div className="text-3xl">{item.icon}</div>
            )}
            <div>
              <h3 className="font-semibold text-lg line-clamp-1">{item.name}</h3>
              <p className="text-sm text-gray-500">{item.creator_name}</p>
            </div>
          </div>
          {item.is_featured && (
            <Badge variant="secondary" className="flex items-center gap-1">
              <Star className="h-3 w-3" />
              Featured
            </Badge>
          )}
        </div>
      </CardHeader>

      <CardContent>
        <p className="text-sm text-gray-600 line-clamp-2 mb-3">
          {item.description}
        </p>

        {item.category && (
          <Badge variant="outline" className="text-xs">
            {item.category}
          </Badge>
        )}
      </CardContent>

      <CardFooter className="pt-3 border-t flex items-center justify-between text-sm text-gray-500">
        <div className="flex items-center gap-1">
          <Download className="h-4 w-4" />
          {formatInstallCount(item.install_count)} installs
        </div>
        <Badge variant="outline" className="text-xs">
          v{item.version}
        </Badge>
      </CardFooter>
    </Card>
  )
}
