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
      className="cursor-pointer bg-[#1a1a1a] border-gray-800 hover:border-orange-500/50 transition-all duration-200 hover:shadow-lg hover:shadow-orange-500/10"
      onClick={onClick}
    >
      <CardHeader className="pb-3">
        <div className="flex items-start justify-between">
          <div className="flex items-center gap-3 flex-1">
            {item.icon && (
              <div className="text-3xl">{item.icon}</div>
            )}
            <div className="flex-1 min-w-0">
              <h3 className="font-semibold text-lg text-white line-clamp-1">{item.name}</h3>
              <p className="text-sm text-gray-400">{item.creator_name}</p>
            </div>
          </div>
          {item.is_featured && (
            <Badge className="bg-orange-500/20 text-orange-500 border-orange-500/30 flex items-center gap-1 shrink-0">
              <Star className="h-3 w-3" />
              Featured
            </Badge>
          )}
        </div>
      </CardHeader>

      <CardContent>
        <p className="text-sm text-gray-400 line-clamp-2 mb-3">
          {item.description}
        </p>

        {item.category && (
          <Badge variant="outline" className="text-xs border-gray-700 text-gray-300">
            {item.category}
          </Badge>
        )}
      </CardContent>

      <CardFooter className="pt-3 border-t border-gray-800 flex items-center justify-between text-sm text-gray-400">
        <div className="flex items-center gap-1">
          <Download className="h-4 w-4 text-green-500" />
          <span className="text-gray-300">{formatInstallCount(item.install_count)}</span> installs
        </div>
        <Badge variant="outline" className="text-xs border-gray-700 text-gray-400">
          v{item.version}
        </Badge>
      </CardFooter>
    </Card>
  )
}
