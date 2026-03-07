'use client'

import { Star, Download, Puzzle } from 'lucide-react'
import { Card, CardHeader, CardContent, CardFooter } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'

export interface MarketplaceWidgetSummary {
  id: string
  name: string
  display_name: string
  description: string
  developer_name: string
  icon_url: string | null
  categories: string[]
  pricing_type: 'free' | 'one_time' | 'subscription'
  price_cents: number | null
  install_count: number
  rating_average: number
  rating_count: number
  version: string
}

interface WidgetCardProps {
  widget: MarketplaceWidgetSummary
  onClick: () => void
}

function formatInstallCount(count: number): string {
  if (count >= 1_000_000) return `${(count / 1_000_000).toFixed(1)}M`
  if (count >= 1_000) return `${(count / 1_000).toFixed(1)}k`
  return count.toString()
}

function formatPrice(pricingType: string, priceCents: number | null): string {
  if (pricingType === 'free') return 'Free'
  const dollars = ((priceCents ?? 0) / 100).toFixed(2)
  if (pricingType === 'subscription') return `$${dollars}/mo`
  return `$${dollars}`
}

function RatingStars({ average, count }: { average: number; count: number }) {
  return (
    <div className="flex items-center gap-1">
      <div className="flex items-center">
        {[1, 2, 3, 4, 5].map((star) => (
          <Star
            key={star}
            className={`h-3.5 w-3.5 ${
              star <= Math.round(average)
                ? 'fill-yellow-400 text-yellow-400'
                : 'text-muted-foreground/40'
            }`}
          />
        ))}
      </div>
      <span className="text-xs text-muted-foreground">({count})</span>
    </div>
  )
}

export function WidgetCard({ widget, onClick }: WidgetCardProps) {
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
        <div className="flex items-start gap-3">
          <div className="flex-shrink-0 h-10 w-10 rounded-lg bg-secondary flex items-center justify-center overflow-hidden">
            {widget.icon_url ? (
              <img
                src={widget.icon_url}
                alt={widget.display_name}
                className="h-10 w-10 rounded-lg object-cover"
              />
            ) : (
              <Puzzle className="h-5 w-5 text-muted-foreground" />
            )}
          </div>
          <div className="flex-1 min-w-0">
            <h3 className="font-semibold text-lg text-foreground line-clamp-1">
              {widget.display_name}
            </h3>
            <p className="text-sm text-muted-foreground">{widget.developer_name}</p>
          </div>
        </div>
      </CardHeader>

      <CardContent>
        <p className="text-sm text-muted-foreground line-clamp-2 mb-3">
          {widget.description}
        </p>

        <div className="flex items-center gap-2 flex-wrap">
          {widget.categories.length > 0 && (
            <Badge variant="outline" className="text-xs border-border text-muted-foreground">
              {widget.categories[0]}
            </Badge>
          )}
          <Badge
            className={`text-xs ${
              widget.pricing_type === 'free'
                ? 'bg-[hsl(var(--success))]/20 text-[hsl(var(--success))] border-[hsl(var(--success))]/30'
                : 'bg-primary/20 text-primary border-primary/30'
            }`}
          >
            {formatPrice(widget.pricing_type, widget.price_cents)}
          </Badge>
        </div>
      </CardContent>

      <CardFooter className="pt-3 border-t border-border flex items-center justify-between text-sm text-muted-foreground">
        <RatingStars average={widget.rating_average} count={widget.rating_count} />
        <div className="flex items-center gap-1">
          <Download className="h-4 w-4 text-[hsl(var(--success))]" />
          <span>{formatInstallCount(widget.install_count)}</span>
        </div>
      </CardFooter>
    </Card>
  )
}
