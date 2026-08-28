'use client'

import { useState, useEffect } from 'react'
import { Package, Sparkles, Bot, BookOpen, Brain, Wrench, Puzzle, Cpu } from 'lucide-react'
import { Card, CardContent, CardHeader } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { apiClient } from '@/lib/api-client'
import {
  MarketplacePackage,
  MEMBER_TYPE_ORDER,
  MEMBER_TYPE_LABELS,
} from './package-types'
import { PackageDetailModal } from './package-detail-modal'

interface MarketplacePackagesTabProps {
  searchQuery?: string
  /** Tests inject packages directly to skip the fetch; prod fetches the API. */
  packages?: MarketplacePackage[]
}

const TYPE_ICONS: Record<string, any> = {
  agent: Bot,
  playbook: BookOpen,
  skill: Brain,
  tool: Wrench,
  plugin: Puzzle,
  llm: Cpu,
}

// D11 (Gerard, 2026-08-28): the showcase row copy is "Starter teams for your business".
const SHOWCASE_ROW_COPY = 'Starter teams for your business'

function memberCounts(pkg: MarketplacePackage): Array<{ type: string; count: number }> {
  const counts = new Map<string, number>()
  for (const m of pkg.members || []) {
    counts.set(m.type, (counts.get(m.type) || 0) + 1)
  }
  const ordered: Array<{ type: string; count: number }> = []
  for (const t of MEMBER_TYPE_ORDER) {
    if (counts.has(t)) {
      ordered.push({ type: t, count: counts.get(t)! })
      counts.delete(t)
    }
  }
  for (const [t, c] of counts) ordered.push({ type: t, count: c })
  return ordered
}

function PackageCard({
  pkg,
  onClick,
  featured,
}: {
  pkg: MarketplacePackage
  onClick: () => void
  featured?: boolean
}) {
  const counts = memberCounts(pkg)
  return (
    <Card
      data-testid="package-card"
      className={`glass-card card-glow hover:border-primary/20 transition-all cursor-pointer ${
        featured ? 'border-primary/30' : ''
      }`}
      onClick={onClick}
    >
      <CardHeader className="pb-3">
        <div className="flex items-start gap-3">
          <Package className="w-8 h-8 text-primary shrink-0" />
          <div className="flex-1 min-w-0">
            <div className="flex items-center gap-2">
              <h3 className="font-semibold text-foreground line-clamp-1">{pkg.name}</h3>
              {pkg.showcase && (
                <Badge
                  variant="outline"
                  className="text-[10px] border-primary/30 text-primary shrink-0"
                >
                  Starter
                </Badge>
              )}
            </div>
            <p className="text-xs text-muted-foreground line-clamp-2 mt-1">
              {pkg.description || 'No description available'}
            </p>
          </div>
        </div>
      </CardHeader>
      <CardContent className="space-y-3">
        {/* Member composition — grouped counts by type */}
        <div className="flex flex-wrap items-center gap-3 text-xs text-muted-foreground">
          {counts.map(({ type, count }) => {
            const Icon = TYPE_ICONS[type] || Puzzle
            const label = MEMBER_TYPE_LABELS[type]
            return (
              <span key={type} className="flex items-center gap-1">
                <Icon className="w-3.5 h-3.5" />
                {count} {label ? (count === 1 ? label.singular : label.plural) : type}
              </span>
            )
          })}
        </div>
        {/* Vertical tags */}
        {(pkg.vertical_tags || []).length > 0 && (
          <div className="flex flex-wrap gap-1.5">
            {pkg.vertical_tags.slice(0, 4).map((tag) => (
              <Badge
                key={tag}
                variant="outline"
                className="text-[10px] border-border text-muted-foreground"
              >
                {tag}
              </Badge>
            ))}
          </div>
        )}
      </CardContent>
    </Card>
  )
}

export function MarketplacePackagesTab({ searchQuery, packages }: MarketplacePackagesTabProps) {
  const [items, setItems] = useState<MarketplacePackage[]>(packages || [])
  const [loading, setLoading] = useState(!packages)
  const [selected, setSelected] = useState<MarketplacePackage | null>(null)

  useEffect(() => {
    if (packages) {
      setItems(packages)
      setLoading(false)
      return
    }
    let active = true
    setLoading(true)
    apiClient
      .get('/api/marketplace/packages')
      .then((data: MarketplacePackage[]) => {
        if (active) setItems(Array.isArray(data) ? data : [])
      })
      .catch((error) => {
        console.error('Failed to fetch packages:', error)
      })
      .finally(() => {
        if (active) setLoading(false)
      })
    return () => {
      active = false
    }
  }, [packages])

  const q = (searchQuery || '').trim().toLowerCase()
  const filtered = q
    ? items.filter(
        (p) =>
          p.name.toLowerCase().includes(q) ||
          (p.description || '').toLowerCase().includes(q) ||
          (p.vertical_tags || []).some((t) => t.toLowerCase().includes(q)),
      )
    : items

  // Showcase row first (showcase=true), then the full grid.
  const showcased = filtered.filter((p) => p.showcase)

  if (loading) {
    return (
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {[...Array(6)].map((_, i) => (
          <div key={i} className="h-40 glass-card animate-pulse rounded-xl" />
        ))}
      </div>
    )
  }

  if (filtered.length === 0) {
    return (
      <div className="text-center py-12 text-muted-foreground" data-testid="packages-empty">
        <Package className="w-12 h-12 mx-auto mb-4 opacity-50" />
        <p>No packages found. Curated starter teams will appear here.</p>
      </div>
    )
  }

  return (
    <div className="space-y-8">
      {/* Showcase row — starter teams on top (D11 copy) */}
      {showcased.length > 0 && (
        <div data-testid="packages-showcase-row">
          <div className="flex items-center gap-2 mb-3">
            <Sparkles className="h-5 w-5 text-primary" />
            <h2 className="text-lg font-semibold">{SHOWCASE_ROW_COPY}</h2>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {showcased.map((pkg) => (
              <PackageCard key={pkg.slug} pkg={pkg} featured onClick={() => setSelected(pkg)} />
            ))}
          </div>
        </div>
      )}

      {/* Full catalog */}
      <div>
        {showcased.length > 0 && (
          <h2 className="text-lg font-semibold mb-3">All packages</h2>
        )}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {filtered.map((pkg) => (
            <PackageCard key={pkg.slug} pkg={pkg} onClick={() => setSelected(pkg)} />
          ))}
        </div>
      </div>

      {selected && (
        <PackageDetailModal pkg={selected} onClose={() => setSelected(null)} />
      )}
    </div>
  )
}
