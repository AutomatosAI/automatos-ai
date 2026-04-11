'use client'

import { Check, ShoppingBag, Sparkles } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import type { ScanResponse } from '@/hooks/use-wizard-api'

interface Step4Props {
  scan: ScanResponse
  selected: string[]
  onChange: (urls: string[]) => void
  onBack: () => void
  onIngest: () => void
}

const ARCHETYPE_LABELS: Record<string, { label: string; icon: typeof ShoppingBag }> = {
  shopify_catalog: { label: 'Shopify Store', icon: ShoppingBag },
}

export function Step4PageChecklist({ scan, selected, onChange, onBack, onIngest }: Step4Props) {
  const archetypeMeta = scan.archetype ? ARCHETYPE_LABELS[scan.archetype] : null
  const ArchetypeIcon = archetypeMeta?.icon || Sparkles

  const toggle = (url: string) => {
    if (selected.includes(url)) {
      onChange(selected.filter(u => u !== url))
    } else {
      onChange([...selected, url])
    }
  }

  return (
    <div className="space-y-4">
      {/* Detection summary */}
      <Card className="bg-secondary/30 border-border/30">
        <CardContent className="py-4">
          <div className="flex items-center gap-3">
            <ArchetypeIcon className="w-8 h-8 text-primary" />
            <div className="flex-1">
              <div className="font-medium">
                Detected: <span className="gradient-text">{archetypeMeta?.label || 'Unknown'}</span>
              </div>
              <div className="text-sm text-muted-foreground">
                {scan.total_urls.toLocaleString()} pages discovered · confidence{' '}
                {(scan.confidence * 100).toFixed(0)}%
              </div>
            </div>
            <Badge variant="outline">{scan.matched_signals.length} signals</Badge>
          </div>
        </CardContent>
      </Card>

      {/* Must-have checklist */}
      <Card className="bg-secondary/30 border-border/30">
        <CardHeader>
          <CardTitle className="text-base">Pages we recommend reading</CardTitle>
          <p className="text-sm text-muted-foreground">
            We&apos;ve pre-selected the high-value pages. Untick anything you want to skip, or add more
            from the recommended list below.
          </p>
        </CardHeader>
        <CardContent className="space-y-2">
          {scan.must_have_urls.length === 0 && (
            <p className="text-sm text-muted-foreground italic">
              No must-have pages found. Pick from the recommended list below.
            </p>
          )}

          {scan.must_have_urls.map(url => (
            <UrlRow
              key={url}
              url={url}
              checked={selected.includes(url)}
              onToggle={() => toggle(url)}
              tier="must"
            />
          ))}

          {scan.recommended_urls.length > 0 && (
            <>
              <div className="text-xs uppercase tracking-wide text-muted-foreground pt-3">
                Recommended
              </div>
              {scan.recommended_urls.slice(0, 10).map(url => (
                <UrlRow
                  key={url}
                  url={url}
                  checked={selected.includes(url)}
                  onToggle={() => toggle(url)}
                  tier="recommended"
                />
              ))}
            </>
          )}
        </CardContent>
      </Card>

      <div className="flex justify-between">
        <Button variant="ghost" onClick={onBack}>
          Back
        </Button>
        <Button onClick={onIngest} disabled={selected.length === 0}>
          Read {selected.length} page{selected.length === 1 ? '' : 's'} →
        </Button>
      </div>
    </div>
  )
}

function UrlRow({
  url,
  checked,
  onToggle,
  tier,
}: {
  url: string
  checked: boolean
  onToggle: () => void
  tier: 'must' | 'recommended'
}) {
  return (
    <button
      type="button"
      onClick={onToggle}
      className={`w-full text-left p-3 rounded-md border flex items-center gap-3 transition-all ${
        checked
          ? 'border-primary bg-primary/10'
          : 'border-border/30 bg-secondary/20 hover:border-border/60'
      }`}
    >
      <div
        className={`w-4 h-4 rounded border flex items-center justify-center flex-shrink-0 ${
          checked ? 'bg-primary border-primary' : 'border-border/60'
        }`}
      >
        {checked && <Check className="w-3 h-3 text-primary-foreground" />}
      </div>
      <div className="flex-1 truncate text-sm font-mono">{url}</div>
      {tier === 'must' && (
        <Badge variant="outline" className="text-xs">
          Must
        </Badge>
      )}
    </button>
  )
}
