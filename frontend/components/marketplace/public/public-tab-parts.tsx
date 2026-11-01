'use client'

/**
 * One Marketplace page for everyone (PRD-270 §5.7) — the pieces the signed-out visitor's
 * Applications, LLMs, Plugins and Skills tabs share: their read of the tab's public items, its
 * loading and refusal lines, and the member tabs' filter pills, drawn as the member tabs draw them.
 */
import { useQuery } from '@tanstack/react-query'

import { Button } from '@/components/ui/button'
import type { PublicMarketplaceType } from '@/lib/marketplace/public-items'
import type { PublicEntry } from '@/lib/marketplace/source/public-reads'
import { themeErrorMessage } from '@/lib/theme-host/api'

import { publicQueries } from '../access/public-source'
import { LOADING_ITEMS } from './copy'

export const ALL = 'all'
const FILTER_ACTIVE = 'bg-secondary border-primary/50 text-foreground font-semibold'
const FILTER_IDLE = 'border-secondary text-muted-foreground hover:bg-secondary'
const NONE: readonly PublicEntry[] = []

export interface FilterOption {
  readonly id: string
  readonly name: string
  readonly count: number
}

/** A tab's public items, read once for the page (React Query, five minutes). */
export function usePublicEntries(type: PublicMarketplaceType) {
  const read = useQuery(publicQueries.entries(type))
  return { read, entries: read.data ?? NONE }
}

/** The tab's loading line or the server's own refusal; null once its items are in. */
export function PublicReadState({ read }: { read: ReturnType<typeof usePublicEntries>['read'] }) {
  if (read.isLoading) return <p className="text-sm text-muted-foreground">{LOADING_ITEMS}</p>
  if (read.isError) {
    return (
      <p className="text-sm text-destructive" role="alert">
        {themeErrorMessage(read.error)}
      </p>
    )
  }
  return null
}

/** Each value's count, most first, then by name; at most `limit`. */
export function countedOptions(values: readonly (string | null | undefined)[], limit: number): FilterOption[] {
  const counts = new Map<string, number>()
  for (const value of values) {
    if (value) counts.set(value, (counts.get(value) ?? 0) + 1)
  }
  return Array.from(counts, ([id, count]) => ({ id, name: id, count }))
    .sort((a, b) => b.count - a.count || a.name.localeCompare(b.name))
    .slice(0, limit)
}

interface FilterPillsProps {
  readonly allLabel: string
  readonly options: readonly FilterOption[]
  readonly selected: string
  readonly onSelect: (id: string) => void
}

/** "All …" then one pill per option with its count, as the member tabs' category filters. */
export function FilterPills({ allLabel, options, selected, onSelect }: FilterPillsProps) {
  const pill = (id: string, name: string, count?: number) => (
    <Button
      key={id}
      variant={selected === id ? 'default' : 'outline'}
      size="sm"
      onClick={() => onSelect(id)}
      className={`whitespace-nowrap flex-shrink-0 ${selected === id ? FILTER_ACTIVE : FILTER_IDLE}`}
    >
      {name}
      {count !== undefined && <span className="ml-1.5 text-[10px] opacity-60">{count}</span>}
    </Button>
  )
  return (
    <div className="flex gap-2 overflow-x-auto pb-1 scrollbar-thin scrollbar-thumb-secondary scrollbar-track-transparent">
      {pill(ALL, allLabel)}
      {options.map((option) => pill(option.id, option.name, option.count))}
    </div>
  )
}
