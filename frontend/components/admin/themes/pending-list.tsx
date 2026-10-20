/**
 * PRD-261 C-03b — the versions waiting for review (`GET /api/admin/themes/pending`), oldest first:
 * what each is, its scan verdict and whether approve would pass now (the server's `approval`,
 * with a blocked or not-run scan never ready).
 */
import { Badge } from '@/components/ui/badge'
import type { ThemePendingVersion } from '@/lib/api-client'
import { cn } from '@/lib/utils'

import { QUEUE_EMPTY, VERDICT_LABELS, approveRefusal, labelOf } from './review-words'

export const READY = 'Ready to approve'
export const NOT_READY = 'Not ready'

interface PendingListProps {
  readonly pending: readonly ThemePendingVersion[]
  readonly selected: string | null
  readonly onSelect: (packageId: string) => void
}

function PendingRow({ row, active, onSelect }: { row: ThemePendingVersion; active: boolean; onSelect: () => void }) {
  const ready = approveRefusal(row.scanVerdict, row.approval) === null
  return (
    <li>
      <button
        type="button"
        onClick={onSelect}
        aria-pressed={active}
        data-testid={`pending-theme-${row.id}`}
        className={cn(
          'w-full rounded-lg border px-3 py-2 text-left text-sm transition-colors',
          active ? 'border-primary bg-primary/5' : 'border-border/50 hover:bg-muted/50',
        )}
      >
        <span className="block font-medium">{row.name || row.themeId}</span>
        <span className="block text-xs text-muted-foreground">
          {row.themeId} · {row.version} · {row.publisher}
        </span>
        <span className="mt-1 flex flex-wrap gap-1">
          <Badge variant="outline">{labelOf(VERDICT_LABELS, row.scanVerdict)}</Badge>
          <Badge variant={ready ? 'default' : 'secondary'}>{ready ? READY : NOT_READY}</Badge>
        </span>
      </button>
    </li>
  )
}

export function PendingList({ pending, selected, onSelect }: PendingListProps) {
  if (pending.length === 0) {
    return <p className="text-sm text-muted-foreground" data-testid="theme-queue-empty">{QUEUE_EMPTY}</p>
  }
  return (
    <ul aria-label="Pending theme versions" className="space-y-2">
      {pending.map((row) => (
        <PendingRow key={row.id} row={row} active={row.id === selected} onSelect={() => onSelect(row.id)} />
      ))}
    </ul>
  )
}
