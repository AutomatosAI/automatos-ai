'use client'

/**
 * PRD-261 C-03b — the admin review queue for Marketplace theme versions (`/admin/themes`).
 *
 * A person reviews every version (D3, 19 Oct). The list of pending versions sits beside the chosen
 * version's report, with Approve (disabled for a blocked or not-run scan, and until the server's
 * `approval` allows it) and Reject with a reason; after approval, the signature hand-off and
 * take-down.
 *
 * Both editions: the local operator is a super admin (`hybrid.py:642`, `contexts/role-context.tsx`),
 * so the queue works on their machine with no isolated-job note (`hosted` false). On the hosted
 * edition a person whose system role is not admin sees the admin-only notice and no request is
 * made; the server's `caller_is_admin` is still the rule, and its 403 sentence is shown as is.
 */
import { useState } from 'react'

import { Button } from '@/components/ui/button'
import { useSystemRoleOptional } from '@/contexts/role-context'
import { isSaaS } from '@/lib/auth-edition'
import { themeErrorMessage } from '@/lib/theme-host/api'

import { PendingList } from './pending-list'
import { ReviewReport } from './review-report'
import { ADMIN_ONLY, PICK_A_VERSION } from './review-words'
import { usePendingThemes } from './use-theme-review'

export const REFRESH = 'Refresh'

function QueueBody({ hosted }: { hosted: boolean }) {
  const pending = usePendingThemes()
  const [selected, setSelected] = useState<string | null>(null)
  if (pending.isLoading) return <p className="text-sm text-muted-foreground">Loading the queue…</p>
  if (pending.isError || !pending.data) {
    return (
      <p role="alert" className="text-sm text-destructive">
        {themeErrorMessage(pending.error)}
      </p>
    )
  }
  return (
    <div className="grid gap-6 lg:grid-cols-[minmax(16rem,22rem)_1fr]">
      <div className="space-y-2">
        <Button variant="outline" size="sm" onClick={() => void pending.refetch()} disabled={pending.isFetching}>
          {REFRESH}
        </Button>
        <PendingList pending={pending.data.pending} selected={selected} onSelect={setSelected} />
      </div>
      {selected ? (
        <ReviewReport packageId={selected} hosted={hosted} />
      ) : (
        <p className="text-sm text-muted-foreground">{PICK_A_VERSION}</p>
      )}
    </div>
  )
}

export function ThemeReviewQueue({ hosted = isSaaS }: { hosted?: boolean }) {
  const role = useSystemRoleOptional()
  if (role?.isLoading) return <p className="text-sm text-muted-foreground">Loading…</p>
  if (role && !role.isAdmin) {
    return (
      <p className="text-sm text-muted-foreground" data-testid="theme-queue-admin-only">
        {ADMIN_ONLY}
      </p>
    )
  }
  return <QueueBody hosted={hosted} />
}
