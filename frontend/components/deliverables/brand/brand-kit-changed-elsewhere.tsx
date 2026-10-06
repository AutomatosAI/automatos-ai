'use client'

/**
 * F372 (night 10c) — the Brand kit page when the kit changes elsewhere (Auto, the Brand
 * designer, the API) while it is open. The kit's stamp (useBrandKitStamp, read on focus and on
 * a poll) moving past the one the page loaded means:
 *
 * * no unsaved edits on the page: the fields are read again, quietly;
 * * unsaved edits: a notice offers to reload (dropping them) or to keep them and save over
 *   the change. A save the server refuses (409: the kit moved since the page loaded it)
 *   offers the same.
 *
 * A stamp moved by a change the page stored itself (an upload, a role reset) is its own.
 */
import { useEffect, useState } from 'react'
import { AlertTriangle } from 'lucide-react'

import { Button } from '@/components/ui/button'
import { useBrandKitStamp } from '@/hooks/use-brand-kit-stamp'
import type { BrandKitForm } from './use-brand-kit-form'

export const CHANGED_ELSEWHERE = 'Changed elsewhere — reload, or keep your edits and save over it'

interface KitChangedElsewhereProps {
  form: BrandKitForm
  canEdit: boolean
}

export function KitChangedElsewhere({ form, canEdit }: KitChangedElsewhereProps) {
  const stamp = useBrandKitStamp(form.boardVersion)
  const { loadedStamp, edited, conflict, adopt } = form.sync
  const { reload } = form
  // The page's stamp when a change elsewhere was seen while it held edits.
  const [staleOver, setStaleOver] = useState<string | null>(null)

  // Only a new stamp from the server is news: the page's own save moves `loadedStamp` first.
  useEffect(() => {
    if (stamp === undefined || loadedStamp === undefined || stamp === loadedStamp || adopt(stamp)) return
    if (edited) setStaleOver(loadedStamp)
    else void reload()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [stamp])

  // Hidden once the page reads or saves the kit again (its stamp moves on).
  const stale = staleOver !== null && staleOver === loadedStamp
  if (!conflict && !stale) return null
  return (
    <div role="alert" className="flex flex-wrap items-center gap-3 rounded-md border border-amber-500/50 bg-amber-500/10 p-3 text-sm">
      <AlertTriangle className="h-4 w-4 shrink-0 text-amber-600" aria-hidden />
      <span className="flex-1">{CHANGED_ELSEWHERE}</span>
      <Button type="button" size="sm" variant="outline" onClick={() => void reload()}>Reload</Button>
      {canEdit && (
        <Button type="button" size="sm" disabled={form.saving} onClick={() => void form.saveOver()}>
          Keep my edits and save over it
        </Button>
      )}
    </div>
  )
}
