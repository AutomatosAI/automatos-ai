'use client'

/**
 * F091-C2 — the owner said no while a request of Auto's still waits on a yes.
 * Night 3: a "no" in chat never reached Auto's delete card, which sat pending
 * for its whole 24 h. The reply carries this card: Withdraw denies the grant,
 * Keep it leaves the request as it is. Nothing is withdrawn without the click.
 */

import { useState } from 'react'
import { toast } from 'sonner'
import { Button } from '@/components/ui/button'
import { useDenyApproval } from '@/hooks/use-approval-grants'
import type { WithdrawOffer } from '@/types/chat'

export function WithdrawOfferCard({ offer }: { offer: WithdrawOffer }) {
  const deny = useDenyApproval()
  const [settled, setSettled] = useState<Record<number, 'withdrawn' | 'kept'>>({})

  const withdraw = async (grantId: number) => {
    try {
      await deny.mutateAsync(grantId)
      setSettled((s) => ({ ...s, [grantId]: 'withdrawn' }))
      toast.success('Request withdrawn')
    } catch {
      toast.error('Could not withdraw the request')
    }
  }

  return (
    <div className="rounded border border-border bg-background/50 p-3 space-y-2" role="group" aria-label="Withdraw request">
      <p className="text-xs text-muted-foreground">{offer.message}</p>
      {offer.requests.map((r) => (
        <div key={r.grant_id} className="flex flex-col gap-1.5">
          <p className="text-sm">{r.reason || r.action || `Request #${r.grant_id}`}</p>
          {settled[r.grant_id] ? (
            <p className="text-xs text-muted-foreground">
              {settled[r.grant_id] === 'withdrawn' ? 'Withdrawn.' : 'Kept — it still waits for your yes.'}
            </p>
          ) : (
            <div className="flex gap-2">
              <Button size="sm" disabled={deny.isLoading} onClick={() => withdraw(r.grant_id)}>
                Withdraw
              </Button>
              <Button
                size="sm"
                variant="outline"
                onClick={() => setSettled((s) => ({ ...s, [r.grant_id]: 'kept' }))}
              >
                Keep it
              </Button>
            </div>
          )}
        </div>
      ))}
    </div>
  )
}
