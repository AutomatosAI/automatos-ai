'use client'

/**
 * PRD-251 S0.5 — the one card the Socials tab shows while this workspace's
 * switch is off. An owner or admin (workspace:manage) turns it on here; anyone
 * else is told to ask one. Turning it on refetches the workspace, so the list
 * replaces this card without a page reload.
 */
import { Loader2, Share2 } from 'lucide-react'

import { Button } from '@/components/ui/button'
import type { Workspace } from '@/components/workspace-provider'
import { useEnableSocials } from '@/hooks/use-socials-api'
import { canTurnOnSocials } from './socials-status'

export function SocialsTurnOnCard({ role }: { role: Workspace['role'] }) {
  const enable = useEnableSocials()

  return (
    <div className="mx-auto flex max-w-xl flex-col items-center gap-3 rounded-xl border border-dashed border-border/60 bg-card/30 px-6 py-10 text-center">
      <Share2 className="h-8 w-8 text-muted-foreground" aria-hidden />
      <h2 className="text-lg font-semibold tracking-tight text-foreground">Socials is off for this workspace</h2>
      <p className="max-w-md text-sm text-muted-foreground">
        On-brand posts for your channels, each one approved before it can go out.
      </p>
      {canTurnOnSocials(role) ? (
        <Button onClick={() => enable.mutate()} disabled={enable.isLoading}>
          {enable.isLoading && <Loader2 className="mr-2 h-4 w-4 animate-spin" aria-hidden />}
          Turn on Socials for this workspace
        </Button>
      ) : (
        <p className="text-sm font-medium text-foreground">Ask an admin to turn on Socials</p>
      )}
    </div>
  )
}
