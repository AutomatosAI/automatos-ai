'use client'

/**
 * PRD-270 W2 US-270-14b (§5.5, §8-Q3a, K19) — an agent's, Playbook's, package's or plugin's two Automatos
 * Cloud badges: "From Automatos Cloud" (W1's words and badge style, themes/listing-badges.tsx) and
 * "Reviewed on Automatos Cloud", never "Official".
 *
 * `CloudItemBadges({kind, id})` sits in the agent cards (grid and list), the package card, the plugin
 * cards (grid and featured) and the agent and Playbook detail header. It renders them only for a row
 * the Cloud items read lists (hooks/use-cloud-items.ts), by the key its install takes; nothing for any
 * other row, nothing while the list loads or if it fails. On the hosted edition it renders nothing and
 * mounts no query, so a card there needs no QueryClientProvider and asks nothing.
 */
import { Badge } from '@/components/ui/badge'
import { useCloudItems } from '@/hooks/use-cloud-items'
import { isLocal } from '@/lib/auth-edition'
import { isCloudItem } from '@/lib/marketplace/cloud-items'
import type { CloudItemKey, CloudItemKind } from '@/lib/marketplace/cloud-items'
import { cn } from '@/lib/utils'

import { CLOUD_BADGE, REVIEWED_BADGE } from './item-copy'

interface CloudItemBadgesProps {
  kind: CloudItemKind
  /** The key its install takes: an agent's or a Playbook's id, a package's slug, a plugin's uuid. */
  id: CloudItemKey
  className?: string
}

/** The two badges, for an item already known to be from Automatos Cloud (the disclosure's header). */
export function CloudBadgePair({ className }: { className?: string }) {
  return (
    <div className={cn('flex flex-wrap gap-1.5', className)}>
      <Badge variant="secondary">{CLOUD_BADGE}</Badge>
      <Badge variant="outline">{REVIEWED_BADGE}</Badge>
    </div>
  )
}

function HeldCloudItemBadges({ kind, id, className }: CloudItemBadgesProps) {
  const { data } = useCloudItems()
  return isCloudItem(data, kind, id) ? <CloudBadgePair className={className} /> : null
}

export function CloudItemBadges(props: CloudItemBadgesProps) {
  return isLocal ? <HeldCloudItemBadges {...props} /> : null
}
