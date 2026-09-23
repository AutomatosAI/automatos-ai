'use client'

/**
 * PRD-251 S0.5 — the Socials tab body, shared by both Deliverables shells.
 *
 * It follows D1. The shells render the tab only while `socials.available` (the
 * platform master switch); with this workspace's switch off the body is ONE
 * card (turn it on, or ask an admin); with both on it is the post list.
 */
import { useWorkspace } from '@/components/workspace-provider'
import { SocialsPostList } from './socials-post-list'
import { SocialsTurnOnCard } from './socials-turn-on-card'

export function SocialsTab() {
  const { workspace } = useWorkspace()
  const socials = workspace?.socials
  if (!workspace || !socials?.available) return null
  if (!socials.enabled) return <SocialsTurnOnCard role={workspace.role} />
  return <SocialsPostList role={workspace.role} />
}
