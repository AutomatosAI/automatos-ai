'use client'

/**
 * PRD-251 S0.5 — the Socials tab body, shared by both Deliverables shells.
 *
 * It follows D1. The shells render the tab only while `socials.available` (the
 * platform master switch); with this workspace's switch off the body is ONE
 * card (turn it on, or ask an admin); with both on it is the post list.
 * S2.1: `postId` is the page's `?post=` (global search links a post there), and
 * the list opens that post.
 * PRD-251B US-B107: with both on it is the Socials Studio (studio/studio-shell.tsx):
 * Calendar · Queue · Plans · Brand kit, the view held in the URL.
 */
import { useWorkspace } from '@/components/workspace-provider'
import { SocialsStudio } from './studio/studio-shell'
import { SocialsTurnOnCard } from './socials-turn-on-card'

export function SocialsTab({ postId = null }: { postId?: string | null }) {
  const { workspace } = useWorkspace()
  const socials = workspace?.socials
  if (!workspace || !socials?.available) return null
  if (!socials.enabled) return <SocialsTurnOnCard role={workspace.role} />
  return <SocialsStudio role={workspace.role} postId={postId} />
}
