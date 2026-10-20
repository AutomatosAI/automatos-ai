/**
 * PRD-260 T-004 — Play's sidebar nav, built from the core nav (the SDK's useCoreNav: the Studio
 * menu after the filters StudioSidebar applies — the edition's route list, the admin role gate,
 * the plan-tier exposure), plus Play's two pages: Conversations right after Chat, and Look last,
 * on its own. Pure data and helpers, no React; the routes stay in the menu, never a second list
 * here. PRD-261 A-14f: the core only through '@automatos/theme-sdk'.
 */
import { LOOKS_HREF, resolveActiveNavId, type CoreNav, type CoreNavItem } from '@automatos/theme-sdk'

import { PLAY_CONVERSATIONS_HREF } from './play-routes'

export interface PlayNavItem {
  readonly id: string
  readonly label: string
  readonly href: string
  /** Opens in a new tab (Docs). */
  readonly external?: boolean
}

export interface PlayNavGroup {
  readonly id: string
  /** The small caps label over the group; null for Look, which stands alone. */
  readonly label: string | null
  readonly items: readonly PlayNavItem[]
}

export interface PlayNav {
  readonly groups: readonly PlayNavGroup[]
  /** Docs and Settings, at the foot of the sidebar where Studio shows them. */
  readonly footer: readonly PlayNavItem[]
}

/** Play's conversation list (T-006): right after Chat, at its theme route (/conversations only redirects there). */
export const PLAY_CONVERSATIONS_ITEM: PlayNavItem = { id: 'conversations', label: 'Conversations', href: PLAY_CONVERSATIONS_HREF }
/** Make it yours (T-008a): last, on its own. */
export const PLAY_LOOK_ITEM: PlayNavItem = { id: 'looks', label: 'Look', href: LOOKS_HREF }
const CONVERSATIONS_FOLLOW = 'chat'
const LOOK_GROUP_ID = 'look'

function toPlayItem({ id, label, href, external }: CoreNavItem): PlayNavItem {
  return external ? { id, label, href, external } : { id, label, href }
}

/** 'OPERATIONS' → 'Operations': the sidebar sets it in small caps itself. */
function groupLabel(group: string): string {
  return group.charAt(0) + group.slice(1).toLowerCase()
}

/** The menu's groups in the menu's own order (OPERATIONS, WORKFORCE, WORKSPACE): the core nav keeps it. */
function groupsInOrder(items: readonly CoreNavItem[]): string[] {
  return Array.from(new Set(items.flatMap((item) => (item.group === null ? [] : [item.group]))))
}

function withConversations(items: readonly PlayNavItem[]): PlayNavItem[] {
  return items.flatMap((item) => (item.id === CONVERSATIONS_FOLLOW ? [item, PLAY_CONVERSATIONS_ITEM] : [item]))
}

/** Play's nav from the core nav for this edition, role and plan: its groups, Play's two pages added. */
export function buildPlayNav({ primary, footer }: CoreNav): PlayNav {
  const groups: PlayNavGroup[] = groupsInOrder(primary).map((group) => ({
    id: group.toLowerCase(),
    label: groupLabel(group),
    items: withConversations(primary.filter((item) => item.group === group).map(toPlayItem)),
  }))
  return { groups: [...groups, { id: LOOK_GROUP_ID, label: null, items: [PLAY_LOOK_ITEM] }], footer: footer.map(toPlayItem) }
}

function isUnder(pathname: string, href: string): boolean {
  return pathname === href || pathname.startsWith(`${href}/`)
}

/** The nav item a path belongs to: Play's two pages, else the core nav's resolver. */
export function resolvePlayActiveId(pathname: string): string | null {
  if (isUnder(pathname, PLAY_CONVERSATIONS_ITEM.href)) return PLAY_CONVERSATIONS_ITEM.id
  if (isUnder(pathname, PLAY_LOOK_ITEM.href)) return PLAY_LOOK_ITEM.id
  return resolveActiveNavId(pathname)
}
