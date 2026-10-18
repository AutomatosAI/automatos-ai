/**
 * PRD-261 A-10a — the core nav items a theme's frame may draw: the Studio menu
 * (lib/studio-menu.ts) after the same three filters StudioSidebar applies (the edition's routes,
 * the admin role gate, the plan-tier exposure). Pure data, no React and no icons: a theme draws
 * its own. The routes stay in the menu, never a second list here.
 */
import type { WorkspaceExposure } from '@/components/workspace-provider'
import { filterNavForEdition, navExposureForEdition } from '@/lib/auth-edition'
import { isNavItemVisible } from '@/lib/nav-exposure'
import { STUDIO_MENU_FOOTER, STUDIO_MENU_PRIMARY } from '@/lib/studio-menu'

export interface CoreNavItem {
  readonly id: string
  readonly label: string
  /** One line about the page; null when the menu has none. */
  readonly description: string | null
  readonly href: string
  /** The menu group ('OPERATIONS', 'WORKFORCE', 'WORKSPACE'); null for the footer. */
  readonly group: string | null
  /** Opens in a new tab (Docs). */
  readonly external: boolean
}

export interface CoreNav {
  /** The menu's items in the menu's own order, groups included. */
  readonly primary: readonly CoreNavItem[]
  /** Docs and Settings, where Studio shows them at the foot of its sidebar. */
  readonly footer: readonly CoreNavItem[]
}

export interface CoreNavAccess {
  /** A platform admin; unknown is not one. */
  readonly isAdmin: boolean
  /** The workspace's plan-tier exposure, as the workspace provider has it. */
  readonly exposure: WorkspaceExposure | null | undefined
}

interface MenuEntry {
  readonly id: string
  readonly label: string
  readonly desc?: string
  readonly href: string
  readonly group?: string
  readonly external?: boolean
  readonly requiredRole?: 'admin'
}

function roleAllows(item: MenuEntry, isAdmin: boolean): boolean {
  return !item.requiredRole || isAdmin
}

function toCoreItem(item: MenuEntry): CoreNavItem {
  return Object.freeze({
    id: item.id,
    label: item.label,
    description: item.desc ?? null,
    href: item.href,
    group: item.group ?? null,
    external: item.external === true,
  })
}

/** The core nav for this edition, role and plan. */
export function buildCoreNav({ isAdmin, exposure }: CoreNavAccess): CoreNav {
  const navExposure = navExposureForEdition(exposure)
  const primary = filterNavForEdition(STUDIO_MENU_PRIMARY)
    .filter((item) => roleAllows(item, isAdmin) && isNavItemVisible(item.requiredExposure, navExposure))
    .map(toCoreItem)
  const footer = filterNavForEdition(STUDIO_MENU_FOOTER)
    .filter((item) => roleAllows(item, isAdmin))
    .map(toCoreItem)
  return Object.freeze({ primary: Object.freeze(primary), footer: Object.freeze(footer) })
}
