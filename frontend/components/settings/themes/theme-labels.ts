/**
 * PRD-261 A-07d — the words Settings › Appearance › Themes shows, and which role sees what.
 *
 * The opt-in wording is NOT here: the dialog shows the server's own sentence
 * (`services/themes/optin.py` OPT_IN_WORDING, sent as `optIn.wording`), so there is one copy.
 */
import type { Workspace } from '@/components/workspace-provider'
import type { InstalledTheme } from '@/lib/api-client'

/** PRD §5: the label of a theme nobody at Automatos reviewed. */
export const PRIVATE_NOT_REVIEWED = 'Private · not reviewed'
export const NEEDS_UPDATE = 'Needs update'
export const NEEDS_UPDATE_HINT =
  'A core page this theme copies changed in this release. It may look out of date until its author updates it.'
export const UPDATE_AVAILABLE = 'Update available'
export const DEFAULT_BADGE = 'Workspace default'
export const IN_USE = 'In use'
export const MEMBERS_SEE_ONLY = 'A workspace owner or admin installs, updates and removes themes.'
export const NO_INSTALLS = 'No themes are installed in this workspace yet.'
export const UNREVIEWED_NOT_DEFAULT = 'A theme that was not reviewed can’t be the workspace default.'

/** Who may install, update and remove themes: `workspace:manage`, the owner and admin roles. */
const MANAGE_ROLES: ReadonlySet<Workspace['role']> = new Set<Workspace['role']>(['owner', 'admin'])

export function canManageThemes(role: Workspace['role'] | undefined): boolean {
  return !!role && MANAGE_ROLES.has(role)
}

/** True for the label PRD §5 puts on private code: not reviewed. */
export function isPrivateNotReviewed(install: Pick<InstalledTheme, 'reviewed'>): boolean {
  return !install.reviewed
}

/** A revocation that stops the theme from running: it can't be chosen. */
export function isDisabled(install: Pick<InstalledTheme, 'revoked'>): boolean {
  return install.revoked?.level === 'disable'
}

export function themeName(install: Pick<InstalledTheme, 'name' | 'themeId'>): string {
  return install.name || install.themeId
}
