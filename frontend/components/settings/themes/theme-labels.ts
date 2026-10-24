/**
 * PRD-261 A-07d — the words Settings › Appearance › Themes shows, and which role sees what.
 *
 * The opt-in wording is NOT here: the dialog shows the server's own sentence
 * (`services/themes/optin.py` OPT_IN_WORDING, sent as `optIn.wording`), so there is one copy.
 */
import type { Workspace } from '@/components/workspace-provider'
import type { InstalledTheme, ThemeSignatureState, ThemeSignatureVerdict } from '@/lib/api-client'

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

// ===== PRD-261 C-05k: signatures (C-05d's signed upload, C-05b's listing verdicts) =====
/**
 * QB-24: a package whose signature an active Automatos signing key verifies. On an installed row it
 * stands in place of PRIVATE_NOT_REVIEWED; it is never shown for `invalid` or `unsigned`. Every
 * surface takes it from here (the one-line edit should QB-24's answer change).
 */
export const SIGNED_BY_AUTOMATOS = 'Signed by Automatos'
/** The listing detail's fact, and the words of each state. */
export const SIGNATURE_FACT = 'Signature'
export const SIGNATURE_WORDS: Readonly<Record<ThemeSignatureState, string>> = {
  verified: SIGNED_BY_AUTOMATOS,
  invalid: 'Signed, but the signature doesn’t check out',
  unsigned: 'Not signed',
}
/** Install theme's optional second file: the detached `.sig.json` released beside an `.autotheme`. */
export const SIGNATURE_FILE_ACCEPT = '.sig.json,application/json'
export const ADD_SIGNATURE_FILE = 'Add its signature file'
export const REMOVE_SIGNATURE_FILE = 'Remove'
export const SIGNATURE_FILE_HINT =
  'Optional: the .sig.json released with the theme. Automatos checks it before anything is installed.'
export const SIGNATURE_FILE_CHOSEN = 'Signature file:'

/** Who may install, update and remove themes: `workspace:manage`, the owner and admin roles. */
const MANAGE_ROLES: ReadonlySet<Workspace['role']> = new Set<Workspace['role']>(['owner', 'admin'])

export function canManageThemes(role: Workspace['role'] | undefined): boolean {
  return !!role && MANAGE_ROLES.has(role)
}

/** True for the label PRD §5 puts on private code: not reviewed. */
export function isPrivateNotReviewed(install: Pick<InstalledTheme, 'reviewed'>): boolean {
  return !install.reviewed
}

/** True only when an active Automatos key verifies the signature (`invalid` never reads as signed). */
export function isSignedByAutomatos(item: { readonly signature?: ThemeSignatureVerdict | null }): boolean {
  return item.signature?.state === 'verified'
}

/** A signature verdict in words; a verdict the server did not send reads as not signed. */
export function signatureWords(verdict: ThemeSignatureVerdict | null | undefined): string {
  return SIGNATURE_WORDS[verdict?.state ?? 'unsigned'] ?? SIGNATURE_WORDS.unsigned
}

/**
 * Whether the install may be the workspace default: the server's own rule (`defaults.can_be_default`,
 * which is "not `choices.is_private`", the row's `private`), so a signed upload QB-24 counts as
 * reviewed may be, and private code never is.
 */
export function canBeDefault(install: Pick<InstalledTheme, 'private'>): boolean {
  return !install.private
}

/** A revocation that stops the theme from running: it can't be chosen. */
export function isDisabled(install: Pick<InstalledTheme, 'revoked'>): boolean {
  return install.revoked?.level === 'disable'
}

export function themeName(install: Pick<InstalledTheme, 'name' | 'themeId'>): string {
  return install.name || install.themeId
}
