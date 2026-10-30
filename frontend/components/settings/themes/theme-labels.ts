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
/**
 * The theme menu's and ⌘K's note on a private version this member has not opted in to: choosing it
 * opens the opt-in dialog first (choose-install.tsx). Once opted in, it reads PRIVATE_NOT_REVIEWED.
 */
export const PRIVATE_OPT_IN_FIRST = 'Private · opt in first'
export const NEEDS_UPDATE = 'Needs update'
export const NEEDS_UPDATE_HINT =
  'A core page this theme copies changed in this release. It may look out of date until its author updates it.'
export const UPDATE_AVAILABLE = 'Update available'
/**
 * A Marketplace install that is not pinned moves to each version approved after it by itself, and back
 * to the previous listed one when its version is taken down (services/themes/auto_update.py). A pin
 * holds the install on its version: it keeps "Update available" instead.
 */
export const UPDATES_ITSELF = 'Updates itself; pin to hold'
export const UPDATES_ITSELF_HINT =
  'Each new Marketplace version of this theme replaces this one once it is approved. Pin a version under Pinned versions to keep it.'
/** The note of the Pinned versions list (theme-pins.tsx). */
export const PINS_NOTE =
  'Marketplace themes update themselves to each new version. A pinned theme stays on that version until it is unpinned.'
/** `InstalledTheme.ownerType` of a version installed from the Marketplace (services/themes/choices.py). */
const MARKETPLACE_OWNER = 'marketplace'
export const DEFAULT_BADGE = 'Workspace default'
export const IN_USE = 'In use'
export const MEMBERS_SEE_ONLY = 'A workspace owner or admin installs, updates and removes themes.'
export const NO_INSTALLS = 'No themes are installed in this workspace yet.'
/** Make default's reason when it is off: shown beside the button, and as its title. */
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

// ===== The scan report of an upload (scan-report.tsx) =====
/**
 * Every check the review stage stores (`services/themes/review_stage.py` run_review_stage: the
 * reader's `sizes`, `cssScope`, `looks`, then `contrast`, `uses` from sdk_uses.py, `audioCapture` from
 * audio_capture.py (theme voice), `clonedFrom` from drift.py and `llm` from llm_stage.py, which member_view
 * leaves out), and `automated`, which only a report stored before the review stage has (`scan.py`).
 */
export const CHECK_LABELS: Readonly<Record<string, string>> = {
  sizes: 'Sizes',
  cssScope: 'Style sheet scope',
  looks: 'Looks',
  contrast: 'Contrast (AA)',
  uses: 'SDK areas',
  audioCapture: 'Microphone and speech',
  clonedFrom: 'Cloned core pages',
  llm: 'AI code read',
  automated: 'Automated review',
}

/** A check's `level` in the review stage's report (review_stage.LEVELS). */
export const LEVEL_LABELS: Readonly<Record<string, string>> = {
  pass: 'Passed',
  warning: 'Warning',
  error: 'Error',
  not_measured: 'Not measured',
}

/** A check's `status` in a report stored before the review stage (`scan.py`), which has no level. */
export const STATUS_LABELS: Readonly<Record<string, string>> = {
  pass: 'Passed',
  absent: 'No style sheet',
  not_run: 'Not run',
}

/** A check that carries neither a level nor a status. */
export const NO_RESULT = 'No result'

/** The report's verdict (review_stage.verdict_of; `not_run` for a report stored before it). */
export const VERDICT_LABELS: Readonly<Record<string, string>> = {
  pass: 'Passed',
  review_required: 'Needs a reviewer',
  blocked: 'Blocked',
  not_run: 'Not run',
}

/**
 * Why the verdict is worse than every check shown: `member_view` leaves the AI code read (`llm`)
 * out. It is a warning when it did not read all of the code (no ANTHROPIC_API_KEY, a failed call or
 * reply, the deadline, the day's budget, too many chunks) or its risk score reached REVIEW_AT, and
 * an error at BLOCK_AT. A private upload is stored and installed whatever the verdict (`scan.py`).
 */
export const HIDDEN_AI_REVIEW =
  'Every check shown here passed or was not measured. The verdict comes from the AI code read, which this report leaves out: it could not read all of the code (for example, this install has no Anthropic API key), or it asked for a person to look. That review is only for a Marketplace listing: a private upload installs here whatever the verdict.'
export const HIDDEN_AI_BLOCKED =
  'No check shown here found an error. The verdict comes from the AI code read, which this report leaves out: it rated this code high risk. A private upload installs here whatever the verdict.'

/**
 * Under a Needs a reviewer or Blocked verdict that a check shown explains: this screen is a private upload's
 * (install-theme.tsx), which nobody at Automatos reviews (review.py queues Marketplace submissions only) and which
 * is stored and installed whatever the verdict (records.save_private_upload).
 */
export const PRIVATE_INSTALLS_ANYWAY =
  'A private upload is not reviewed by Automatos: it installs here whatever the verdict. The checks above say what to look at before you use it.'

/** A cloned page whose core file this release changed or no longer ships (drift.stale_clones). */
export function staleCloneWords(route: string | null, path: string | null, stillShipped: boolean): string {
  const page = `${route ?? 'A page'} copies ${path ?? 'a core file'}`
  return stillShipped ? `${page}, which this release has changed since.` : `${page}, which this release no longer ships.`
}

/** Who may install, update and remove themes: `workspace:manage`, the owner and admin roles. */
const MANAGE_ROLES: ReadonlySet<Workspace['role']> = new Set<Workspace['role']>(['owner', 'admin'])

export function canManageThemes(role: Workspace['role'] | undefined): boolean {
  return !!role && MANAGE_ROLES.has(role)
}

/** True for the label PRD §5 puts on private code: not reviewed. */
export function isPrivateNotReviewed(install: Pick<InstalledTheme, 'reviewed'>): boolean {
  return !install.reviewed
}

/**
 * True while this member must opt in to the install's current version before it runs for them: the
 * server's own answer (`services/themes/optin.py` opt_in_state, private code not yet opted in to).
 */
export function needsOptIn(install: Pick<InstalledTheme, 'optIn'>): boolean {
  return install.optIn.required
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

/** True for a Marketplace install with no pin: the server moves it to each new version by itself. */
export function updatesItself(install: Pick<InstalledTheme, 'ownerType' | 'pinnedVersion'>): boolean {
  return install.ownerType === MARKETPLACE_OWNER && !install.pinnedVersion
}

/** A revocation that stops the theme from running: it can't be chosen. */
export function isDisabled(install: Pick<InstalledTheme, 'revoked'>): boolean {
  return install.revoked?.level === 'disable'
}

export function themeName(install: Pick<InstalledTheme, 'name' | 'themeId'>): string {
  return install.name || install.themeId
}
