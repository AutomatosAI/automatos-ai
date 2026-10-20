/**
 * PRD-261 C-01b — why the Marketplace refused a submitted package, in the author's words.
 *
 * `POST /api/marketplace/themes/submit` checks the package as a private upload is checked
 * (`orchestrator/services/themes/package.py` `read_theme_package`) and refuses it with
 * `{ detail: { message, errors: [{ code, path, message }] } }`: 400, or 413 when the package is over
 * the cap (`api/marketplace_themes_submit.py` `_read_upload`). The form lists every error under the
 * part of the package it is about, with the server's sentence, the file or field and the code.
 *
 * The groups follow where each code is raised: the manifest codes are `MANIFEST_ERROR_CODES`
 * (lib/theme-host/package-manifest.ts, the mirror of `services/themes/manifest.py`); the style sheet
 * codes are `css_scope.py`'s `CSS_*`; the looks codes are `look_schema.py`'s and the preset checks in
 * `package.py`; the size codes are `package.py`'s caps. Any other code, including one added later,
 * is listed under the package file with the server's own sentence, never dropped.
 *
 * The body-size middleware answers an oversized request before the route with a bare
 * `{ detail: 'Payload too large' }` 413 (`main.py`); that one gets a plain sentence of its own.
 */
import { MANIFEST_ERROR_CODES } from '@/lib/theme-host/package-manifest'

export type PublishErrorGroup = 'archive' | 'sizes' | 'manifest' | 'styleSheet' | 'looks'

/** One refusal as `services/themes/contract.py` `theme_error` writes it. */
export interface PackageError {
  readonly code: string
  readonly path: string
  readonly message: string
}

/** A submission refused by the package checks: the server's headline and each error. */
export interface SubmissionRefusal {
  readonly message: string
  readonly errors: readonly PackageError[]
}

/** The headings, in the order the form shows them. */
export const PUBLISH_ERROR_GROUPS: ReadonlyArray<{ readonly id: PublishErrorGroup; readonly title: string }> = [
  { id: 'archive', title: 'The package file' },
  { id: 'sizes', title: 'Sizes' },
  { id: 'manifest', title: 'The manifest (theme.json)' },
  { id: 'styleSheet', title: 'The style sheet' },
  { id: 'looks', title: 'Looks and presets' },
]

export const PACKAGE_TOO_LARGE_MESSAGE = "This package is over the platform's size limit for a theme."
const REFUSED_FALLBACK = "This theme can't be submitted."

const PACKAGE_CHECK_STATUSES: ReadonlySet<number> = new Set([400, 413])
const TOO_LARGE_STATUS = 413
const CSS_PREFIX = 'CSS_'
const LOOK_PREFIX = 'LOOK'
const TOO_LARGE_SUFFIX = '_TOO_LARGE'
const LOOKS_CODES: ReadonlySet<string> = new Set([
  'BAD_KNOB', 'BAD_LOOK_SCHEMA', 'NO_PRESET', 'PRESET_ID_MISMATCH', 'UNKNOWN_DEFAULT_LOOK', 'TOO_MANY_LOOKS',
])
const SIZE_CODES: ReadonlySet<string> = new Set(['TOO_MANY_FILES'])
const MANIFEST_CODES: ReadonlySet<string> = new Set(MANIFEST_ERROR_CODES)

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

/** The part of the package a refusal code is about. */
export function publishErrorGroupOf(code: string): PublishErrorGroup {
  if (MANIFEST_CODES.has(code)) return 'manifest'
  if (code.startsWith(CSS_PREFIX)) return 'styleSheet'
  if (code.startsWith(LOOK_PREFIX) || LOOKS_CODES.has(code)) return 'looks'
  if (code.endsWith(TOO_LARGE_SUFFIX) || SIZE_CODES.has(code)) return 'sizes'
  return 'archive'
}

function packageErrorOf(value: unknown): PackageError | null {
  if (!isRecord(value) || typeof value.code !== 'string' || typeof value.message !== 'string') return null
  return { code: value.code, path: typeof value.path === 'string' ? value.path : '', message: value.message }
}

/**
 * The package checks' refusal a thrown `apiClient` error carries (400 or 413), or null for any
 * other error. A 413 without the checks' body is the middleware's: one error, the plain sentence.
 */
export function submissionRefusalOf(error: unknown): SubmissionRefusal | null {
  if (!isRecord(error) || typeof error.status !== 'number' || !PACKAGE_CHECK_STATUSES.has(error.status)) return null
  const detail = isRecord(error.body) ? error.body.detail : null
  const raw = isRecord(detail) && Array.isArray(detail.errors) ? detail.errors : []
  const errors = raw.map(packageErrorOf).filter((item): item is PackageError => item !== null)
  if (errors.length) {
    const message = isRecord(detail) && typeof detail.message === 'string' ? detail.message : REFUSED_FALLBACK
    return { message, errors }
  }
  return error.status === TOO_LARGE_STATUS ? { message: PACKAGE_TOO_LARGE_MESSAGE, errors: [] } : null
}

/** The errors under each heading, headings with none left out, in the form's order. */
export function groupPackageErrors(
  errors: readonly PackageError[],
): Array<{ readonly id: PublishErrorGroup; readonly title: string; readonly errors: PackageError[] }> {
  return PUBLISH_ERROR_GROUPS.map((group) => ({
    ...group,
    errors: errors.filter((error) => publishErrorGroupOf(error.code) === group.id),
  })).filter((group) => group.errors.length > 0)
}

export function PublishErrors({ refusal }: { refusal: SubmissionRefusal }) {
  const groups = groupPackageErrors(refusal.errors)
  return (
    <div role="alert" className="space-y-3 rounded-xl border border-destructive/40 px-4 py-3" data-testid="publish-errors">
      <p className="text-sm font-medium text-destructive">{refusal.message}</p>
      {groups.map((group) => (
        <section key={group.id} aria-label={group.title} className="space-y-1.5" data-testid={`publish-errors-${group.id}`}>
          <p className="text-xs font-medium text-foreground">{group.title}</p>
          <ul className="space-y-1.5">
            {group.errors.map((error, index) => (
              <li key={`${error.code}-${error.path}-${index}`} className="text-sm">
                <p className="text-foreground">{error.message}</p>
                <p className="text-xs text-muted-foreground">
                  {error.path ? `${error.path} · ` : ''}
                  <code>{error.code}</code>
                </p>
              </li>
            ))}
          </ul>
        </section>
      ))}
    </div>
  )
}
