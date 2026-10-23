'use client'

/**
 * PRD-261 C-05j — revoke a theme: a package digest, or a theme id with an optional version range,
 * at `warn` or `disable`, with the reason every member reads in the host's notice (admin + step-up).
 *
 * The body is exactly what `POST /api/admin/themes/revocations` takes (`RevocationBody`, which
 * refuses any other field): `{digest, level, reason}` or `{themeId[, versionRange], level, reason}`.
 * The server checks every field (64 hex for a digest, `publisher/slug` for an id, `1.2.0` or `^1.0.0`
 * for a range, one line of plain text for the reason) and its 400 sentence is shown as it is; the
 * form only keeps a blank target or reason from being sent.
 */
import { useState } from 'react'

import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Textarea } from '@/components/ui/textarea'
import type { ThemeRevocationCreate, ThemeRevocationLevel } from '@/lib/api-client'
import { themeErrorMessage } from '@/lib/theme-host/api'

import { LEVEL_HINTS, REVOCATION_REASON_MAX, REVOKE } from './review-words'
import { useCreateRevocation } from './use-theme-revocations'

export type RevocationTargetKind = 'digest' | 'theme'

/** What the form holds, as typed. */
export interface RevocationFields {
  readonly kind: RevocationTargetKind
  readonly digest: string
  readonly themeId: string
  readonly versionRange: string
  readonly level: ThemeRevocationLevel
  readonly reason: string
}

const EMPTY_FIELDS: RevocationFields = {
  kind: 'digest', digest: '', themeId: '', versionRange: '', level: 'disable', reason: '',
}
const KINDS: ReadonlyArray<{ readonly id: RevocationTargetKind; readonly label: string }> = [
  { id: 'digest', label: 'A package digest' },
  { id: 'theme', label: 'A theme id' },
]
const LEVELS: readonly ThemeRevocationLevel[] = ['warn', 'disable']

/** The request body for `fields`, trimmed; a theme id without a range revokes every version. */
export function revocationRequest(fields: RevocationFields): ThemeRevocationCreate {
  const terms = { level: fields.level, reason: fields.reason.trim() }
  if (fields.kind === 'digest') return { digest: fields.digest.trim(), ...terms }
  const range = fields.versionRange.trim()
  return range ? { themeId: fields.themeId.trim(), versionRange: range, ...terms } : { themeId: fields.themeId.trim(), ...terms }
}

/** True while there is nothing to send: no target, or no reason. */
export function revocationIncomplete(fields: RevocationFields): boolean {
  const target = fields.kind === 'digest' ? fields.digest : fields.themeId
  return target.trim() === '' || fields.reason.trim() === ''
}

function Choice<T extends string>({ name, value, options, onChange }: {
  name: string
  value: T
  options: ReadonlyArray<{ readonly id: T; readonly label: string }>
  onChange: (next: T) => void
}) {
  return (
    <fieldset className="flex flex-wrap gap-4 text-xs">
      {options.map((option) => (
        <label key={option.id} className="flex items-center gap-1">
          <input type="radio" name={name} checked={value === option.id} onChange={() => onChange(option.id)} />
          {option.label}
        </label>
      ))}
    </fieldset>
  )
}

function TargetInputs({ fields, set }: { fields: RevocationFields; set: (change: Partial<RevocationFields>) => void }) {
  if (fields.kind === 'digest') {
    // No maxLength: a browser cuts a paste to it before the trim, so a digest copied with a space
    // around it would lose hex characters off its end. The server's 64-hex check is the bound.
    return (
      <Input
        aria-label="Package digest"
        placeholder="The package's sha256, 64 hex characters"
        value={fields.digest}
        spellCheck={false}
        autoComplete="off"
        onChange={(event) => set({ digest: event.target.value })}
      />
    )
  }
  return (
    <div className="grid gap-2 sm:grid-cols-2">
      <Input aria-label="Theme id" placeholder="acme/ink" value={fields.themeId} spellCheck={false}
        autoComplete="off" onChange={(event) => set({ themeId: event.target.value })} />
      <Input aria-label="Version range (optional)" placeholder="1.2.0 or ^1.0.0; empty for every version"
        value={fields.versionRange} spellCheck={false} autoComplete="off"
        onChange={(event) => set({ versionRange: event.target.value })} />
    </div>
  )
}

export function RevocationForm() {
  const create = useCreateRevocation()
  const [fields, setFields] = useState<RevocationFields>(EMPTY_FIELDS)
  const set = (change: Partial<RevocationFields>) => setFields((current) => ({ ...current, ...change }))
  const blocked = revocationIncomplete(fields) || create.isLoading
  const levelOptions = LEVELS.map((level) => ({ id: level, label: LEVEL_HINTS[level] }))
  return (
    <form
      aria-label="Revoke a theme"
      className="space-y-2 rounded-lg border border-border/50 p-3"
      onSubmit={(event) => {
        event.preventDefault()
        if (!blocked) create.mutate(revocationRequest(fields), { onSuccess: () => setFields(EMPTY_FIELDS) })
      }}
    >
      <p className="text-sm font-medium">Revoke a theme</p>
      <Choice name="theme-revocation-kind" value={fields.kind} options={KINDS} onChange={(kind) => set({ kind })} />
      <TargetInputs fields={fields} set={set} />
      <Choice name="theme-revocation-level" value={fields.level} options={levelOptions} onChange={(level) => set({ level })} />
      <label className="block text-xs font-medium" htmlFor="theme-revocation-reason">
        Reason (every member of an affected workspace reads it)
      </label>
      <Textarea
        id="theme-revocation-reason"
        value={fields.reason}
        maxLength={REVOCATION_REASON_MAX}
        rows={2}
        onChange={(event) => set({ reason: event.target.value })}
      />
      <Button type="submit" variant="destructive" disabled={blocked}>
        {REVOKE}
      </Button>
      {create.error ? (
        <p role="alert" className="text-xs text-destructive" data-testid="theme-revocation-error">
          {themeErrorMessage(create.error)}
        </p>
      ) : null}
    </form>
  )
}
