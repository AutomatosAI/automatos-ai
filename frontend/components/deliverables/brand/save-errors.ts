/**
 * Why PUT /api/documents/brand-kit refused a save. A 422 names each refused field and why:
 * the kit's own check (`{message, errors}`, PRD-255's contrast check among them) or the
 * request's (a list); apiClient carries the detail as the error's message.
 */
import type { BrandPaletteRole } from '@/components/documents/blocks/types'

interface FieldError {
  loc: unknown[]
  msg: string
}

export const PALETTE_ROLES: readonly BrandPaletteRole[] = [
  'ink', 'heading', 'paper', 'surface', 'surface_2', 'accent', 'accent_2', 'muted', 'rule',
]
export type RoleErrors = Partial<Record<BrandPaletteRole, string>>
const PALETTE_LOC = 'palette'
const VALUE_ERROR_PREFIX = /^Value error, /

/** Each refused field (its loc without "body") and the rule it broke; none when ``e`` is no validation detail. */
export function fieldErrors(e: any): FieldError[] {
  let detail: any
  try {
    detail = JSON.parse(e?.message ?? '')
  } catch {
    return []
  }
  const errors: Array<{ loc?: unknown[]; msg?: string }> = Array.isArray(detail)
    ? detail
    : Array.isArray(detail?.errors) ? detail.errors : []
  return errors.map((err) => ({
    loc: (err.loc ?? []).filter((part) => part !== 'body'),
    msg: (err.msg ?? '').replace(VALUE_ERROR_PREFIX, ''),
  }))
}

/** The refusal in one line: each field and why, or the message as it came. */
export function saveErrorMessage(e: any): string {
  const errors = fieldErrors(e)
  if (errors.length) return errors.map((err) => `${err.loc.join('.')}: ${err.msg}`).join('; ')
  return e?.message || 'Failed to save brand kit'
}

/** The colour roles a refusal names (loc ['palette', role]), each with its message. */
export function roleErrorsFrom(e: any): RoleErrors {
  const found: RoleErrors = {}
  for (const { loc, msg } of fieldErrors(e)) {
    const role = loc[1] as BrandPaletteRole
    if (loc[0] === PALETTE_LOC && PALETTE_ROLES.includes(role)) found[role] = msg
  }
  return found
}
