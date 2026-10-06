'use client'

/**
 * PRD-255 US-007 — the colour roles on the Brand kit page: each role's swatch and hex, its
 * contrast badge (computed here from the hexes; the save's check is the server's), whether
 * it is set or derived from the kit's four colours, "Reset to derived" for a set role, and
 * the save's contrast error under the role it names. Then how far the accent goes.
 */
import { Loader2, RotateCcw } from 'lucide-react'

import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import type { BrandAccentUse, BrandKit, BrandPaletteRole } from '@/components/documents/blocks/types'
import { roleContrast } from './contrast'
import { SELECT_CLASS, SectionTitle } from './brand-kit-inputs'
import { PALETTE_ROLES } from './save-errors'
import type { BrandPaletteEdits } from './use-brand-palette'

const HEX6 = /^#[0-9a-fA-F]{6}$/
const EMPTY_SWATCH = '#ffffff'
// Every kit's default (PRD-255 Decision Q1), as the server answers a kit without one.
const DEFAULT_ACCENT_USE: BrandAccentUse = 'sparing'

export const ROLE_LABELS: Record<BrandPaletteRole, string> = {
  ink: 'Body text',
  heading: 'Headings',
  paper: 'Page',
  surface: 'Cards and zebra rows',
  surface_2: 'Table header fill',
  accent: 'Accent (highlights)',
  accent_2: 'Second accent',
  muted: 'Secondary text',
  rule: 'Hairlines',
}

export const ACCENT_USE_LABELS: Record<BrandAccentUse, string> = {
  sparing: 'Sparing: highlights only (the title rule, key numbers, links)',
  bold: 'Bold: highlights and table header fills',
}

interface RoleRowProps {
  kit: BrandKit
  role: BrandPaletteRole
  edits: BrandPaletteEdits
  canEdit: boolean
}

function ContrastBadge({ kit, role }: { kit: BrandKit; role: BrandPaletteRole }) {
  const badge = roleContrast(kit.palette ?? {}, role)
  if (!badge) return null
  const tone = badge.passes === false ? 'border-destructive text-destructive' : 'text-muted-foreground'
  return (
    <span className={`rounded border px-1.5 py-0.5 text-[10px] ${tone}`} data-testid={`contrast-${role}`}>
      {badge.label}{badge.passes === false ? ' — too low' : ''}
    </span>
  )
}

function RoleRow({ kit, role, edits, canEdit }: RoleRowProps) {
  const value = kit.palette?.[role] ?? ''
  const source = kit.palette_source?.[role] ?? 'derived'
  const error = edits.roleErrors[role]
  const label = ROLE_LABELS[role]
  return (
    <div className="min-w-0 rounded-md border p-2" data-testid={`role-${role}`}>
      <div className="flex items-center justify-between gap-2">
        <Label className="text-xs">{label}</Label>
        <span className="text-[10px] uppercase tracking-wide text-muted-foreground">{source}</span>
      </div>
      <div className="mt-1 flex items-center gap-2">
        <input
          type="color" aria-label={`${label} colour`} value={HEX6.test(value) ? value : EMPTY_SWATCH}
          onChange={(e) => edits.setRole(role, e.target.value)} className="h-8 w-9 shrink-0 cursor-pointer rounded border bg-transparent"
        />
        <Input
          aria-label={`${label} hex`} value={value} aria-invalid={error ? true : undefined}
          onChange={(e) => edits.setRole(role, e.target.value)} className="h-8 font-mono text-xs"
        />
      </div>
      <div className="mt-1 flex flex-wrap items-center gap-2">
        <ContrastBadge kit={kit} role={role} />
        {canEdit && source === 'set' && (
          <Button
            type="button" size="sm" variant="ghost" className="h-6 gap-1 px-1.5 text-[11px]"
            disabled={edits.resetting !== null} onClick={() => void edits.resetRole(role)}
          >
            {edits.resetting === role ? <Loader2 className="h-3 w-3 animate-spin" aria-hidden /> : <RotateCcw className="h-3 w-3" aria-hidden />}
            Reset to derived
          </Button>
        )}
      </div>
      {error && <p className="mt-1 text-xs text-destructive" role="alert">{error}</p>}
    </div>
  )
}

interface BrandKitColoursProps {
  kit: BrandKit
  edits: BrandPaletteEdits
  canEdit: boolean
  patch: (p: Partial<BrandKit>) => void
}

export function BrandKitColours({ kit, edits, canEdit, patch }: BrandKitColoursProps) {
  return (
    <section aria-label="Colours">
      <SectionTitle title="Colours">
        Each colour has a job. A derived role follows the four colours above; set one to pin it.
      </SectionTitle>
      <div className="grid grid-cols-1 gap-2 sm:grid-cols-2 xl:grid-cols-3">
        {PALETTE_ROLES.map((role) => <RoleRow key={role} kit={kit} role={role} edits={edits} canEdit={canEdit} />)}
      </div>
      <div className="mt-3 max-w-md">
        <Label htmlFor="brand-accent-use" className="text-xs">How far the accent goes</Label>
        <select
          id="brand-accent-use" className={SELECT_CLASS} value={kit.accent_use ?? DEFAULT_ACCENT_USE}
          onChange={(e) => patch({ accent_use: e.target.value as BrandAccentUse })}
        >
          {(Object.keys(ACCENT_USE_LABELS) as BrandAccentUse[]).map((use) => (
            <option key={use} value={use}>{ACCENT_USE_LABELS[use]}</option>
          ))}
        </select>
      </div>
    </section>
  )
}
