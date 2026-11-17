'use client'

/**
 * PRD-251D US-D301 (E10) — the Look card's look options, under its template gallery: light or
 * dark, the accent strong or sparing, and the logo on or off, for this post only (the brand kit
 * stays as it is). Light or dark shows only for a template built on paper, so not for "Let Auto
 * pick"; the accent and the logo show for every template (look-options-model.ts). The look saves
 * with the draft and is a render input (preview-model.ts), so a change marks the preview out of date.
 */
import type { ReactNode } from 'react'

import type { SocialTemplateSummary } from '@/lib/api-client'
import { Hint, Segmented, type Choice } from './editor-ui'
import {
  lookKeys, lookSegments, withLookChoice, type AccentValue, type LogoValue, type ModeValue, type PostLook,
} from './look-options-model'

export const LOOK_OPTIONS_LABEL = 'Look options'
export const MODE_LABEL = 'Light or dark'
export const ACCENT_LABEL = 'Accent'
export const LOGO_LABEL = 'Logo'
export const LOOK_HINT = 'For this post only: your brand kit stays as it is.'
/** Under the choices, when the chosen template takes no light or dark. */
export const NO_MODE_NOTE = 'This template is set on a dark background or a photo, so it has no light or dark choice.'

export const MODE_CHOICES: ReadonlyArray<Choice<ModeValue>> = [
  { value: 'light', label: 'Light' },
  { value: 'dark', label: 'Dark' },
]
export const ACCENT_CHOICES: ReadonlyArray<Choice<AccentValue>> = [
  { value: 'kit', label: 'As your kit' },
  { value: 'bold', label: 'Strong' },
  { value: 'sparing', label: 'Sparing' },
]
export const LOGO_CHOICES: ReadonlyArray<Choice<LogoValue>> = [
  { value: 'on', label: 'On' },
  { value: 'off', label: 'Off' },
]

function LookRow({ caption, children }: { caption: string; children: ReactNode }) {
  return (
    <div className="flex flex-wrap items-center gap-3">
      <span className="w-24 text-[13px] font-medium text-foreground">{caption}</span>
      {children}
    </div>
  )
}

export interface LookOptionsProps {
  /** The chosen template; null for "Let Auto pick". */
  template: Pick<SocialTemplateSummary, 'looks'> | null
  look: PostLook
  onChange: (look: PostLook) => void
}

export function LookOptions({ template, look, onChange }: LookOptionsProps) {
  const keys = lookKeys(template)
  const pressed = lookSegments(look)
  return (
    <div role="group" aria-label={LOOK_OPTIONS_LABEL} className="flex flex-col gap-2.5 border-t border-border pt-3">
      <Hint>{LOOK_HINT}</Hint>
      {keys.includes('mode') && (
        <LookRow caption={MODE_LABEL}>
          <Segmented label={MODE_LABEL} choices={MODE_CHOICES} value={pressed.mode} onChange={(v) => onChange(withLookChoice(look, 'mode', v))} />
        </LookRow>
      )}
      {keys.includes('accent') && (
        <LookRow caption={ACCENT_LABEL}>
          <Segmented label={ACCENT_LABEL} choices={ACCENT_CHOICES} value={pressed.accent} onChange={(v) => onChange(withLookChoice(look, 'accent', v))} />
        </LookRow>
      )}
      {keys.includes('logo') && (
        <LookRow caption={LOGO_LABEL}>
          <Segmented label={LOGO_LABEL} choices={LOGO_CHOICES} value={pressed.logo} onChange={(v) => onChange(withLookChoice(look, 'logo', v))} />
        </LookRow>
      )}
      {template && !keys.includes('mode') && <Hint>{NO_MODE_NOTE}</Hint>}
    </div>
  )
}
