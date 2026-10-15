'use client'

import { type ChangeEvent, type ReactNode, useId } from 'react'

import {
  ACCENTS,
  ACCENT_KEYS,
  GROUNDS,
  GROUND_KEYS,
  LIGHTS,
  LIGHT_KEYS,
  SHELLS,
  SHELL_KEYS,
  TYPE_STYLES,
  TYPE_STYLE_KEYS,
} from '@/lib/looks/catalogue'
import { safeGlow } from '@/lib/looks/css'
import { clampGlow, glowLabel, type LookValues } from '@/lib/looks/editor'
import { GLOW_MIN, type Look } from '@/lib/looks/types'

import styles from './editor.module.css'
import { colourStyle, typeStyle } from './swatch-style'

/**
 * PRD-260 — the editor's option groups (Ember looks-01), ported from Automatos
 * Play's components/looks/LookOptions.tsx: accent swatches, background, type,
 * Auto's shell, Auto's light and the glow slider. Each option is a toggle
 * button (aria-pressed) named after its catalogue entry; each group is named
 * by its visible label. The values are lib/looks/editor.ts's EDITABLE_FIELDS.
 */

export const TYPE_SAMPLE = 'Laptop shortlist'
/** The light colours Auto's ring (and the connected dot); his eyes are part of the clips, so the shell tints them (F12). */
export const LIGHT_LABEL = "Auto's light · ring"
const GLOW_STEP = 1

export interface OptionsProps {
  readonly look: Look
  readonly onChange: (values: Partial<LookValues>) => void
}

interface OptionGroupProps {
  readonly label: string
  /** The class that lays the options out. */
  readonly layout?: string
  readonly children: ReactNode
}

function OptionGroup({ label, layout, children }: OptionGroupProps) {
  const labelId = useId()
  return (
    <div className={styles.section} role="group" aria-labelledby={labelId}>
      <p id={labelId} className={styles.label}>
        {label}
      </p>
      <div className={layout}>{children}</div>
    </div>
  )
}

export function AccentOptions({ look, onChange }: OptionsProps) {
  return (
    <OptionGroup label={`Accent · ${ACCENTS[look.accent].name}`} layout={styles.swatches}>
      {ACCENT_KEYS.map((key) => (
        <button
          key={key}
          type="button"
          className={styles.swatch}
          style={colourStyle(ACCENTS[key].hex)}
          aria-label={ACCENTS[key].name}
          aria-pressed={key === look.accent}
          onClick={() => onChange({ accent: key })}
        />
      ))}
    </OptionGroup>
  )
}

export function GroundOptions({ look, onChange }: OptionsProps) {
  return (
    <OptionGroup label="Background" layout={styles.grounds}>
      {GROUND_KEYS.map((key) => (
        <button
          key={key}
          type="button"
          className={styles.option}
          aria-pressed={key === look.ground}
          onClick={() => onChange({ ground: key })}
        >
          <span className={styles.groundSwatch} style={colourStyle(GROUNDS[key].bg)} aria-hidden="true" />
          {GROUNDS[key].name}
        </button>
      ))}
    </OptionGroup>
  )
}

/** Each type style is set in its own face; the button's name is the style's name. */
export function TypeOptions({ look, onChange }: OptionsProps) {
  return (
    <OptionGroup label="Type" layout={styles.types}>
      {TYPE_STYLE_KEYS.map((key) => (
        <button
          key={key}
          type="button"
          className={styles.typeOption}
          aria-pressed={key === look.font}
          onClick={() => onChange({ font: key })}
        >
          <span className={styles.typeSample} style={typeStyle(key)} aria-hidden="true">
            {TYPE_SAMPLE}
          </span>
          <span className={styles.typeName}>{TYPE_STYLES[key].name}</span>
        </button>
      ))}
    </OptionGroup>
  )
}

export function ShellOptions({ look, onChange }: OptionsProps) {
  return (
    <OptionGroup label="Auto's shell" layout={styles.chips}>
      {SHELL_KEYS.map((key) => (
        <button
          key={key}
          type="button"
          className={styles.chip}
          aria-pressed={key === look.shell}
          onClick={() => onChange({ shell: key })}
        >
          <span className={styles.dot} style={colourStyle(SHELLS[key].hex)} aria-hidden="true" />
          {SHELLS[key].name}
        </button>
      ))}
    </OptionGroup>
  )
}

export function LightOptions({ look, onChange }: OptionsProps) {
  return (
    <OptionGroup label={LIGHT_LABEL} layout={styles.chips}>
      {LIGHT_KEYS.map((key) => (
        <button
          key={key}
          type="button"
          className={styles.chip}
          aria-pressed={key === look.light}
          onClick={() => onChange({ light: key })}
        >
          <span className={styles.lightDot} style={colourStyle(LIGHTS[key].hex)} aria-hidden="true" />
          {LIGHTS[key].name}
        </button>
      ))}
    </OptionGroup>
  )
}

/**
 * GLOW_MIN to the look's safe glow (GLOW_MAX at most: the strongest glow its
 * ground and accent keep readable), 'Off' at 0. It shows the glow the page
 * paints, so a stored glow past the cap reads as the cap. The value is read
 * out as '24%' or 'Off'.
 */
export function GlowSlider({ look, onChange }: OptionsProps) {
  const inputId = useId()
  const max = safeGlow(look.ground, look.accent)
  const glow = clampGlow(look.glow, max)
  const value = glowLabel(glow)
  const onInput = (event: ChangeEvent<HTMLInputElement>) => onChange({ glow: clampGlow(Number(event.target.value), max) })
  return (
    <div className={styles.section}>
      <div className={styles.glowHead}>
        <label htmlFor={inputId} className={styles.label}>
          Glow
        </label>
        {/* Screen readers hear the value from the slider's aria-valuetext, so this is shown only. */}
        <span className={styles.label} aria-hidden="true">
          {value}
        </span>
      </div>
      <input
        id={inputId}
        type="range"
        className={styles.slider}
        min={GLOW_MIN}
        max={max}
        step={GLOW_STEP}
        value={glow}
        aria-valuetext={value}
        onChange={onInput}
      />
    </div>
  )
}
