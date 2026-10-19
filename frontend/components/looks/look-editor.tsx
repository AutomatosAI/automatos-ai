'use client'

import type { LookValues } from '@/lib/looks/editor'
import type { Look } from '@/lib/looks/types'

import styles from './editor.module.css'
import { AccentOptions, GlowSlider, GroundOptions, LightOptions, ShellOptions, TypeOptions } from './look-options'

/**
 * PRD-260 — the editor panel on the Looks page (Ember looks-01), ported from
 * Automatos Play's components/looks/LookEditor.tsx. Every control changes the
 * active look at once (and so the whole app, through useLook); 'Save as new
 * look' keeps a named copy, 'Reset' goes back to the picked look's saved values.
 */

export const EDITOR_LABEL = 'Edit the look'
export const SAVE_AS_LABEL = 'Save as new look'
export const RESET_LABEL = 'Reset'
export const NOT_SAVED_TITLE = "That look wasn't saved."

export interface LookEditorProps {
  readonly look: Look
  readonly onChange: (values: Partial<LookValues>) => void
  readonly onSaveAs: () => void
  readonly onReset: () => void
  /** False when the active look has no preset or saved look to go back to. */
  readonly canReset: boolean
  /** Why the last save failed, in plain words. */
  readonly errors: readonly string[]
}

export function LookEditor({ look, onChange, onSaveAs, onReset, canReset, errors }: LookEditorProps) {
  return (
    <aside className={styles.editor} aria-label={EDITOR_LABEL}>
      <AccentOptions look={look} onChange={onChange} />
      <GroundOptions look={look} onChange={onChange} />
      <TypeOptions look={look} onChange={onChange} />
      <ShellOptions look={look} onChange={onChange} />
      <LightOptions look={look} onChange={onChange} />
      <GlowSlider look={look} onChange={onChange} />
      {errors.length > 0 && (
        <div className={styles.errors} role="alert">
          <p className={styles.errorTitle}>{NOT_SAVED_TITLE}</p>
          <ul className={styles.errorList}>
            {errors.map((error, index) => (
              <li key={`${index}-${error}`}>{error}</li>
            ))}
          </ul>
        </div>
      )}
      <div className={styles.actions}>
        <button type="button" className={styles.save} onClick={onSaveAs}>
          {SAVE_AS_LABEL}
        </button>
        <button type="button" className={styles.reset} onClick={onReset} disabled={!canReset}>
          {RESET_LABEL}
        </button>
      </div>
    </aside>
  )
}
