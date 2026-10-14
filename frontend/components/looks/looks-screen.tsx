'use client'

import { useState } from 'react'
import Link from 'next/link'

import { asNewLook, baseLook, statusLine, withValues, type LookValues } from '@/lib/looks/editor'
import { PRESETS } from '@/lib/looks/presets'
import type { Look } from '@/lib/looks/types'

import { LookEditor } from './look-editor'
import { LookImport } from './look-import'
import { LookManager } from './look-manager'
import { LookPreview } from './look-preview'
import { useLook } from './look-provider'
import { LooksRow } from './looks-row'
import styles from './looks.module.css'

/**
 * PRD-260 §4 — Make it yours (/looks, Ember looks-01), ported from Automatos
 * Play's components/looks/LooksScreen.tsx: the heading with the active look's
 * name, a live preview with Auto, the Looks row (the five presets, then the
 * user's saved looks), the picked look's actions (Save, Rename, Duplicate,
 * Delete, Export, Copy JSON), Import, and the editor over EDITABLE_FIELDS.
 * Every change goes through useLook().setLook, which re-themes the whole app
 * at once, not only the preview. 'See the style sheet' opens /looks/style.
 */

/** The page's name, and what /looks says under Classic or Studio (K9: app/looks/page.tsx). */
export const LOOKS_TITLE = 'Make it yours'
export const LOOKS_PLAY_ONLY_NOTE = 'Make it yours is part of the Play theme.'

export const LOOKS_HEADING = 'Make it yours.'
export const LOOKS_LEAD = 'Pick a look, then tweak it. Auto changes with it.'
export const STYLE_SHEET_LINK = 'See the style sheet'
export const STYLE_SHEET_HREF = '/looks/style'

/** The editor's state and actions: picking, editing, Save as new look and Reset. */
function useLookEditing() {
  const { look, saved, setLook, saveLook } = useLook()
  const [errors, setErrors] = useState<readonly string[]>([])

  const pick = (next: Look) => {
    setErrors([])
    setLook(next)
  }

  const change = (values: Partial<LookValues>) => setLook(withValues(look, values))

  const saveAsNew = () => {
    const result = saveLook(asNewLook(look, saved))
    if (result.ok) {
      setErrors([])
      setLook(result.look)
      return
    }
    console.warn('[looks] Save as new look was refused:', result.errors)
    setErrors(result.errors)
  }

  const base = baseLook(look, saved)
  const reset = () => {
    setErrors([])
    if (base) setLook(base)
  }

  return { look, saved, errors, pick, change, saveAsNew, reset, canReset: base !== undefined }
}

export function LooksScreen() {
  const { look, saved, errors, pick, change, saveAsNew, reset, canReset } = useLookEditing()
  return (
    <div className={styles.screen}>
      <div className={styles.main}>
        <div className={styles.header}>
          <div className={styles.headingText}>
            <h1 className={styles.heading}>{LOOKS_HEADING}</h1>
            <p className={styles.lead}>{LOOKS_LEAD}</p>
          </div>
          <div className={styles.headerSide}>
            <p className={styles.status} role="status">
              {statusLine(look, saved)}
            </p>
            <Link href={STYLE_SHEET_HREF} className={styles.styleLink}>
              {STYLE_SHEET_LINK}
            </Link>
          </div>
        </div>
        <LookPreview />
        <LooksRow looks={[...PRESETS, ...saved]} activeId={look.id} onPick={pick} />
        <LookManager />
        <LookImport />
      </div>
      <LookEditor
        look={look}
        onChange={change}
        onSaveAs={saveAsNew}
        onReset={reset}
        canReset={canReset}
        errors={errors}
      />
    </div>
  )
}
