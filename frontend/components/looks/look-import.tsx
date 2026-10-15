'use client'

import { type ChangeEvent, type FormEvent, useEffect, useId, useRef, useState } from 'react'

import { type Told, useNoticeSeq } from '@/hooks/use-notice-seq'
import { importLook, readLookFile } from '@/lib/looks/transfer'
import type { ValidationResult } from '@/lib/looks/types'

import { LookNotice, type Notice, failed, storedNotice } from './look-notice'
import { useLook } from './look-provider'
import styles from './manage.module.css'

/**
 * PRD-260 — 'Import a look' on the Looks page, ported from Automatos Play's
 * components/looks/LookImport.tsx. Pasted JSON or a chosen .json file is
 * untrusted: it goes through lib/looks/transfer.ts (the size check before a
 * byte is read, JSON, validateLook, a fresh id). A valid look is added to the
 * saved looks and picked; an invalid one adds nothing and lists transfer's
 * reasons in words. The success says when this browser isn't keeping looks
 * (LookNotice, M98). Each outcome is numbered (useNoticeSeq), so the same
 * import or the same refusal twice is announced again (M22).
 */

export const IMPORT_LABEL = 'Import a look'
export const PASTE_LABEL = "Paste a Look's JSON"
export const FILE_LABEL = 'Or choose a .json file'
export const IMPORT_PASTED = 'Import pasted JSON'
export const NOT_ADDED = "That look wasn't added."

const PASTE_ROWS = 4
const FILE_ACCEPT = '.json,application/json'

/** The paste box's text, the last outcome, and the two ways in. */
function useImport() {
  const { saved, saveLook, setLook } = useLook()
  const [text, setText] = useState('')
  const [notice, setNotice] = useState<Told<Notice> | null>(null)
  const stamp = useNoticeSeq()
  // Reading a file takes a moment: its fresh id must come from the saved looks after the read.
  const savedNow = useRef(saved)
  useEffect(() => {
    savedNow.current = saved
  }, [saved])

  const add = (imported: ValidationResult): boolean => {
    const result = imported.ok ? saveLook(imported.look) : imported
    if (!result.ok) {
      setNotice(stamp(failed(NOT_ADDED, result.errors)))
      return false
    }
    setLook(result.look)
    setNotice(stamp(storedNotice(`Added ${result.look.name} to your Looks.`)))
    return true
  }

  const importPasted = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault()
    if (add(importLook(text, saved))) setText('')
  }

  const importFile = async (event: ChangeEvent<HTMLInputElement>) => {
    const input = event.currentTarget
    const file = input.files?.[0]
    // Cleared at once, so choosing the same file again (after fixing it) imports it again.
    input.value = ''
    if (!file) return
    const read = await readLookFile(file)
    add(read.ok ? importLook(read.text, savedNow.current) : read)
  }

  return { text, setText, notice, importPasted, importFile }
}

export function LookImport() {
  const { text, setText, notice, importPasted, importFile } = useImport()
  const headingId = useId()
  const pasteId = useId()
  const fileId = useId()

  return (
    <section className={styles.manage} aria-labelledby={headingId}>
      <h2 id={headingId} className={styles.label}>
        {IMPORT_LABEL}
      </h2>
      <form className={styles.form} onSubmit={importPasted}>
        <label htmlFor={pasteId} className={styles.fieldLabel}>
          {PASTE_LABEL}
        </label>
        <textarea
          id={pasteId}
          className={styles.paste}
          value={text}
          onChange={(event) => setText(event.target.value)}
          rows={PASTE_ROWS}
          spellCheck={false}
          autoComplete="off"
        />
        <div className={styles.formActions}>
          <button type="submit" className={styles.action}>
            {IMPORT_PASTED}
          </button>
        </div>
      </form>
      <div className={styles.form}>
        <label htmlFor={fileId} className={styles.fieldLabel}>
          {FILE_LABEL}
        </label>
        <input
          id={fileId}
          type="file"
          accept={FILE_ACCEPT}
          className={styles.file}
          onChange={(event) => void importFile(event)}
        />
      </div>
      <LookNotice notice={notice} />
    </section>
  )
}
