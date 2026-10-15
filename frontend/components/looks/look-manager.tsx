'use client'

import { useEffect, useId, useRef, useState } from 'react'

import { asDuplicate, baseLook, isEdited, isSavedLook } from '@/lib/looks/editor'
import { DEFAULT_LOOK } from '@/lib/looks/presets'
import { LOOK_FILE_TYPE, lookFileName, lookJson } from '@/lib/looks/transfer'
import type { Look } from '@/lib/looks/types'
import { copyToClipboard } from '@/lib/utils'

import { downloadText } from './download'
import { DeleteConfirm, RenameForm } from './look-manage-forms'
import { LookNotice, type Notice, failed } from './look-notice'
import { useLook } from './look-provider'
import styles from './manage.module.css'

/**
 * PRD-260 — 'This look' on the Looks page, ported from Automatos Play's
 * components/looks/LookManager.tsx: the picked look's own actions. A saved
 * look can be saved with its edits, renamed, duplicated and deleted; a preset
 * can only be duplicated (or kept with 'Save as new look' in the editor).
 * Duplicate, Export and Copy JSON take the active look as it is now, edits
 * included; Save writes those edits over the saved look it was picked from;
 * Rename and Delete act on that saved look. Everything goes through useLook(),
 * so every page follows at once. When the control just used leaves the page
 * (Save, Save name, Cancel, Keep it, Delete look), focus goes to a toolbar
 * button that stays, never to <body> (P260-FX-22, M20).
 */

export const MANAGE_LABEL = 'This look'
export const SAVE_LABEL = 'Save'
export const RENAME_LABEL = 'Rename'
export const DUPLICATE_LABEL = 'Duplicate'
export const DELETE_LABEL = 'Delete'
export const EXPORT_LABEL = 'Export'
export const COPY_JSON_LABEL = 'Copy JSON'

type Mode = 'rename' | 'delete'

/** The toolbar's buttons, each marked data-tool so focus can be handed to one. */
type Tool = 'save' | 'rename' | 'duplicate' | 'delete' | 'export' | 'copy-json'

/** A message belongs to one look: picking another look hides it. */
interface ForLook<T> {
  readonly id: string
  readonly value: T
}

type Tell = (id: string, notice: Notice) => void

type ReturnFocus = (tool: Tool) => void

/**
 * Focus to a toolbar button after the re-render that removed the control just
 * used: an effect, since the button may only exist once that render is done.
 */
function useReturnFocus() {
  const toolbar = useRef<HTMLDivElement>(null)
  const [focusTo, setFocusTo] = useState<Tool | null>(null)
  useEffect(() => {
    if (focusTo === null) return
    toolbar.current?.querySelector<HTMLElement>(`[data-tool="${focusTo}"]`)?.focus()
    setFocusTo(null)
  }, [focusTo])
  const returnFocus: ReturnFocus = setFocusTo
  return { toolbar, returnFocus }
}

/** Export and Copy JSON: the active look as a .look.json file, or as text on the clipboard. */
function useTransfer(look: Look, tell: Tell) {
  const exportFile = () => {
    const fileName = lookFileName(look)
    try {
      downloadText(fileName, lookJson(look), LOOK_FILE_TYPE)
      tell(look.id, { kind: 'done', text: `Downloaded ${fileName}.` })
    } catch (error) {
      console.error('[looks] export failed', error)
      tell(look.id, failed("That look wasn't exported.", ["The download didn't start. Use Copy JSON instead."]))
    }
  }

  // lib/utils' copyToClipboard logs a refusal and answers false.
  const copyJson = async () => {
    if (await copyToClipboard(lookJson(look))) {
      tell(look.id, { kind: 'done', text: 'Copied the JSON.' })
      return
    }
    tell(look.id, failed("The JSON wasn't copied.", ['The browser blocked the clipboard. Use Export instead.']))
  }

  return { exportFile, copyJson }
}

/**
 * Save, duplicate, rename and delete, each through the LookProvider so the whole app follows.
 * Save and a rename hand focus to Rename (Save is gone once saved, the form once renamed);
 * a delete to Duplicate, the one button every look has.
 */
function useManage(tell: Tell, closeForm: () => void, returnFocus: ReturnFocus) {
  const { look, saved, setLook, saveLook, deleteLook, renameLook } = useLook()

  const save = (edited: Look) => {
    const result = saveLook(edited)
    if (!result.ok) return tell(edited.id, failed("That look wasn't saved.", result.errors))
    setLook(result.look)
    returnFocus('rename')
    tell(result.look.id, { kind: 'done', text: `Saved ${result.look.name}.` })
  }

  const duplicate = (from: Look) => {
    const result = saveLook(asDuplicate(from, saved))
    if (!result.ok) return tell(look.id, failed("That look wasn't duplicated.", result.errors))
    setLook(result.look)
    tell(result.look.id, { kind: 'done', text: `Duplicated as ${result.look.name}.` })
  }

  const rename = (from: Look, name: string) => {
    const result = renameLook(from.id, name)
    if (!result.ok) return tell(from.id, failed("That name wasn't saved.", result.errors))
    closeForm()
    returnFocus('rename')
    tell(from.id, { kind: 'done', text: `Renamed to ${result.look.name}.` })
  }

  // Deleting the active look takes Play back to the default (lib/looks/store.ts).
  const remove = (from: Look) => {
    deleteLook(from.id)
    closeForm()
    returnFocus('duplicate')
    tell(DEFAULT_LOOK.id, { kind: 'done', text: `Deleted ${from.name}. Play is back to ${DEFAULT_LOOK.name}.` })
  }

  return { save, duplicate, rename, remove }
}

export function LookManager() {
  const { look, saved } = useLook()
  const [mode, setMode] = useState<Mode | null>(null)
  const [notice, setNotice] = useState<ForLook<Notice> | null>(null)
  const [shownFor, setShownFor] = useState(look.id)
  const headingId = useId()

  // Another look picked: an open form, or a message about the old one, goes away for good
  // (reset while rendering, React's pattern for state that follows a prop).
  if (shownFor !== look.id) {
    setShownFor(look.id)
    setMode(null)
    if (notice !== null && notice.id !== look.id) setNotice(null)
  }

  const tell: Tell = (id, value) => setNotice({ id, value })
  const close = () => setMode(null)
  const open = (value: Mode) => {
    setNotice(null)
    setMode(value)
  }
  const { toolbar, returnFocus } = useReturnFocus()
  // Cancel hands focus back to Rename, Keep it to Delete: the buttons that opened them.
  const closeTo = (tool: Tool) => {
    close()
    returnFocus(tool)
  }
  const { save, duplicate, rename, remove } = useManage(tell, close, returnFocus)
  const { exportFile, copyJson } = useTransfer(look, tell)

  const target = baseLook(look, saved)
  // Save, Rename and Delete need one of the user's saved looks; a preset has none of them.
  const own = target !== undefined && isSavedLook(target, saved) ? target : undefined
  const unsaved = own !== undefined && isEdited(look, saved)

  return (
    <section className={styles.manage} aria-labelledby={headingId}>
      <h2 id={headingId} className={styles.label}>
        {MANAGE_LABEL}
      </h2>
      <div ref={toolbar} className={styles.toolbar}>
        {unsaved && <ToolButton tool="save" label={SAVE_LABEL} onClick={() => save(look)} />}
        {own && <ToolButton tool="rename" label={RENAME_LABEL} onClick={() => open('rename')} />}
        <ToolButton tool="duplicate" label={DUPLICATE_LABEL} onClick={() => duplicate(look)} />
        {own && <ToolButton tool="delete" label={DELETE_LABEL} onClick={() => open('delete')} />}
        <ToolButton tool="export" label={EXPORT_LABEL} onClick={exportFile} />
        <ToolButton tool="copy-json" label={COPY_JSON_LABEL} onClick={() => void copyJson()} />
      </div>
      {own && mode === 'rename' && (
        <RenameForm key={own.id} look={own} onRename={(name) => rename(own, name)} onCancel={() => closeTo('rename')} />
      )}
      {own && mode === 'delete' && (
        <DeleteConfirm look={own} onDelete={() => remove(own)} onCancel={() => closeTo('delete')} />
      )}
      <LookNotice notice={notice?.id === look.id ? notice.value : null} />
    </section>
  )
}

interface ToolButtonProps {
  readonly tool: Tool
  readonly label: string
  readonly onClick: () => void
}

function ToolButton({ tool, label, onClick }: ToolButtonProps) {
  return (
    <button type="button" className={styles.action} onClick={onClick} data-tool={tool}>
      {label}
    </button>
  )
}
