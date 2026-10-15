'use client'

import { type FormEvent, useId, useState } from 'react'

import type { Look } from '@/lib/looks/types'

import styles from './manage.module.css'

/**
 * PRD-260 — the two small forms of 'This look' on the Looks page, ported from
 * Automatos Play's components/looks/LookManager.tsx (split out so the manager
 * stays small): renaming a saved look, and confirming its deletion. The name
 * typed here goes through validateLook (useLook().renameLook) before it is kept.
 * Each takes focus as it opens: the name field, or Keep it, the safe choice, so
 * the question is read out with it (P260-FX-22, M20).
 */

export const NEW_NAME_LABEL = 'New name'
export const SAVE_NAME_LABEL = 'Save name'
export const DELETE_LOOK_LABEL = 'Delete look'
export const KEEP_IT_LABEL = 'Keep it'

interface RenameFormProps {
  readonly look: Look
  readonly onRename: (name: string) => void
  readonly onCancel: () => void
}

export function RenameForm({ look, onRename, onCancel }: RenameFormProps) {
  const [name, setName] = useState(look.name)
  const fieldId = useId()
  const submit = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault()
    onRename(name)
  }
  return (
    <form className={styles.form} onSubmit={submit}>
      <label htmlFor={fieldId} className={styles.fieldLabel}>
        {NEW_NAME_LABEL}
      </label>
      <input
        id={fieldId}
        type="text"
        className={styles.field}
        value={name}
        onChange={(event) => setName(event.target.value)}
        autoComplete="off"
        autoFocus
      />
      <div className={styles.formActions}>
        <button type="submit" className={styles.action}>
          {SAVE_NAME_LABEL}
        </button>
        <button type="button" className={styles.action} onClick={onCancel}>
          Cancel
        </button>
      </div>
    </form>
  )
}

interface DeleteConfirmProps {
  readonly look: Look
  readonly onDelete: () => void
  readonly onCancel: () => void
}

export function DeleteConfirm({ look, onDelete, onCancel }: DeleteConfirmProps) {
  const textId = useId()
  return (
    <div className={styles.form} role="group" aria-labelledby={textId}>
      <p id={textId} className={styles.confirmText}>
        Delete {look.name}? This can&apos;t be undone. Export it first to keep a copy.
      </p>
      <div className={styles.formActions}>
        <button type="button" className={styles.danger} onClick={onDelete}>
          {DELETE_LOOK_LABEL}
        </button>
        <button type="button" className={styles.action} onClick={onCancel} autoFocus>
          {KEEP_IT_LABEL}
        </button>
      </div>
    </div>
  )
}
