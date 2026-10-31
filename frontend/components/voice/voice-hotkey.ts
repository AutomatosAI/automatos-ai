'use client'

/**
 * Theme voice — a voice control's hotkey (Space by default, as a KeyboardEvent.code): pressed anywhere on the page it
 * presses the control, held it holds, except while the member types (the composer, any input, select or editable
 * text) or while focus is on another control the key would press (a button, a link, a menu item). A control that
 * already handled the key (preventDefault) is left alone, so a focused voice button is never pressed twice. Letting
 * go of the key, or leaving the window while it is down, ends the hold.
 */
import { useEffect, useRef } from 'react'

import type { VoicePress } from './voice-press'

export const DEFAULT_VOICE_HOTKEY = 'Space'

/** Where the member types: a keypress there is theirs, never the hotkey. */
const TYPING_TARGETS = 'input, textarea, select, [contenteditable]:not([contenteditable="false"]), [role="textbox"]'
/** Controls the key itself presses when focused. */
const KEY_TARGETS =
  'button, a[href], summary, [role="button"], [role="link"], [role="menuitem"], [role="option"], [role="tab"], [role="checkbox"], [role="switch"]'

/** The key belongs to what has focus: a field the member types in, or a control the key presses. */
export function keyBelongsToFocus(target: EventTarget | null): boolean {
  if (!(target instanceof Element)) return false
  return target.closest(`${TYPING_TARGETS}, ${KEY_TARGETS}`) !== null
}

function isHotkeyPress(event: KeyboardEvent, hotkey: string): boolean {
  if (event.code !== hotkey || event.defaultPrevented) return false
  if (event.altKey || event.ctrlKey || event.metaKey) return false
  return !keyBelongsToFocus(event.target)
}

/** Presses `press` on `hotkey` (null: no hotkey) while `enabled`. */
export function useVoiceHotkey(hotkey: string | null, press: Pick<VoicePress, 'down' | 'up'>, enabled: boolean): void {
  const latest = useRef(press)
  useEffect(() => {
    latest.current = press
  }, [press])
  useEffect(() => {
    if (hotkey === null || !enabled) return undefined
    const onKeyDown = (event: KeyboardEvent) => {
      if (!isHotkeyPress(event, hotkey)) return
      event.preventDefault()
      if (!event.repeat) latest.current.down()
    }
    const onKeyUp = (event: KeyboardEvent) => {
      if (event.code === hotkey) latest.current.up()
    }
    const onBlur = () => latest.current.up()
    document.addEventListener('keydown', onKeyDown)
    document.addEventListener('keyup', onKeyUp)
    window.addEventListener('blur', onBlur)
    return () => {
      document.removeEventListener('keydown', onKeyDown)
      document.removeEventListener('keyup', onKeyUp)
      window.removeEventListener('blur', onBlur)
    }
  }, [hotkey, enabled])
}
