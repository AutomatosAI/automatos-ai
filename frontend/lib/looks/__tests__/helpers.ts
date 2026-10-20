import { MOTION_ATTRIBUTE } from '@/lib/looks/document'
import { LOOK_VAR_NAMES, STUDIO_TOKEN_NAMES } from '@/lib/looks/engine'
import { EMBER } from '@/lib/looks/presets'
import type { Look } from '@/lib/looks/types'
import { BUILTIN_PLAY_HTML_CLASS as PLAY_HTML_CLASS } from '@/lib/theme-host/builtin/play-entry'
import { STUDIO_HTML_CLASS } from '@/lib/ui-style'

/** A Storage kept in memory, for tests that shouldn't share jsdom's localStorage. */
export function memoryStorage(initial: Readonly<Record<string, string>> = {}): Storage {
  const items = new Map(Object.entries(initial))
  return {
    get length() {
      return items.size
    },
    clear: () => items.clear(),
    getItem: (key: string) => items.get(key) ?? null,
    key: (index: number) => Array.from(items.keys())[index] ?? null,
    removeItem: (key: string) => {
      items.delete(key)
    },
    setItem: (key: string, value: string) => {
      items.set(key, String(value))
    },
  }
}

/** Storage that refuses every access, like a sandboxed frame or blocked site data. */
export function blockedStorage(): Storage {
  throw new DOMException('The operation is insecure.', 'SecurityError')
}

/** A valid saved look of the user's own. */
export function myLook(fields: Partial<Look> = {}): Look {
  return { ...EMBER, id: 'my-look-1', name: 'My look 1', sub: 'Saved just now', accent: 'rose', ground: 'indigo', ...fields }
}

/** Play's variables and Studio's tokens currently inline on <html> ('' where unset). */
export function rootVars(): Readonly<Record<string, string>> {
  const { style } = document.documentElement
  return Object.fromEntries(
    [...LOOK_VAR_NAMES, ...STUDIO_TOKEN_NAMES].map((name) => [name, style.getPropertyValue(name).trim()]),
  )
}

/** The look's names that are still set inline on <html>. */
export function lookNamesLeft(): readonly string[] {
  return Object.entries(rootVars()).filter(([, value]) => value !== '').map(([name]) => name)
}

/** <html> as a fresh page has it: no inline style, no motion mark, no style or tone class. */
export function clearRoot(): void {
  const root = document.documentElement
  root.removeAttribute('style')
  root.removeAttribute(MOTION_ATTRIBUTE)
  root.classList.remove(PLAY_HTML_CLASS, STUDIO_HTML_CLASS, 'light', 'dark')
}
