/**
 * PRD-261 A-19f2 — what the host look runtime's tests read off the page, on hello-a09b's table
 * (./hello-looks.ts): the table's variables inline on <html>, the reduced-motion marks, the tone
 * class next-themes puts there and a cookie's value; and <html> as a fresh page has it. They replace
 * the helpers (rootVars, lookNamesLeft, clearRoot) of the look tests A-19f1 deleted with builtin
 * Play; myLook and the storages live in ./hello-looks.ts.
 */
import type { PrepaintTable } from '@/lib/theme-host/boot-types'
import { THEME_MOTION_ATTRIBUTE, THEME_MOTION_VAR } from '@/lib/theme-host/looks/motion'
import { paintNames } from '@/lib/theme-host/looks/paint'
import { STUDIO_HTML_CLASS } from '@/lib/ui-style'

import { HELLO_ENTRY, HELLO_RULES } from './hello-looks'

/** The tone classes next-themes puts on <html> (attribute="class"). */
const TONE_CLASSES: readonly string[] = ['light', 'dark']
/** A cookie written with this Max-Age is deleted. */
const DELETE_COOKIE = 'Max-Age=0; Path=/'

function root(): HTMLElement {
  return document.documentElement
}

/** The table's variables inline on <html> now, trimmed; a variable that isn't set is left out. */
export function rootVars(table: PrepaintTable = HELLO_RULES.table): Readonly<Record<string, string>> {
  const entries = paintNames(table).map((name): [string, string] => [name, root().style.getPropertyValue(name).trim()])
  return Object.fromEntries(entries.filter(([, value]) => value !== ''))
}

/** The table's variables still set inline on <html>. */
export function lookNamesLeft(table: PrepaintTable = HELLO_RULES.table): readonly string[] {
  return Object.keys(rootVars(table))
}

export interface MotionMarks {
  readonly attribute: string | null
  readonly variable: string
}

/** The generic runtime's reduced-motion marks on <html>: data-theme-motion and --theme-motion. */
export function motionMarks(): MotionMarks {
  return { attribute: root().getAttribute(THEME_MOTION_ATTRIBUTE), variable: root().style.getPropertyValue(THEME_MOTION_VAR) }
}

/** The tone class on <html> ('light', 'dark', or '' with none). */
export function htmlTone(): string {
  return TONE_CLASSES.filter((tone) => root().classList.contains(tone)).join(' ')
}

/** A cookie's value on this document, or undefined when it isn't set. */
export function cookieValue(name: string): string | undefined {
  const prefix = `${name}=`
  return document.cookie.split('; ').find((entry) => entry.startsWith(prefix))?.slice(prefix.length)
}

export function dropCookies(names: readonly string[]): void {
  names.forEach((name) => {
    document.cookie = `${name}=; ${DELETE_COOKIE}`
  })
}

/** <html> as a fresh page has it: no inline style, no motion mark, no style, theme or tone class. */
export function clearRoot(): void {
  root().removeAttribute('style')
  root().removeAttribute(THEME_MOTION_ATTRIBUTE)
  root().classList.remove(STUDIO_HTML_CLASS, HELLO_ENTRY.htmlClass, ...TONE_CLASSES)
}
