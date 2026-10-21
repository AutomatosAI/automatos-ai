/**
 * PRD-261 A-19d1 — the generic look runtime's tests run on hello-a09b's look data, not on any real
 * theme's: e2e/browser/fixtures/hello-a09b/boot.json is what the boot route answers for that small
 * theme (A-19c: three presets, a ground and an accent knob, a glow), passed through the browser's own
 * gate (boot-sanitize.ts) as the root layout does. Also a Storage kept in memory and one that is
 * blocked, so no test shares jsdom's localStorage.
 *
 * A-19d2: the whole boot of that theme as the root layout writes it into the page (helloThemeBoot:
 * its entry, look keys, guard, table and presets), and the page's JSON element itself (putPageBoot),
 * which the generic look runtime reads after mount.
 */
import { readFileSync } from 'fs'
import path from 'path'

import { THEME_BOOT_ELEMENT_ID, type ThemeBoot } from '@/lib/theme-host/boot'
import { sanitizeBootLooks, sanitizeLookGuard, sanitizePrepaint, sanitizeThemeBoot } from '@/lib/theme-host/boot-sanitize'
import type { InstalledThemeEntryData, ThemeLookStorage } from '@/lib/theme-host/boot-types'
import { themeLookRulesOf, type ThemeLook, type ThemeLookBoot, type ThemeLookRules } from '@/lib/theme-host/looks/spec'
import { bootLookStorage, lookTheme, type LookTheme } from '@/lib/theme-host/looks/storage'
import { scriptSafeJson } from '@/lib/theme-host/script-safe-json'

const HELLO_BOOT_PATH = path.resolve(__dirname, '../../../../..', 'e2e/browser/fixtures/hello-a09b/boot.json')

interface HelloBootFile {
  readonly themeJson: { readonly id: string; readonly look: string }
  readonly prepaint: unknown
  readonly lookGuard: unknown
  readonly looks: unknown
}

/** The fixture as the boot route answers it. */
export const HELLO_BOOT_FILE: HelloBootFile = JSON.parse(readFileSync(HELLO_BOOT_PATH, 'utf8'))

/** The theme's id (theme.json `id`): the looks are kept under it, here and on the server. */
export const HELLO_THEME_ID = HELLO_BOOT_FILE.themeJson.id

/** The boot's look parts as the browser keeps them. */
export function helloBoot(): ThemeLookBoot {
  return {
    prepaint: sanitizePrepaint(HELLO_BOOT_FILE.prepaint),
    lookGuard: sanitizeLookGuard(HELLO_BOOT_FILE.lookGuard),
    looks: sanitizeBootLooks(HELLO_BOOT_FILE.looks),
  }
}

function rulesOf(boot: ThemeLookBoot): ThemeLookRules {
  const rules = themeLookRulesOf(boot)
  if (rules === null) throw new Error("hello-a09b's boot has no look rules")
  return rules
}

export const HELLO_RULES: ThemeLookRules = rulesOf(helloBoot())
export const HELLO: LookTheme = lookTheme(HELLO_THEME_ID, HELLO_RULES)

function preset(id: string): ThemeLook {
  const found = HELLO_RULES.presets.find((look) => look.id === id)
  if (found === undefined) throw new Error(`hello-a09b has no preset "${id}"`)
  return found
}

/** The presets: paper is the default look. */
export const PAPER = preset('paper')
export const DUSK = preset('dusk')
export const INK = preset('ink')

/** A valid saved look of the user's own. */
export function myLook(fields: Readonly<Record<string, string | number>> = {}): ThemeLook {
  return { ...PAPER, id: 'my-look-1', name: 'My look 1', sub: 'Saved just now', ground: 'dusk', accent: 'green', glow: 18, ...fields } as ThemeLook
}

/** hello-a09b's entry as the boot route answers it (serve.py theme_entry): looks, and a tone its look sets. */
export const HELLO_ENTRY: InstalledThemeEntryData = {
  themeId: HELLO_THEME_ID,
  name: 'Hello looks',
  base: 'studio',
  htmlClass: 'hello-a09b',
  minWidth: 0,
  tone: 'from-look',
  reviewed: false,
  ownsPages: [],
  routes: [],
  nav: [],
  slots: ['frame@1', 'lookEditor@1', 'styleSheet@1'],
}

/** hello-a09b's whole boot as the root layout writes it into the page, through the browser's gate. */
export function helloThemeBoot(fields: Partial<ThemeBoot> = {}): ThemeBoot {
  return sanitizeThemeBoot({
    addClasses: ['studio', HELLO_ENTRY.htmlClass],
    removeClasses: [],
    tone: { policy: 'from-look', value: null },
    ...helloBoot(),
    lookStorage: bootLookStorage(HELLO.keys),
    integrity: null,
    entry: HELLO_ENTRY,
    ...fields,
  })
}

/** Puts `boot` into the page as the root layout's JSON element; the returned function takes it out. */
export function putPageBoot(boot: unknown): () => void {
  const element = document.createElement('script')
  element.type = 'application/json'
  element.id = THEME_BOOT_ELEMENT_ID
  element.textContent = scriptSafeJson(boot)
  document.head.appendChild(element)
  return () => element.remove()
}

/** The keys the backend's boot names for an installed theme (orchestrator serve.py look_storage). */
const SHARED_LOOK_STORAGE_PATH = path.resolve(__dirname, '../../../../..', 'orchestrator/tests/fixtures/themes/look-storage.json')

export function sharedLookStorage(): { readonly themeId: string; readonly lookStorage: ThemeLookStorage } {
  return JSON.parse(readFileSync(SHARED_LOOK_STORAGE_PATH, 'utf8'))
}

/** A Storage kept in memory. */
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

/** A Storage that reads but refuses every write, like a full one. */
export function fullStorage(initial: Readonly<Record<string, string>> = {}): Storage {
  const full = memoryStorage(initial)
  full.setItem = () => {
    throw new DOMException('The quota has been exceeded.', 'QuotaExceededError')
  }
  return full
}
