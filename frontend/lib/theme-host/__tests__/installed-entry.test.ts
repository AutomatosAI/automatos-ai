/**
 * PRD-261 W-A6b (gap G1) — an installed theme's registry entry comes from its boot. The boot
 * route's entry data passes its own check with the rest of the boot (a single field that fails its
 * theme.json rule drops the entry, a list item that fails is dropped alone), and the root layout
 * makes it the registry entry of the style's install: its ref, kind 'installed', and the desktop
 * width the host draws an installed theme from in Phase A. Gap G2: the entry names the theme's id
 * (theme.json `id`), which the host's look store keeps the theme's looks under.
 */
import { describe, expect, it } from 'vitest'

import { baseThemeBoot } from '@/lib/theme-host/boot'
import { sanitizeThemeBoot } from '@/lib/theme-host/boot-sanitize'
import {
  INSTALLED_FRAME_MIN_WIDTH_PX,
  installedThemeEntry,
  sanitizeInstalledEntry,
} from '@/lib/theme-host/installed-entry'
import { findThemeEntry } from '@/lib/theme-host/registry'
import type { UiStyle } from '@/lib/theme-host/registry-types'

import { HELLO_PACKAGE_ENTRY, INSTALLED_HELLO_STYLE, INSTALLED_PLAY_STYLE, PLAY_PACKAGE_ENTRY } from './installed-entry-fixture'

const INSTALL_ID = INSTALLED_PLAY_STYLE.slice('theme:install:'.length)
/** A theme style in no installed theme's exact shape: it names no theme this build knows. */
const UNKNOWN_THEME_STYLE: UiStyle = 'theme:install:not-a-uuid'
/** Every style but an installed theme's: Classic, Studio and a theme style that names none. */
const OTHER_STYLES: readonly UiStyle[] = ['classic', 'studio', UNKNOWN_THEME_STYLE]

describe('PRD-261 W-A6b — the boot route entry, checked', () => {
  it("keeps the Play package's entry whole", () => {
    expect(sanitizeInstalledEntry(PLAY_PACKAGE_ENTRY)).toEqual(PLAY_PACKAGE_ENTRY)
    expect(sanitizeInstalledEntry(HELLO_PACKAGE_ENTRY)).toEqual(HELLO_PACKAGE_ENTRY)
  })

  it.each<[string, unknown]>([
    ['themeId', undefined],
    ['themeId', 'play'],
    ['themeId', 'Automatos/Play'],
    ['themeId', 'automatos/play/../x'],
    ['name', ''],
    ['name', 'a'.repeat(61)],
    ['name', 'two\nlines'],
    ['base', 'matte'],
    ['htmlClass', '</style>'],
    ['minWidth', -1],
    ['minWidth', 1.5],
    ['minWidth', 3841],
    ['tone', 'sepia'],
    ['reviewed', 'yes'],
  ])('drops the whole entry when %s is %j', (field, value) => {
    expect(sanitizeInstalledEntry({ ...PLAY_PACKAGE_ENTRY, [field]: value })).toBeNull()
  })

  it('reads nothing that is not an object as an entry', () => {
    for (const value of [null, undefined, 'Play', 7, [PLAY_PACKAGE_ENTRY]]) expect(sanitizeInstalledEntry(value)).toBeNull()
  })

  it('drops a list item that fails its rule, and keeps the rest', () => {
    const entry = sanitizeInstalledEntry({
      ...PLAY_PACKAGE_ENTRY,
      ownsPages: ['/chat', '/settings', '/chat', 7],
      routes: ['conversations', '../up', 'Bad'],
      nav: [...PLAY_PACKAGE_ENTRY.nav, { id: 'x', label: 'Off site', path: 'https://x.test' }, 'nav'],
      slots: ['frame@1', 'header@9', 'lookEditor@1'],
    })
    expect(entry).toMatchObject({
      ownsPages: ['/chat'],
      routes: ['conversations'],
      nav: PLAY_PACKAGE_ENTRY.nav,
      slots: ['frame@1', 'lookEditor@1'],
    })
  })

  it('is part of the boot gate: kept for a theme that runs, null for every other boot', () => {
    const running = sanitizeThemeBoot({ ...baseThemeBoot(INSTALLED_PLAY_STYLE), entry: PLAY_PACKAGE_ENTRY })
    expect(running.entry).toEqual(PLAY_PACKAGE_ENTRY)
    expect(sanitizeThemeBoot({ ...running, entry: { ...PLAY_PACKAGE_ENTRY, tone: 'sepia' } as never }).entry).toBeNull()
    expect(baseThemeBoot(INSTALLED_PLAY_STYLE).entry).toBeNull()
    for (const style of OTHER_STYLES) expect(baseThemeBoot(style).entry).toBeNull()
  })
})

describe("PRD-261 W-A6b — the installed theme's registry entry", () => {
  it("is the install's entry: its ref, kind 'installed' and everything the boot route answered", () => {
    const entry = installedThemeEntry(INSTALLED_PLAY_STYLE, PLAY_PACKAGE_ENTRY)
    expect(entry).toEqual({ ...PLAY_PACKAGE_ENTRY, ref: `install:${INSTALL_ID}`, kind: 'installed' })
    // W-A6b (G2): the id the host's look store keeps the theme's looks under.
    expect(entry?.themeId).toBe('automatos/play')
    expect(findThemeEntry(INSTALLED_PLAY_STYLE, entry === null ? [] : [entry])).toBe(entry)
  })

  it('draws from the desktop width up, whatever minWidth the theme names (Phase A)', () => {
    expect(installedThemeEntry(INSTALLED_HELLO_STYLE, HELLO_PACKAGE_ENTRY)?.minWidth).toBe(INSTALLED_FRAME_MIN_WIDTH_PX)
    expect(installedThemeEntry(INSTALLED_PLAY_STYLE, { ...PLAY_PACKAGE_ENTRY, minWidth: 1440 })?.minWidth).toBe(INSTALLED_FRAME_MIN_WIDTH_PX)
  })

  it('is null for every other style, and for an install whose boot carried no entry', () => {
    for (const style of OTHER_STYLES) expect(installedThemeEntry(style, PLAY_PACKAGE_ENTRY)).toBeNull()
    expect(installedThemeEntry(INSTALLED_PLAY_STYLE, null)).toBeNull()
  })
})
