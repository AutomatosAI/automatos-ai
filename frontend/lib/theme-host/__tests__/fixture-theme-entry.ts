/**
 * PRD-261 A-19b — a neutral theme for the core's tests of the host: an installed theme's registry
 * entry and what its code registered, for every host test that needs "a theme". No theme is
 * compiled into the app (A-19f1), so a host test drives the host through an installed theme.
 *
 * The entry is what installedThemeEntry (lib/theme-host/installed-entry.ts) makes of a boot's entry
 * data: kind 'installed', the host's frame width as its minWidth. It lists every slot the host
 * draws a theme's screen in (frame, lookEditor, styleSheet, notFound, asleep), owns the pages /chat
 * and /deliverables, and has one route of its own, `fixture-route`, at /x/fixture-route. Its <html>
 * class is `fixture`.
 *
 * An installed theme's registrations reach the slots through the theme host's context, never from
 * the entry, so a test under this entry also provides the host's value (fixtureThemeHost). Each
 * registered component draws one element with a test id of FIXTURE_THEME_TEST_IDS; the frame
 * draws the page and Auto's widget MainLayout hands it, with no wrapper of its own around the page.
 *
 * Plain .ts (no JSX), as the boot harness beside it: it is a helper, never a test file.
 */
import { createElement, type ComponentType } from 'react'
import { vi } from 'vitest'

import type { ThemeHostContextValue } from '@/components/theme-host/theme-host-context'
import { FRAME_SLOT } from '@/lib/theme-host/constants'
import { INSTALLED_FRAME_MIN_WIDTH_PX } from '@/lib/theme-host/installed-entry'
import type { SlotId, ThemeEntry, ThemeRef, UiStyle } from '@/lib/theme-host/registry-types'
import type { FrameProps, ThemeHostStatus, ThemeRegistrations } from '@/lib/theme-host/types'
import { CHAT_HREF, DELIVERABLES_HREF } from '@/lib/ticket-links'

export const FIXTURE_THEME_REF: ThemeRef = 'install:00000000-0000-4000-8000-00000000a19b'
export const FIXTURE_THEME_STYLE: UiStyle = `theme:${FIXTURE_THEME_REF}`
export const FIXTURE_THEME_ID = 'automatos-e2e/fixture'
export const FIXTURE_THEME_HTML_CLASS = 'fixture'
/** The fixture's own route (registerRoute), served at /x/fixture-route. */
export const FIXTURE_THEME_ROUTE = 'fixture-route'
export const FIXTURE_THEME_ROUTE_HREF = `/x/${FIXTURE_THEME_ROUTE}`

/** Every slot the fixture lists and registers, frame@1 included. */
export const FIXTURE_THEME_SLOTS: readonly SlotId[] = [FRAME_SLOT, 'lookEditor@1', 'styleSheet@1', 'notFound@1', 'asleep@1']

/** The test id of each element the fixture's registered components draw. */
export const FIXTURE_THEME_TEST_IDS = {
  frame: 'fixture-frame',
  chat: 'fixture-chat',
  deliverables: 'fixture-deliverables',
  lookEditor: 'fixture-look-editor',
  styleSheet: 'fixture-style-sheet',
  notFound: 'fixture-not-found',
  asleep: 'fixture-asleep',
  route: 'fixture-route',
} as const

export const FIXTURE_THEME_ENTRY: ThemeEntry = {
  ref: FIXTURE_THEME_REF,
  name: 'Fixture theme',
  base: 'studio',
  htmlClass: FIXTURE_THEME_HTML_CLASS,
  minWidth: INSTALLED_FRAME_MIN_WIDTH_PX,
  tone: 'user',
  kind: 'installed',
  reviewed: false,
  ownsPages: [CHAT_HREF, DELIVERABLES_HREF],
  routes: [FIXTURE_THEME_ROUTE],
  nav: [{ id: 'nav-fixture-route', label: 'Fixture route', path: FIXTURE_THEME_ROUTE_HREF }],
  slots: FIXTURE_THEME_SLOTS,
  themeId: FIXTURE_THEME_ID,
}

/** frame@1: the page, then Auto's widget, inside one element and nothing else. */
function FixtureFrame({ children, fullBleed, autoWidget }: FrameProps) {
  return createElement(
    'section',
    { 'data-testid': FIXTURE_THEME_TEST_IDS.frame, 'data-full-bleed': String(Boolean(fullBleed)) },
    children,
    autoWidget,
  )
}

/** A registered screen: one element with its test id and a line of text. */
function screenOf(testId: string, text: string): ComponentType {
  function FixtureScreen() {
    return createElement('div', { 'data-testid': testId }, text)
  }
  return FixtureScreen
}

const ids = FIXTURE_THEME_TEST_IDS

/** What the fixture's register(host) registered, as the theme host hands it over once loaded. */
export const FIXTURE_THEME_REGISTRATIONS: ThemeRegistrations = {
  frame: FixtureFrame,
  slots: new Map<SlotId, unknown>([
    [FRAME_SLOT, FixtureFrame],
    ['lookEditor@1', screenOf(ids.lookEditor, "The fixture theme's look editor")],
    ['styleSheet@1', screenOf(ids.styleSheet, "The fixture theme's style sheet")],
    ['notFound@1', screenOf(ids.notFound, "The fixture theme's 404")],
    ['asleep@1', screenOf(ids.asleep, "The fixture theme's asleep screen")],
  ]),
  pages: new Map<string, unknown>([
    [CHAT_HREF, screenOf(ids.chat, "The fixture theme's chat")],
    [DELIVERABLES_HREF, screenOf(ids.deliverables, "The fixture theme's Deliverables")],
  ]),
  routes: new Map<string, unknown>([[FIXTURE_THEME_ROUTE, screenOf(ids.route, "The fixture theme's route")]]),
}

/**
 * The theme host's value under the fixture, as the provider gives it: its registrations while it
 * is active (each part replaceable through `parts`), none while it loads, is off, failed or in
 * safe mode. `reportRenderFailure` is a fresh mock each time.
 */
export function fixtureThemeHost(
  status: ThemeHostStatus = 'active',
  parts: Partial<ThemeRegistrations> = {},
): ThemeHostContextValue {
  const reason = status === 'failed' ? 'load' : null
  if (status !== 'active') {
    return { status, reason, frame: null, slots: null, pages: null, routes: null, reportRenderFailure: vi.fn() }
  }
  const registrations = { ...FIXTURE_THEME_REGISTRATIONS, ...parts }
  return {
    status,
    reason,
    frame: registrations.frame,
    slots: registrations.slots,
    pages: registrations.pages,
    routes: registrations.routes,
    reportRenderFailure: vi.fn(),
  }
}
