/**
 * PRD-261 W-A6b (gap G1) — the root layout hands Providers an installed theme's registry entry,
 * built from the boot the server resolved for the style (lib/theme-host/installed-entry.ts), so the
 * server render and the client's first render name the same active theme. A base boot (the theme
 * may not run, or the boot failed) and every other style hand none.
 *
 * Gap G2: an installed theme whose look sets the tone starts on the tone its look last wrote to the
 * tone cookie, so Providers forces that tone from the server render on. PRD-261 A-19d2: the generic
 * look runtime's cookie (`automatos-theme-tone`), for a theme whose boot carries its looks and a
 * tone rule. The value names its theme (lib/theme-host/looks/tone-cookie.ts), so another installed
 * theme's tone, left by a switch from it, or the old bare tone is not read.
 *
 * The installed resolver (lib/theme-host/boot-server.ts, which fetches the backend's boot route) is
 * stood in for here by one that answers the boot the test gives; its own tests are boot-server.test.ts.
 */
import { Children, isValidElement, type ReactElement, type ReactNode } from 'react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import type { ThemeBoot } from '@/lib/theme-host/boot'

const cookieJar = vi.hoisted(() => new Map<string, string>())
const installedBoot = vi.hoisted(() => ({ current: null as ThemeBoot | null }))

vi.mock('next/headers', () => ({
  cookies: async () => ({
    get: (name: string) => {
      const value = cookieJar.get(name)
      return value === undefined ? undefined : { name, value }
    },
  }),
}))
// next/font is a compile-time transform; under vitest the local loader is a plain function.
vi.mock('next/font/local', () => ({
  default: ({ variable }: { readonly variable: string }) => ({ variable: `font${variable}`, className: 'font-face', style: {} }),
}))
vi.mock('@/components/providers', () => ({ Providers: ({ children }: { children: ReactNode }) => children }))
vi.mock('@/lib/theme-host/boot-server', async () => {
  const { baseThemeBoot } = await import('@/lib/theme-host/boot')
  return { installedThemeBootResolver: async (style: Parameters<typeof baseThemeBoot>[0]) => installedBoot.current ?? baseThemeBoot(style) }
})

import RootLayout from '../layout'
import { baseThemeBoot } from '@/lib/theme-host/boot'
import { installedThemeEntry } from '@/lib/theme-host/installed-entry'
import { HELLO_ENTRY, helloThemeBoot } from '@/lib/theme-host/looks/__tests__/hello-looks'
import { THEME_TONE_COOKIE, themeToneCookieValue } from '@/lib/theme-host/looks/tone-cookie'
import { UI_STYLE_COOKIE } from '@/lib/ui-style'

import { INSTALLED_PLAY_STYLE, PLAY_PACKAGE_ENTRY } from '@/lib/theme-host/__tests__/installed-entry-fixture'

interface ElementProps {
  readonly children?: ReactNode
  readonly initialUiStyle?: string
  readonly initialTheme?: unknown
  readonly initialTone?: string
}
type LayoutElement = ReactElement<ElementProps>

/** A style that names no theme this build knows: it reads as the default style, Studio. */
const NO_THEME_STYLE = 'theme:install:not-a-uuid'

function childElements(element: LayoutElement): LayoutElement[] {
  return Children.toArray(element.props.children).filter((child): child is LayoutElement => isValidElement(child))
}

/** The props the layout hands Providers (the <body>'s one child) under the style cookie `style`. */
async function providersProps(style: string, cookies: Readonly<Record<string, string>> = {}): Promise<ElementProps | undefined> {
  cookieJar.clear()
  cookieJar.set(UI_STYLE_COOKIE, style)
  Object.entries(cookies).forEach(([name, value]) => cookieJar.set(name, value))
  const html = (await RootLayout({ children: null })) as LayoutElement
  const body = childElements(html).find((child) => child.type === 'body')
  return body === undefined ? undefined : childElements(body)[0]?.props
}

afterEach(() => {
  cookieJar.clear()
  installedBoot.current = null
})

describe("PRD-261 W-A6b — the root layout hands Providers the installed theme's entry", () => {
  it("builds it from the install's boot", async () => {
    installedBoot.current = { ...baseThemeBoot(INSTALLED_PLAY_STYLE), entry: PLAY_PACKAGE_ENTRY }
    const props = await providersProps(INSTALLED_PLAY_STYLE)
    expect(props?.initialUiStyle).toBe(INSTALLED_PLAY_STYLE)
    expect(props?.initialTheme).toEqual(installedThemeEntry(INSTALLED_PLAY_STYLE, PLAY_PACKAGE_ENTRY))
  })

  it('hands none for a base boot: the theme may not run, or its boot failed', async () => {
    expect((await providersProps(INSTALLED_PLAY_STYLE))?.initialTheme).toBeNull()
  })

  it.each(['classic', 'studio', NO_THEME_STYLE])('hands none under %s', async (style) => {
    installedBoot.current = { ...baseThemeBoot(INSTALLED_PLAY_STYLE), entry: PLAY_PACKAGE_ENTRY }
    expect((await providersProps(style))?.initialTheme).toBeNull()
  })
})

describe("PRD-261 W-A6b (gap G2) — an installed theme's look tone from the first render", () => {
  it("hands Providers the tone cookie's tone for a 'from-look' theme whose boot carries its looks and a tone rule", async () => {
    // A theme with looks as the boot route answers it (hello-a09b's table, guard and presets), under Play's package's entry.
    installedBoot.current = helloThemeBoot({ entry: PLAY_PACKAGE_ENTRY })
    const lightFor = (themeId: string) => themeToneCookieValue(themeId, 'light')
    expect((await providersProps(INSTALLED_PLAY_STYLE, { [THEME_TONE_COOKIE]: lightFor(PLAY_PACKAGE_ENTRY.themeId) }))?.initialTone).toBe('light')
    expect((await providersProps(INSTALLED_PLAY_STYLE))?.initialTone).toBeUndefined()
    // Another installed theme's tone (a switch from it reloads into this one) and the old bare tone: none.
    expect((await providersProps(INSTALLED_PLAY_STYLE, { [THEME_TONE_COOKIE]: lightFor(HELLO_ENTRY.themeId) }))?.initialTone).toBeUndefined()
    expect((await providersProps(INSTALLED_PLAY_STYLE, { [THEME_TONE_COOKIE]: 'light' }))?.initialTone).toBeUndefined()
  })
})
