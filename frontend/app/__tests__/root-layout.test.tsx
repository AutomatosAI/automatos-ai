/**
 * PRD-260 M07 — the root layout's <html className> is the font variables only,
 * the same string for every style, so a server re-render (router.refresh after a
 * client-side style switch) never rewrites the class attribute and drops
 * next-themes' tone class.
 *
 * PRD-261 A-03 — the style's classes, its look and its tone come from the theme
 * boot: <head> is the import map first (on every page, PA-13), then one JSON
 * element holding the style's boot and one constant script that applies it.
 *
 * PRD-261 A-19f1 — no theme is compiled into the app. Under the Classic and the Studio style
 * cookie the first paint carries only that style's class (none for Classic, `studio` for Studio):
 * no `play` class and no font of Play's. <html className> is the core's font variables
 * (app/core-fonts.ts, CORE_FONT_VARIABLE_CLASSES) and nothing else, and a missing or tampered
 * style cookie paints Studio, the default style.
 */
import { Children, isValidElement, type ReactElement, type ReactNode } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import { afterEach, describe, expect, it, vi } from 'vitest'

const cookieJar = vi.hoisted(() => new Map<string, string>())
/** The CSS variable each core face binds, and the name its mocked class carries. */
const FONT_NAMES = vi.hoisted((): Readonly<Record<string, string>> => ({
  '--font-geist-sans': 'geist',
  '--font-geist-mono': 'geist-mono',
  '--font-newsreader': 'newsreader',
}))
vi.mock('next/headers', () => ({
  cookies: async () => ({
    get: (name: string) => {
      const value = cookieJar.get(name)
      return value === undefined ? undefined : { name, value }
    },
  }),
}))
// next/font is a compile-time transform; under vitest the local loader is a plain function, so
// app/core-fonts.ts runs for real and each face's class is named by the CSS variable it binds.
vi.mock('next/font/local', () => {
  const loader = ({ variable }: { readonly variable: string }) => {
    const name = FONT_NAMES[variable] ?? `unknown${variable}`
    return { variable: `font-${name}`, className: `font-${name}-face`, style: {} }
  }
  return { default: loader }
})
vi.mock('@/components/providers', () => ({ Providers: ({ children }: { children: ReactNode }) => children }))

import RootLayout from '../layout'
import { CORE_FONT_VARIABLE_CLASSES } from '@/app/core-fonts'
import { ThemeBootScripts } from '@/components/theme-host/theme-boot'
import { ThemeImportMap } from '@/components/theme-host/theme-import-map'
import { runThemeBoot } from '@/lib/theme-host/__tests__/boot-harness'
import { FIXTURE_THEME_ID } from '@/lib/theme-host/__tests__/fixture-theme-entry'
import { THEME_BOOT_ELEMENT_ID, THEME_BOOT_SCRIPT, baseThemeBoot, type ThemeBoot } from '@/lib/theme-host/boot'
import { IMPORT_MAP_SCRIPT_BODY } from '@/lib/theme-host/import-map'
import { THEME_TONE_COOKIE, themeToneCookieValue } from '@/lib/theme-host/looks/tone-cookie'
import {
  DEFAULT_UI_STYLE,
  STUDIO_HTML_CLASS,
  STYLE_HTML_CLASS_NAMES,
  UI_STYLE_COOKIE,
  UI_STYLES,
  parseUiStyle,
  type UiStyle,
} from '@/lib/ui-style'

interface ElementProps {
  readonly children?: ReactNode
  readonly className?: string
  readonly initialUiStyle?: string
  readonly initialTone?: string
}
type LayoutElement = ReactElement<ElementProps>

const INSTALL_STYLE = 'theme:install:0b7e4c1a-9f3d-4c2e-8a51-3d2f6b7a9c10'
/** Every cookie value the layout must boot: each core style and an installed theme. */
const COOKIES = [...UI_STYLES, INSTALL_STYLE]
/** Values the style axis does not take, each read as the default style. */
const TAMPERED_COOKIES = ['classic;studio', 'theme:install:x', 'theme:other']
/** The class builtin Play put on <html>; no style's first paint carries it. */
const PLAY_CLASS = 'play'
/** The <html> classes app/core-fonts.ts defines, as the mocked loader names them. */
const CORE_FONT_CLASSES = Object.values(FONT_NAMES).map((name) => `font-${name}`)
/** The style classes each core style's first paint carries (A-19f1): Classic none, Studio `studio`. */
const FIRST_PAINT_CLASSES: ReadonlyArray<readonly [UiStyle, readonly string[]]> = [
  ['classic', []],
  ['studio', [STUDIO_HTML_CLASS]],
]
/** A tone the generic look runtime left in its cookie for some theme: no core style reads it. */
const LEFT_TONE_COOKIE = { [THEME_TONE_COOKIE]: themeToneCookieValue(FIXTURE_THEME_ID, 'light') }
const IMPORT_MAP_OPEN = '<script type="importmap">'
const SCRIPT_CLOSE = '</script>'

function childElements(element: LayoutElement): LayoutElement[] {
  return Children.toArray(element.props.children).filter((child): child is LayoutElement => isValidElement(child))
}

async function layoutFor(cookies: Readonly<Record<string, string>>): Promise<LayoutElement> {
  cookieJar.clear()
  Object.entries(cookies).forEach(([name, value]) => cookieJar.set(name, value))
  return (await RootLayout({ children: null })) as LayoutElement
}

function part(html: LayoutElement, tag: 'head' | 'body'): LayoutElement {
  const found = childElements(html).find((child) => child.type === tag)
  expect(found, `<${tag}>`).toBeDefined()
  return found as LayoutElement
}

interface HeadScript {
  readonly type: string | null
  readonly id: string | null
  readonly body: string
}

/** The <head> as the browser receives it: each script element's type, id and body, in order. */
function headScripts(html: LayoutElement): HeadScript[] {
  const markup = renderToStaticMarkup(part(html, 'head'))
  const doc = new DOMParser().parseFromString(`<!doctype html><html>${markup}<body></body></html>`, 'text/html')
  return Array.from(doc.head.children).map((element) => ({
    type: element.getAttribute('type'),
    id: element.getAttribute('id'),
    body: element.textContent ?? '',
  }))
}

/** The boot the layout wrote into its JSON element. */
function bootOf(html: LayoutElement): ThemeBoot {
  const [, element] = headScripts(html)
  return JSON.parse(element?.body ?? 'null') as ThemeBoot
}

/** The props the layout hands Providers (the <body>'s one child). */
function providersProps(html: LayoutElement): ElementProps | undefined {
  return childElements(part(html, 'body'))[0]?.props
}

afterEach(() => {
  cookieJar.clear()
  document.documentElement.className = ''
})

describe('PRD-260 M07 — the root layout leaves the style classes to the <head> boot', () => {
  it('renders the same <html className> for every style', async () => {
    const studio = (await layoutFor({ [UI_STYLE_COOKIE]: 'studio' })).props.className
    for (const value of COOKIES) {
      expect({ value, className: (await layoutFor({ [UI_STYLE_COOKIE]: value })).props.className }).toEqual({ value, className: studio })
    }
  })

  it("puts only the core's font variables in <html className>, never a style class", async () => {
    for (const value of COOKIES) {
      const classes = ((await layoutFor({ [UI_STYLE_COOKIE]: value })).props.className ?? '').split(' ')
      expect({ value, classes }).toEqual({ value, classes: [...CORE_FONT_VARIABLE_CLASSES] })
      expect(classes).not.toContain(STUDIO_HTML_CLASS)
      expect(classes).not.toContain(PLAY_CLASS)
    }
    // Not vacuous: the core's three faces, by the variables they bind.
    expect(CORE_FONT_VARIABLE_CLASSES).toEqual(CORE_FONT_CLASSES)
  })
})

describe('PRD-261 A-19f1 — Classic and Studio paint only their own class', () => {
  it.each(FIRST_PAINT_CLASSES)("under the %s cookie the boot adds only that style's classes and takes the others off", async (style, classes) => {
    const boot = bootOf(await layoutFor({ [UI_STYLE_COOKIE]: style }))
    expect(boot.addClasses).toEqual(classes)
    expect(boot.removeClasses).toEqual(STYLE_HTML_CLASS_NAMES.filter((name) => !classes.includes(name)))
    expect([...boot.addClasses, ...boot.removeClasses]).not.toContain(PLAY_CLASS)
    // Nothing of a theme: no look to paint, no look storage, no entry, the user's own tone.
    expect(boot).toMatchObject({ prepaint: null, lookStorage: null, lookGuard: null, entry: null, tone: { policy: 'user', value: null } })
  })

  it.each(FIRST_PAINT_CLASSES)(
    'under the %s cookie the first paint is the font variables and that style\'s classes: no play class, no Play font',
    async (style, classes) => {
      const html = await layoutFor({ [UI_STYLE_COOKIE]: style })
      const [, element] = headScripts(html)
      // <html> as the server sends it, then the constant script over the page's own JSON element.
      document.documentElement.className = html.props.className ?? ''
      runThemeBoot(null, element?.body ?? '')
      const painted = Array.from(document.documentElement.classList)
      expect(painted).toEqual([...CORE_FONT_CLASSES, ...classes])
      expect(painted).not.toContain(PLAY_CLASS)
    },
  )

  it('hands Providers the style and no tone under Classic and Studio, whatever tone cookie is left', async () => {
    for (const style of UI_STYLES) {
      const props = providersProps(await layoutFor({ [UI_STYLE_COOKIE]: style, ...LEFT_TONE_COOKIE }))
      expect({ style, props: { initialUiStyle: props?.initialUiStyle, initialTone: props?.initialTone } }).toEqual({
        style,
        props: { initialUiStyle: style, initialTone: undefined },
      })
    }
  })
})

describe('PRD-261 A-03 — the <head>: the import map, one boot element, one constant script', () => {
  it('is exactly those three, in that order, for every style', async () => {
    for (const value of COOKIES) {
      const html = await layoutFor({ [UI_STYLE_COOKIE]: value })
      expect(childElements(part(html, 'head')).map((child) => child.type)).toEqual([ThemeImportMap, ThemeBootScripts])
      const scripts = headScripts(html)
      expect({ value, kinds: scripts.map(({ type, id }) => [type, id]) }).toEqual({
        value,
        kinds: [['importmap', null], ['application/json', THEME_BOOT_ELEMENT_ID], [null, null]],
      })
      expect(scripts[0]?.body).toBe(IMPORT_MAP_SCRIPT_BODY)
      expect(scripts[2]?.body).toBe(THEME_BOOT_SCRIPT)
    }
  })

  it("writes the style's boot, and nothing else, into the JSON element", async () => {
    for (const value of COOKIES) {
      const boot = bootOf(await layoutFor({ [UI_STYLE_COOKIE]: value }))
      const expected = baseThemeBoot(parseUiStyle(value))
      expect({ value, boot }).toEqual({ value, boot: JSON.parse(JSON.stringify(expected)) })
    }
  })

  it('reads a missing or tampered style cookie as Studio, the default style', async () => {
    expect(DEFAULT_UI_STYLE).toBe('studio')
    const expected = JSON.stringify(baseThemeBoot(DEFAULT_UI_STYLE))
    const cases = [...TAMPERED_COOKIES.map((value) => ({ [UI_STYLE_COOKIE]: value })), {}]
    for (const cookies of cases) {
      const html = await layoutFor(cookies)
      const [, element] = headScripts(html)
      expect({ cookies, boot: element?.body }).toEqual({ cookies, boot: expected })
      expect({ cookies, addClasses: bootOf(html).addClasses }).toEqual({ cookies, addClasses: [STUDIO_HTML_CLASS] })
      expect({ cookies, style: providersProps(html)?.initialUiStyle }).toEqual({ cookies, style: DEFAULT_UI_STYLE })
    }
  })

  it('renders the import map as one inline map holding the constant body', () => {
    const markup = renderToStaticMarkup(<ThemeImportMap />)
    expect(markup.startsWith(IMPORT_MAP_OPEN)).toBe(true)
    expect(markup.endsWith(SCRIPT_CLOSE)).toBe(true)
    expect(markup.slice(IMPORT_MAP_OPEN.length, markup.length - SCRIPT_CLOSE.length)).toBe(IMPORT_MAP_SCRIPT_BODY)
  })
})
