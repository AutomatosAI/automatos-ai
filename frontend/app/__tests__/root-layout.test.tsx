/**
 * PRD-260 M07 — the root layout's <html className> is the font variables only,
 * the same string for every style, so a server re-render (router.refresh after a
 * client-side style switch) never rewrites the class attribute and drops
 * next-themes' tone class.
 *
 * PRD-261 A-03 — the style's classes, its look and its tone come from the theme
 * boot: <head> is the import map first (on every page, PA-13), then one JSON
 * element holding the style's boot and one constant script that applies it.
 */
import { Children, isValidElement, type ReactElement, type ReactNode } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import { afterEach, describe, expect, it, vi } from 'vitest'

const cookieJar = vi.hoisted(() => new Map<string, string>())
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
  const names: Readonly<Record<string, string>> = {
    '--font-geist-sans': 'geist',
    '--font-geist-mono': 'geist-mono',
    '--font-newsreader': 'newsreader',
  }
  const loader = ({ variable }: { readonly variable: string }) => {
    const name = names[variable] ?? `unknown${variable}`
    return { variable: `font-${name}`, className: `font-${name}-face`, style: {} }
  }
  return { default: loader }
})
vi.mock('@/app/play-fonts', () => ({ PLAY_FONT_VARIABLE_CLASSES: ['font-instrument', 'font-bricolage', 'font-fredoka'] }))
vi.mock('@/components/providers', () => ({ Providers: ({ children }: { children: ReactNode }) => children }))

import RootLayout from '../layout'
import { PLAY_TONE_COOKIE } from '@/lib/looks/tone'
import { ThemeBootScripts } from '@/components/theme-host/theme-boot'
import { ThemeImportMap } from '@/components/theme-host/theme-import-map'
import { THEME_BOOT_ELEMENT_ID, THEME_BOOT_SCRIPT, builtinThemeBoot } from '@/lib/theme-host/boot'
import { BUILTIN_PLAY_HTML_CLASS as PLAY_HTML_CLASS, BUILTIN_PLAY_STYLE } from '@/lib/theme-host/builtin/play-entry'
import { IMPORT_MAP_SCRIPT_BODY } from '@/lib/theme-host/import-map'
import { DEFAULT_UI_STYLE, STUDIO_HTML_CLASS, UI_STYLE_COOKIE, UI_STYLES, parseUiStyle } from '@/lib/ui-style'

interface ElementProps {
  readonly children?: ReactNode
  readonly className?: string
  readonly initialUiStyle?: string
  readonly initialTone?: string
}
type LayoutElement = ReactElement<ElementProps>

const INSTALL_STYLE = 'theme:install:0b7e4c1a-9f3d-4c2e-8a51-3d2f6b7a9c10'
/** Every cookie value the layout must boot: each style, the legacy 'play' and an installed theme. */
const COOKIES = [...UI_STYLES, 'play', INSTALL_STYLE]
const IMPORT_MAP_OPEN = '<script type="importmap">'
const SCRIPT_CLOSE = '</script>'
const NO_COOKIES = () => undefined

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

afterEach(() => {
  cookieJar.clear()
})

describe('PRD-260 M07 — the root layout leaves the style classes to the <head> boot', () => {
  it('renders the same <html className> for every style', async () => {
    const studio = (await layoutFor({ [UI_STYLE_COOKIE]: 'studio' })).props.className
    for (const value of COOKIES) {
      expect({ value, className: (await layoutFor({ [UI_STYLE_COOKIE]: value })).props.className }).toEqual({ value, className: studio })
    }
  })

  it('puts only the font variables in <html className>, never a style class', async () => {
    const classes = ((await layoutFor({ [UI_STYLE_COOKIE]: 'play' })).props.className ?? '').split(' ')
    expect(classes).toContain('font-geist')
    expect(classes).toContain('font-instrument')
    expect(classes).not.toContain(STUDIO_HTML_CLASS)
    expect(classes).not.toContain(PLAY_HTML_CLASS)
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
      const [, element] = headScripts(await layoutFor({ [UI_STYLE_COOKIE]: value }))
      const expected = builtinThemeBoot(parseUiStyle(value), NO_COOKIES)
      expect({ value, boot: JSON.parse(element?.body ?? 'null') }).toEqual({ value, boot: JSON.parse(JSON.stringify(expected)) })
    }
  })

  it('reads a missing or tampered style cookie as the default style', async () => {
    const expected = JSON.stringify(builtinThemeBoot(DEFAULT_UI_STYLE, NO_COOKIES))
    for (const cookies of [{ [UI_STYLE_COOKIE]: 'play;studio' }, { [UI_STYLE_COOKIE]: 'theme:install:x' }, {}]) {
      const [, element] = headScripts(await layoutFor(cookies))
      expect({ cookies, boot: element?.body }).toEqual({ cookies, boot: expected })
    }
  })

  it("hands the boot's tone to Providers: the look's tone under Play only (PRD-260 K4)", async () => {
    const toneOf = async (style: string) =>
      childElements(part(await layoutFor({ [UI_STYLE_COOKIE]: style, [PLAY_TONE_COOKIE]: 'light' }), 'body'))[0]?.props
    expect(await toneOf(BUILTIN_PLAY_STYLE)).toMatchObject({ initialUiStyle: BUILTIN_PLAY_STYLE, initialTone: 'light' })
    expect((await toneOf('studio'))?.initialTone).toBeUndefined()
    expect((await toneOf('classic'))?.initialTone).toBeUndefined()
  })

  it('renders the import map as one inline map holding the constant body', () => {
    const markup = renderToStaticMarkup(<ThemeImportMap />)
    expect(markup.startsWith(IMPORT_MAP_OPEN)).toBe(true)
    expect(markup.endsWith(SCRIPT_CLOSE)).toBe(true)
    expect(markup.slice(IMPORT_MAP_OPEN.length, markup.length - SCRIPT_CLOSE.length)).toBe(IMPORT_MAP_SCRIPT_BODY)
  })
})
