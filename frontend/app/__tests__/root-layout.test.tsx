/**
 * PRD-260 M07 — the root layout's <html className> is the font variables only,
 * the same string for every style, so a server re-render (router.refresh after a
 * client-side style switch) never rewrites the class attribute and drops
 * next-themes' tone class. The style's classes come from STYLE_CLASS_SCRIPT in
 * <head>, ahead of the look's pre-paint script under Play.
 */
import { Children, isValidElement, type ReactElement, type ReactNode } from 'react'
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
// next/font is a compile-time transform; under vitest each loader is a plain function.
vi.mock('next/font/google', () => {
  const loader = (name: string) => () => ({ variable: `font-${name}`, className: `font-${name}-face`, style: {} })
  return { Geist: loader('geist'), Geist_Mono: loader('geist-mono'), Newsreader: loader('newsreader') }
})
vi.mock('@/app/play-fonts', () => ({ PLAY_FONT_VARIABLE_CLASSES: ['font-instrument', 'font-bricolage', 'font-fredoka'] }))
vi.mock('@/components/providers', () => ({ Providers: ({ children }: { children: ReactNode }) => children }))

import RootLayout from '../layout'
import { LOOK_PREPAINT_SCRIPT } from '@/lib/looks/prepaint'
import {
  DEFAULT_UI_STYLE,
  PLAY_HTML_CLASS,
  STUDIO_HTML_CLASS,
  STYLE_CLASS_SCRIPT,
  UI_STYLE_COOKIE,
  UI_STYLES,
  type UiStyle,
} from '@/lib/ui-style'
import { renderToStaticMarkup } from 'react-dom/server'
import { ThemeImportMap } from '@/components/theme-host/theme-import-map'
import { IMPORT_MAP_SCRIPT_BODY } from '@/lib/theme-host/import-map'

interface ElementProps {
  readonly children?: ReactNode
  readonly className?: string
  readonly dangerouslySetInnerHTML?: { readonly __html: string }
}
type LayoutElement = ReactElement<ElementProps>

function childElements(element: LayoutElement): LayoutElement[] {
  return Children.toArray(element.props.children).filter((child): child is LayoutElement => isValidElement(child))
}

async function layoutFor(style: UiStyle): Promise<LayoutElement> {
  cookieJar.set(UI_STYLE_COOKIE, style)
  return (await RootLayout({ children: null })) as LayoutElement
}

function headScripts(html: LayoutElement): string[] {
  const head = childElements(html).find((child) => child.type === 'head')
  expect(head, '<head>').toBeDefined()
  return childElements(head as LayoutElement).map((script) => script.props.dangerouslySetInnerHTML?.__html ?? '')
}

afterEach(() => {
  cookieJar.clear()
})

describe('PRD-260 M07 — the root layout leaves the style classes to a <head> script', () => {
  it('renders the same <html className> for Studio and Play (and Classic)', async () => {
    const studio = (await layoutFor('studio')).props.className
    const play = (await layoutFor('play')).props.className
    const classic = (await layoutFor('classic')).props.className
    expect(play).toBe(studio)
    expect(classic).toBe(studio)
  })

  it('puts only the font variables in <html className>, never a style class', async () => {
    const classes = ((await layoutFor('play')).props.className ?? '').split(' ')
    expect(classes).toContain('font-geist')
    expect(classes).toContain('font-instrument')
    expect(classes).not.toContain(STUDIO_HTML_CLASS)
    expect(classes).not.toContain(PLAY_HTML_CLASS)
  })

  it("runs the style's class script first in <head>, then the look's script under Play only", async () => {
    for (const style of UI_STYLES) {
      const expected = style === 'play' ? [STYLE_CLASS_SCRIPT.play, LOOK_PREPAINT_SCRIPT] : [STYLE_CLASS_SCRIPT[style]]
      expect({ style, scripts: headScripts(await layoutFor(style)) }).toEqual({ style, scripts: expected })
    }
  })

  it('reads a missing or tampered style cookie as the default style', async () => {
    cookieJar.set(UI_STYLE_COOKIE, 'play;studio')
    const tampered = headScripts((await RootLayout({ children: null })) as LayoutElement)
    cookieJar.clear()
    const missing = headScripts((await RootLayout({ children: null })) as LayoutElement)
    expect(tampered).toEqual([STYLE_CLASS_SCRIPT[DEFAULT_UI_STYLE]])
    expect(missing).toEqual([STYLE_CLASS_SCRIPT[DEFAULT_UI_STYLE]])
  })
})

// PRD-261 A-01: with the spike built in, the theme import map is the head's first child, ahead
// of every script above; without it (every shipped build) the head is exactly as before.
const SPIKE_FLAG = 'AUTOMATOS_THEME_HOST_SPIKE'
const IMPORT_MAP_OPEN = '<script type="importmap">'
const SCRIPT_CLOSE = '</script>'

function headChildren(html: LayoutElement): LayoutElement[] {
  const head = childElements(html).find((child) => child.type === 'head')
  expect(head, '<head>').toBeDefined()
  return childElements(head as LayoutElement)
}

function scriptBodies(children: readonly LayoutElement[]): string[] {
  return children.map((script) => script.props.dangerouslySetInnerHTML?.__html ?? '')
}

describe('PRD-261 A-01 — the theme import map', () => {
  afterEach(() => {
    vi.unstubAllEnvs()
  })

  it('is not in <head> when the spike is off, for every style', async () => {
    vi.stubEnv(SPIKE_FLAG, '0')
    for (const style of UI_STYLES) {
      const types = headChildren(await layoutFor(style)).map((child) => child.type)
      expect({ style, hasMap: types.includes(ThemeImportMap) }).toEqual({ style, hasMap: false })
    }
  })

  it('is the first child of <head> when the spike is on, ahead of the same scripts, for every style', async () => {
    for (const style of UI_STYLES) {
      vi.stubEnv(SPIKE_FLAG, '0')
      const off = scriptBodies(headChildren(await layoutFor(style)))
      vi.stubEnv(SPIKE_FLAG, '1')
      const [first, ...rest] = headChildren(await layoutFor(style))
      expect({ style, first: first?.type === ThemeImportMap }).toEqual({ style, first: true })
      expect({ style, scripts: scriptBodies(rest) }).toEqual({ style, scripts: off })
    }
  })

  it('renders as one inline import map holding the constant body', () => {
    const markup = renderToStaticMarkup(<ThemeImportMap />)
    expect(markup.startsWith(IMPORT_MAP_OPEN)).toBe(true)
    expect(markup.endsWith(SCRIPT_CLOSE)).toBe(true)
    expect(markup.slice(IMPORT_MAP_OPEN.length, markup.length - SCRIPT_CLOSE.length)).toBe(IMPORT_MAP_SCRIPT_BODY)
  })
})
