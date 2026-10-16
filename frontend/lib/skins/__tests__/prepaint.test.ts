/**
 * PRD-259 S-004a — the pre-paint script, run as the string it is in jsdom: it applies the
 * cached stylesheet only when the cache is for <html data-skin>'s skin and matches the
 * grammar; anything else, including storage that throws, leaves the page alone.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { SKIN_CSS_CACHE_KEY, readSkinCache, writeSkinCache } from '@/lib/skins/cache'
import { skinToCss } from '@/lib/skins/css'
import { SKIN_PREPAINT_SCRIPT, SKIN_STYLE_ELEMENT_ID } from '@/lib/skins/prepaint'

const ID = '0f8c2b6e-4d1a-4c3e-9b7a-2e5d6f8a1c90'
const OTHER_ID = '11111111-2222-4333-8444-555555555555'
const CSS = skinToCss({ tokens: { light: { background: '38 50% 92%', foreground: '30 12% 9%' } }, shape: { radius: 6 } }, ID)

function runScript(): void {
  // The script is a constant built from constants; running it is what the <head> does.
  new Function(SKIN_PREPAINT_SCRIPT)()
}

function styleElement(): HTMLElement | null {
  return document.getElementById(SKIN_STYLE_ELEMENT_ID)
}

/** Storage that refuses every access, as a blocked or private-mode browser does. */
function blockStorage(): void {
  const refuse = () => {
    throw new DOMException('blocked', 'SecurityError')
  }
  vi.spyOn(Storage.prototype, 'getItem').mockImplementation(refuse)
  vi.spyOn(Storage.prototype, 'setItem').mockImplementation(refuse)
}

function cache(entry: unknown): void {
  window.localStorage.setItem(SKIN_CSS_CACHE_KEY, JSON.stringify(entry))
}

beforeEach(() => {
  window.localStorage.clear()
  styleElement()?.remove()
  document.documentElement.setAttribute('data-skin', ID)
})

afterEach(() => {
  vi.restoreAllMocks()
  document.documentElement.removeAttribute('data-skin')
})

describe('SKIN_PREPAINT_SCRIPT', () => {
  it('puts exactly the cached stylesheet in the style element for a valid cache', () => {
    cache({ id: ID, version: '1.0.0', css: CSS })
    runScript()
    expect(styleElement()?.tagName).toBe('STYLE')
    expect(styleElement()?.textContent).toBe(CSS)
  })

  it('does nothing for a cache made for another skin', () => {
    cache({ id: OTHER_ID, version: '1.0.0', css: skinToCss({ tokens: {}, shape: { radius: 4 } }, OTHER_ID) })
    runScript()
    expect(styleElement()).toBeNull()
  })

  it("does nothing when the cache's id matches but its stylesheet names another skin", () => {
    cache({ id: ID, version: '1.0.0', css: skinToCss({ tokens: {}, shape: { radius: 4 } }, OTHER_ID) })
    runScript()
    expect(styleElement()).toBeNull()
  })

  it("does nothing when the cached stylesheet holds 'url('", () => {
    cache({ id: ID, version: '1.0.0', css: `html[data-skin="${ID}"]{--paper:url(https://evil.example/x.png)}` })
    runScript()
    expect(styleElement()).toBeNull()
  })

  it('does nothing, and throws nothing, when localStorage throws', () => {
    blockStorage()
    expect(runScript).not.toThrow()
    expect(styleElement()).toBeNull()
  })

  it('does nothing without a data-skin, or with one that is not a skin id', () => {
    cache({ id: ID, version: '1.0.0', css: CSS })
    document.documentElement.removeAttribute('data-skin')
    runScript()
    expect(styleElement()).toBeNull()
    document.documentElement.setAttribute('data-skin', 'x"]{}')
    runScript()
    expect(styleElement()).toBeNull()
  })

  it('is a constant that can sit inside a <script> element, with no innerHTML', () => {
    expect(SKIN_PREPAINT_SCRIPT).not.toContain('<')
    expect(SKIN_PREPAINT_SCRIPT).not.toContain('innerHTML')
    expect(SKIN_PREPAINT_SCRIPT).toContain('textContent')
  })
})

describe('the skin stylesheet cache', () => {
  it('reads back what it wrote, and null for a corrupt entry', () => {
    expect(writeSkinCache({ id: ID, version: '1.0.0', css: CSS })).toBe(true)
    expect(readSkinCache()).toEqual({ id: ID, version: '1.0.0', css: CSS })
    window.localStorage.setItem(SKIN_CSS_CACHE_KEY, '{not json')
    vi.spyOn(console, 'warn').mockImplementation(() => undefined)
    expect(readSkinCache()).toBeNull()
    cache({ id: ID, css: CSS })
    expect(readSkinCache()).toBeNull()
  })

  it('returns null and false, never throws, when storage refuses', () => {
    vi.spyOn(console, 'warn').mockImplementation(() => undefined)
    blockStorage()
    expect(readSkinCache()).toBeNull()
    expect(writeSkinCache({ id: ID, version: '1.0.0', css: CSS })).toBe(false)
  })
})
