/**
 * PRD-261 A-13c — SDK area `assets` (lib/theme-sdk/assets.ts): assetUrl names a file in the active installed
 * theme's package, under the same-origin files route and the digest the host loaded for that install. It
 * refuses a path that could leave the package's folder, and answers null where there is no installed theme
 * (Classic, Studio, a style that names no install) or the host has not named the package's digest for this install.
 */
import { afterEach, describe, expect, it } from 'vitest'

import { assetUrl } from '../assets'
import { THEME_FILES_PATH } from '@/lib/theme-host/constants'
import { THEME_DIGEST_COOKIE, THEME_DIGEST_HINT_SEPARATOR } from '@/lib/theme-host/types'
import { UI_STYLE_COOKIE } from '@/lib/ui-style'

const INSTALL = '0f8e2c3a-4b5d-4e6f-8a9b-0c1d2e3f4a5b'
const OTHER_INSTALL = '1a2b3c4d-5e6f-4a7b-8c9d-0e1f2a3b4c5d'
const DIGEST = 'a'.repeat(64)
const ROOT = `${window.location.origin}${THEME_FILES_PATH}${DIGEST}/`

function setCookie(name: string, value: string): void {
  document.cookie = `${name}=${value}; Path=/`
}

function clearCookie(name: string): void {
  document.cookie = `${name}=; Max-Age=0; Path=/`
}

function installed(installId: string = INSTALL, digestFor: string = INSTALL, digest: string = DIGEST): void {
  setCookie(UI_STYLE_COOKIE, `theme:install:${installId}`)
  setCookie(THEME_DIGEST_COOKIE, `${digestFor}${THEME_DIGEST_HINT_SEPARATOR}${digest}`)
}

afterEach(() => {
  clearCookie(UI_STYLE_COOKIE)
  clearCookie(THEME_DIGEST_COOKIE)
})

describe('PRD-261 A-13c — assetUrl under an installed theme', () => {
  it("names the file under the package's folder on this origin", () => {
    installed()
    expect(assetUrl('assets/mascot/wave.mp4')).toBe(`${ROOT}assets/mascot/wave.mp4`)
    expect(assetUrl('looks/presets/default.look.json')).toBe(`${ROOT}looks/presets/default.look.json`)
  })

  it.each([
    ['a parent segment', '../other/file.png'],
    ['a nested parent segment', 'assets/../../x.png'],
    ['a leading slash', '/etc/passwd'],
    ['a protocol-relative path', '//evil.example/x.png'],
    ['an absolute URL', 'https://evil.example/x.png'],
    ['a backslash', 'assets\\x.png'],
    ['an empty path', ''],
    ['a double slash inside', 'assets//x.png'],
  ])('refuses %s', (_case, path) => {
    installed()
    expect(assetUrl(path)).toBeNull()
  })
})

describe('PRD-261 A-13c — assetUrl with no package to name', () => {
  it.each(['classic', 'studio', 'theme:install:not-a-uuid'])('is null under %s', (style) => {
    setCookie(UI_STYLE_COOKIE, style)
    setCookie(THEME_DIGEST_COOKIE, `${INSTALL}${THEME_DIGEST_HINT_SEPARATOR}${DIGEST}`)
    expect(assetUrl('assets/mascot/wave.mp4')).toBeNull()
  })

  it('is null when the digest the host left is for another install, malformed, or missing', () => {
    installed(INSTALL, OTHER_INSTALL)
    expect(assetUrl('assets/x.png')).toBeNull()
    installed(INSTALL, INSTALL, 'not-a-digest')
    expect(assetUrl('assets/x.png')).toBeNull()
    clearCookie(THEME_DIGEST_COOKIE)
    expect(assetUrl('assets/x.png')).toBeNull()
  })
})
