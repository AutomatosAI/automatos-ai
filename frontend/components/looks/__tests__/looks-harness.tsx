/**
 * PRD-260 T-008a — what the Make it yours tests share: /looks rendered inside
 * the appearance providers as components/providers.tsx nests them (the style,
 * then the look), and the page's regions by role and name. Each test file
 * mocks MainLayout itself (vi.mock is per file).
 */
import { render, screen, waitFor, within } from '@testing-library/react'
import { expect, vi, type Mock } from 'vitest'

import LooksPage from '@/app/looks/page'
import { LookProvider } from '@/lib/theme-host/builtin/play-look-provider'
import { UiStyleProvider } from '@/contexts/ui-style-context'
import { clearRoot, rootVars } from '@/lib/looks/__tests__/helpers'
import { lookDocumentVars } from '@/lib/looks/document'
import { EMBER } from '@/lib/looks/presets'
import { DEFAULT_LOOK_SCOPE } from '@/lib/looks/scope'
import { lookKeys, type StorageSource } from '@/lib/theme-host/builtin/play-look-storage'
import { PLAY_TONE_COOKIE } from '@/lib/looks/tone'
import type { Look } from '@/lib/looks/types'
import { BUILTIN_PLAY_STYLE } from '@/lib/theme-host/builtin/play-entry'
import { APPEARANCE_DEFAULTS_KEY, APPEARANCE_DEFAULTS_VERSION, UI_STYLE_COOKIE, type UiStyle } from '@/lib/ui-style'

/** Where the looks are kept here: no AccountLookScope above the provider, so the default scope (F5). */
export const KEYS = lookKeys(DEFAULT_LOOK_SCOPE)

/** /looks as the app nests it; `storage` stands in for localStorage (blocked, full). */
export function looksTree(style: UiStyle = BUILTIN_PLAY_STYLE, storage?: StorageSource) {
  return (
    <UiStyleProvider initialStyle={style}>
      <LookProvider storage={storage}>
        <LooksPage />
      </LookProvider>
    </UiStyleProvider>
  )
}

export function renderLooks(style: UiStyle = BUILTIN_PLAY_STYLE, storage?: StorageSource) {
  return render(looksTree(style, storage))
}

/** A browser already moved to the current appearance defaults, so the provider keeps the given style. */
export function freshBrowser(): void {
  window.localStorage.clear()
  window.localStorage.setItem(APPEARANCE_DEFAULTS_KEY, APPEARANCE_DEFAULTS_VERSION)
  document.cookie = `${UI_STYLE_COOKIE}=; Max-Age=0; Path=/`
  document.cookie = `${PLAY_TONE_COOKIE}=; Max-Age=0; Path=/`
  clearRoot()
}

/** The look on <html> (Play's variables and Studio's tokens) is this one. */
export async function lookOnPage(look: Look = EMBER): Promise<void> {
  await waitFor(() => expect(rootVars()).toEqual(lookDocumentVars(look)))
}

export function rootVar(name: string): string {
  return document.documentElement.style.getPropertyValue(name).trim()
}

export function stored(key: string): unknown {
  const raw = window.localStorage.getItem(key)
  return raw === null ? null : JSON.parse(raw)
}

/** The line beside the heading: the active look's name, '· edited' once changed. */
export const status = () => screen.getByRole('status')
export const looksRow = () => within(screen.getByRole('region', { name: 'Looks' }))
export const manage = () => within(screen.getByRole('region', { name: 'This look' }))
export const importer = () => within(screen.getByRole('region', { name: 'Import a look' }))
export const group = (name: string | RegExp) => within(screen.getByRole('group', { name }))

/** A card in the Looks row, by the look's name ('Ember Warm, editorial'). */
export function card(name: string): HTMLElement {
  return looksRow().getByRole('button', { name: new RegExp(`^${name}\\b`) })
}

export function pressedCard(): HTMLElement {
  const pressed = looksRow().getAllByRole('button', { pressed: true })
  expect(pressed).toHaveLength(1)
  return pressed[0] as HTMLElement
}

export const BLOB_URL = 'blob:automatos/look'

export interface DownloadStub {
  /** Every Blob Export made a URL for. */
  readonly blobs: Blob[]
  /** Every link Export clicked: its href and file name. */
  readonly downloads: { readonly href: string; readonly download: string }[]
  readonly revoke: Mock
  /** Puts URL's own functions back (the click spy goes with vi.restoreAllMocks). */
  restore(): void
}

/** jsdom implements no object URLs and no downloads: records what Export does instead. */
export function stubDownloads(): DownloadStub {
  const originals = {
    createObjectURL: Object.getOwnPropertyDescriptor(URL, 'createObjectURL'),
    revokeObjectURL: Object.getOwnPropertyDescriptor(URL, 'revokeObjectURL'),
  }
  const blobs: Blob[] = []
  const downloads: { href: string; download: string }[] = []
  const revoke = vi.fn()
  const create = (blob: Blob) => {
    blobs.push(blob)
    return BLOB_URL
  }
  Object.defineProperty(URL, 'createObjectURL', { value: create, configurable: true, writable: true })
  Object.defineProperty(URL, 'revokeObjectURL', { value: revoke, configurable: true, writable: true })
  vi.spyOn(HTMLAnchorElement.prototype, 'click').mockImplementation(function (this: HTMLAnchorElement) {
    downloads.push({ href: this.getAttribute('href') ?? '', download: this.download })
  })
  const restore = () => {
    for (const [name, original] of Object.entries(originals)) {
      if (original) Object.defineProperty(URL, name, original)
      else Reflect.deleteProperty(URL, name)
    }
  }
  return { blobs, downloads, revoke, restore }
}

/** A Blob's text through FileReader (jsdom's Blob may have no text()). */
export function blobText(blob: Blob): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader()
    reader.onload = () => resolve(String(reader.result))
    reader.onerror = () => reject(reader.error)
    reader.readAsText(blob)
  })
}
