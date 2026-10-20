/**
 * PRD-261 A-14d — what Play's look editor copied from the core when it stopped importing it
 * (docs/theme-host/sdk-surface-map.md: "copied into Play"), held to the core's own, so the two
 * never drift apart while both exist:
 *
 * - downloadText (lib/download-text.ts → components/looks/download-text.ts): the same file, the
 *   same link taken away, the URL let go after the same delay;
 * - copyToClipboard (lib/utils.ts → components/looks/clipboard.ts): the same answer and the same
 *   log when the browser refuses;
 * - the sample card's kind word (lib/chat/deliverables.ts KIND_WORDS.document → the style sheet's
 *   SAMPLE_KIND);
 * - the mark on a Play-only screen's h1 (components/looks/screen-heading.ts), which the host's
 *   ThemeSlotPage finds by its SCREEN_HEADING_SELECTOR (lib/theme-host/builtin/play-only-route.tsx).
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { copyToClipboard } from '@/components/looks/clipboard'
import { DOWNLOAD_REVOKE_AFTER_MS, downloadText } from '@/components/looks/download-text'
import { PLAY_ONLY_HEADING } from '@/components/looks/screen-heading'
import { SAMPLE_KIND } from '@/components/looks/style/style-components'
import { KIND_WORDS } from '@/lib/chat/deliverables'
import { DOWNLOAD_REVOKE_AFTER_MS as CORE_REVOKE_AFTER_MS, downloadText as coreDownloadText } from '@/lib/download-text'
import { SCREEN_HEADING_SELECTOR } from '@/lib/theme-host/builtin/play-only-route'
import { copyToClipboard as coreCopyToClipboard } from '@/lib/utils'

const BLOB_URL = 'blob:a14d-download'

interface DownloadRecord {
  readonly blobTypes: readonly string[]
  readonly downloads: readonly { readonly href: string; readonly download: string }[]
  readonly linksLeft: number
  readonly revokedBefore: number
  readonly revokedAfter: readonly unknown[][]
}

const urlOriginals = {
  createObjectURL: Object.getOwnPropertyDescriptor(URL, 'createObjectURL'),
  revokeObjectURL: Object.getOwnPropertyDescriptor(URL, 'revokeObjectURL'),
}
const clipboardOriginal = Object.getOwnPropertyDescriptor(navigator, 'clipboard')

beforeEach(() => {
  vi.useFakeTimers()
})

afterEach(() => {
  vi.useRealTimers()
  vi.restoreAllMocks()
  for (const [name, original] of Object.entries(urlOriginals)) {
    if (original) Object.defineProperty(URL, name, original)
    else Reflect.deleteProperty(URL, name)
  }
  if (clipboardOriginal) Object.defineProperty(navigator, 'clipboard', clipboardOriginal)
  else Reflect.deleteProperty(navigator, 'clipboard')
})

/** What one download helper does with jsdom's missing object URLs and downloads stood in. */
function recordDownload(download: typeof downloadText, revokeAfterMs: number): DownloadRecord {
  const blobTypes: string[] = []
  const downloads: { href: string; download: string }[] = []
  const revoke = vi.fn()
  const create = (blob: Blob) => {
    blobTypes.push(blob.type)
    return BLOB_URL
  }
  Object.defineProperty(URL, 'createObjectURL', { value: create, configurable: true, writable: true })
  Object.defineProperty(URL, 'revokeObjectURL', { value: revoke, configurable: true, writable: true })
  const click = vi.spyOn(HTMLAnchorElement.prototype, 'click').mockImplementation(function (this: HTMLAnchorElement) {
    downloads.push({ href: this.getAttribute('href') ?? '', download: this.download })
  })
  download('ember.look.json', '{}', 'application/json')
  const linksLeft = document.querySelectorAll('a[download]').length
  vi.advanceTimersByTime(revokeAfterMs - 1)
  const revokedBefore = revoke.mock.calls.length
  vi.advanceTimersByTime(1)
  click.mockRestore()
  return { blobTypes, downloads, linksLeft, revokedBefore, revokedAfter: revoke.mock.calls }
}

function stubClipboard(writeText: (text: string) => Promise<void>): void {
  Object.defineProperty(navigator, 'clipboard', { value: { writeText }, configurable: true })
}

describe("Play's copies of the core's helpers", () => {
  it('downloadText saves the same file and lets the URL go after the same delay as the core', () => {
    expect(DOWNLOAD_REVOKE_AFTER_MS).toBe(CORE_REVOKE_AFTER_MS)
    const play = recordDownload(downloadText, DOWNLOAD_REVOKE_AFTER_MS)
    const core = recordDownload(coreDownloadText, CORE_REVOKE_AFTER_MS)
    expect(play).toEqual(core)
    expect(play.downloads).toEqual([{ href: BLOB_URL, download: 'ember.look.json' }])
    expect(play.revokedAfter).toEqual([[BLOB_URL]])
  })

  it('copyToClipboard answers as the core does, when the browser takes the text and when it refuses', async () => {
    const written = vi.fn(async () => undefined)
    stubClipboard(written)
    expect([await copyToClipboard('{"a":1}'), await coreCopyToClipboard('{"a":1}')]).toEqual([true, true])
    expect(written.mock.calls).toEqual([['{"a":1}'], ['{"a":1}']])

    const refusal = new Error('blocked')
    stubClipboard(async () => {
      throw refusal
    })
    const logged = vi.spyOn(console, 'error').mockImplementation(() => undefined)
    expect([await copyToClipboard('x'), await coreCopyToClipboard('x')]).toEqual([false, false])
    expect(logged.mock.calls[0]).toEqual(logged.mock.calls[1])
  })

  it("the style sheet's sample card names its kind in the core's words", () => {
    expect(SAMPLE_KIND).toBe(KIND_WORDS.document)
  })

  it("marks a Play-only screen's heading where the host's ThemeSlotPage looks for it", () => {
    const heading = document.createElement('h1')
    for (const [name, value] of Object.entries(PLAY_ONLY_HEADING)) heading.setAttribute(name, String(value))
    expect(heading.matches(SCREEN_HEADING_SELECTOR)).toBe(true)
  })
})
