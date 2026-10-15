/**
 * PRD-260 P260-FX-27 (M88) — one Blob download helper (lib/download-text.ts)
 * behind a look's Export and a knowledge document's markdown download. The
 * markdown download used to revoke its URL in the same tick as the click;
 * through the shared helper it lets the URL go only once the download has
 * started, which Safari needs. The Blob URL and the link's click are stand-ins.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { downloadMarkdownFile } from '@/components/widgets/DocumentWidget/document-content'
import { DOWNLOAD_REVOKE_AFTER_MS, downloadText } from '@/lib/download-text'

const BLOB_URL = 'blob:fx27-download'

const originals = {
  createObjectURL: Object.getOwnPropertyDescriptor(URL, 'createObjectURL'),
  revokeObjectURL: Object.getOwnPropertyDescriptor(URL, 'revokeObjectURL'),
}
let blobs: Blob[] = []
let downloads: { href: string; download: string }[] = []
const revoke = vi.fn()

beforeEach(() => {
  vi.useFakeTimers()
  blobs = []
  downloads = []
  const create = (blob: Blob) => {
    blobs.push(blob)
    return BLOB_URL
  }
  Object.defineProperty(URL, 'createObjectURL', { value: create, configurable: true, writable: true })
  Object.defineProperty(URL, 'revokeObjectURL', { value: revoke, configurable: true, writable: true })
  vi.spyOn(HTMLAnchorElement.prototype, 'click').mockImplementation(function (this: HTMLAnchorElement) {
    downloads.push({ href: this.getAttribute('href') ?? '', download: this.download })
  })
})

afterEach(() => {
  vi.useRealTimers()
  vi.restoreAllMocks()
  revoke.mockReset()
  for (const [name, original] of Object.entries(originals)) {
    if (original) Object.defineProperty(URL, name, original)
    else Reflect.deleteProperty(URL, name)
  }
})

describe('downloadText', () => {
  it('saves the text under the file name, takes its link away, and lets the URL go a moment later', () => {
    downloadText('ember.look.json', '{}', 'application/json')
    expect(downloads).toEqual([{ href: BLOB_URL, download: 'ember.look.json' }])
    expect(blobs.map((blob) => blob.type)).toEqual(['application/json'])
    expect(document.querySelector('a[download]')).toBeNull()
    vi.advanceTimersByTime(DOWNLOAD_REVOKE_AFTER_MS - 1)
    expect(revoke).not.toHaveBeenCalled()
    vi.advanceTimersByTime(1)
    expect(revoke).toHaveBeenCalledWith(BLOB_URL)
  })
})

describe("DocumentWidget's markdown download", () => {
  it('goes through the shared helper: a markdown file, its URL let go only once the download has started', () => {
    downloadMarkdownFile('# Notes', 'notes.md')
    expect(downloads).toEqual([{ href: BLOB_URL, download: 'notes.md' }])
    expect(blobs.map((blob) => blob.type)).toEqual(['text/markdown'])
    expect(revoke).not.toHaveBeenCalled()
    vi.advanceTimersByTime(DOWNLOAD_REVOKE_AFTER_MS)
    expect(revoke).toHaveBeenCalledWith(BLOB_URL)
  })
})
