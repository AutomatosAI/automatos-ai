/**
 * PRD-261 A-08c — the browser's own fetch of an installed theme's boot, for a page whose boot
 * carries no entry (the server's boot fetch timed out, was refused or failed). Pinned: one request
 * per install for the page's life, every caller handed the same answer; the answer through the boot
 * gate the server's passes (parseInstalledBoot, sanitizeThemeBoot); a failed request, or an answer
 * that isn't a boot, gives null with one warning and is not asked again; and the page's source goes
 * through apiClient's named method for GET /api/themes/boot/{installId}.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { apiClient } from '@/lib/api-client'
import { createInstalledBootSource, pageInstalledBoot } from '@/lib/theme-host/boot-client'
import { sanitizeThemeBoot } from '@/lib/theme-host/boot-sanitize'
import { helloThemeBoot, HELLO_ENTRY } from '@/lib/theme-host/looks/__tests__/hello-looks'

const INSTALL_ID = '6e1f3b8f-2d5c-4fab-a4e7-9a2b3d4f5c6e'
const OTHER_INSTALL_ID = '5d0f2a7e-1c4b-4e9a-b3d6-8f1a2c3e4b5d'
/** An install no other test here asks for: the page's source keeps its answer for the module's life. */
const PAGE_INSTALL_ID = '0b9a3c3e-6a1d-4f9e-9a43-0c1f2d3e4f50'
const LOG_PREFIX = '[theme-boot]'
/** A class the boot gate refuses (HTML_CLASS_PATTERN) and a value its CSS grammar refuses. */
const BAD_CLASS = 'Not A Class'
const BAD_VALUE = 'url(https://example.com/x.png)'

/** The warnings this module logged (its prefix first). */
function bootWarnings(): unknown[][] {
  return vi.mocked(console.warn).mock.calls.filter(([prefix]) => prefix === LOG_PREFIX)
}

beforeEach(() => {
  vi.spyOn(console, 'warn').mockImplementation(() => undefined)
  vi.spyOn(console, 'error').mockImplementation(() => undefined)
  vi.spyOn(console, 'log').mockImplementation(() => undefined)
})

afterEach(() => {
  vi.restoreAllMocks()
})

describe('createInstalledBootSource', () => {
  it('asks once per install, and hands every caller the same answer', async () => {
    const fetchBoot = vi.fn(async () => helloThemeBoot())
    const source = createInstalledBootSource(fetchBoot)
    const [first, second] = await Promise.all([source(INSTALL_ID), source(INSTALL_ID)])
    const later = await source(INSTALL_ID)
    expect(fetchBoot.mock.calls).toEqual([[INSTALL_ID]])
    expect(first?.entry).toEqual(HELLO_ENTRY)
    expect(second).toBe(first)
    expect(later).toBe(first)
    await source(OTHER_INSTALL_ID)
    expect(fetchBoot.mock.calls).toEqual([[INSTALL_ID], [OTHER_INSTALL_ID]])
  })

  it('passes the answer through the boot gate the server uses: a refused class or value never reaches the page', async () => {
    const boot = helloThemeBoot()
    const table = boot.prepaint
    if (table === null) throw new Error("hello-a09b's boot has no prepaint table")
    const [knob] = Object.keys(table.knobs)
    const [option] = Object.keys(table.knobs[knob])
    const hostile = {
      ...boot,
      addClasses: [...boot.addClasses, BAD_CLASS],
      prepaint: { ...table, knobs: { ...table.knobs, [knob]: { ...table.knobs[knob], [option]: { '--bad': BAD_VALUE } } } },
    }
    const answer = await createInstalledBootSource(async () => hostile)(INSTALL_ID)
    expect(answer).toEqual(sanitizeThemeBoot(hostile))
    expect(answer?.addClasses).not.toContain(BAD_CLASS)
    expect(JSON.stringify(answer)).not.toContain(BAD_VALUE)
    expect(bootWarnings()).toEqual([])
  })

  it('a failed request gives null with one warning, and is not asked again for the page', async () => {
    const fetchBoot = vi.fn(() => Promise.reject(new Error('Failed to fetch')))
    const source = createInstalledBootSource(fetchBoot)
    await expect(source(INSTALL_ID)).resolves.toBeNull()
    await expect(source(INSTALL_ID)).resolves.toBeNull()
    expect(fetchBoot).toHaveBeenCalledTimes(1)
    expect(bootWarnings()).toHaveLength(1)
  })

  it.each<[string, unknown]>([
    ['null', null],
    ['a list', []],
    ['text', 'not a boot'],
  ])('an answer that is %s gives null with one warning', async (_label, answer) => {
    await expect(createInstalledBootSource(async () => answer)(INSTALL_ID)).resolves.toBeNull()
    expect(bootWarnings()).toHaveLength(1)
  })
})

describe("the page's source", () => {
  it('asks GET /api/themes/boot/<id> through apiClient, the id as one encoded segment', async () => {
    const request = vi.spyOn(apiClient, 'request').mockResolvedValue(helloThemeBoot() as never)
    await apiClient.getThemeBoot('../files')
    const answer = await pageInstalledBoot(PAGE_INSTALL_ID)
    expect(request.mock.calls).toEqual([['/api/themes/boot/..%2Ffiles'], [`/api/themes/boot/${PAGE_INSTALL_ID}`]])
    expect(answer?.entry).toEqual(HELLO_ENTRY)
    await pageInstalledBoot(PAGE_INSTALL_ID)
    expect(request).toHaveBeenCalledTimes(2)
  })
})
