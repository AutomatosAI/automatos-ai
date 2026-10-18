// @vitest-environment node
/**
 * PRD-262 US-046: the standalone build carries only what the Next server runs.
 *
 * - Image optimisation is off (§9-Q12), so the server never loads sharp. The only image Next
 *   optimised was the 48 px brand mark on the hosted sign-in pages, so public/brand/ holds it at
 *   that size: 96 px wide, the 2x copy Next's optimiser served to high-density screens.
 * - sharp and its libvips builds, TypeScript and Next's AMP validator stay out of the trace.
 * - test.yml's prod-image-parity job boots both built editions and looks for every excluded
 *   path (scripts/check-standalone-image.sh), so the script's list is this config's list.
 */
import fs from 'fs'
import path from 'path'
import { createRequire } from 'module'
import { describe, expect, it } from 'vitest'

type NextConfig = {
  images?: { unoptimized?: boolean }
  outputFileTracingExcludes?: Record<string, string[]>
}

const load = createRequire(import.meta.url)
const CONFIG_PATH = load.resolve('../../next.config.js')
const config = load(CONFIG_PATH) as NextConfig
const FRONTEND = path.dirname(CONFIG_PATH)

const EXCLUDED = [
  'node_modules/@img/**',
  'node_modules/sharp/**',
  'node_modules/typescript/**',
  'node_modules/next/dist/compiled/amphtml-validator/**',
]
const MARK_SRC = '/brand/automatos-mark.png'
const MARK_DISPLAY_PX = 48
const HIGH_DENSITY_SCALE = 2
// The pages that show the mark through next/image (git grep "src=\"/brand/automatos-mark.png\"").
const MARK_PAGES = [
  'components/auth/sign-in-form.tsx',
  'components/auth/sign-up-form.tsx',
  'app/reset-password/page.tsx',
]
const PNG_SIGNATURE = Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a])

function read(relative: string): Buffer {
  return fs.readFileSync(path.join(FRONTEND, relative))
}

/** A PNG's pixel size, from its IHDR chunk. */
function pngSize(bytes: Buffer): { width: number; height: number } {
  expect(bytes.subarray(0, PNG_SIGNATURE.length).equals(PNG_SIGNATURE)).toBe(true)
  expect(bytes.toString('ascii', 12, 16)).toBe('IHDR')
  return { width: bytes.readUInt32BE(16), height: bytes.readUInt32BE(20) }
}

/** The paths scripts/check-standalone-image.sh looks for, from its TRACE_EXCLUDED array. */
function checkedPaths(): string[] {
  const script = read('scripts/check-standalone-image.sh').toString('utf8')
  const block = /^TRACE_EXCLUDED=\(\n([\s\S]*?)\n\)$/m.exec(script)
  return (block?.[1] ?? '').split('\n').map((line) => line.trim()).filter(Boolean)
}

describe('the standalone build without sharp, libvips or TypeScript', () => {
  it('turns Next image optimisation off', () => {
    expect(config.images?.unoptimized).toBe(true)
  })

  it('keeps sharp, libvips, TypeScript and the AMP validator out of every trace', () => {
    // '*' matches every route and the server's own trace ('next-server').
    expect(config.outputFileTracingExcludes).toEqual({ '*': EXCLUDED })
  })

  it('has the parity check look for every path the trace leaves out', () => {
    expect(checkedPaths().map((dir) => `${dir}/**`)).toEqual(config.outputFileTracingExcludes?.['*'])
  })
})

describe('the brand mark, pre-resized for the sign-in pages', () => {
  it(`is a PNG no wider than ${HIGH_DENSITY_SCALE}x its ${MARK_DISPLAY_PX} px display`, () => {
    const { width } = pngSize(read(`public${MARK_SRC}`))
    expect(width).toBeGreaterThanOrEqual(MARK_DISPLAY_PX)
    expect(width).toBeLessThanOrEqual(MARK_DISPLAY_PX * HIGH_DENSITY_SCALE)
  })

  it.each(MARK_PAGES)('is shown at its display size or less by %s', (page) => {
    const source = read(page).toString('utf8')
    const shown = new RegExp(`src="${MARK_SRC}"[^>]*?width=\\{(\\d+)\\}`).exec(source)
    expect(shown, `${page} no longer shows ${MARK_SRC}`).not.toBeNull()
    expect(Number(shown?.[1])).toBeLessThanOrEqual(MARK_DISPLAY_PX)
  })
})
