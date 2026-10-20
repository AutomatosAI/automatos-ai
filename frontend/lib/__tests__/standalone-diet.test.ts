// @vitest-environment node
/**
 * PRD-262 US-046: the standalone build carries only what the Next server runs.
 *
 * - Image optimisation is off (§9-Q12), so the server never loads sharp. The only image Next
 *   optimised was the 48 px brand mark on the hosted sign-in pages, so public/brand/ holds it at
 *   that size: 96 px wide, the 2x copy Next's optimiser served to high-density screens.
 * - sharp and its libvips builds, TypeScript and Next's AMP validator stay out of the trace, and
 *   so do Next's own build and dev internals (PRD-262 #29) and React's development builds (#47).
 * - test.yml's prod-image-parity job and images.yml boot each built image and look for every
 *   excluded glob (scripts/check-standalone-image.sh), so the script's list is this config's list.
 * - The server builds resolve the client-only Retell, LiveKit and SheetJS packages to an empty
 *   module, so their code stays out of the server chunks (PRD-262 #47); the browser keeps them.
 */
import fs from 'fs'
import path from 'path'
import { createRequire } from 'module'
import { describe, expect, it } from 'vitest'

type WebpackConfig = { mode?: string; resolve?: { alias?: Record<string, string | false> } }
type NextConfig = {
  images?: { unoptimized?: boolean }
  outputFileTracingExcludes?: Record<string, string[]>
  webpack?: (config: WebpackConfig, options: { isServer: boolean }) => WebpackConfig
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
  'node_modules/next/dist/server/capsize-font-metrics.json',
  'node_modules/next/dist/server/font-utils.js',
  'node_modules/next/dist/compiled/@next/font/**',
  'node_modules/next/dist/compiled/babel/bundle.js',
  'node_modules/next/dist/compiled/babel-packages/**',
  'node_modules/next/dist/compiled/browserslist/**',
  'node_modules/caniuse-lite/**',
  'node_modules/next/dist/compiled/postcss-preset-env/**',
  'node_modules/next/dist/compiled/cssnano-simple/**',
  'node_modules/next/dist/compiled/crypto-browserify/**',
  'node_modules/next/dist/compiled/@vercel/nft/**',
  'node_modules/next/dist/compiled/next-devtools/**',
  'node_modules/next/dist/compiled/next-server/*-experimental.runtime.prod.js',
  'node_modules/next/dist/compiled/next-server/*-turbo.runtime.prod.js',
  'node_modules/react/cjs/*.development.js',
  'node_modules/react-dom/cjs/*.development.js',
  'node_modules/scheduler/cjs/*.development.js',
  'node_modules/next/dist/compiled/react-is/cjs/*.development.js',
  'node_modules/next/dist/compiled/react-refresh/**',
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

/** The globs scripts/check-standalone-image.sh looks for, from its TRACE_EXCLUDED array (each quoted). */
function checkedGlobs(): string[] {
  const script = read('scripts/check-standalone-image.sh').toString('utf8')
  const block = /^TRACE_EXCLUDED=\(\n([\s\S]*?)\n\)$/m.exec(script)
  return (block?.[1] ?? '')
    .split('\n')
    .map((line) => line.trim().replace(/^'(.*)'$/, '$1'))
    .filter(Boolean)
}

describe('the standalone build without sharp, libvips or TypeScript', () => {
  it('turns Next image optimisation off', () => {
    expect(config.images?.unoptimized).toBe(true)
  })

  it("keeps sharp, libvips, TypeScript, the AMP validator and Next's build internals out of every trace", () => {
    // '*' matches every route and the server's own trace ('next-server').
    expect(config.outputFileTracingExcludes).toEqual({ '*': EXCLUDED })
  })

  it('has the image check look for every glob the trace leaves out', () => {
    expect(checkedGlobs()).toEqual(config.outputFileTracingExcludes?.['*'])
  })
})

describe('the client-only packages, out of the server bundle', () => {
  const CLIENT_ONLY = { 'retell-client-js-sdk': false, 'livekit-client': false, xlsx: false }

  it('resolves each to an empty module in the server builds and keeps the other aliases', () => {
    const base: WebpackConfig = { mode: 'production', resolve: { alias: { react: '/react' } } }
    const server = config.webpack?.(base, { isServer: true })
    expect(server?.resolve?.alias).toEqual({ react: '/react', ...CLIENT_ONLY })
    expect(server?.mode).toBe('production')
    expect(base.resolve?.alias).toEqual({ react: '/react' })
  })

  it('leaves the browser build as it is', () => {
    const base: WebpackConfig = { resolve: { alias: { react: '/react' } } }
    expect(config.webpack?.(base, { isServer: false })).toBe(base)
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
