// @vitest-environment node
/**
 * PRD-262 US-049: public/ holds only what the app serves, at the size it's shown.
 *
 * - The logos are resized to at most 128 px and stored as real PNGs under their old names,
 *   which the backend's seed and lib/logo-service.ts build from brand names. Before this
 *   story 73 were ICO files named .png, Microsoft.png and Nginx.png were HTML pages, and
 *   Sentry.png was in no image format.
 * - scripts/check-public-assets.js, a hard step in test.yml's frontend-ci, fails when public/
 *   grows past scripts/public-budget.json, when a file under public/logos is not a PNG or SVG
 *   by its magic bytes (and the one its name says), or when a logo PNG is over 256 px.
 */
import { execFileSync } from 'child_process'
import fs from 'fs'
import os from 'os'
import path from 'path'
import { createRequire } from 'module'
import { afterEach, describe, expect, it } from 'vitest'

type Budget = { max_bytes: number; reason: string }
type Checked = { total: number; problems: string[] }
type Gate = {
  sniff: (buf: Buffer) => 'png' | 'svg' | null
  pngSize: (buf: Buffer) => { width: number; height: number }
  logoProblems: (rel: string, buf: Buffer) => string[]
  listPublicFiles: (frontendDir: string) => string[]
  checkPublic: (frontendDir: string, files: string[], budget: Budget) => Checked
  readBudget: (file?: string) => Budget
  MAX_LOGO_SIDE: number
}

const load = createRequire(import.meta.url)
const SCRIPT = load.resolve('../check-public-assets.js')
const gate = load(SCRIPT) as Gate
const FRONTEND = path.resolve(path.dirname(SCRIPT), '..')
const LOGOS = path.join(FRONTEND, 'public', 'logos')
// The resize bar: logos are shown at 24-64 px, so 128 px covers a 2x screen.
const LOGO_MAX_PX = 128
const PNG_SIGNATURE = Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a])
const ICO_HEADER = Buffer.from([0x00, 0x00, 0x01, 0x00, 0x01, 0x00, 0x10, 0x10, 0x00, 0x00])
// The first bytes of the old Sentry.png and Microsoft.png.
const SENTRY_OLD = Buffer.from('152d3c00c0ffe7dfb6fa9af5e6abbffa', 'hex')
const HTML_PAGE = Buffer.from('\r\n<!DOCTYPE html>\r\n<html lang="en"><head><title>Microsoft</title>')
const SVG_PLAIN = Buffer.from('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24"><path d="M0 0h24v24H0z"/></svg>')
const SVG_WITH_PROLOG = Buffer.from(
  '\uFEFF<?xml version="1.0" encoding="UTF-8"?>\n<!-- exported -->\n' +
    '<!DOCTYPE svg PUBLIC "-//W3C//DTD SVG 1.1//EN" "http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd">\n' +
    '<svg\n  xmlns="http://www.w3.org/2000/svg"></svg>',
)
const SVG_ILLUSTRATOR = Buffer.from(
  '<?xml version="1.0"?>\n<!DOCTYPE svg PUBLIC "-//W3C//DTD SVG 1.1//EN" "svg11.dtd" [\n' +
    '  <!ENTITY ns_svg "http://www.w3.org/2000/svg">\n]>\n<svg version="1.1" xmlns="&ns_svg;"></svg>',
)
// An HTML page that starts with a doctype is still not an SVG, even with an <svg> in it.
const HTML_WITH_SVG = Buffer.from('<!DOCTYPE html>\n<svg></svg>')

/** The first 33 bytes of a PNG: the signature and an IHDR chunk (enough for the gate). */
function png(width: number, height: number, firstChunk = 'IHDR'): Buffer {
  const ihdr = Buffer.alloc(25)
  ihdr.writeUInt32BE(13, 0)
  ihdr.write(firstChunk, 4, 'latin1')
  ihdr.writeUInt32BE(width, 8)
  ihdr.writeUInt32BE(height, 12)
  ihdr.writeUInt8(8, 16)
  ihdr.writeUInt8(6, 17)
  return Buffer.concat([PNG_SIGNATURE, ihdr])
}

const dirs: string[] = []

function tempDir(files: Record<string, Buffer | string>): string {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'public-assets-'))
  dirs.push(dir)
  for (const [rel, body] of Object.entries(files)) {
    fs.mkdirSync(path.dirname(path.join(dir, rel)), { recursive: true })
    fs.writeFileSync(path.join(dir, rel), body)
  }
  return dir
}

afterEach(() => {
  for (const dir of dirs.splice(0)) fs.rmSync(dir, { recursive: true, force: true })
})

describe('sniff', () => {
  it('knows a PNG by its signature and its IHDR chunk', () => {
    expect(gate.sniff(png(16, 16))).toBe('png')
    expect(gate.sniff(png(16, 16, 'IDAT'))).toBeNull()
    expect(gate.sniff(PNG_SIGNATURE)).toBeNull()
  })

  it('knows an SVG by its root element, after a BOM, XML declaration, comment or doctype', () => {
    expect(gate.sniff(SVG_PLAIN)).toBe('svg')
    expect(gate.sniff(SVG_WITH_PROLOG)).toBe('svg')
    expect(gate.sniff(SVG_ILLUSTRATOR)).toBe('svg')
    expect(gate.sniff(Buffer.from('<svgfoo/>'))).toBeNull()
  })

  it('rejects what public/logos held before: ICO, HTML and bytes in no image format', () => {
    expect(gate.sniff(ICO_HEADER)).toBeNull()
    expect(gate.sniff(HTML_PAGE)).toBeNull()
    expect(gate.sniff(HTML_WITH_SVG)).toBeNull()
    expect(gate.sniff(SENTRY_OLD)).toBeNull()
    expect(gate.sniff(Buffer.alloc(0))).toBeNull()
  })
})

describe('logoProblems', () => {
  const rel = (name: string) => `public/logos/${name}`

  it('passes a PNG up to 256 px on each side, and an SVG', () => {
    expect(gate.logoProblems(rel('A.png'), png(128, 128))).toEqual([])
    expect(gate.logoProblems(rel('A.png'), png(gate.MAX_LOGO_SIDE, 40))).toEqual([])
    expect(gate.logoProblems(rel('A.svg'), SVG_PLAIN)).toEqual([])
  })

  it('fails a PNG wider or taller than 256 px', () => {
    expect(gate.logoProblems(rel('Wide.png'), png(257, 10))).toEqual([
      'public/logos/Wide.png: 257x10 px, over the 256 px limit (logos are shown at 64 px or less)',
    ])
    expect(gate.logoProblems(rel('Tall.png'), png(64, 400))).toHaveLength(1)
  })

  it('fails a .png that is not a PNG, and names what it found', () => {
    const [ico] = gate.logoProblems(rel('AWS.png'), ICO_HEADER)
    expect(ico).toBe('public/logos/AWS.png: named .png, but by its magic bytes it is neither a PNG nor an SVG (it starts 00 00 01 00 01 00 10 10)')
    expect(gate.logoProblems(rel('Microsoft.png'), HTML_PAGE)).toHaveLength(1)
    expect(gate.logoProblems(rel('Sentry.png'), SENTRY_OLD)[0]).toContain('it starts 15 2d 3c 00')
    expect(gate.logoProblems(rel('Empty.png'), Buffer.alloc(0))[0]).toContain('it starts with nothing')
  })

  it('fails a real image under the other extension, and any other kind of file', () => {
    expect(gate.logoProblems(rel('A.png'), SVG_PLAIN)[0]).toContain('it is an SVG')
    expect(gate.logoProblems(rel('A.svg'), png(32, 32))[0]).toContain('it is a PNG')
    expect(gate.logoProblems(rel('A.jpg'), png(32, 32))).toEqual([
      'public/logos/A.jpg: only .png and .svg files belong in public/logos/',
    ])
  })
})

describe('checkPublic', () => {
  const files = {
    'public/logos/Good.png': png(64, 64),
    'public/logos/nested/Also.svg': SVG_PLAIN,
    'public/modal-fix.js': 'x'.repeat(100),
  }
  const total = Object.values(files).reduce((sum, body) => sum + Buffer.byteLength(body), 0)
  const budget = (max_bytes: number): Budget => ({ max_bytes, reason: 'test' })

  it('passes a tree within budget whose logos are real, and checks only files under public/logos', () => {
    const dir = tempDir(files)
    expect(gate.checkPublic(dir, Object.keys(files), budget(total))).toEqual(
      expect.objectContaining({ total, problems: [] }),
    )
  })

  it('fails one byte over the budget', () => {
    const dir = tempDir(files)
    expect(gate.checkPublic(dir, Object.keys(files), budget(total - 1)).problems).toEqual([
      `public/ is ${total} bytes, over its budget of ${total - 1} (scripts/public-budget.json)`,
    ])
  })

  it('reports every bad logo, and skips a listed file that is gone from the working tree', () => {
    const dir = tempDir({ ...files, 'public/logos/Nginx.png': HTML_PAGE, 'public/logos/Big.png': png(512, 512) })
    const { problems, total: counted } = gate.checkPublic(
      dir,
      [...Object.keys(files), 'public/logos/Nginx.png', 'public/logos/Big.png', 'public/logos/Deleted.png'],
      budget(10 ** 9),
    )
    expect(problems).toHaveLength(2)
    expect(counted).toBe(total + HTML_PAGE.length + png(512, 512).length)
  })
})

describe('listPublicFiles', () => {
  it('counts what git tracks or would track, and leaves out files the .gitignore drops', () => {
    const dir = tempDir({
      '.gitignore': '/public/theme-sdk/\n',
      'public/logos/A.png': png(16, 16),
      'public/theme-sdk/v1/react.js': 'export default 1',
      'lib/x.ts': 'export {}',
    })
    execFileSync('git', ['init', '--quiet'], { cwd: dir })
    expect(gate.listPublicFiles(dir)).toEqual(['public/logos/A.png'])
  })
})

describe('readBudget', () => {
  it('reads the committed budget: a positive whole number of bytes, with a reason', () => {
    const budget = gate.readBudget()
    expect(Number.isInteger(budget.max_bytes) && budget.max_bytes > 0).toBe(true)
    expect(budget.reason).toContain('US-049')
  })

  it('refuses a budget without a whole number of bytes or a reason', () => {
    const dir = tempDir({
      'text.json': JSON.stringify({ max_bytes: '10', reason: 'r' }),
      'bare.json': JSON.stringify({ max_bytes: 10 }),
    })
    expect(() => gate.readBudget(path.join(dir, 'text.json'))).toThrow('positive whole number')
    expect(() => gate.readBudget(path.join(dir, 'bare.json'))).toThrow('needs a reason')
  })
})

describe('the committed logos', () => {
  const names = fs.readdirSync(LOGOS).filter((name) => !name.startsWith('.'))

  it('are all real PNGs or SVGs, named for what they are', () => {
    const problems = names.flatMap((name) =>
      gate.logoProblems(`public/logos/${name}`, fs.readFileSync(path.join(LOGOS, name))),
    )
    expect(problems).toEqual([])
    expect(names.length).toBeGreaterThanOrEqual(283)
  })

  it('are resized to at most 128 px on each side', () => {
    const over = names
      .map((name) => ({ name, bytes: fs.readFileSync(path.join(LOGOS, name)) }))
      .filter(({ bytes }) => gate.sniff(bytes) === 'png')
      .map(({ name, bytes }) => ({ name, ...gate.pngSize(bytes) }))
      .filter(({ width, height }) => Math.max(width, height) > LOGO_MAX_PX)
    expect(over).toEqual([])
  })

  it('include real images under the names the seed and logo-service use for Microsoft, Nginx and Sentry', () => {
    for (const name of ['Microsoft.png', 'Nginx.png', 'Sentry.png']) {
      expect(gate.sniff(fs.readFileSync(path.join(LOGOS, name)))).toBe('png')
    }
  })
})
