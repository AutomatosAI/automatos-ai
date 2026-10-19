/**
 * PRD-260 — looks as files. Export writes the look pretty-printed; Import
 * size-checks, parses and validates (a look file is untrusted input), and gives
 * the look a fresh id. (jsdom, for File and FileReader.)
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { AURORA, EMBER } from '@/lib/looks/presets'
import {
  LOOK_FILE_MAX_BYTES,
  NOT_JSON,
  NOTHING_TO_IMPORT,
  TOO_BIG,
  UNREADABLE,
  importLook,
  importLookFile,
  lookFileName,
  lookJson,
  tooBig,
} from '@/lib/looks/transfer'
import { LOOK_FIELDS } from '@/lib/looks/types'

import { myLook } from './helpers'

function jsonFile(text: string, name = 'mine.look.json'): File {
  return new File([text], name, { type: 'application/json' })
}

function errorsOf(text: string): readonly string[] {
  const result = importLook(text, [])
  return result.ok ? [] : result.errors
}

beforeEach(() => {
  vi.spyOn(console, 'warn').mockImplementation(() => undefined)
})

afterEach(() => {
  vi.restoreAllMocks()
})

describe('Export', () => {
  it('writes the look pretty-printed, its fields in format order, ending in a newline', () => {
    const text = lookJson(myLook())
    expect(text.endsWith('}\n')).toBe(true)
    expect(text.split('\n')[1]).toBe('  "schema": "automatos-interface-look/v1",')
    expect(Object.keys(JSON.parse(text))).toEqual([...LOOK_FIELDS])
    expect(JSON.parse(text)).toEqual(myLook())
  })

  it("leaves out anything that isn't part of a look", () => {
    const extra = { ...EMBER, note: 'not a field' }
    expect(JSON.parse(lookJson(extra))).toEqual(EMBER)
  })

  it('names the file after the look, without characters a file system refuses', () => {
    expect(lookFileName(EMBER)).toBe('Ember.look.json')
    expect(lookFileName(myLook({ name: 'Night / Day: v2?' }))).toBe('Night - Day- v2-.look.json')
    expect(lookFileName(myLook({ name: '.hidden' }))).toBe('hidden.look.json')
    expect(lookFileName(myLook({ name: '...' }))).toBe('look.look.json')
  })
})

describe('Import', () => {
  it('round-trips an exported look to an equal look under a fresh id', () => {
    const saved = [myLook()]
    const result = importLook(lookJson(myLook()), saved)
    expect(result).toEqual({ ok: true, look: { ...myLook(), id: 'look-1' } })

    const preset = importLook(lookJson(AURORA), [])
    expect(preset).toEqual({ ok: true, look: { ...AURORA, id: 'look-1' } })
  })

  it('gives a look whose id clashes with a saved one a fresh id', () => {
    const saved = [myLook({ id: 'look-1' })]
    const result = importLook(lookJson(myLook({ id: 'look-1', name: 'Clash' })), saved)
    expect(result).toEqual({ ok: true, look: { ...myLook({ name: 'Clash' }), id: 'look-2' } })
  })

  it('round-trips through a file too', async () => {
    const result = await importLookFile(jsonFile(lookJson(EMBER)), [])
    expect(result).toEqual({ ok: true, look: { ...EMBER, id: 'look-1' } })
  })

  it("says in plain words why a look wasn't imported", () => {
    expect(importLook('', [])).toEqual({ ok: false, errors: [NOTHING_TO_IMPORT] })
    expect(importLook('   \n', [])).toEqual({ ok: false, errors: [NOTHING_TO_IMPORT] })
    expect(importLook('{ name: Ember', [])).toEqual({ ok: false, errors: [NOT_JSON] })
    expect(console.warn).toHaveBeenCalledWith('[looks] import: not JSON', expect.any(SyntaxError))
    expect(importLook('[1, 2]', [])).toEqual({ ok: false, errors: ['A Look must be a JSON object.'] })

    const raw = importLook(JSON.stringify({ ...EMBER, accent: '#ff0000', glow: 99 }), [])
    expect(raw.ok).toBe(false)
    if (raw.ok) return
    expect(raw.errors).toEqual([
      expect.stringContaining('"accent" must be one of'),
      expect.stringContaining('"glow" must be a whole number'),
    ])
  })

  it('refuses an unknown field, a non-catalogue key and a __proto__ field', () => {
    expect(errorsOf(JSON.stringify({ ...EMBER, css: 'body{}' }))).toEqual([`"css" isn't part of a Look. Remove it.`])
    expect(errorsOf(JSON.stringify({ ...EMBER, shell: 'toString' }))).toEqual([
      expect.stringContaining(`"shell" must be one of the catalogue's shells`),
    ])
    const proto = `${lookJson(EMBER).trimEnd().slice(0, -1)},"__proto__":{"accent":"rose"}}`
    expect(errorsOf(proto)).toEqual([`"__proto__" isn't part of a Look. Remove it.`])
  })

  it('says the file could not be read when the read is aborted, instead of waiting forever', async () => {
    vi.spyOn(FileReader.prototype, 'readAsText').mockImplementation(function (this: FileReader) {
      this.dispatchEvent(new ProgressEvent('abort'))
    })
    expect(await importLookFile(jsonFile(lookJson(EMBER)), [])).toEqual({ ok: false, errors: [UNREADABLE] })
  })

  it("refuses a file that isn't a look", async () => {
    const result = await importLookFile(jsonFile(JSON.stringify({ schema: 'something-else' })), [])
    expect(result.ok).toBe(false)
    if (result.ok) return
    expect(result.errors).toContain(`This isn't a v1 Look: "schema" must be "automatos-interface-look/v1".`)
  })
})

describe('the 1 MB limit', () => {
  it('is 1 MB, and a size at or over it is too big', () => {
    expect(LOOK_FILE_MAX_BYTES).toBe(1024 * 1024)
    expect(tooBig(LOOK_FILE_MAX_BYTES)).toBe(true)
    expect(tooBig(LOOK_FILE_MAX_BYTES + 1)).toBe(true)
    expect(tooBig(LOOK_FILE_MAX_BYTES - 1)).toBe(false)
  })

  it('rejects a file over the limit before reading or parsing it', async () => {
    const read = vi.spyOn(FileReader.prototype, 'readAsText')
    const parse = vi.spyOn(JSON, 'parse')
    const big = new File([new Uint8Array(LOOK_FILE_MAX_BYTES + 1)], 'big.look.json', { type: 'application/json' })
    expect(big.size).toBeGreaterThan(LOOK_FILE_MAX_BYTES)
    const result = await importLookFile(big, [])
    expect(result).toEqual({ ok: false, errors: [TOO_BIG] })
    expect(read).not.toHaveBeenCalled()
    expect(parse).not.toHaveBeenCalled()
  })

  it('rejects 1 MB of pasted text before parsing it', () => {
    const parse = vi.spyOn(JSON, 'parse')
    const result = importLook(' '.repeat(LOOK_FILE_MAX_BYTES), [])
    expect(result).toEqual({ ok: false, errors: [TOO_BIG] })
    expect(parse).not.toHaveBeenCalled()
  })
})
