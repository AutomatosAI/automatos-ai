/**
 * PRD-261 A-19d1 — a theme's looks as files, in the generic look runtime, on hello-a09b's data.
 * Export writes the look as the theme's checks keep it, pretty-printed; Import size-checks, parses
 * and checks (a look file is untrusted input) and gives the look a fresh id that no preset and no
 * saved look uses. (jsdom, for File and FileReader.)
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import {
  importLook,
  importLookFile,
  LOOK_FILE_MAX_BYTES,
  LOOK_FILE_TYPE,
  lookFile,
  lookFileName,
  NOT_JSON,
  NOTHING_TO_IMPORT,
  TOO_BIG,
  tooBig,
  UNREADABLE,
} from '@/lib/theme-host/looks/transfer'

import { DUSK, HELLO_RULES, myLook, PAPER } from './hello-looks'

function jsonFile(text: string, name = 'mine.look.json'): File {
  return new File([text], name, { type: LOOK_FILE_TYPE })
}

function textOf(input: unknown): string {
  const file = lookFile(HELLO_RULES, input)
  if (file === null) throw new Error("the look didn't pass the theme's checks")
  return file.text
}

function errorsOf(text: string): readonly string[] {
  const result = importLook(HELLO_RULES, text, [])
  return result.ok ? [] : result.errors
}

beforeEach(() => {
  vi.spyOn(console, 'warn').mockImplementation(() => undefined)
})

afterEach(() => {
  vi.restoreAllMocks()
})

describe('Export', () => {
  it("writes the look pretty-printed, the spec's fields in its order, ending in a newline", () => {
    const text = textOf(myLook())
    expect(text.endsWith('}\n')).toBe(true)
    expect(text.split('\n')[1]).toBe('  "schema": "automatos-e2e-a09b-look/v1",')
    expect(Object.keys(JSON.parse(text))).toEqual(['schema', 'id', 'name', 'sub', 'interface', 'ground', 'accent', 'glow'])
    expect(JSON.parse(text)).toEqual(myLook())
  })

  it("offers a file only for a look the theme's checks take", () => {
    expect(lookFile(HELLO_RULES, PAPER)).toEqual({ fileName: 'Paper.look.json', text: textOf(PAPER), type: LOOK_FILE_TYPE })
    expect(lookFile(HELLO_RULES, { ...PAPER, note: 'not a field' })).toBeNull()
    expect(lookFile(HELLO_RULES, { ...PAPER, accent: 'red' })).toBeNull()
  })

  it('names the file after the look, without characters a file system refuses', () => {
    expect(lookFileName(PAPER)).toBe('Paper.look.json')
    expect(lookFileName(myLook({ name: 'Night / Day: v2?' }))).toBe('Night - Day- v2-.look.json')
    expect(lookFileName(myLook({ name: '.hidden' }))).toBe('hidden.look.json')
    expect(lookFileName(myLook({ name: '...' }))).toBe('look.look.json')
  })
})

describe('Import', () => {
  it('round-trips an exported look to an equal look under a fresh id', () => {
    expect(importLook(HELLO_RULES, textOf(myLook()), [myLook()])).toEqual({ ok: true, look: { ...myLook(), id: 'look-1' } })
    expect(importLook(HELLO_RULES, textOf(DUSK), [])).toEqual({ ok: true, look: { ...DUSK, id: 'look-1' } })
  })

  it('gives a look whose id clashes with a saved one a fresh id', () => {
    const saved = [myLook({ id: 'look-1' })]
    const result = importLook(HELLO_RULES, textOf(myLook({ id: 'look-1', name: 'Clash' })), saved)
    expect(result).toEqual({ ok: true, look: { ...myLook({ name: 'Clash' }), id: 'look-2' } })
  })

  it('round-trips through a file too', async () => {
    const result = await importLookFile(HELLO_RULES, jsonFile(textOf(PAPER)), [])
    expect(result).toEqual({ ok: true, look: { ...PAPER, id: 'look-1' } })
  })

  it("says in plain words why a look wasn't imported", () => {
    expect(importLook(HELLO_RULES, '', [])).toEqual({ ok: false, errors: [NOTHING_TO_IMPORT] })
    expect(importLook(HELLO_RULES, '   \n', [])).toEqual({ ok: false, errors: [NOTHING_TO_IMPORT] })
    expect(importLook(HELLO_RULES, '{ name: Paper', [])).toEqual({ ok: false, errors: [NOT_JSON] })
    expect(console.warn).toHaveBeenCalledWith('[looks] import: not JSON', expect.any(SyntaxError))
    expect(importLook(HELLO_RULES, '[1, 2]', [])).toEqual({ ok: false, errors: ['A Look must be a JSON object.'] })
    expect(errorsOf(JSON.stringify({ ...PAPER, accent: '#ff0000', glow: 99 }))).toEqual([
      expect.stringContaining('"accent" must be one of'),
      expect.stringContaining('"glow" must be a whole number'),
    ])
  })

  it("refuses an unknown field, a key outside the knob's choices and a __proto__ field", () => {
    expect(errorsOf(JSON.stringify({ ...PAPER, css: 'body{}' }))).toEqual([`"css" isn't part of a Look. Remove it.`])
    expect(errorsOf(JSON.stringify({ ...PAPER, ground: 'toString' }))).toEqual([expect.stringContaining('"ground" must be one of')])
    const proto = `${textOf(PAPER).trimEnd().slice(0, -1)},"__proto__":{"accent":"green"}}`
    expect(errorsOf(proto)).toEqual([`"__proto__" isn't part of a Look. Remove it.`])
  })

  it('says the file could not be read when the read is aborted, instead of waiting forever', async () => {
    vi.spyOn(FileReader.prototype, 'readAsText').mockImplementation(function (this: FileReader) {
      this.dispatchEvent(new ProgressEvent('abort'))
    })
    expect(await importLookFile(HELLO_RULES, jsonFile(textOf(PAPER)), [])).toEqual({ ok: false, errors: [UNREADABLE] })
  })

  it("refuses a file that isn't one of the theme's looks", async () => {
    const result = await importLookFile(HELLO_RULES, jsonFile(JSON.stringify({ schema: 'something-else' })), [])
    expect(result.ok).toBe(false)
    if (result.ok) return
    expect(result.errors).toContain(`This isn't a v1 Look: "schema" must be "automatos-e2e-a09b-look/v1".`)
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
    const big = new File([new Uint8Array(LOOK_FILE_MAX_BYTES + 1)], 'big.look.json', { type: LOOK_FILE_TYPE })
    expect(big.size).toBeGreaterThan(LOOK_FILE_MAX_BYTES)
    expect(await importLookFile(HELLO_RULES, big, [])).toEqual({ ok: false, errors: [TOO_BIG] })
    expect(read).not.toHaveBeenCalled()
    expect(parse).not.toHaveBeenCalled()
  })

  it('rejects 1 MB of pasted text before parsing it', () => {
    const parse = vi.spyOn(JSON, 'parse')
    expect(importLook(HELLO_RULES, ' '.repeat(LOOK_FILE_MAX_BYTES), [])).toEqual({ ok: false, errors: [TOO_BIG] })
    expect(parse).not.toHaveBeenCalled()
  })
})
