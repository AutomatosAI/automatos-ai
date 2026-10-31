/**
 * PRD-261 E3a — the theme-state store (lib/theme-sdk/internal/theme-state-store.ts): the key scheme,
 * JSON-only values, the caps and the eviction of plan §2.2.1, against an in-memory Storage. The store
 * takes its storage as a parameter, so nothing here touches the browser's.
 */
import { readFileSync } from 'fs'
import path from 'path'
import { afterEach, describe, expect, it, vi } from 'vitest'

import {
  checkThemeStateValue,
  isThemeStateKey,
  parseThemeStateBlob,
  planThemeStateWrite,
  readThemeStateBlob,
  readThemeStateEntry,
  removeThemeStateEntry,
  THEME_STATE_KEY_PREFIX,
  THEME_STATE_LIMITS,
  themeStateScope,
  themeStateStorageKey,
  type ThemeStateStorage,
  writeThemeStateEntry,
} from '../internal/theme-state-store'

const NOW = 1760000000000
const UUID = '3f2b8c1e-9a4d-4c2b-8e6f-1a2b3c4d5e6f'
const KEY = 'automatos-theme-state:v1:b:automatos.play:u:user_1'
/** The installed-looks and theme-choice prefixes (lib/theme-host/looks/storage.ts, choice-sync.ts), named as the plan names them. */
const LOOKS_PREFIX = 'automatos-theme.'
const CHOICE_PREFIX = 'automatos-theme-choice:'

/** A Storage held in a Map, in insertion order, like the browser's. */
class MemoryStorage implements ThemeStateStorage {
  readonly items = new Map<string, string>()
  get length(): number {
    return this.items.size
  }
  key(index: number): string | null {
    return Array.from(this.items.keys())[index] ?? null
  }
  getItem(key: string): string | null {
    return this.items.has(key) ? (this.items.get(key) as string) : null
  }
  setItem(key: string, value: string): void {
    this.items.set(key, value)
  }
  removeItem(key: string): void {
    this.items.delete(key)
  }
}

function blob(at: number, data: Record<string, unknown>): string {
  return JSON.stringify({ v: 1, at, data })
}

function nested(levels: number): unknown {
  let value: unknown = 1
  for (let level = 0; level < levels; level += 1) value = [value]
  return value
}

function jsonString(chars: number): string {
  return 'x'.repeat(chars - 2)
}

function expectDeepFrozen(value: unknown): void {
  if (value === null || typeof value !== 'object') return
  expect(Object.isFrozen(value)).toBe(true)
  Object.values(value).forEach(expectDeepFrozen)
}

afterEach(() => {
  vi.restoreAllMocks()
})

describe('E3a key scheme and limits', () => {
  it('names a blob by lifetime, theme and scope', () => {
    const account = themeStateScope({ userKey: 'user_1', workspaceKey: 'ws-1' }, 'account')
    const workspace = themeStateScope({ userKey: 'local', workspaceKey: 'ws-1' }, 'workspace')
    expect(account).toBe('u:user_1')
    expect(workspace).toBe('w:ws-1:u:local')
    expect(themeStateStorageKey('browser', 'automatos.play', account as string)).toBe(KEY)
    expect(themeStateStorageKey('session', 'automatos.play', workspace as string)).toBe(
      'automatos-theme-state:v1:s:automatos.play:w:ws-1:u:local',
    )
  })

  it('has no scope without the ids it needs, or with an id holding a separator', () => {
    expect(themeStateScope({ userKey: 'user_1', workspaceKey: null }, 'workspace')).toBeNull()
    expect(themeStateScope({ userKey: '', workspaceKey: 'ws-1' }, 'account')).toBeNull()
    expect(themeStateScope({ userKey: 'a:b', workspaceKey: 'ws-1' }, 'account')).toBeNull()
    expect(themeStateScope({ userKey: 'user_1', workspaceKey: 'w:x' }, 'workspace')).toBeNull()
    expect(themeStateStorageKey('browser', '', 'u:local')).toBeNull()
    expect(themeStateStorageKey('browser', 'a:b', 'u:local')).toBeNull()
  })

  it('freezes the limits at the contract values', () => {
    expect(Object.isFrozen(THEME_STATE_LIMITS)).toBe(true)
    expect(THEME_STATE_LIMITS).toEqual({
      keyPattern: '^[a-z][a-z0-9-]{0,47}$',
      keysPerBlob: 32,
      valueChars: 32768,
      blobChars: 65536,
      depth: 8,
      blobsPerLifetime: 8,
    })
  })

  it('uses a prefix that overlaps neither the installed looks nor the theme choice', () => {
    expect(THEME_STATE_KEY_PREFIX).toBe('automatos-theme-state:v1:')
    for (const other of [LOOKS_PREFIX, CHOICE_PREFIX]) {
      expect(THEME_STATE_KEY_PREFIX.startsWith(other)).toBe(false)
      expect(other.startsWith(THEME_STATE_KEY_PREFIX)).toBe(false)
      expect(KEY.startsWith(other)).toBe(false)
    }
  })
})

describe('E3a theme keys', () => {
  it.each(['mascot-paused', `seen-${UUID}`, 'a', `a${'b'.repeat(47)}`])('accepts %s', (key) => {
    expect(isThemeStateKey(key)).toBe(true)
    expect(writeThemeStateEntry(new MemoryStorage(), KEY, key, true, NOW).ok).toBe(true)
  })

  it.each(['', 'Bad', 'a:b', `a${'b'.repeat(48)}`, '1st', 'a_b', 'mascot\n'])('refuses %j as bad-key', (key) => {
    const storage = new MemoryStorage()
    expect(isThemeStateKey(key)).toBe(false)
    expect(writeThemeStateEntry(storage, KEY, key, true, NOW)).toMatchObject({ ok: false, reason: 'bad-key' })
    expect(readThemeStateEntry(storage, KEY, key)).toMatchObject({ ok: false, reason: 'bad-key' })
    expect(removeThemeStateEntry(storage, KEY, key, NOW)).toMatchObject({ ok: false, reason: 'bad-key' })
    expect(storage.items.size).toBe(0)
  })
})

describe('E3a values are JSON only', () => {
  class Instance {
    readonly n = 1
  }
  const cyclic: Record<string, unknown> = { a: 1 }
  cyclic.self = cyclic
  const throwing = Object.defineProperty({}, 'boom', { enumerable: true, get: () => { throw new Error('boom') } })
  const bad: [string, unknown][] = [
    ['undefined', undefined],
    ['a function', () => 1],
    ['NaN', Number.NaN],
    ['Infinity', Number.POSITIVE_INFINITY],
    ['a Date', new Date(NOW)],
    ['a Map', new Map([['a', 1]])],
    ['a class instance', new Instance()],
    ['depth 9', nested(9)],
    ['a cyclic object', cyclic],
    ['a nested undefined', { a: [1, undefined] }],
    ['a symbol', Symbol('s')],
    ['a bigint', BigInt(1)],
    ['an array with holes', new Array<number>(2)],
    ['a throwing getter', throwing],
  ]

  it.each(bad)('refuses %s as bad-value, never throwing', (_name, value) => {
    const storage = new MemoryStorage()
    let result: unknown
    expect(() => {
      result = writeThemeStateEntry(storage, KEY, 'mascot-paused', value, NOW)
    }).not.toThrow()
    expect(result).toMatchObject({ ok: false, reason: 'bad-value' })
    expect(storage.items.size).toBe(0)
  })

  const good: [string, unknown][] = [
    ['null', null],
    ['true', true],
    ['false', false],
    ['a finite number', -12.5],
    ['zero', 0],
    ['a string', 'hello'],
    ['an array', [1, 'two', null, [true]]],
    ['a plain object', { a: { b: [1, { c: 'd' }] } }],
    ['a null-prototype object', Object.assign(Object.create(null), { a: 1 })],
    ['depth 8', nested(8)],
  ]

  it.each(good)('keeps %s and reads it back', (_name, value) => {
    const storage = new MemoryStorage()
    expect(writeThemeStateEntry(storage, KEY, 'mascot-paused', value, NOW)).toMatchObject({ ok: true, evicted: [] })
    const read = readThemeStateEntry(storage, KEY, 'mascot-paused')
    expect(read).toEqual({ ok: true, value: JSON.parse(JSON.stringify(value)) })
  })

  it('stores the blob as {v, at, data}', () => {
    const storage = new MemoryStorage()
    writeThemeStateEntry(storage, KEY, 'mascot-paused', true, NOW)
    writeThemeStateEntry(storage, KEY, `seen-${UUID}`, [1, 2], NOW + 1)
    expect(JSON.parse(storage.getItem(KEY) as string)).toEqual({
      v: 1,
      at: NOW + 1,
      data: { 'mascot-paused': true, [`seen-${UUID}`]: [1, 2] },
    })
  })
})

describe('E3a caps leave the stored blob byte-identical', () => {
  function seeded(): MemoryStorage {
    const storage = new MemoryStorage()
    writeThemeStateEntry(storage, KEY, 'mascot-paused', true, NOW)
    return storage
  }

  function expectRefused(storage: MemoryStorage, key: string, value: unknown): void {
    const before = storage.getItem(KEY)
    const setItem = vi.spyOn(storage, 'setItem')
    expect(writeThemeStateEntry(storage, KEY, key, value, NOW + 5)).toMatchObject({ ok: false, reason: 'too-large' })
    expect(setItem).not.toHaveBeenCalled()
    expect(storage.getItem(KEY)).toBe(before)
  }

  it('takes a value of exactly 32,768 JSON chars and refuses 32,769', () => {
    expect(checkThemeStateValue(jsonString(THEME_STATE_LIMITS.valueChars)).ok).toBe(true)
    expectRefused(seeded(), 'big', jsonString(THEME_STATE_LIMITS.valueChars + 1))
  })

  it('refuses a 33rd key, and still overwrites one of the 32', () => {
    const storage = new MemoryStorage()
    for (let n = 0; n < THEME_STATE_LIMITS.keysPerBlob; n += 1) writeThemeStateEntry(storage, KEY, `k-${n}`, n, NOW)
    expectRefused(storage, 'one-more', 1)
    expect(writeThemeStateEntry(storage, KEY, 'k-0', 'again', NOW).ok).toBe(true)
  })

  it('refuses a write that takes the blob over 65,536 chars', () => {
    const storage = seeded()
    const half = jsonString(32000)
    expect(writeThemeStateEntry(storage, KEY, 'first', half, NOW).ok).toBe(true)
    expect(writeThemeStateEntry(storage, KEY, 'second', half, NOW).ok).toBe(true)
    expect((storage.getItem(KEY) as string).length).toBeLessThanOrEqual(THEME_STATE_LIMITS.blobChars)
    expectRefused(storage, 'third', jsonString(2000))
  })
})

describe('E3a corrupt blobs', () => {
  it.each([
    ['not JSON', '{not json'],
    ['another version', JSON.stringify({ v: 2, at: NOW, data: { a: 1 } })],
    ['no time', JSON.stringify({ v: 1, data: { a: 1 } })],
    ['array data', JSON.stringify({ v: 1, at: NOW, data: [1] })],
    ['a bad key inside', blob(NOW, { Bad: 1 })],
    ['too deep inside', blob(NOW, { a: nested(9) })],
    ['a JSON scalar', '42'],
  ])('reads %s as empty and replaces it on the next write', (_name, raw) => {
    const storage = new MemoryStorage()
    storage.setItem(KEY, raw)
    expect(readThemeStateEntry(storage, KEY, 'a')).toEqual({ ok: true, value: null })
    expect(readThemeStateBlob(storage, KEY)).toMatchObject({ ok: true, raw, at: null, data: {} })
    expect(writeThemeStateEntry(storage, KEY, 'mascot-paused', true, NOW).ok).toBe(true)
    expect(JSON.parse(storage.getItem(KEY) as string)).toEqual({ v: 1, at: NOW, data: { 'mascot-paused': true } })
  })
})

describe('E3a eviction', () => {
  const themes = ['t-a', 't-b', 't-c', 't-d', 't-e', 't-f', 't-g', 't-h']
  const ats = [500, 200, 800, 100, 700, 300, 600, 400]
  const keyOf = (theme: string) => themeStateStorageKey('browser', theme, 'u:user_1') as string

  function withEightOthers(): MemoryStorage {
    const storage = new MemoryStorage()
    themes.forEach((theme, index) => storage.setItem(keyOf(theme), blob(ats[index], { a: index })))
    storage.setItem(`${LOOKS_PREFIX}t-z.looks`, blob(1, { a: 1 }))
    storage.setItem(`${CHOICE_PREFIX}ws-1`, 'checked')
    storage.setItem('automatos-theme-state:v2:b:t-y:u:user_1', blob(2, { a: 1 }))
    storage.setItem('xautomatos-theme-state:v1:b:t-x:u:user_1', blob(3, { a: 1 }))
    return storage
  }

  it("names the lowest-'at' blob when a 9th theme writes, counting only the exact prefix", () => {
    const storage = withEightOthers()
    const plan = planThemeStateWrite(storage, keyOf('t-new'), 'a', 1, NOW)
    expect(plan).toEqual({ ok: true, raw: blob(NOW, { a: 1 }), evict: [keyOf('t-d')] })
    expect(writeThemeStateEntry(storage, keyOf('t-new'), 'a', 1, NOW)).toMatchObject({ ok: true, evicted: [keyOf('t-d')] })
    expect(storage.getItem(keyOf('t-d'))).toBeNull()
    expect(storage.items.size).toBe(12)
    expect(storage.getItem(`${LOOKS_PREFIX}t-z.looks`)).not.toBeNull()
    expect(storage.getItem('automatos-theme-state:v2:b:t-y:u:user_1')).not.toBeNull()
  })

  it('never evicts the blob being written, even when it is the oldest', () => {
    const storage = withEightOthers()
    storage.setItem(keyOf('t-new'), blob(1, { a: 0 }))
    expect(planThemeStateWrite(storage, keyOf('t-new'), 'a', 1, NOW)).toMatchObject({ ok: true, evict: [keyOf('t-d')] })
    storage.removeItem(keyOf('t-h'))
    expect(planThemeStateWrite(storage, keyOf('t-new'), 'a', 1, NOW)).toMatchObject({ ok: true, evict: [] })
  })

  it('evicts nothing with seven others, and an unreadable blob first', () => {
    const storage = withEightOthers()
    storage.removeItem(keyOf('t-a'))
    expect(planThemeStateWrite(storage, keyOf('t-new'), 'a', 1, NOW)).toMatchObject({ ok: true, evict: [] })
    storage.setItem(keyOf('t-a'), '{broken')
    expect(planThemeStateWrite(storage, keyOf('t-new'), 'a', 1, NOW)).toMatchObject({ ok: true, evict: [keyOf('t-a')] })
  })
})

describe('E3a reads and inputs', () => {
  it('reads deeply frozen values, the same objects for the same blob', () => {
    const storage = new MemoryStorage()
    writeThemeStateEntry(storage, KEY, 'shelf', { items: [{ id: 'a', tags: ['x'] }] }, NOW)
    const first = readThemeStateEntry(storage, KEY, 'shelf')
    const second = readThemeStateEntry(storage, KEY, 'shelf')
    if (!first.ok || !second.ok) throw new Error('the reads failed')
    expect(first.value).toEqual({ items: [{ id: 'a', tags: ['x'] }] })
    expectDeepFrozen(first.value)
    expect(first.value).toBe(second.value)
    expect(parseThemeStateBlob(storage.getItem(KEY)).data).toBe(parseThemeStateBlob(storage.getItem(KEY)).data)
  })

  it('does not mutate or freeze what a theme hands in', () => {
    const storage = new MemoryStorage()
    const value = { items: [{ id: 'a' }] }
    const copy = JSON.parse(JSON.stringify(value))
    writeThemeStateEntry(storage, KEY, 'shelf', value, NOW)
    writeThemeStateEntry(storage, KEY, 'bad', { when: new Date(NOW), items: value.items }, NOW)
    expect(value).toEqual(copy)
    expect(Object.isFrozen(value) || Object.isFrozen(value.items)).toBe(false)
  })

  it('removes a key, and the blob with its last key; an absent key writes nothing', () => {
    const storage = new MemoryStorage()
    writeThemeStateEntry(storage, KEY, 'a', 1, NOW)
    writeThemeStateEntry(storage, KEY, 'b', 2, NOW)
    const setItem = vi.spyOn(storage, 'setItem')
    expect(removeThemeStateEntry(storage, KEY, 'missing', NOW)).toMatchObject({ ok: true, evicted: [] })
    expect(setItem).not.toHaveBeenCalled()
    expect(removeThemeStateEntry(storage, KEY, 'a', NOW + 1)).toEqual({ ok: true, raw: blob(NOW + 1, { b: 2 }), evicted: [] })
    expect(removeThemeStateEntry(storage, KEY, 'b', NOW + 2)).toEqual({ ok: true, raw: null, evicted: [] })
    expect(storage.getItem(KEY)).toBeNull()
  })

  it('reports a storage that throws as unavailable, never throwing', () => {
    const storage = new MemoryStorage()
    const blocked = () => {
      throw new DOMException('blocked', 'SecurityError')
    }
    vi.spyOn(storage, 'getItem').mockImplementation(blocked)
    vi.spyOn(storage, 'setItem').mockImplementation(blocked)
    expect(readThemeStateEntry(storage, KEY, 'a')).toMatchObject({ ok: false, reason: 'unavailable' })
    expect(writeThemeStateEntry(storage, KEY, 'a', 1, NOW)).toMatchObject({ ok: false, reason: 'unavailable' })
    expect(removeThemeStateEntry(storage, KEY, 'a', NOW)).toMatchObject({ ok: false, reason: 'unavailable' })
  })
})

describe('E3a stays pure and ES2020', () => {
  it("never touches the browser's storage", () => {
    const touched = [
      vi.spyOn(Storage.prototype, 'getItem'),
      vi.spyOn(Storage.prototype, 'setItem'),
      vi.spyOn(Storage.prototype, 'removeItem'),
      vi.spyOn(Storage.prototype, 'key'),
    ]
    const storage = new MemoryStorage()
    writeThemeStateEntry(storage, KEY, 'a', 1, NOW)
    readThemeStateEntry(storage, KEY, 'a')
    removeThemeStateEntry(storage, KEY, 'a', NOW)
    touched.forEach((spy) => expect(spy).not.toHaveBeenCalled())
    const source = readFileSync(path.resolve(__dirname, '../internal/theme-state-store.ts'), 'utf8')
    expect(source).not.toMatch(/\bwindow\.|\bglobalThis\b|\blocalStorage\b|\bsessionStorage\b|\bdocument\./)
  })

  it('is type-checked against ES2020, so tsc refuses replaceAll, .at(, Object.hasOwn, Promise.any and { cause }', () => {
    const tsconfig = JSON.parse(readFileSync(path.resolve(__dirname, '../../../tsconfig.json'), 'utf8'))
    const libs: string[] = tsconfig.compilerOptions.lib.map((lib: string) => lib.toLowerCase())
    expect(tsconfig.compilerOptions.target.toLowerCase()).toBe('es2020')
    const later = libs.filter((lib) => lib === 'esnext' || (/^es(\d{4})/.test(lib) && Number(/^es(\d{4})/.exec(lib)?.[1]) > 2020))
    expect(later).toEqual([])
  })
})
