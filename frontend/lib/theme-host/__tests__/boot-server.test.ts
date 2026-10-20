/**
 * PRD-261 A-08c — first paint for an installed theme: the root layout's resolver for
 * `theme:install:*` fetches the install's boot from the backend with the user's session within
 * 300 ms, keeps it 60 s per user, install and package digest (fetched again at once when the
 * client host saw another digest active), and drops any
 * integrity entry that is not a files-route module with a sha256 value. Anything slow or broken
 * paints the base boot (Studio's classes), and the client host takes over.
 */
import { afterEach, describe, expect, it, vi, type Mock } from 'vitest'

import { baseThemeBoot, builtinThemeBoot, createThemeBootResolver, type ThemeBoot } from '@/lib/theme-host/boot'
import {
  INSTALLED_BOOT_CACHE_TTL_MS,
  INSTALLED_BOOT_TIMEOUT_MS,
  THEME_BOOT_PATH,
  bootDigest,
  createInstalledBootResolver,
  parseDigestHint,
  type BootCaller,
  type InstalledBootDeps,
} from '@/lib/theme-host/boot-server'
import { BUILTIN_PLAY_STYLE } from '@/lib/theme-host/builtin/play-entry'
import { importMapScriptBody } from '@/lib/theme-host/import-map'
import type { UiStyle } from '@/lib/theme-host/registry-types'
import { THEME_DIGEST_COOKIE } from '@/lib/theme-host/types'

const clerk = vi.hoisted(() => ({ auth: vi.fn() }))

const INSTALL_ID = '0b7e4c1a-9f3d-4c2e-8a51-3d2f6b7a9c10'
const INSTALL_STYLE: UiStyle = `theme:install:${INSTALL_ID}`
const OTHER_INSTALL_STYLE: UiStyle = 'theme:install:5d0f2a7e-1c4b-4e9a-b3d6-8f1a2c3e4b5d'
const DIGEST = 'a'.repeat(64)
const NEW_DIGEST = 'b'.repeat(64)
const SHA = `sha256-${'A'.repeat(43)}=`
const OTHER_SHA = `sha256-${'B'.repeat(43)}=`
const NO_COOKIES = () => undefined
const TINY_TIMEOUT_MS = 20

type BootCall = (url: string, init: RequestInit) => Promise<Response>
type BootFetch = Mock<BootCall>

function moduleUrl(digest: string, file = 'dist/entry.js'): string {
  return `/api/themes/files/${digest}/${file}`
}

/** What the backend's boot route (A-08a) answers for a theme that runs. */
function runningBoot(digest = DIGEST, htmlClass = 'hello'): Record<string, unknown> {
  return {
    addClasses: ['studio', htmlClass],
    removeClasses: ['play'],
    tone: { policy: 'user', value: null },
    prepaint: null,
    lookStorage: null,
    lookGuard: null,
    integrity: { [moduleUrl(digest)]: SHA, [moduleUrl(digest, 'dist/chunks/a.js')]: OTHER_SHA },
  }
}

/** The base boot the route answers when the caller may not run the theme (no opt-in, revoked). */
function baseAnswer(): Record<string, unknown> {
  return { ...runningBoot(), addClasses: ['studio'], removeClasses: ['hello', 'play'], integrity: null }
}

function json(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), { status, headers: { 'Content-Type': 'application/json' } })
}

function answering(...bodies: Array<() => Response>): BootFetch {
  const queue = [...bodies]
  return vi.fn<BootCall>(async () => (queue.length > 1 ? queue.shift() : queue[0])?.() ?? json(null, 500))
}

function caller(user = 'user_1', authorization: string | null = 'Bearer token-1'): BootCaller {
  return { user, credential: vi.fn(async () => ({ ok: true as const, authorization })) }
}

function deps(fetchBoot: BootFetch, fields: Partial<InstalledBootDeps> = {}): Partial<InstalledBootDeps> {
  const noHint = async (): Promise<string | null> => null
  return { caller: async () => caller(), digestHint: noHint, fetchBoot, cache: new Map(), now: () => 1_000, ...fields }
}

function requestHeaders(fetchBoot: BootFetch, call = 0): Record<string, string> {
  return fetchBoot.mock.calls[call][1].headers as Record<string, string>
}

afterEach(() => {
  vi.restoreAllMocks()
  clerk.auth.mockReset()
})

describe('PRD-261 A-08c — the installed theme boot, fetched on the server', () => {
  it('asks the boot route for the install with the session token, uncached by Next, and paints its classes', async () => {
    const fetchBoot = answering(() => json(runningBoot()))
    const boot = await createInstalledBootResolver(deps(fetchBoot))(INSTALL_STYLE)
    expect(boot.addClasses).toEqual(['studio', 'hello'])
    expect(boot.integrity).toEqual({ [moduleUrl(DIGEST)]: SHA, [moduleUrl(DIGEST, 'dist/chunks/a.js')]: OTHER_SHA })
    const [url, init] = fetchBoot.mock.calls[0]
    expect(url.endsWith(`${THEME_BOOT_PATH}${INSTALL_ID}`)).toBe(true)
    expect(init).toMatchObject({ cache: 'no-store', redirect: 'manual' })
    expect(requestHeaders(fetchBoot)).toMatchObject({ Authorization: 'Bearer token-1' })
  })

  it('sends no Authorization in the local edition (the backend serves the local operator)', async () => {
    const fetchBoot = answering(() => json(runningBoot()))
    await createInstalledBootResolver(deps(fetchBoot, { caller: async () => caller('local-operator', null) }))(INSTALL_STYLE)
    expect(requestHeaders(fetchBoot)).not.toHaveProperty('Authorization')
  })

  it('has a 300 ms budget and a 60 s cache', () => {
    expect(INSTALLED_BOOT_TIMEOUT_MS).toBe(300)
    expect(INSTALLED_BOOT_CACHE_TTL_MS).toBe(60_000)
  })

  it('paints the base boot when the fetch is slower than the budget, and aborts it', async () => {
    vi.spyOn(console, 'error').mockImplementation(() => undefined)
    let signal: AbortSignal | undefined
    const hanging: BootFetch = vi.fn<BootCall>((_url, init) => {
      signal = init.signal ?? undefined
      return new Promise<Response>(() => undefined)
    })
    const installed = createInstalledBootResolver(deps(hanging, { timeoutMs: TINY_TIMEOUT_MS }))
    expect(await createThemeBootResolver(NO_COOKIES, installed)(INSTALL_STYLE)).toEqual(baseThemeBoot(INSTALL_STYLE))
    expect(signal?.aborted).toBe(true)
    expect(console.error).toHaveBeenCalled()
  })

  it('paints the base boot when the backend fails or answers something that is not a boot', async () => {
    vi.spyOn(console, 'error').mockImplementation(() => undefined)
    const answers = [() => json({ detail: 'down' }, 503), () => json(['not', 'a', 'boot']), () => json('text')]
    for (const answer of answers) {
      const installed = createInstalledBootResolver(deps(answering(answer)))
      expect(await createThemeBootResolver(NO_COOKIES, installed)(INSTALL_STYLE)).toEqual(baseThemeBoot(INSTALL_STYLE))
    }
    const unreachable: BootFetch = vi.fn<BootCall>(async () => Promise.reject(new TypeError('fetch failed')))
    const installed = createInstalledBootResolver(deps(unreachable))
    expect(await createThemeBootResolver(NO_COOKIES, installed)(INSTALL_STYLE)).toEqual(baseThemeBoot(INSTALL_STYLE))
  })

  it('reads 401, 403 and 404 as no theme to paint: the base boot, nothing logged', async () => {
    const logged = vi.spyOn(console, 'error').mockImplementation(() => undefined)
    for (const status of [401, 403, 404]) {
      const installed = createInstalledBootResolver(deps(answering(() => json({ detail: 'no' }, status))))
      expect(await createThemeBootResolver(NO_COOKIES, installed)(INSTALL_STYLE)).toEqual(baseThemeBoot(INSTALL_STYLE))
    }
    expect(logged).not.toHaveBeenCalled()
  })

  it('asks nothing when signed out or when the session has no token', async () => {
    const fetchBoot = answering(() => json(runningBoot()))
    expect(await createInstalledBootResolver(deps(fetchBoot, { caller: async () => null }))(INSTALL_STYLE)).toEqual(
      baseThemeBoot(INSTALL_STYLE),
    )
    const noToken: BootCaller = { user: 'user_1', credential: async () => ({ ok: false }) }
    expect(await createInstalledBootResolver(deps(fetchBoot, { caller: async () => noToken }))(INSTALL_STYLE)).toEqual(
      baseThemeBoot(INSTALL_STYLE),
    )
    expect(fetchBoot).not.toHaveBeenCalled()
  })

  it('is never asked for Classic, Studio or a builtin theme', async () => {
    const fetchBoot = answering(() => json(runningBoot()))
    const installed = createInstalledBootResolver(deps(fetchBoot))
    const resolve = createThemeBootResolver(NO_COOKIES, installed)
    for (const style of ['classic', 'studio', BUILTIN_PLAY_STYLE] as const) {
      expect(await resolve(style)).toEqual(builtinThemeBoot(style, NO_COOKIES))
      expect(await installed(style)).toEqual(baseThemeBoot(style))
    }
    expect(fetchBoot).not.toHaveBeenCalled()
  })
})

describe('PRD-261 A-08c — the per-process cache, per user and install, pinned to one digest', () => {
  it('serves a boot again for 60 s without asking, then asks again', async () => {
    let now = 1_000
    const fetchBoot = answering(() => json(runningBoot()))
    const user = caller()
    const resolve = createInstalledBootResolver(deps(fetchBoot, { caller: async () => user, now: () => now }))
    const first = await resolve(INSTALL_STYLE)
    now += INSTALLED_BOOT_CACHE_TTL_MS - 1
    expect(await resolve(INSTALL_STYLE)).toBe(first)
    expect(fetchBoot).toHaveBeenCalledTimes(1)
    expect(user.credential).toHaveBeenCalledTimes(1)
    now += 1
    await resolve(INSTALL_STYLE)
    expect(fetchBoot).toHaveBeenCalledTimes(2)
  })

  it("never serves one user's boot to another, nor one install's to another", async () => {
    const fetchBoot = answering(
      () => json(runningBoot(DIGEST, 'first')),
      () => json(runningBoot(NEW_DIGEST, 'second')),
      () => json(runningBoot(NEW_DIGEST, 'third')),
    )
    let current = caller('user_1')
    const resolve = createInstalledBootResolver(deps(fetchBoot, { caller: async () => current }))
    expect((await resolve(INSTALL_STYLE)).addClasses).toContain('first')
    current = caller('user_2', 'Bearer token-2')
    expect((await resolve(INSTALL_STYLE)).addClasses).toContain('second')
    expect(requestHeaders(fetchBoot, 1)).toMatchObject({ Authorization: 'Bearer token-2' })
    expect((await resolve(OTHER_INSTALL_STYLE)).addClasses).toContain('third')
    expect(fetchBoot).toHaveBeenCalledTimes(3)
  })

  it('replaces the boot with the new digest after an update, never mixing the two', async () => {
    let now = 1_000
    const fetchBoot = answering(() => json(runningBoot(DIGEST)), () => json(runningBoot(NEW_DIGEST)))
    const cache = new Map()
    const resolve = createInstalledBootResolver(deps(fetchBoot, { cache, now: () => now }))
    expect(bootDigest(await resolve(INSTALL_STYLE))).toBe(DIGEST)
    now += INSTALLED_BOOT_CACHE_TTL_MS
    const updated = await resolve(INSTALL_STYLE)
    expect(bootDigest(updated)).toBe(NEW_DIGEST)
    expect(Object.keys(updated.integrity ?? {}).every((url) => url.includes(NEW_DIGEST))).toBe(true)
    expect(Array.from(cache.values()).map((entry) => entry.digest)).toEqual([NEW_DIGEST])
  })

  it('fetches again at once when the client host saw another digest active (an update, a rollback)', async () => {
    const fetchBoot = answering(() => json(runningBoot(DIGEST)), () => json(runningBoot(NEW_DIGEST)))
    let hint: string | null = DIGEST
    const resolve = createInstalledBootResolver(deps(fetchBoot, { digestHint: async () => hint }))
    expect(bootDigest(await resolve(INSTALL_STYLE))).toBe(DIGEST)
    expect(bootDigest(await resolve(INSTALL_STYLE))).toBe(DIGEST)
    expect(fetchBoot).toHaveBeenCalledTimes(1)
    hint = NEW_DIGEST
    expect(bootDigest(await resolve(INSTALL_STYLE))).toBe(NEW_DIGEST)
    expect(bootDigest(await resolve(INSTALL_STYLE))).toBe(NEW_DIGEST)
    expect(fetchBoot).toHaveBeenCalledTimes(2)
  })

  it("reads the hint only for this install, with a sha256 digest", () => {
    expect(parseDigestHint(`${INSTALL_ID}.${DIGEST}`, INSTALL_ID)).toBe(DIGEST)
    expect(parseDigestHint(undefined, INSTALL_ID)).toBeNull()
    expect(parseDigestHint(`5d0f2a7e-1c4b-4e9a-b3d6-8f1a2c3e4b5d.${DIGEST}`, INSTALL_ID)).toBeNull()
    expect(parseDigestHint(`${INSTALL_ID}.${'A'.repeat(64)}`, INSTALL_ID)).toBeNull()
    expect(parseDigestHint(`${INSTALL_ID}.${DIGEST}0`, INSTALL_ID)).toBeNull()
    expect(parseDigestHint(DIGEST, INSTALL_ID)).toBeNull()
  })

  it('keeps no base boot, so an opt-in is painted on the next load', async () => {
    const fetchBoot = answering(() => json(baseAnswer()), () => json(runningBoot()))
    const resolve = createInstalledBootResolver(deps(fetchBoot))
    expect((await resolve(INSTALL_STYLE)).addClasses).toEqual(['studio'])
    expect((await resolve(INSTALL_STYLE)).addClasses).toEqual(['studio', 'hello'])
    expect(fetchBoot).toHaveBeenCalledTimes(2)
  })

  it('stays within its bound: the oldest entry goes first', async () => {
    const fetchBoot = answering(() => json(runningBoot()))
    const cache = new Map()
    let user = 'user_1'
    const resolve = createInstalledBootResolver(deps(fetchBoot, { cache, maxEntries: 2, caller: async () => caller(user) }))
    for (const next of ['user_1', 'user_2', 'user_3']) {
      user = next
      await resolve(INSTALL_STYLE)
    }
    expect(Array.from(cache.keys())).toEqual([JSON.stringify(['user_2', INSTALL_ID]), JSON.stringify(['user_3', INSTALL_ID])])
  })
})

describe('PRD-261 A-08c — the integrity block written into the import map (PA-13)', () => {
  it('drops an entry whose value is not sha256 or whose URL is not a files-route module', async () => {
    const integrity = {
      [moduleUrl(DIGEST)]: SHA,
      [moduleUrl(DIGEST, 'dist/b.js')]: 'sha384-' + 'A'.repeat(64),
      [moduleUrl(DIGEST, 'dist/c.js')]: 'sha256-short=',
      [moduleUrl(DIGEST, 'dist/theme.css')]: SHA,
      [moduleUrl(DIGEST, 'dist/../escape.js')]: SHA,
      [`https://evil.example${moduleUrl(DIGEST)}`]: SHA,
      [`/api/themes/files/${DIGEST.toUpperCase()}/dist/d.js`]: SHA,
      [moduleUrl(DIGEST, 'dist/</script>.js')]: SHA,
    }
    const fetchBoot = answering(() => json({ ...runningBoot(), integrity }))
    const boot = await createInstalledBootResolver(deps(fetchBoot))(INSTALL_STYLE)
    expect(boot.integrity).toEqual({ [moduleUrl(DIGEST)]: SHA })
    const body = importMapScriptBody(boot.integrity)
    expect(body).not.toContain('</script>')
    expect(JSON.parse(body).integrity).toEqual({ [moduleUrl(DIGEST)]: SHA })
  })

  it('writes the map as scriptSafeJson of the whole object', async () => {
    const boot = await createInstalledBootResolver(deps(answering(() => json(runningBoot()))))(INSTALL_STYLE)
    const body = importMapScriptBody(boot.integrity)
    const parsed = JSON.parse(body) as { imports: Record<string, string>; integrity: Record<string, string> }
    expect(Object.keys(parsed.imports).length).toBeGreaterThan(0)
    expect(parsed.integrity).toEqual(boot.integrity)
    expect(body).toBe(body.replace(/</g, '\\u003c'))
  })

  it('refuses a boot whose integrity names two package digests: the base boot paints', async () => {
    vi.spyOn(console, 'error').mockImplementation(() => undefined)
    const mixed = { ...runningBoot(), integrity: { [moduleUrl(DIGEST)]: SHA, [moduleUrl(NEW_DIGEST)]: SHA } }
    const cache = new Map()
    const installed = createInstalledBootResolver(deps(answering(() => json(mixed)), { cache }))
    expect(await createThemeBootResolver(NO_COOKIES, installed)(INSTALL_STYLE)).toEqual(baseThemeBoot(INSTALL_STYLE))
    expect(cache.size).toBe(0)
  })

  it('drops a class the theme-class pattern refuses, as every boot does', async () => {
    const answer = { ...runningBoot(), addClasses: ['studio', 'hello', '</script>', 'Bad Class'] }
    const boot: ThemeBoot = await createInstalledBootResolver(deps(answering(() => json(answer))))(INSTALL_STYLE)
    expect(boot.addClasses).toEqual(['studio', 'hello'])
  })
})

describe('PRD-261 A-08c — whose session the boot is fetched with', () => {
  async function sessionCaller(saas: boolean): Promise<() => Promise<BootCaller | null>> {
    vi.resetModules()
    vi.doMock('@/lib/auth-edition', () => ({ isSaaS: saas }))
    vi.doMock('@clerk/nextjs/server', () => ({ auth: clerk.auth }))
    return (await import('@/lib/theme-host/boot-server')).sessionBootCaller
  }

  it('local edition: the local operator, no token, and Clerk is never asked', async () => {
    const found = await (await sessionCaller(false))()
    expect(found?.user).toBe('local-operator')
    expect(await found?.credential()).toEqual({ ok: true, authorization: null })
    expect(clerk.auth).not.toHaveBeenCalled()
  })

  it('SaaS: the Clerk user and its token as Bearer, read only when asked', async () => {
    const getToken = vi.fn(async () => 'session-token')
    clerk.auth.mockResolvedValue({ userId: 'user_1', getToken })
    const found = await (await sessionCaller(true))()
    expect(found?.user).toBe('user_1')
    expect(getToken).not.toHaveBeenCalled()
    expect(await found?.credential()).toEqual({ ok: true, authorization: 'Bearer session-token' })
  })

  it('SaaS: nobody when signed out or the session cannot be read; not ok without a token', async () => {
    const read = await sessionCaller(true)
    clerk.auth.mockResolvedValue({ userId: null, getToken: vi.fn(async () => null) })
    expect(await read()).toBeNull()
    clerk.auth.mockResolvedValue({ userId: 'user_1', getToken: vi.fn(async () => null) })
    expect(await (await read())?.credential()).toEqual({ ok: false })
    vi.spyOn(console, 'error').mockImplementation(() => undefined)
    clerk.auth.mockRejectedValue(new Error('auth() was called outside a request'))
    expect(await read()).toBeNull()
  })
})

describe('PRD-261 A-08c — the digest hint the request carries', () => {
  async function hintReader(cookies: () => Promise<unknown>): Promise<(installId: string) => Promise<string | null>> {
    vi.resetModules()
    vi.doMock('next/headers', () => ({ cookies }))
    return (await import('@/lib/theme-host/boot-server')).requestDigestHint
  }

  it("reads the install's digest from THEME_DIGEST_COOKIE; null outside a request", async () => {
    const jar = { get: (name: string) => (name === THEME_DIGEST_COOKIE ? { name, value: `${INSTALL_ID}.${DIGEST}` } : undefined) }
    expect(await (await hintReader(async () => jar))(INSTALL_ID)).toBe(DIGEST)
    vi.spyOn(console, 'error').mockImplementation(() => undefined)
    const outside = await hintReader(async () => {
      throw new Error('cookies() was called outside a request scope')
    })
    expect(await outside(INSTALL_ID)).toBeNull()
  })
})
