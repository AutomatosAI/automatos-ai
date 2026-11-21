// @vitest-environment node
/**
 * PRD-277 D-12 (#1605): the enterprise web app's content security policy names only what that
 * edition uses. No Clerk host and no Cloudflare challenge (it mounts no Clerk), no Automatos
 * origin (its API is the page's own), no payment host (no billing). Automatos Cloud's screenshot
 * path is in img-src only while ENTERPRISE_CALLS_CLOUD_SYNC is on, read when the server runs, the
 * way MEDIA_ORIGINS is: refused at start-up when malformed, read on each response.
 * The local and hosted policies are pinned, untouched, in runtime-csp.test.ts and the csp-*.test.ts files.
 */
import { NextRequest } from 'next/server'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { CLOUD_SYNC_REFUSED, CLOUD_SYNC_SETTING, parseCloudSync } from '@/lib/enterprise/cloud-sync-setting'
import { CSP_HEADER, enterprisePolicy, runningPolicy } from '@/lib/enterprise/content-security-policy'
import { enterpriseMiddleware } from '@/lib/enterprise/session-middleware'
import { enterpriseStartUpProblem } from '@/lib/enterprise/start-up'

const CLOUD_IMAGES = 'https://api.automatos.app/api/marketplace/catalog/files/'
const MEDIA = 'https://media.example.test'
const READY = { BACKEND_INTERNAL_URL: 'http://automatos-api:8000' }

const POLICY_OFF = [
  "default-src 'self'",
  "script-src 'self' 'unsafe-inline' 'unsafe-eval' https://cdn.jsdelivr.net",
  "style-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net",
  "img-src 'self' data: blob: https://*.googleusercontent.com https://logos.composio.dev",
  "media-src 'self' blob:",
  "font-src 'self' data: https://cdn.jsdelivr.net",
  "connect-src 'self' https://cdn.jsdelivr.net wss: ws:",
  "frame-src 'self' blob:",
  "worker-src 'self' blob:",
  'object-src blob:',
  "base-uri 'self'",
  "form-action 'self'",
  "frame-ancestors 'none'",
].join('; ')

const POLICY_ON = POLICY_OFF.replace(
  "https://logos.composio.dev;",
  `https://logos.composio.dev ${CLOUD_IMAGES};`,
)

/** Hosts the enterprise policy must never name: Automatos's, the hosted sign-in provider's, its challenge's, the payment provider's. */
const NEVER = ['automatos.app', 'clerk', 'cloudflare', 'stripe', 'link.com', 'maps.googleapis.com']

afterEach(() => {
  vi.unstubAllEnvs()
  vi.restoreAllMocks()
})

describe('the enterprise policy', () => {
  it('with Cloud sync off is this string, and names no Automatos, sign-in or payment host', () => {
    expect(enterprisePolicy([])).toBe(POLICY_OFF)
    expect(runningPolicy('', '')).toBe(POLICY_OFF)
    for (const host of NEVER) expect(POLICY_OFF).not.toContain(host)
  })

  it('with Cloud sync on is this string: only the Cloud files path is added, in img-src alone', () => {
    expect(POLICY_ON).not.toBe(POLICY_OFF)
    expect(enterprisePolicy([], true)).toBe(POLICY_ON)
    expect(runningPolicy('', 'true')).toBe(POLICY_ON)
    const rest = POLICY_ON.replace(` ${CLOUD_IMAGES}`, '')
    expect(rest).toBe(POLICY_OFF)
    for (const host of NEVER) expect(rest).not.toContain(host)
    expect(POLICY_ON).not.toContain('https://api.automatos.app ')
    expect(POLICY_ON).not.toContain('https://api.automatos.app;')
  })

  it('reads the setting as the API does: trimmed, lower-cased, and its switch forms', () => {
    for (const on of ['1', 'true', ' TRUE ', 'yes', 'On']) expect(runningPolicy('', on)).toBe(POLICY_ON)
    for (const off of [undefined, '', '0', 'false', 'no', ' OFF ']) expect(runningPolicy('', off)).toBe(POLICY_OFF)
  })

  it('keeps the media origins with Cloud sync on, after the Cloud path', () => {
    const img = enterprisePolicy([MEDIA], true).split('; ').find((part) => part.startsWith('img-src '))
    expect(img).toBe(`img-src 'self' data: blob: https://*.googleusercontent.com https://logos.composio.dev ${CLOUD_IMAGES} ${MEDIA}`)
  })

  it('names no Cloud screenshot for a malformed value, and logs the setting once, without the value', () => {
    const logged = vi.spyOn(console, 'error').mockImplementation(() => {})
    for (let n = 0; n < 3; n += 1) expect(runningPolicy('', 'enabled-please')).toBe(POLICY_OFF)
    expect(runningPolicy('', 'maybe-later')).toBe(POLICY_OFF)
    expect(logged).toHaveBeenCalledTimes(1)
    expect(String(logged.mock.calls[0][0])).toContain(CLOUD_SYNC_SETTING)
    expect(String(logged.mock.calls[0][0])).not.toContain('enabled-please')
  })
})

describe('the middleware reads the setting the running server has', () => {
  const middleware = enterpriseMiddleware(() => false)
  const policyOf = () => middleware(new NextRequest(new URL('/sign-in', 'https://automatos.example.test'))).headers.get(CSP_HEADER)

  it('on each response', () => {
    expect(policyOf()).toBe(POLICY_OFF)
    vi.stubEnv(CLOUD_SYNC_SETTING, 'true')
    expect(policyOf()).toBe(POLICY_ON)
    vi.stubEnv(CLOUD_SYNC_SETTING, 'false')
    expect(policyOf()).toBe(POLICY_OFF)
  })
})

describe('start-up', () => {
  it('starts with the setting unset, off or on', () => {
    for (const value of [undefined, '', 'false', 'true', ' Yes ']) {
      expect(enterpriseStartUpProblem({ ...READY, ENTERPRISE_CALLS_CLOUD_SYNC: value })).toBeNull()
    }
  })

  it('refuses a malformed value, naming the setting and not the value', () => {
    for (const value of ['enabled-please', 'truee', "'unsafe-inline'", '2']) {
      const problem = enterpriseStartUpProblem({ ...READY, ENTERPRISE_CALLS_CLOUD_SYNC: value })
      expect(problem).toBe(CLOUD_SYNC_REFUSED)
      expect(problem?.startsWith(CLOUD_SYNC_SETTING)).toBe(true)
      expect(problem).not.toContain(value)
    }
    expect(parseCloudSync('enabled-please')).toEqual({ ok: false, error: CLOUD_SYNC_REFUSED })
  })
})
