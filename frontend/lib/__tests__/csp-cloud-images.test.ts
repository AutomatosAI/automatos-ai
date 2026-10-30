/**
 * PRD-270 US-270-05a (K15) — a Cloud listing's screenshots load from Automatos Cloud's catalogue files
 * route, straight from the browser. The Content-Security-Policy's img-src names that route on Automatos
 * Cloud's origin and on the build's API origin (the path, not the whole origin), and `themeFileUrl`
 * (the one choke point every storefront image goes through) keeps a Cloud files URL on exactly those
 * origins and still drops every other one.
 */
import { createRequire } from 'module'
import { afterEach, describe, expect, it } from 'vitest'

import { themeFileUrl } from '@/components/marketplace/themes/listing-format'
import { AUTOMATOS_CLOUD_ORIGIN, CLOUD_FILES_PATH, isCloudFileUrl } from '@/lib/marketplace/cloud-files'

const load = createRequire(import.meta.url)
const csp = load('../../csp-sources.js') as {
  AUTOMATOS_CLOUD_ORIGIN: string; CLOUD_FILES_PATH: string; cloudImageSources: (api: unknown) => string
}
const CONFIG = load.resolve('../../next.config.js')
const BEFORE = process.env.NEXT_PUBLIC_API_URL

const DIGEST = 'c'.repeat(64)
const CLOUD_SOURCE = 'https://api.automatos.app/api/marketplace/catalog/files/'
const LOCAL_API_SOURCE = 'http://localhost:8000/api/marketplace/catalog/files/'
const CLOUD_FILE = `https://api.automatos.app/api/marketplace/catalog/files/${DIGEST}/assets/previews/default.png`

function setApiUrl(value: string | undefined) {
  if (value === undefined) delete process.env.NEXT_PUBLIC_API_URL
  else process.env.NEXT_PUBLIC_API_URL = value
}

async function imgSrcWith(apiUrl: string | undefined): Promise<string[]> {
  setApiUrl(apiUrl)
  delete load.cache[CONFIG]
  const config = load(CONFIG) as { headers: () => Promise<Array<{ headers: Array<{ key: string; value: string }> }>> }
  const all = (await config.headers()).flatMap((rule) => rule.headers)
  const policy = all.find((header) => header.key === 'Content-Security-Policy')?.value ?? ''
  return (policy.split('; ').find((part) => part.startsWith('img-src ')) ?? '').split(' ')
}

afterEach(() => {
  setApiUrl(BEFORE)
  delete load.cache[CONFIG]
})

describe('img-src', () => {
  it('names the same Automatos Cloud origin and files route as the web', () => {
    expect(csp.AUTOMATOS_CLOUD_ORIGIN).toBe(AUTOMATOS_CLOUD_ORIGIN)
    expect(csp.CLOUD_FILES_PATH).toBe(CLOUD_FILES_PATH)
    expect(AUTOMATOS_CLOUD_ORIGIN).toBe('https://api.automatos.app')
  })

  it('holds the Cloud origin’s files route with no API origin set, not the whole origin', async () => {
    const sources = await imgSrcWith(undefined)
    expect(sources).toContain(CLOUD_SOURCE)
    expect(sources).not.toContain('https://api.automatos.app')
  })

  it('holds the API origin’s files route too (the local edition’s API), once each', async () => {
    const sources = await imgSrcWith('http://localhost:8000/')
    expect(sources).toContain(CLOUD_SOURCE)
    expect(sources).toContain(LOCAL_API_SOURCE)
    expect(sources).not.toContain('http://localhost:8000')
    expect(sources.filter((source) => source === CLOUD_SOURCE)).toHaveLength(1)
  })

  it('names the hosted API once when it is Automatos Cloud', async () => {
    const sources = await imgSrcWith('https://api.automatos.app')
    expect(sources.filter((source) => source === CLOUD_SOURCE)).toHaveLength(1)
  })

  it('adds nothing for an API origin that is not a bare origin', () => {
    expect(csp.cloudImageSources("'unsafe-inline'")).toBe(CLOUD_SOURCE)
    expect(csp.cloudImageSources('*')).toBe(CLOUD_SOURCE)
    expect(csp.cloudImageSources(undefined)).toBe(CLOUD_SOURCE)
  })
})

describe('themeFileUrl and Cloud files', () => {
  it('takes a Cloud catalogue files URL', () => {
    expect(CLOUD_FILES_PATH).toBe('/api/marketplace/catalog/files/')
    expect(themeFileUrl(CLOUD_FILE)).toBe(CLOUD_FILE)
    expect(isCloudFileUrl(CLOUD_FILE)).toBe(true)
  })

  it('takes one on the build’s API origin, and only while the build names it', () => {
    const local = `http://localhost:8000/api/marketplace/catalog/files/${DIGEST}/assets/previews/default.png`
    setApiUrl(undefined)
    expect(themeFileUrl(local)).toBeNull()
    setApiUrl('http://localhost:8000')
    expect(themeFileUrl(local)).toBe(local)
  })

  it('still drops another origin and a protocol-relative URL', () => {
    expect(themeFileUrl('https://evil.example/x.png')).toBeNull()
    expect(themeFileUrl(`//evil.example/api/marketplace/catalog/files/${DIGEST}/x.png`)).toBeNull()
    expect(themeFileUrl(`https://evil.example/api/marketplace/catalog/files/${DIGEST}/x.png`)).toBeNull()
    expect(themeFileUrl(`https://api.automatos.app.evil.example/api/marketplace/catalog/files/${DIGEST}/x.png`)).toBeNull()
    expect(themeFileUrl(`https://api.automatos.app@evil.example/api/marketplace/catalog/files/${DIGEST}/x.png`)).toBeNull()
    expect(themeFileUrl(`http://api.automatos.app/api/marketplace/catalog/files/${DIGEST}/x.png`)).toBeNull()
  })

  it('drops a Cloud-origin URL outside the files path', () => {
    expect(themeFileUrl('https://api.automatos.app/api/themes/files/x.png')).toBeNull()
    expect(themeFileUrl('https://api.automatos.app/api/marketplace/catalog')).toBeNull()
    expect(themeFileUrl(`https://api.automatos.app/api/marketplace/catalog/files/${DIGEST}`)).toBeNull()
    expect(themeFileUrl(`https://api.automatos.app/api/marketplace/catalog/files/${DIGEST}/`)).toBeNull()
    expect(themeFileUrl('https://api.automatos.app/api/marketplace/catalog/files/not-a-digest/x.png')).toBeNull()
    expect(themeFileUrl(`https://api.automatos.app/api/marketplace/catalog/files/${DIGEST}/../../../admin`)).toBeNull()
    expect(themeFileUrl(`https://api.automatos.app/api/marketplace/catalog/files/${DIGEST}/%2e%2e/%2e%2e/%2e%2e/x`)).toBeNull()
    expect(themeFileUrl(`https://api.automatos.app/api/marketplace/catalog/files/${DIGEST}\\..\\x`)).toBeNull()
  })

  it('drops a percent-encoded dot, slash or backslash a server could decode into a climb', () => {
    const files = `https://api.automatos.app/api/marketplace/catalog/files/${DIGEST}`
    expect(themeFileUrl(`${files}/%2e%2e%2f%2e%2e%2fadmin`)).toBeNull()
    expect(themeFileUrl(`${files}/a%2Fb.png`)).toBeNull()
    expect(themeFileUrl(`${files}/a%5cb.png`)).toBeNull()
    expect(themeFileUrl(`${files}/default%2Epng`)).toBeNull()
    expect(themeFileUrl(`${files}/assets/previews/default.png`)).toBe(`${files}/assets/previews/default.png`)
  })
})
