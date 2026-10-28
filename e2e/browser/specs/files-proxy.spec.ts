import { execFileSync } from 'child_process'
import fs from 'fs'
import os from 'os'
import path from 'path'
import { expect, test, type APIRequestContext, type Page } from '@playwright/test'

// PRD-261 A-08b (@host): an installed theme's files reach the browser through the app's own
// origin, /api/themes/files/<digest>/<path> (frontend/app/api/themes/files/[...path]/route.ts),
// which forwards to the backend's files route (A-08a). Unlike the other theme specs, nothing is
// intercepted: fixtures/hello-a08b/build.py builds A-06a's hello-clip package under its own id,
// it is uploaded to the local backend and opted in to, and the standalone server proxies it. The
// package carries two clips of the same source: HEVC (assets/mascot/idle.mp4, Safari's alpha
// format) and VP9 (assets/mascot/idle-vp9.webm, an extra member, not a declared mascot state).
//
// 1. A Range request for the HEVC clip comes back 206 with its Content-Range, through the proxy.
// 2. A <video> on the app's page loads the HEVC clip through the proxy, with no failed answer,
//    and plays it where this browser build actually decodes it (§"decodable", below).
//    Chromium's media loader asks in ranges, so there the 206 is required; Playwright's Linux
//    WebKit loads the clip with one plain request (the CI run saw [0, 200]), so there the Range
//    proof is test 1's explicit request only.
// 3. A request made to register a service worker (Service-Worker: script) is refused with 403.
// 4. A <video> loads the VP9 clip through the proxy and plays it where this build decodes VP9 —
//    a playback proof that doesn't depend on HEVC, which 1.64.1's WebKit claims it can decode
//    (canPlayType answers non-empty) but can't actually play (#1363): "decodable" below is never
//    canPlayType's optimistic non-empty check alone.
//
// decodable: MediaCapabilities.decodingInfo() where the browser has it (it was built for exactly
// this — an honest "can this engine actually decode this stream" answer, not canPlayType's
// best-effort guess), else canPlayType(...) === 'probably'. A clip whose "decodable" comes back
// false is recorded as an unproven playback, not a failure: some CI browser builds genuinely
// can't decode one codec or the other, and that is not what this spec is proving.

const BACKEND_URL = process.env.E2E_BACKEND_URL ?? 'http://localhost:8000'
const BUILDER = path.resolve(__dirname, '..', 'fixtures', 'hello-a08b', 'build.py')
const PACKAGE_NAME = 'hello-a08b.autotheme'
const PACKAGE_TYPE = 'application/zip'
const CLIP_PATH = 'assets/mascot/idle.mp4'
const CLIP_TYPE = 'video/mp4'
// The clip is PRD-260's idle.mp4, HEVC with alpha (Safari's format for it): 512×512, ~24fps,
// ~5.04s (read from its moov box — mdhd's timescale/duration, stsd's hvc1 sample entry, stts).
const CLIP_CODEC = 'video/mp4; codecs="hvc1"'
const CLIP_WIDTH = 512
const CLIP_HEIGHT = 512
const CLIP_FRAMERATE = 24
const CLIP_BITRATE = 244_865
// The same clip, VP9: automatos-design-kit's public/auto/idle.webm, same 512×512/~24fps/~5.04s
// (read from its EBML: Tracks' PixelWidth/PixelHeight, Info's Duration/TimecodeScale). A second,
// non-HEVC playback proof (#1363) — an extra package member, not a declared mascot state.
const CLIP2_PATH = 'assets/mascot/idle-vp9.webm'
const CLIP2_TYPE = 'video/webm'
const CLIP2_CODEC = 'video/webm; codecs="vp9"'
const CLIP2_BITRATE = 115_960
const FIRST_BYTES = 'bytes=0-99'
const FIRST_BYTES_LENGTH = 100
const PAGE_PATH = '/chat'
const PARTIAL_CONTENT = 206
const OK = 200
/** Playwright's status for a media response the browser served without a status line (WebKit). */
const NO_STATUS = 0
/** The project whose media loader sends Range for a <video>, so the 206 is proven in a browser. */
const RANGED_MEDIA_PROJECT = 'chromium'
const PLAY_TIMEOUT_MS = 10_000

interface UploadAnswer {
  readonly package: { readonly id: string; readonly digest: string }
  readonly install: { readonly id: string } | null
}

function buildPackage(): Buffer {
  const folder = fs.mkdtempSync(path.join(os.tmpdir(), 'hello-a08b-'))
  const out = path.join(folder, PACKAGE_NAME)
  execFileSync('python3', [BUILDER, '--out', out], { stdio: 'pipe' })
  return fs.readFileSync(out)
}

/** Uploads the clip package (201, or 200 when a parallel project already did) and opts in; its digest. */
async function installClip(request: APIRequestContext): Promise<string> {
  const upload = await request.post(`${BACKEND_URL}/api/themes/import`, {
    multipart: { file: { name: PACKAGE_NAME, mimeType: PACKAGE_TYPE, buffer: buildPackage() } },
  })
  expect(upload.ok(), `POST /api/themes/import answered ${upload.status()}: ${await upload.text()}`).toBe(true)
  const answer = (await upload.json()) as UploadAnswer
  expect(answer.install, 'the upload installs the theme in the workspace').not.toBeNull()
  const optIn = await request.post(`${BACKEND_URL}/api/themes/installs/${answer.install?.id}/opt-in`, {
    data: { packageId: answer.package.id },
  })
  expect(optIn.ok(), `POST …/opt-in answered ${optIn.status()}: ${await optIn.text()}`).toBe(true)
  return answer.package.digest
}

function clipUrl(digest: string, clipPath: string = CLIP_PATH): string {
  return `/api/themes/files/${digest}/${clipPath}`
}

interface VideoOutcome {
  readonly decodable: boolean
  readonly played: boolean
  readonly error: number | null
}

/** What a browser is asked whether it can actually decode, in MediaCapabilities' own shape. */
interface ClipProof {
  readonly codec: string
  readonly width: number
  readonly height: number
  readonly bitrate: number
  readonly framerate: number
}

const HEVC_CLIP: ClipProof = {
  codec: CLIP_CODEC, width: CLIP_WIDTH, height: CLIP_HEIGHT, bitrate: CLIP_BITRATE, framerate: CLIP_FRAMERATE,
}
const VP9_CLIP: ClipProof = {
  codec: CLIP2_CODEC, width: CLIP_WIDTH, height: CLIP_HEIGHT, bitrate: CLIP2_BITRATE, framerate: CLIP_FRAMERATE,
}

/**
 * Puts a muted <video> for `src` on the page; whether this build actually decodes `proof`'s
 * codec, and whether it played. "Actually": MediaCapabilities.decodingInfo() where the browser
 * has it — the API built for this question — else canPlayType() === 'probably'; never
 * canPlayType() !== '', whose 'maybe' is WebKit 1.64.1 claiming HEVC it then can't play (#1363).
 */
function playVideo(page: Page, src: string, proof: ClipProof): Promise<VideoOutcome> {
  return page.evaluate(
    async ({ src, proof, timeoutMs }) => {
      const video = document.createElement('video')
      const capabilities = navigator.mediaCapabilities
      const videoConfig = {
        contentType: proof.codec, width: proof.width, height: proof.height, bitrate: proof.bitrate, framerate: proof.framerate,
      }
      const decodable = capabilities
        ? await capabilities
            .decodingInfo({ type: 'file', video: videoConfig })
            .then((result) => result.supported)
            .catch(() => video.canPlayType(proof.codec) === 'probably')
        : video.canPlayType(proof.codec) === 'probably'
      return new Promise<VideoOutcome>((resolve) => {
        const settle = (played: boolean) => resolve({ decodable, played, error: video.error?.code ?? null })
        const timer = window.setTimeout(() => settle(false), timeoutMs)
        video.addEventListener('timeupdate', () => {
          if (video.currentTime <= 0) return
          window.clearTimeout(timer)
          settle(true)
        })
        video.addEventListener('error', () => {
          window.clearTimeout(timer)
          settle(false)
        })
        video.muted = true
        video.playsInline = true
        video.src = src
        document.body.append(video)
        video.play().catch(() => undefined)
      })
    },
    { src, proof, timeoutMs: PLAY_TIMEOUT_MS },
  )
}

test.describe('PRD-261 A-08b — the same-origin files proxy', () => {
  test('answers a Range request for a clip with 206 and its Content-Range', { tag: '@host' }, async ({ request }) => {
    const digest = await installClip(request)
    const response = await request.get(clipUrl(digest), { headers: { Range: FIRST_BYTES } })
    expect(response.status()).toBe(PARTIAL_CONTENT)
    expect(response.headers()['content-range']).toMatch(/^bytes 0-99\/\d+$/)
    expect(response.headers()['accept-ranges']).toBe('bytes')
    expect(response.headers()['content-type']).toBe(CLIP_TYPE)
    expect(response.headers()['x-content-type-options']).toBe('nosniff')
    expect(response.headers()['set-cookie']).toBeUndefined()
    expect((await response.body()).length).toBe(FIRST_BYTES_LENGTH)
  })

  test('loads a clip in <video> through the proxy, over 206 where the browser asks in ranges', { tag: '@host' }, async ({ page, request }, testInfo) => {
    const src = clipUrl(await installClip(request))
    const statuses: number[] = []
    page.on('response', (response) => {
      if (new URL(response.url()).pathname === src) statuses.push(response.status())
    })
    await page.goto(PAGE_PATH)
    const outcome = await playVideo(page, src, HEVC_CLIP)
    await expect.poll(() => statuses.length, 'the <video> asked the proxy for the clip').toBeGreaterThan(0)
    if (testInfo.project.name === RANGED_MEDIA_PROJECT) {
      await expect.poll(() => statuses).toContain(PARTIAL_CONTENT)
      expect(statuses.filter((status) => status !== PARTIAL_CONTENT && status !== OK)).toEqual([])
    } else {
      expect(statuses.filter((status) => status !== PARTIAL_CONTENT && status !== OK && status !== NO_STATUS)).toEqual([])
      testInfo.annotations.push({ type: 'unproven', description: `${testInfo.project.name} loaded the <video> without Range; test 1 proves the 206` })
    }
    if (outcome.decodable) {
      expect(outcome, 'a browser that decodes the clip plays it through the proxy').toMatchObject({ played: true })
    } else {
      // Unproven here, not passed: this build cannot decode HEVC, so only the 206 is checked.
      test.info().annotations.push({ type: 'unproven', description: `${CLIP_CODEC} is not decodable in this build` })
    }
  })

  test('refuses a service-worker registration with 403', { tag: '@host' }, async ({ request }) => {
    const digest = await installClip(request)
    const response = await request.get(`/api/themes/files/${digest}/dist/entry.js`, { headers: { 'Service-Worker': 'script' } })
    expect(response.status()).toBe(403)
  })

  test('plays a VP9 clip through the proxy, proving playback without depending on HEVC', { tag: '@host' }, async ({ page, request }) => {
    const src = clipUrl(await installClip(request), CLIP2_PATH)
    const response = await request.get(src)
    expect(response.ok(), `GET ${src} answered ${response.status()}: ${await response.text()}`).toBe(true)
    expect(response.headers()['content-type']).toBe(CLIP2_TYPE)
    await page.goto(PAGE_PATH)
    const outcome = await playVideo(page, src, VP9_CLIP)
    if (outcome.decodable) {
      expect(outcome, 'a browser that decodes VP9 plays the clip through the proxy').toMatchObject({ played: true })
    } else {
      // Unproven here, not failed: VP9 isn't decodable in this build; the HEVC test covers 206/Range.
      test.info().annotations.push({ type: 'unproven', description: `${CLIP2_CODEC} is not decodable in this build` })
    }
  })
})
