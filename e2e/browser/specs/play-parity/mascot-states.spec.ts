import { expect, test, type Locator } from '@playwright/test'
import { describeParity, openPlay, PARITY_TAG, type AvailablePlay } from '../../support/play-parity'
import { watchPage, type ResponseSeen } from '../../support/watch'

// PRD-261 A-17b (@parity): Auto's 12 states, on the style sheet (/looks/style), under builtin Play
// and under the Design Kit's package. Each slot names its state, draws its still (the state's PNG,
// served from builtin Play's /auto/ or from the package's assets/mascot/ through the files proxy)
// and rings it in the state's tone (lib/mascot/ring.ts); "Play Auto's states" plays the 12 clips.
// Chromium plays the VP9-alpha .webm; WebKit gets the HEVC-alpha .mp4 (lib/mascot/media.ts), which
// a Linux WebKit may not decode, and then each slot falls back to its still, as AutoMascot does
// after a clip that won't play: there every slot must show its clip or its still.

const STYLE_SHEET_PATH = '/looks/style'
/** components/looks/style/style-mascots.tsx MASCOT_LABEL and PLAY_STATES. */
const MASCOT_SECTION = 'Mascot slot · 12 states · ring drawn by the page'
const PLAY_STATES = "Play Auto's states"
const STILL_EXTENSION = 'png'
const CLIP_EXTENSION: Readonly<Record<string, string>> = { chromium: 'webm', webkit: 'mp4' }
/** HTMLMediaElement.HAVE_CURRENT_DATA: a frame is there to draw. */
const HAVE_CURRENT_DATA = 2
const SERVED_STATUSES = [200, 206, 304]
/**
 * WebKit plays HEVC clips through its own media stack, which fetches them outside the network layer
 * Playwright watches, so their responses report status 0 (browser-e2e run 39382509598, builtin and
 * installed Play alike). Status 0 says nothing about the file: a refused one reports its real status
 * (404, 403), which this check still catches, and expectClips checks what each slot draws. Only that
 * case is excused; every other mascot file, and every clip in Chromium, must be served.
 */
const MEDIA_STACK_STATUS = 0
const MEDIA_STACK_CLIP = /\.mp4$/
/** The 12 states in the style sheet's order, each with its ring tone (lib/mascot/types.ts, lib/mascot/ring.ts). */
const STATES: ReadonlyArray<readonly [string, string]> = [
  ['hello', 'light'],
  ['idle', 'light'],
  ['listening', 'light'],
  ['thinking', 'accent'],
  ['working', 'accent'],
  ['searching', 'accent'],
  ['reading', 'accent'],
  ['celebrate', 'light'],
  ['oops', 'error'],
  ['sleeping', 'dusk'],
  ['lost', 'dusk'],
  ['loading', 'quiet'],
]

interface SlotDrawing {
  readonly still: string | null
  readonly stillDrawn: boolean
  readonly clip: string | null
  readonly clipReady: boolean
}

/** What a slot draws now: its still (loaded?) or its clip (a frame ready?), each by URL path. */
function slotDrawing(slot: Locator): Promise<SlotDrawing> {
  return slot.evaluate((item, ready) => {
    const img = item.querySelector('img')
    const video = item.querySelector('video')
    const pathOf = (url: string) => new URL(url, document.baseURI).pathname
    return {
      still: img ? pathOf(img.currentSrc || img.src) : null,
      stillDrawn: img !== null && img.complete && img.naturalWidth > 0,
      clip: video ? pathOf(video.currentSrc || video.src) : null,
      clipReady: video !== null && video.readyState >= ready,
    }
  }, HAVE_CURRENT_DATA)
}

/** Every mascot file response that was not served, as '<status> <path>'. */
function servedByMediaStack(status: number, pathname: string, browserName: string): boolean {
  return browserName === 'webkit' && status === MEDIA_STACK_STATUS && MEDIA_STACK_CLIP.test(pathname)
}

function refusedMascotFiles(responses: readonly ResponseSeen[], play: AvailablePlay, browserName: string): string[] {
  const root = play.mascotPath('')
  return responses
    .map(({ url, status }) => ({ status, pathname: new URL(url).pathname }))
    .filter(({ status, pathname }) => pathname.startsWith(root) && !SERVED_STATUSES.includes(status))
    .filter(({ status, pathname }) => !servedByMediaStack(status, pathname, browserName))
    .map(({ status, pathname }) => `${status} ${pathname}`)
}

async function expectStills(section: Locator, play: AvailablePlay): Promise<void> {
  const slots = section.getByRole('listitem')
  await expect(slots).toHaveCount(STATES.length)
  for (const [index, [state, ring]] of STATES.entries()) {
    const slot = slots.nth(index)
    await expect(slot.getByText(state, { exact: true })).toBeVisible()
    await expect(slot.getByTestId('auto-stage')).toHaveAttribute('data-ring', ring)
    await slot.scrollIntoViewIfNeeded()
    await expect.poll(() => slotDrawing(slot)).toEqual({
      still: play.mascotPath(`${state}.${STILL_EXTENSION}`),
      stillDrawn: true,
      clip: null,
      clipReady: false,
    })
  }
}

async function expectClips(section: Locator, play: AvailablePlay, browserName: string): Promise<void> {
  const slots = section.getByRole('listitem')
  const extension = CLIP_EXTENSION[browserName] ?? CLIP_EXTENSION.chromium
  for (const [index, [state]] of STATES.entries()) {
    const slot = slots.nth(index)
    await slot.scrollIntoViewIfNeeded()
    const clip = { still: null, stillDrawn: false, clip: play.mascotPath(`${state}.${extension}`), clipReady: true }
    const still = { still: play.mascotPath(`${state}.${STILL_EXTENSION}`), stillDrawn: true, clip: null, clipReady: false }
    // A clip WebKit can't decode falls back to the still; Chromium decodes VP9 alpha, so it must play.
    const allowed = browserName === 'chromium' ? [clip] : [clip, still]
    await expect.poll(() => slotDrawing(slot).then((drawn) => allowed.some((one) => sameDrawing(one, drawn)))).toBe(true)
  }
}

function sameDrawing(a: SlotDrawing, b: SlotDrawing): boolean {
  return a.still === b.still && a.stillDrawn === b.stillDrawn && a.clip === b.clip && a.clipReady === b.clipReady
}

describeParity("Auto's 12 states on the style sheet", (variant) => {
  test("each state is named, ringed in its tone and drawn from Play's own files, still and playing", { tag: PARITY_TAG }, async ({ page, request, browserName }) => {
    const watch = watchPage(page)
    const play = await openPlay(page, request, variant, STYLE_SHEET_PATH)
    const section = page.getByRole('region', { name: MASCOT_SECTION })
    await expect(section).toBeVisible()

    await expectStills(section, play)

    const toggle = section.getByRole('button', { name: PLAY_STATES })
    await toggle.click()
    await expect(toggle).toHaveAttribute('aria-pressed', 'true')
    await expectClips(section, play, browserName)

    expect(refusedMascotFiles(watch.responses, play, browserName)).toEqual([])
    expect(watch.pageErrors).toEqual([])
  })
})
