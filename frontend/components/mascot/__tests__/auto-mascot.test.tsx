/**
 * PRD-260 D6 — Auto on screen, ported from Automatos Play's
 * tests/mascot/AutoMascot.test.tsx: the .webm under Chrome, the .mp4 under
 * Safari, the PNG as poster, only the PNG under reduced motion, no loop
 * attribute (the player swaps at the clip boundary), and an accessible name
 * saying what Auto is doing. jsdom plays no media: the tests fire the media
 * events by hand, and play() is a mock that resolves (AutoMascot starts each
 * clip itself, to hear when the browser blocks autoplay: P260-FX-15). Roles,
 * attributes and text only (CSS modules are proxies).
 */
import { act, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { AutoMascot } from '@/components/mascot/auto-mascot'
import { CLIP_MS } from '@/lib/mascot/player'

const CHROME =
  'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36'
const SAFARI =
  'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.0 Safari/605.1.15'
/** Every clip is 5 s long. */
const CLIP_SECONDS = CLIP_MS / 1000

const originalMatchMedia = Object.getOwnPropertyDescriptor(window, 'matchMedia')

function preferReducedMotion(matches: boolean): void {
  const matchMedia = vi.fn((media: string) => ({
    matches,
    media,
    addEventListener: vi.fn(),
    removeEventListener: vi.fn(),
  }))
  Object.defineProperty(window, 'matchMedia', { configurable: true, writable: true, value: matchMedia })
}

function setUserAgent(userAgent: string): void {
  Object.defineProperty(window.navigator, 'userAgent', { configurable: true, value: userAgent })
}

function videoOf(container: HTMLElement): HTMLVideoElement {
  const video = container.querySelector('video')
  if (!video) throw new Error('no video')
  return video
}

function stillOf(container: HTMLElement): HTMLImageElement {
  const image = container.querySelector('img')
  if (!image) throw new Error('no still')
  return image
}

/** Puts the clip at a time (in seconds) of its 5 s, then fires timeupdate. */
function playTo(video: HTMLVideoElement, seconds: number): void {
  Object.defineProperty(video, 'duration', { configurable: true, value: CLIP_SECONDS })
  Object.defineProperty(video, 'currentTime', { configurable: true, writable: true, value: seconds })
  fireEvent.timeUpdate(video)
}

const ring = () => screen.getByTestId('auto-stage').getAttribute('data-ring')
const glow = () => screen.getByTestId('auto-stage').getAttribute('data-glow')

/** jsdom has no play(): every <video> AutoMascot mounts gets this one, resolved unless a test says otherwise. */
const mockPlay = () => vi.spyOn(HTMLMediaElement.prototype, 'play').mockResolvedValue(undefined)
let play: ReturnType<typeof mockPlay>

/** Lets play()'s promise settle, inside act: what AutoMascot does with a refusal is a state update. */
async function settlePlay(): Promise<void> {
  await act(async () => {
    await Promise.resolve()
  })
}

beforeEach(() => {
  play = mockPlay()
})

afterEach(() => {
  if (originalMatchMedia) Object.defineProperty(window, 'matchMedia', originalMatchMedia)
  else Reflect.deleteProperty(window, 'matchMedia')
  Reflect.deleteProperty(window.navigator, 'userAgent')
  vi.restoreAllMocks()
  vi.useRealTimers()
})

describe('AutoMascot: the clip for the browser', () => {
  it('plays the .webm under Chrome, muted and inline, with the PNG as poster and no loop attribute', () => {
    setUserAgent(CHROME)
    const { container } = render(<AutoMascot state="idle" />)
    const video = videoOf(container)
    expect(video.getAttribute('src')).toBe('/auto/idle.webm')
    expect(video.getAttribute('poster')).toBe('/auto/idle.png')
    expect(video.hasAttribute('loop')).toBe(false)
    expect(video.loop).toBe(false)
    expect(video.muted).toBe(true)
    expect(video.hasAttribute('playsinline')).toBe(true)
    expect(container.querySelector('img')).toBeNull()
  })

  it('plays the .mp4 under Safari', () => {
    setUserAgent(SAFARI)
    const { container } = render(<AutoMascot state="thinking" />)
    expect(videoOf(container).getAttribute('src')).toBe('/auto/thinking.mp4')
    expect(videoOf(container).getAttribute('poster')).toBe('/auto/thinking.png')
  })

  it('stays a still, loaded lazily, where the page has not asked him to play (playing={false})', () => {
    setUserAgent(CHROME)
    const { container } = render(<AutoMascot state="idle" playing={false} />)
    expect(container.querySelector('video')).toBeNull()
    expect(stillOf(container).getAttribute('src')).toBe('/auto/idle.png')
    expect(stillOf(container).getAttribute('loading')).toBe('lazy')
    expect(play).not.toHaveBeenCalled()
  })

  it('falls back to the still when a clip will not play', () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => undefined)
    const { container } = render(<AutoMascot state="idle" />)
    fireEvent.error(videoOf(container))
    expect(container.querySelector('video')).toBeNull()
    expect(stillOf(container).getAttribute('src')).toBe('/auto/idle.png')
    expect(warn).toHaveBeenCalledTimes(1)
  })
})

describe('AutoMascot when the browser blocks autoplay', () => {
  it('shows the still at once and ends a one-shot on the timer, as under reduced motion', async () => {
    setUserAgent(CHROME)
    vi.useFakeTimers({ toFake: ['setTimeout', 'clearTimeout'] })
    play.mockRejectedValue(new DOMException('blocked', 'NotAllowedError'))
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => undefined)
    const onClipEnded = vi.fn()
    const { container } = render(<AutoMascot state="hello" onClipEnded={onClipEnded} />)
    await settlePlay()
    expect(container.querySelector('video')).toBeNull()
    expect(stillOf(container).getAttribute('src')).toBe('/auto/hello.png')
    expect(warn).toHaveBeenCalledTimes(1)
    act(() => {
      vi.advanceTimersByTime(CLIP_MS)
    })
    expect(onClipEnded).toHaveBeenCalledTimes(1)
    expect(onClipEnded).toHaveBeenCalledWith('hello')
  })

  it('keeps the clip when a start is only cut short (an AbortError from a swap)', async () => {
    setUserAgent(CHROME)
    play.mockRejectedValue(new DOMException('interrupted', 'AbortError'))
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => undefined)
    const { container } = render(<AutoMascot state="idle" />)
    await settlePlay()
    expect(videoOf(container).getAttribute('src')).toBe('/auto/idle.webm')
    expect(container.querySelector('img')).toBeNull()
    expect(warn).not.toHaveBeenCalled()
  })
})

describe('AutoMascot under reduced motion', () => {
  it('shows only the PNG and swaps it at once', () => {
    preferReducedMotion(true)
    const { container, rerender } = render(<AutoMascot state="idle" />)
    expect(container.querySelector('video')).toBeNull()
    expect(stillOf(container).getAttribute('src')).toBe('/auto/idle.png')
    rerender(<AutoMascot state="searching" />)
    expect(container.querySelector('video')).toBeNull()
    expect(stillOf(container).getAttribute('src')).toBe('/auto/searching.png')
    expect(ring()).toBe('accent')
  })

  it('ends a one-shot on a timer, since no clip plays', () => {
    preferReducedMotion(true)
    vi.useFakeTimers()
    const onClipEnded = vi.fn()
    render(<AutoMascot state="celebrate" onClipEnded={onClipEnded} />)
    act(() => {
      vi.advanceTimersByTime(CLIP_MS - 1)
    })
    expect(onClipEnded).not.toHaveBeenCalled()
    act(() => {
      vi.advanceTimersByTime(1)
    })
    expect(onClipEnded).toHaveBeenCalledWith('celebrate')
  })
})

describe('AutoMascot: the clip boundary', () => {
  it('queues a new state and swaps it at the boundary', () => {
    const { container, rerender } = render(<AutoMascot state="idle" />)
    rerender(<AutoMascot state="working" />)
    expect(screen.getByText('Working on it')).toBeInTheDocument()
    expect(videoOf(container).getAttribute('src')).toBe('/auto/idle.webm')
    expect(ring()).toBe('light')

    playTo(videoOf(container), CLIP_SECONDS / 2)
    expect(videoOf(container).getAttribute('src')).toBe('/auto/idle.webm')

    playTo(videoOf(container), CLIP_SECONDS - 0.05)
    expect(videoOf(container).getAttribute('src')).toBe('/auto/working.webm')
    expect(ring()).toBe('accent')
  })

  it('swaps on ended when timeupdate missed the boundary, and swaps oops at once', () => {
    const { container, rerender } = render(<AutoMascot state="thinking" />)
    rerender(<AutoMascot state="searching" />)
    fireEvent.ended(videoOf(container))
    expect(videoOf(container).getAttribute('src')).toBe('/auto/searching.webm')
    rerender(<AutoMascot state="oops" />)
    expect(videoOf(container).getAttribute('src')).toBe('/auto/oops.webm')
    expect(ring()).toBe('error')
  })

  it('plays a loop again at the boundary when nothing is queued', () => {
    const { container } = render(<AutoMascot state="reading" />)
    const video = videoOf(container)
    // Once as the clip mounts (AutoMascot starts it, to hear a blocked autoplay), once more at the boundary.
    expect(play).toHaveBeenCalledTimes(1)
    playTo(video, CLIP_SECONDS - 0.03)
    expect(video.currentTime).toBe(0)
    expect(play).toHaveBeenCalledTimes(2)
    expect(videoOf(container)).toBe(video)
  })

  it("takes a clip's boundary once: its last timeupdate and its ended swap a queued celebrate in, never finish it", () => {
    setUserAgent(CHROME)
    const onClipEnded = vi.fn()
    const { container, rerender } = render(<AutoMascot state="idle" onClipEnded={onClipEnded} />)
    rerender(<AutoMascot state="celebrate" onClipEnded={onClipEnded} />)
    const idle = videoOf(container)
    // In a browser both land before React re-renders: one act, so neither sees the swap.
    act(() => {
      playTo(idle, CLIP_SECONDS)
      fireEvent.ended(idle)
    })
    const celebrate = videoOf(container)
    expect(celebrate.getAttribute('src')).toBe('/auto/celebrate.webm')
    expect(onClipEnded).not.toHaveBeenCalled()
    fireEvent.ended(celebrate)
    expect(onClipEnded).toHaveBeenCalledTimes(1)
    expect(onClipEnded).toHaveBeenCalledWith('celebrate')
  })

  it("reports a one-shot's end once, then swaps to the next state at once", () => {
    const onClipEnded = vi.fn()
    const { container, rerender } = render(<AutoMascot state="celebrate" onClipEnded={onClipEnded} />)
    playTo(videoOf(container), CLIP_SECONDS - 0.04)
    fireEvent.ended(videoOf(container))
    expect(onClipEnded).toHaveBeenCalledTimes(1)
    expect(onClipEnded).toHaveBeenCalledWith('celebrate')
    rerender(<AutoMascot state="idle" onClipEnded={onClipEnded} />)
    expect(videoOf(container).getAttribute('src')).toBe('/auto/idle.webm')
  })
})

describe('AutoMascot: what Auto is doing, in words', () => {
  it('names the stage by the caption and says it in a polite live region', () => {
    render(<AutoMascot state="searching" />)
    expect(screen.getByRole('img', { name: 'Searching the web' })).toBeInTheDocument()
    const caption = screen.getByText('Searching the web', { selector: 'span' })
    expect(caption.closest('p')).toHaveAttribute('aria-live', 'polite')
  })

  it('takes a caption and a detail line from the page, and keeps the name without the caption', () => {
    const { rerender } = render(<AutoMascot state="celebrate" caption="Your report is ready" detail="Laptop shortlist" />)
    expect(screen.getByRole('img', { name: 'Your report is ready' })).toBeInTheDocument()
    expect(screen.getByText('Laptop shortlist')).toBeInTheDocument()
    rerender(<AutoMascot state="hello" showCaption={false} />)
    expect(screen.getByRole('img', { name: "Hi, I'm Auto." })).toBeInTheDocument()
    expect(screen.queryByText("Hi, I'm Auto.")).toBeNull()
  })

  it('tells the page glow what he is showing (app/play-theme.css reads data-glow)', () => {
    preferReducedMotion(true)
    const cases = [['searching', 'busy'], ['celebrate', 'bright'], ['oops', 'snag'], ['sleeping', 'off'], ['idle', 'calm']] as const
    for (const [state, expected] of cases) {
      const { unmount } = render(<AutoMascot state={state} />)
      expect({ state, glow: glow() }).toEqual({ state, glow: expected })
      unmount()
    }
  })
})
