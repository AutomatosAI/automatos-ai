/**
 * PRD-261 A-13b (PRD-260 review F10, M24; WCAG 2.2.2 Pause, Stop, Hide) — the
 * Pause Auto control under Auto: it stops his loop (his still shows, his ring
 * stops), it survives a state change, a remount and another tab, and under the
 * OS reduced-motion setting he is the still with no control, since nothing
 * moves. jsdom plays no media: play() is a mock that resolves, as in
 * auto-mascot.test.tsx. Roles, attributes and text only (CSS modules are proxies).
 */
import { act, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { AutoMascot } from '@/components/mascot/auto-mascot'
import { PAUSE_AUTO } from '@/components/mascot/mascot-pause-button'
import { PAUSE_KEY } from '@/lib/mascot/pause'
import { CLIP_MS } from '@/lib/mascot/player'

const CHROME =
  'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36'

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

const pauseButton = () => screen.getByRole('button', { name: PAUSE_AUTO })
const stills = (container: HTMLElement) => Array.from(container.querySelectorAll('img'), (image) => image.getAttribute('src'))
const clips = (container: HTMLElement) => container.querySelectorAll('video')

/** jsdom has no play(): every <video> AutoMascot mounts gets this one. */
const mockPlay = () => vi.spyOn(HTMLMediaElement.prototype, 'play').mockResolvedValue(undefined)
let play: ReturnType<typeof mockPlay>

beforeEach(() => {
  Object.defineProperty(window.navigator, 'userAgent', { configurable: true, value: CHROME })
  play = mockPlay()
})

afterEach(() => {
  window.localStorage.removeItem(PAUSE_KEY)
  if (originalMatchMedia) Object.defineProperty(window, 'matchMedia', originalMatchMedia)
  else Reflect.deleteProperty(window, 'matchMedia')
  Reflect.deleteProperty(window.navigator, 'userAgent')
  vi.restoreAllMocks()
  vi.useRealTimers()
})

describe('Pause Auto (F10, WCAG 2.2.2)', () => {
  it('sits under a moving Auto, not pressed, while his clip plays and his ring pulses', () => {
    const { container } = render(<AutoMascot state="idle" />)
    expect(pauseButton()).toHaveAttribute('aria-pressed', 'false')
    expect(clips(container)).toHaveLength(1)
    expect(container.firstElementChild).not.toHaveAttribute('data-motion')
  })

  it('stops the loop: his clip gives way to his still, his ring stops, and nothing plays again', () => {
    const { container } = render(<AutoMascot state="reading" />)
    expect(play).toHaveBeenCalledTimes(1)
    fireEvent.click(pauseButton())
    expect(pauseButton()).toHaveAttribute('aria-pressed', 'true')
    expect(clips(container)).toHaveLength(0)
    expect(stills(container)).toEqual(['/auto/reading.png'])
    expect(container.firstElementChild).toHaveAttribute('data-motion', 'still')
    expect(play).toHaveBeenCalledTimes(1)
    expect(window.localStorage.getItem(PAUSE_KEY)).toBe('1')
  })

  it('survives a state change: the new state shows as a still at once, and a one-shot still ends on the timer', () => {
    vi.useFakeTimers({ toFake: ['setTimeout', 'clearTimeout'] })
    const onClipEnded = vi.fn()
    const { container, rerender } = render(<AutoMascot state="idle" onClipEnded={onClipEnded} />)
    fireEvent.click(pauseButton())
    rerender(<AutoMascot state="working" onClipEnded={onClipEnded} />)
    expect(clips(container)).toHaveLength(0)
    expect(stills(container)).toEqual(['/auto/working.png'])
    rerender(<AutoMascot state="celebrate" onClipEnded={onClipEnded} />)
    expect(stills(container)).toEqual(['/auto/celebrate.png'])
    expect(pauseButton()).toHaveAttribute('aria-pressed', 'true')
    act(() => {
      vi.advanceTimersByTime(CLIP_MS)
    })
    expect(onClipEnded).toHaveBeenCalledWith('celebrate')
  })

  it('survives a remount, and plays him again when pressed a second time', () => {
    const first = render(<AutoMascot state="idle" />)
    fireEvent.click(pauseButton())
    first.unmount()
    const { container } = render(<AutoMascot state="thinking" />)
    expect(pauseButton()).toHaveAttribute('aria-pressed', 'true')
    expect(stills(container)).toEqual(['/auto/thinking.png'])
    fireEvent.click(pauseButton())
    expect(pauseButton()).toHaveAttribute('aria-pressed', 'false')
    expect(clips(container)[0]?.getAttribute('src')).toBe('/auto/thinking.webm')
    expect(container.firstElementChild).not.toHaveAttribute('data-motion')
  })

  it('pauses every Auto on the page with one press, and follows a pause made in another tab', () => {
    const { container } = render(
      <>
        <AutoMascot state="idle" />
        <AutoMascot state="searching" lightsPage={false} />
      </>,
    )
    fireEvent.click(screen.getAllByRole('button', { name: PAUSE_AUTO })[0] as HTMLElement)
    expect(clips(container)).toHaveLength(0)
    expect(stills(container)).toEqual(['/auto/idle.png', '/auto/searching.png'])
    act(() => {
      window.localStorage.removeItem(PAUSE_KEY)
      window.dispatchEvent(new StorageEvent('storage', { key: PAUSE_KEY }))
    })
    expect(clips(container)).toHaveLength(2)
  })

  it('stills the specimens too, and gives a decorative Auto no control of his own', () => {
    window.localStorage.setItem(PAUSE_KEY, '1')
    const { container } = render(<AutoMascot state="oops" showCaption={false} lightsPage={false} decorative />)
    expect(screen.queryByRole('button', { hidden: true })).toBeNull()
    expect(container.firstElementChild).toHaveAttribute('data-motion', 'still')
  })
})

describe('Auto under reduced motion: the still', () => {
  it('shows the still with his ring stopped and no Pause control, since nothing moves', () => {
    preferReducedMotion(true)
    const { container } = render(<AutoMascot state="working" />)
    expect(clips(container)).toHaveLength(0)
    expect(stills(container)).toEqual(['/auto/working.png'])
    expect(container.firstElementChild).toHaveAttribute('data-motion', 'still')
    expect(screen.queryByRole('button', { name: PAUSE_AUTO })).toBeNull()
    expect(play).not.toHaveBeenCalled()
  })
})
