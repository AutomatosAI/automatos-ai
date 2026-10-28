/**
 * Theme voice, web quick-ask — the server voice's audio (audio-player.ts) over a stand-in <audio> element: a clip
 * plays from an object URL, its playing, timeupdate and end are told and the URL is revoked once it ends or fails;
 * stop() pauses, unloads and revokes, after which the clip's late events do nothing; unlock() loads the element inside
 * the press, and makes it only then.
 */
import { describe, expect, it, vi } from 'vitest'

import { browserAudioPlayer, type AudioElement, type PlaybackEvents } from '@/lib/voice/quick-ask/audio-player'

class FakeElement extends EventTarget {
  src = ''
  error: MediaError | null = null
  readonly play = vi.fn(async () => undefined)
  readonly pause = vi.fn()
  readonly load = vi.fn()
  readonly removeAttribute = vi.fn((name: string) => {
    if (name === 'src') this.src = ''
  })

  fire(type: string): void {
    this.dispatchEvent(new Event(type))
  }
}

function setup() {
  const element = new FakeElement()
  const makeElement = vi.fn(() => element as unknown as AudioElement)
  let serial = 0
  const urls = {
    createObjectURL: vi.fn(() => {
      serial += 1
      return `blob:clip-${serial}`
    }),
    revokeObjectURL: vi.fn(),
  }
  const player = browserAudioPlayer(makeElement, urls)
  return { element, makeElement, urls, player }
}

function playback() {
  return { onPlaying: vi.fn(), onProgress: vi.fn(), onEnded: vi.fn(), onFailed: vi.fn() } satisfies PlaybackEvents
}

const clip = (): Blob => new Blob(['mp3'], { type: 'audio/mpeg' })

describe('browserAudioPlayer', () => {
  it('plays a clip from an object URL, tells playing and each timeupdate, and revokes the URL when it ends', () => {
    const { element, urls, player } = setup()
    const events = playback()

    player.play(clip(), events)
    element.fire('playing')
    element.fire('timeupdate')
    element.fire('timeupdate')
    element.fire('ended')
    element.fire('ended')

    expect(element.src).toBe('blob:clip-1')
    expect(element.play).toHaveBeenCalledTimes(1)
    expect([events.onPlaying, events.onProgress, events.onEnded].map((mock) => mock.mock.calls.length)).toEqual([1, 2, 1])
    expect(urls.revokeObjectURL.mock.calls).toEqual([['blob:clip-1']])
  })

  it('tells a failure once when play() is refused (the autoplay rule) or the clip errors, and revokes its URL', async () => {
    const { element, urls, player } = setup()
    const refused = playback()
    element.play.mockRejectedValueOnce(new DOMException('not allowed', 'NotAllowedError'))

    player.play(clip(), refused)
    await vi.waitFor(() => expect(refused.onFailed).toHaveBeenCalledTimes(1))
    const broken = playback()
    player.play(clip(), broken)
    element.fire('error')
    element.fire('error')

    expect(broken.onFailed).toHaveBeenCalledTimes(1)
    expect(urls.revokeObjectURL.mock.calls).toEqual([['blob:clip-1'], ['blob:clip-2']])
  })

  it('stop() pauses, unloads and revokes the clip, and its late events do nothing', () => {
    const { element, urls, player } = setup()
    const events = playback()
    player.play(clip(), events)

    player.stop()
    element.fire('timeupdate')
    element.fire('ended')

    expect(element.pause).toHaveBeenCalledTimes(1)
    expect(element.removeAttribute).toHaveBeenCalledWith('src')
    expect(element.load).toHaveBeenCalledTimes(1)
    expect(urls.revokeObjectURL.mock.calls).toEqual([['blob:clip-1']])
    expect([events.onProgress, events.onEnded].map((mock) => mock.mock.calls.length)).toEqual([0, 0])
  })

  it('a new clip replaces the one playing: the old one released, its late refusal ignored', async () => {
    const { element, urls, player } = setup()
    let refuseOld: (error: unknown) => void = () => undefined
    element.play.mockImplementationOnce(
      () =>
        new Promise<undefined>((_resolve, reject) => {
          refuseOld = reject
        }),
    )
    const old = playback()
    player.play(clip(), old)

    player.play(clip(), playback())
    refuseOld(new DOMException('replaced', 'AbortError'))
    await new Promise((resolve) => setTimeout(resolve, 0))

    expect(old.onFailed).not.toHaveBeenCalled()
    expect(urls.revokeObjectURL.mock.calls).toEqual([['blob:clip-1']])
    expect(element.src).toBe('blob:clip-2')
  })

  it('makes the element only when first needed, and unlock() loads it inside the press unless a clip plays', () => {
    const { element, makeElement, player } = setup()
    expect(makeElement).not.toHaveBeenCalled()

    player.unlock()
    expect(makeElement).toHaveBeenCalledTimes(1)
    expect(element.load).toHaveBeenCalledTimes(1)
    player.play(clip(), playback())
    player.unlock()

    expect(element.load).toHaveBeenCalledTimes(1)
    expect(makeElement).toHaveBeenCalledTimes(1)
  })

  it('stop() with nothing playing touches nothing', () => {
    const { element, player } = setup()
    player.unlock()

    player.stop()

    expect(element.pause).not.toHaveBeenCalled()
  })
})
