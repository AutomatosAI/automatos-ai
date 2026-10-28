/**
 * Theme voice, web quick-ask: the server voice's audio, on one reusable <audio> element per engine. One clip at a time,
 * from an object URL that is revoked, and its listeners removed, when it ends, fails or is stopped; a stopped clip's
 * late events do nothing. iOS Safari lets an element play without a gesture only once a gesture has loaded or played
 * it: unlock() runs inside the hold's press, seconds before the reply. The element is made on first use, so building
 * the player touches no browser object.
 */
export interface PlaybackEvents {
  /** The clip started playing. */
  readonly onPlaying: () => void
  /** The clip's position moved (timeupdate, a few times a second while it plays). */
  readonly onProgress: () => void
  readonly onEnded: () => void
  /** The clip could not be played (decoding, the browser's autoplay rule), or failed while playing. */
  readonly onFailed: (error: unknown) => void
}

export interface AudioPlayer {
  /** Plays the clip from its start, in place of whatever played. */
  play(clip: Blob, events: PlaybackEvents): void
  /** Stops the clip and releases it. */
  stop(): void
  /** Called inside a user gesture, so the element may play later. */
  unlock(): void
}

/** The part of HTMLAudioElement the player uses. */
export type AudioElement = Pick<
  HTMLAudioElement,
  'src' | 'error' | 'play' | 'pause' | 'load' | 'removeAttribute' | 'addEventListener' | 'removeEventListener'
>

/** Object URLs for a clip (the window's URL). */
export interface ObjectUrls {
  createObjectURL(blob: Blob): string
  revokeObjectURL(url: string): void
}

interface Loaded {
  readonly url: string
  readonly detach: () => void
}

type Listeners = Readonly<Record<'playing' | 'timeupdate' | 'ended' | 'error', () => void>>

function attach(element: AudioElement, listeners: Listeners): () => void {
  const entries = Object.entries(listeners)
  entries.forEach(([type, listener]) => element.addEventListener(type, listener))
  return () => entries.forEach(([type, listener]) => element.removeEventListener(type, listener))
}

/** play() as a promise whatever the browser returns (an old one returns nothing); a throw is a rejection. */
function started(element: AudioElement): Promise<void> {
  try {
    return Promise.resolve(element.play())
  } catch (error) {
    return Promise.reject(error)
  }
}

export function browserAudioPlayer(
  makeElement: () => AudioElement = () => new Audio(),
  urls: ObjectUrls = URL,
): AudioPlayer {
  let element: AudioElement | null = null
  let loaded: Loaded | null = null

  const audio = (): AudioElement => {
    if (element === null) element = makeElement()
    return element
  }

  /** The current clip let go: its listeners off and its URL revoked. True when there was one. */
  function release(url?: string): boolean {
    const current = loaded
    if (current === null || (url !== undefined && current.url !== url)) return false
    loaded = null
    current.detach()
    urls.revokeObjectURL(current.url)
    return true
  }

  function play(clip: Blob, events: PlaybackEvents): void {
    stop()
    const target = audio()
    const url = urls.createObjectURL(clip)
    const fail = (error: unknown): void => {
      if (release(url)) events.onFailed(error)
    }
    const detach = attach(target, {
      playing: () => events.onPlaying(),
      timeupdate: () => events.onProgress(),
      ended: () => {
        if (release(url)) events.onEnded()
      },
      error: () => fail(target.error),
    })
    loaded = Object.freeze({ url, detach })
    target.src = url
    void started(target).catch(fail)
  }

  function stop(): void {
    if (element === null || !release()) return
    element.pause()
    element.removeAttribute('src')
    element.load()
  }

  return Object.freeze({
    play,
    stop,
    unlock: () => {
      if (loaded === null) audio().load()
    },
  })
}
