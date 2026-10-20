'use client'

import { type SyntheticEvent, useCallback, useState } from 'react'

import { type MascotPlayer, useMascotPlayer } from '@/hooks/use-mascot-player'
import { useVideoFormat } from '@/hooks/use-media-preferences'
import { captionFor, clipUrls, pageGlow, posterUrl, type VideoFormat } from '@/lib/mascot/media'
import { ringTone } from '@/lib/mascot/ring'
import type { MascotName } from '@/lib/mascot/types'

import styles from './auto-mascot.module.css'
import { MascotPauseButton } from './mascot-pause-button'
import { useMascotMotion } from './use-mascot-motion'

/**
 * PRD-260 D6 — Auto: the clip for the reducer's state, the light ring under him
 * and a caption that says what he is doing, ported from Automatos Play's
 * components/mascot/AutoMascot.tsx. Safari and iOS play the .mp4, everything
 * else the .webm (lib/mascot/media.ts videoFormat); under reduced motion, on the
 * server and after a clip that won't play, the state's PNG shows instead. The
 * stage is an image named by the caption; the caption itself is a polite live
 * region, so a screen reader hears what he is doing whenever it changes. A
 * browser that blocks muted autoplay gets the stills too, so the one-shots still
 * end on a timer (P260-FX-15). The clips live in public/auto/ (committed), so
 * there is no stand-in for missing files. A `decorative` Auto (a specimen whose
 * page names his state in its own words) is hidden from assistive technology:
 * no image role, no name, no caption read out (P260-FX-24, M74).
 *
 * PRD-261 A-13b (F10, WCAG 2.2.2): under him sits the Pause Auto control,
 * wherever he can move (playing, not decorative, not under the OS reduced-motion
 * setting) unless the caller turns it off (pauseControl). Paused, or never asked
 * to play (playing={false}), he is his still and his ring stops (data-motion="still"),
 * so nothing of his moves without a control; he stays paused across a state change, a remount and a reload, in every
 * tab of this browser (lib/mascot/pause.ts). Under reduced motion he is the
 * same still, with no control, since nothing moves.
 */

export type AutoMascotSize = 'small' | 'regular' | 'large'

export interface AutoMascotProps {
  /** The reducer's state.mascot (useAutoMascot's `mascot`). */
  readonly state: MascotName
  /** Dispatch {type: 'clip-ended', state} to the reducer (useAutoMascot's `onClipEnded`). */
  readonly onClipEnded?: (state: MascotName) => void
  /** Overrides the caption for the state, e.g. captionFor('celebrate', 'report'). */
  readonly caption?: string
  /** A second caption line, e.g. the search query. */
  readonly detail?: string
  /** large: the first-open introduction (mock 06); small: Auto beside his caption. */
  readonly size?: AutoMascotSize
  /** False when the page already says it in words (the first-open headline). */
  readonly showCaption?: boolean
  /**
   * False where he is a specimen, not the page's Auto (the style sheet shows all
   * 12 states at once): his stage then leaves the page glow alone (no data-glow).
   */
  readonly lightsPage?: boolean
  /**
   * False keeps him a still where a clip could play: the style sheet shows all
   * 12 states and plays their clips only when asked (StyleMascots).
   */
  readonly playing?: boolean
  /**
   * True where the page already names the state in visible words (the style
   * sheet's slots): he is aria-hidden, with no role and no name, so a specimen
   * is never announced as a live status ('Something went wrong').
   */
  readonly decorative?: boolean
  /**
   * False drops his own Pause Auto control; he keeps his name and caption and
   * obeys the shared pause. Only for a page that already gives the user a way
   * to stop him (WCAG 2.2.2), as the style sheet's specimens have "Play Auto's
   * states" (with playing={false} his clip and his ring both stand still): a
   * moving Auto with no control anywhere on the page fails 2.2.2.
   * No caller passes it yet; the Looks preview keeps its control.
   */
  readonly pauseControl?: boolean
}

/** The Pause Auto control shows only where a live Auto's clip could be moving. */
function showsPauseControl(props: {
  readonly playing: boolean
  readonly decorative: boolean
  readonly pauseControl: boolean
  readonly reducedMotion: boolean
}): boolean {
  return props.playing && props.pauseControl && !props.decorative && !props.reducedMotion
}

/** play()'s refusal when the browser blocks autoplay. */
const AUTOPLAY_BLOCKED = 'NotAllowedError'
/** play()'s refusal when a swap or an unmount cut the start short: the next clip starts itself. */
const START_INTERRUPTED = 'AbortError'

export function AutoMascot({
  state,
  onClipEnded,
  caption,
  detail,
  size = 'regular',
  showCaption = true,
  lightsPage = true,
  playing = true,
  decorative = false,
  pauseControl = true,
}: AutoMascotProps) {
  const motion = useMascotMotion()
  const format = useVideoFormat()
  // One clip that won't play, or blocked autoplay, means stills from then on, not a stuck clip.
  const [videoFailed, setVideoFailed] = useState(false)
  // Nothing of his moves unless he plays: no clip and no ring pulse (WCAG 2.2.2).
  const still = motion.still || !playing
  const video = !still && !videoFailed ? format : null
  const player = useMascotPlayer({ target: state, still: video === null, onClipEnded })
  const label = caption ?? captionFor(state)
  const pausable = showsPauseControl({ playing, decorative, pauseControl, reducedMotion: motion.reducedMotion })

  const onVideoError = useCallback((event: SyntheticEvent<HTMLVideoElement>) => {
    console.warn(`[mascot] Auto's clip ${event.currentTarget.src} didn't play, so he is a still.`)
    setVideoFailed(true)
  }, [])
  const onVideoBlocked = useCallback(() => setVideoFailed(true), [])

  return (
    <div
      className={styles.auto}
      data-size={size}
      data-motion={still ? 'still' : undefined}
      aria-hidden={decorative ? true : undefined}
    >
      {/* data-glow: app/play-theme.css lights the page behind everything from it (body:has). */}
      <div
        className={styles.stage}
        role={decorative ? undefined : 'img'}
        aria-label={decorative ? undefined : label}
        data-testid="auto-stage"
        data-ring={ringTone(player.showing)}
        data-glow={lightsPage ? pageGlow(player.showing) : undefined}
      >
        <div className={styles.ring} />
        {video !== null ? (
          <MascotVideo
            state={player.showing}
            format={video}
            player={player}
            onError={onVideoError}
            onBlocked={onVideoBlocked}
          />
        ) : (
          <MascotStill state={player.showing} />
        )}
      </div>
      {showCaption && (
        <p className={styles.caption} aria-live="polite" aria-atomic="true">
          <span className={styles.captionTitle}>{label}</span>
          {detail && <span className={styles.captionDetail}>{detail}</span>}
        </p>
      )}
      {pausable && (
        <MascotPauseButton paused={motion.paused} onToggle={motion.togglePaused} />
      )}
    </div>
  )
}

interface MascotVideoProps {
  readonly state: MascotName
  readonly format: VideoFormat
  readonly player: MascotPlayer
  readonly onError: (event: SyntheticEvent<HTMLVideoElement>) => void
  /** The browser refused to play the clip (blocked autoplay): show the still instead. */
  readonly onBlocked: () => void
}

/** The name of what play() threw or rejected with ('' for anything without one). */
function errorName(error: unknown): string {
  return typeof error === 'object' && error !== null && 'name' in error ? String(error.name) : ''
}

/**
 * Starts the clip by hand as well as by autoPlay: a browser that blocks autoplay
 * only says so through play()'s promise. Blocked → onBlocked; a start cut short
 * by a swap is normal and ignored; anything else is logged, and the <video>'s
 * own error event decides whether the clip is broken.
 */
function startClip(video: HTMLVideoElement, onBlocked: () => void): void {
  const refused = (error: unknown): void => {
    const name = errorName(error)
    if (name === START_INTERRUPTED) return
    if (name === AUTOPLAY_BLOCKED) {
      console.warn("[mascot] The browser blocked Auto's clip, so he is a still.", error)
      onBlocked()
      return
    }
    console.warn("[mascot] Auto's clip didn't start:", error)
  }
  try {
    // Older browsers return undefined instead of a promise.
    Promise.resolve(video.play()).catch(refused)
  } catch (error) {
    refused(error)
  }
}

/**
 * No `loop`: the player restarts loops itself, at the boundary (lib/mascot/player.ts).
 * Each handler carries the clip it belongs to (`state`), so a boundary counts once
 * per clip even when timeupdate and ended both land before React re-renders.
 */
function MascotVideo({ state, format, player, onError, onBlocked }: MascotVideoProps) {
  const urls = clipUrls(state, format)
  // Stable, so React calls it once per <video> (a new one per state, by key), never on a re-render:
  // play() on a finished one-shot would start it again.
  const startOnMount = useCallback(
    (video: HTMLVideoElement | null) => {
      if (video) startClip(video, onBlocked)
    },
    [onBlocked],
  )
  return (
    <video
      key={state}
      ref={startOnMount}
      className={styles.figure}
      src={urls.video}
      poster={urls.poster}
      autoPlay
      muted
      playsInline
      preload="auto"
      onTimeUpdate={(event) => player.onTimeUpdate(event, state)}
      onEnded={(event) => player.onEnded(event, state)}
      onError={onError}
    />
  )
}

/** The state's PNG: the poster, and all there is under reduced motion. */
function MascotStill({ state }: { readonly state: MascotName }) {
  // A plain <img>: a 512 px transparent still from public/auto/, never resized. Lazy, so a
  // page of specimens (the style sheet's 12) loads only the ones in view.
  // eslint-disable-next-line @next/next/no-img-element
  return <img className={styles.figure} src={posterUrl(state)} alt="" loading="lazy" />
}
