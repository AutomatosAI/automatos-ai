'use client'

import { type SyntheticEvent, useCallback, useState } from 'react'

import { type MascotPlayer, useMascotPlayer } from '@/hooks/use-mascot-player'
import { usePrefersReducedMotion, useVideoFormat } from '@/hooks/use-media-preferences'
import { captionFor, clipUrls, pageGlow, posterUrl, type VideoFormat } from '@/lib/mascot/media'
import { ringTone } from '@/lib/mascot/ring'
import type { MascotName } from '@/lib/mascot/types'

import styles from './auto-mascot.module.css'

/**
 * PRD-260 D6 — Auto: the clip for the reducer's state, the light ring under him
 * and a caption that says what he is doing, ported from Automatos Play's
 * components/mascot/AutoMascot.tsx. Safari and iOS play the .mp4, everything
 * else the .webm (lib/mascot/media.ts videoFormat); under reduced motion, on the
 * server and after a clip that won't play, the state's PNG shows instead. The
 * stage is an image named by the caption; the caption itself is a polite live
 * region, so a screen reader hears what he is doing whenever it changes. The
 * clips live in public/auto/ (committed), so there is no stand-in for missing
 * files.
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
}

export function AutoMascot({
  state,
  onClipEnded,
  caption,
  detail,
  size = 'regular',
  showCaption = true,
  lightsPage = true,
}: AutoMascotProps) {
  const reducedMotion = usePrefersReducedMotion()
  const format = useVideoFormat()
  // One clip that won't play means the set is broken: stills from then on, not a stuck clip.
  const [videoFailed, setVideoFailed] = useState(false)
  const video = !reducedMotion && !videoFailed ? format : null
  const player = useMascotPlayer({ target: state, still: video === null, onClipEnded })
  const label = caption ?? captionFor(state)

  const onVideoError = useCallback((event: SyntheticEvent<HTMLVideoElement>) => {
    console.warn(`[mascot] Auto's clip ${event.currentTarget.src} didn't play, so he is a still.`)
    setVideoFailed(true)
  }, [])

  return (
    <div className={styles.auto} data-size={size}>
      {/* data-glow: app/play-theme.css lights the page behind everything from it (body:has). */}
      <div
        className={styles.stage}
        role="img"
        aria-label={label}
        data-testid="auto-stage"
        data-ring={ringTone(player.showing)}
        data-glow={lightsPage ? pageGlow(player.showing) : undefined}
      >
        <div className={styles.ring} />
        {video !== null ? (
          <MascotVideo state={player.showing} format={video} player={player} onError={onVideoError} />
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
    </div>
  )
}

interface MascotVideoProps {
  readonly state: MascotName
  readonly format: VideoFormat
  readonly player: MascotPlayer
  readonly onError: (event: SyntheticEvent<HTMLVideoElement>) => void
}

/** No `loop`: the player restarts loops itself, at the boundary (lib/mascot/player.ts). */
function MascotVideo({ state, format, player, onError }: MascotVideoProps) {
  const urls = clipUrls(state, format)
  return (
    <video
      key={state}
      className={styles.figure}
      src={urls.video}
      poster={urls.poster}
      autoPlay
      muted
      playsInline
      preload="auto"
      onTimeUpdate={player.onTimeUpdate}
      onEnded={player.onEnded}
      onError={onError}
    />
  )
}

/** The state's PNG: the poster, and all there is under reduced motion. */
function MascotStill({ state }: { readonly state: MascotName }) {
  // A plain <img>: a 512 px transparent still from public/auto/, never resized.
  // eslint-disable-next-line @next/next/no-img-element
  return <img className={styles.figure} src={posterUrl(state)} alt="" />
}
