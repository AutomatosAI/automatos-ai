'use client'

/**
 * Theme voice — which voice this tab can offer: what this browser can do (read after mount: the server has no
 * browser) and Auto Live's status for the workspace (`GET /api/voice/live-status`, once per mount).
 *
 * VoiceProvider passes useVoiceAvailability to every surface inside it (useHostVoice().available, the SDK's
 * useVoice().available). Its browser half, useBrowserVoiceSupport, is what Settings → Orchestrator's "What Auto
 * depends on" (PRD-268 US-W05) draws the microphone and speech rows from: /settings has no VoiceProvider (one would
 * hold the tab's voice host, and a call would keep running with no controls on screen), and `available.live` also
 * folds in the workspace's plan, which says nothing about this browser's microphone.
 */
import { useEffect, useMemo, useState } from 'react'

import { liveCallSupported } from '@/lib/voice/live/devices'
import { quickAskSupported } from '@/lib/voice/support'

import { LIVE_STATUS_CHECKING, type LiveStatusReading, type VoiceAvailability, readLiveStatus, voiceAvailability } from './availability'

/** What this browser can do for voice. */
export interface BrowserVoiceSupport {
  /** False until read after mount; until then the other two say nothing. */
  readonly read: boolean
  /** Speech recognition and synthesis, which hold to talk needs: `available.quick` itself. */
  readonly quick: boolean
  /** Microphone capture (getUserMedia), which an Auto Live call needs: the browser's half of `available.live`. */
  readonly capture: boolean
}

const UNREAD: BrowserVoiceSupport = Object.freeze({ read: false, quick: false, capture: false })

/** This browser's voice support, read once after mount. */
export function useBrowserVoiceSupport(): BrowserVoiceSupport {
  const [browser, setBrowser] = useState<BrowserVoiceSupport>(UNREAD)
  useEffect(() => {
    setBrowser(Object.freeze({ read: true, quick: quickAskSupported(), capture: liveCallSupported() }))
  }, [])
  return browser
}

/** live-status once per mount, and what this browser can do. */
export function useVoiceAvailability(): VoiceAvailability {
  const browser = useBrowserVoiceSupport()
  const [reading, setReading] = useState<LiveStatusReading>(LIVE_STATUS_CHECKING)
  useEffect(() => {
    let mounted = true
    void readLiveStatus().then((next) => {
      if (mounted) setReading(next)
    })
    return () => {
      mounted = false
    }
  }, [])
  return useMemo(() => voiceAvailability(reading, browser.quick, browser.capture), [reading, browser])
}
