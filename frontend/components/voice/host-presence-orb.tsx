'use client'

/**
 * Studio's room while a voice session runs (PRD-207/208, theme voice): the presence wave (PresenceOrb, the chat's
 * full-bleed background) drawn from the host voice provider: the session's phase as its state, the engine's levels
 * read in its own animation frame (never a render), fading in when a session starts and out when it stops.
 *
 * components/chatbot/chat.tsx imports this in PresenceOrb's place: the chat is far over the code-shape limits, so
 * its own lines are not edited beyond that import (as it imports FencedMultimodalInput for MultimodalInput).
 */
import { useMemo } from 'react'
import { AnimatePresence, motion } from 'framer-motion'

import { PresenceOrb, type VoiceLevelsSource } from '@/components/voice/PresenceOrb'
import { VoiceErrorBoundary } from '@/components/voice/VoiceErrorBoundary'
import { useHostVoice } from '@/components/voice/voice-provider'

const FADE_SECONDS = 0.5

export interface HostPresenceOrbProps {
  /** 0..1 vertical beamline placement; glides on change (the welcome screen sits higher than a thread). */
  readonly horizon?: number
}

export function HostPresenceOrb({ horizon }: HostPresenceOrbProps) {
  const { snapshot, readLevels } = useHostVoice()
  // A getter, so the canvas reads the engine's levels each frame without a render.
  const levels = useMemo<VoiceLevelsSource>(
    () => ({
      get current() {
        return readLevels()
      },
    }),
    [readLevels],
  )
  return (
    <AnimatePresence>
      {snapshot.mode !== null && (
        <motion.div
          key="voice-presence"
          className="absolute inset-0"
          data-testid="host-presence-orb"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          transition={{ duration: FADE_SECONDS }}
        >
          <VoiceErrorBoundary>
            <PresenceOrb state={snapshot.phase} levelsRef={levels} fullBleed horizon={horizon} />
          </VoiceErrorBoundary>
        </motion.div>
      )}
    </AnimatePresence>
  )
}
