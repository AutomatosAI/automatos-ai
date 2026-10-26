// Theme voice tests — a HostVoice (components/voice/voice-provider.tsx) as a test sets it: a session's snapshot,
// what is available, the live call's microphone, and spies for every control. A test mocks useHostVoice to return it.
import { vi } from 'vitest'

import type { HostLiveControls, HostVoice } from '@/components/voice/voice-provider'
import { IDLE_VOICE, type VoiceSnapshot } from '@/lib/voice/engine'
import type { VoiceAvailability } from '@/lib/voice/host/availability'

export interface HostVoiceStubPatch {
  readonly snapshot?: Partial<VoiceSnapshot>
  readonly available?: Partial<VoiceAvailability>
  readonly live?: Partial<HostLiveControls>
  readonly levels?: { readonly user: number; readonly agent: number }
}

/** A live call on the conversation `chat-1`, listening, unless the patch says otherwise. */
export function hostVoiceStub(patch: HostVoiceStubPatch = {}): HostVoice {
  const levels = patch.levels ?? { user: 0, agent: 0 }
  return {
    snapshot: { ...IDLE_VOICE, mode: 'live', phase: 'listening', micOpen: true, chatId: 'chat-1', ...patch.snapshot },
    available: { quick: true, live: true, reason: null, ...patch.available },
    readLevels: vi.fn(() => levels),
    start: vi.fn(async () => undefined),
    release: vi.fn(),
    stop: vi.fn(),
    live: {
      micSilent: false,
      inputDevices: [],
      captureDeviceId: null,
      toggleMute: vi.fn(),
      pickDevice: vi.fn(),
      ...patch.live,
    },
  }
}
