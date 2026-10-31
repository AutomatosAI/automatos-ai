// Theme voice tests — stand-in engines for the voice host: a session that emits the snapshots a test names, and a
// Live one with the call's microphone controls.
import { vi } from 'vitest'

import { IDLE_VOICE, type VoiceEngine, type VoiceSnapshot } from '@/lib/voice/engine'
import type { LiveCallDetails } from '@/lib/voice/live/live-engine'

export class FakeEngine implements VoiceEngine {
  current: VoiceSnapshot = IDLE_VOICE
  private readonly listeners = new Set<(snapshot: VoiceSnapshot) => void>()
  readonly start = vi.fn(async ({ chatId }: { readonly chatId: string | null }) => {
    this.emit({ phase: 'connecting', chatId })
  })
  readonly release = vi.fn()
  readonly stop = vi.fn(() => this.emit({ phase: 'idle' }))
  readonly levels = vi.fn(() => ({ user: 0.2, agent: 0.7 }))
  readonly dispose = vi.fn()
  snapshot() {
    return this.current
  }
  subscribe(listener: (snapshot: VoiceSnapshot) => void) {
    this.listeners.add(listener)
    return () => {
      this.listeners.delete(listener)
    }
  }
  emit(patch: Partial<VoiceSnapshot>) {
    this.current = Object.freeze({ ...this.current, ...patch })
    this.listeners.forEach((listener) => listener(this.current))
  }
}

export const LIVE_DETAILS: LiveCallDetails = Object.freeze({
  micSilent: true,
  inputDevices: Object.freeze([]),
  captureDeviceId: 'usb-1',
})

export class FakeLive extends FakeEngine {
  readonly details = vi.fn(() => LIVE_DETAILS)
  readonly toggleMute = vi.fn()
  readonly pickDevice = vi.fn()
}
