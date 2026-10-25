'use client'

/**
 * PRD-207 S5 — the one place the browser talks to Retell: its WebRTC SDK (`retell-client-js-sdk`, which brings
 * LiveKit) carries a live call's audio both ways. The package loads on demand, inside the call's start, and never in a
 * server render (next.config.js CLIENT_ONLY_PACKAGES resolves it to an empty module in the server builds).
 *
 * The SDK catches its own start failure and reports it as an `error` event ("Error starting call"), so a start that
 * resolves has not necessarily connected: the Live engine reads the events, not the promise.
 */

export interface RetellCallEvents {
  readonly onStarted: () => void
  readonly onEnded: () => void
  readonly onError: (message: unknown) => void
  readonly onAgentTalking: (talking: boolean) => void
  /** Auto's voice as raw PCM, each animation frame. */
  readonly onAgentAudio: (samples: Float32Array) => void
  /** Retell's running transcript (its `update` event's `transcript`), unchecked. */
  readonly onTranscript: (transcript: unknown) => void
}

export interface RetellCall {
  /** Connects with the mint's access token (it dies in ~30s unused), capturing from the device given. */
  readonly start: (accessToken: string, captureDeviceId: string | null) => Promise<void>
  readonly stop: () => void
  readonly mute: () => void
  readonly unmute: () => void
}

/** A Retell web client with the call's events wired to `events`; nothing connects until `start`. */
export async function openRetellCall(events: RetellCallEvents): Promise<RetellCall> {
  const { RetellWebClient } = await import('retell-client-js-sdk')
  const client = new RetellWebClient()
  client.on('call_started', events.onStarted)
  client.on('call_ended', events.onEnded)
  client.on('error', events.onError)
  client.on('agent_start_talking', () => events.onAgentTalking(true))
  client.on('agent_stop_talking', () => events.onAgentTalking(false))
  client.on('audio', events.onAgentAudio)
  client.on('update', (update: { transcript?: unknown } | null) => events.onTranscript(update?.transcript))
  return {
    start: (accessToken, captureDeviceId) =>
      client.startCall({
        accessToken,
        emitRawAudioSamples: true,
        // Explicit capture binding: the default device is exactly what fails silently.
        ...(captureDeviceId ? { captureDeviceId } : {}),
      }),
    stop: () => client.stopCall(),
    mute: () => client.mute(),
    unmute: () => client.unmute(),
  }
}
