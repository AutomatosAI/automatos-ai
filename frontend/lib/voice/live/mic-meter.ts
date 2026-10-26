/**
 * PRD-207 — the member's side of a live call's levels and the mic-health meter: a local analyser tap on the SAME
 * device the Retell SDK captures (the SDK owns the track it actually sends; this one only measures it).
 */
import { rmsLevel } from '@/lib/voice/orb-state'

/** How often the tap reads the microphone. */
export const MIC_METER_INTERVAL_MS = 60
const ANALYSER_FFT_SIZE = 512

export interface MicMeter {
  readonly stop: () => void
}

/** The picked device's stream, or the default's when the picked device vanished. */
async function captureStream(captureDeviceId: string | null): Promise<MediaStream> {
  const media = navigator.mediaDevices
  if (!captureDeviceId) return media.getUserMedia({ audio: true })
  try {
    return await media.getUserMedia({ audio: { deviceId: { exact: captureDeviceId } } })
  } catch (error) {
    console.warn('[voice] the picked microphone is gone; metering the default instead', error)
    return media.getUserMedia({ audio: true })
  }
}

/**
 * Opens the tap and calls `onLevel` with the microphone's RMS level (0 to 1) every MIC_METER_INTERVAL_MS. Rejects
 * when the microphone can't be opened (a `NotAllowedError` is the member's refusal) or the browser has no Web Audio.
 */
export async function openMicMeter(captureDeviceId: string | null, onLevel: (level: number) => void): Promise<MicMeter> {
  const stream = await captureStream(captureDeviceId)
  let context: AudioContext
  try {
    context = new AudioContext()
  } catch (error) {
    stream.getTracks().forEach((track) => track.stop())
    throw error
  }
  const analyser = context.createAnalyser()
  analyser.fftSize = ANALYSER_FFT_SIZE
  context.createMediaStreamSource(stream).connect(analyser)
  const samples = new Float32Array(analyser.fftSize)
  const timer = setInterval(() => {
    analyser.getFloatTimeDomainData(samples)
    onLevel(rmsLevel(samples))
  }, MIC_METER_INTERVAL_MS)
  return {
    stop: () => {
      clearInterval(timer)
      stream.getTracks().forEach((track) => track.stop())
      if (context.state !== 'closed') void context.close()
    },
  }
}

/** The member refused the microphone (the browser's NotAllowedError, or a message that says so). */
export function isMicPermissionRefusal(error: unknown): boolean {
  const name = (error as { name?: unknown } | null)?.name
  if (name === 'NotAllowedError' || name === 'SecurityError') return true
  return /permission|denied|notallowed/i.test(String((error as { message?: unknown } | null)?.message ?? error ?? ''))
}
