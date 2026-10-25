/**
 * PRD-207 — the microphone a live call captures from: whether the browser can capture at all, the inputs it offers,
 * and the one the member picked, which survives page loads (a mic that worked stays picked).
 *
 * The classic silent-call cause is the default binding to a continuity iPhone or a virtual/recorder device that
 * delivers flat zeros, so the Live engine binds the call's capture explicitly to the picked device.
 */

export interface MicInputDevice {
  readonly deviceId: string
  readonly label: string
}

export const CAPTURE_DEVICE_STORAGE_KEY = 'automatos.voice.capture-device'

/** A live call needs the browser's microphone capture. False on the server. */
export function liveCallSupported(): boolean {
  return typeof navigator !== 'undefined' && typeof navigator.mediaDevices?.getUserMedia === 'function'
}

/** The device the member picked last; null for the browser default (or when storage can't be read). */
export function readStoredCaptureDevice(): string | null {
  if (typeof window === 'undefined') return null
  try {
    return window.localStorage.getItem(CAPTURE_DEVICE_STORAGE_KEY)
  } catch (error) {
    console.warn('[voice] the picked microphone could not be read; the default is used', error)
    return null
  }
}

/** Keeps the picked device for the next call; null forgets it. */
export function storeCaptureDevice(deviceId: string | null): void {
  if (typeof window === 'undefined') return
  try {
    if (deviceId) window.localStorage.setItem(CAPTURE_DEVICE_STORAGE_KEY, deviceId)
    else window.localStorage.removeItem(CAPTURE_DEVICE_STORAGE_KEY)
  } catch (error) {
    console.warn('[voice] the picked microphone could not be kept for the next call', error)
  }
}

/**
 * The audio inputs the browser offers, labelled once a permission grant exists. Listing is a nicety (the call works
 * without it), so a failure leaves the list empty and says so in the console.
 */
export async function listInputDevices(): Promise<readonly MicInputDevice[]> {
  if (typeof navigator === 'undefined' || !navigator.mediaDevices?.enumerateDevices) return []
  try {
    const all = await navigator.mediaDevices.enumerateDevices()
    return Object.freeze(
      all
        .filter((device) => device.kind === 'audioinput' && device.deviceId)
        .map((device, index) => Object.freeze({ deviceId: device.deviceId, label: device.label || `Microphone ${index + 1}` })),
    )
  } catch (error) {
    console.warn('[voice] the microphones could not be listed', error)
    return []
  }
}

/** Calls `onChange` when an input is plugged in or out; returns what stops listening. */
export function watchInputDevices(onChange: () => void): () => void {
  const media = typeof navigator !== 'undefined' ? navigator.mediaDevices : undefined
  if (!media?.addEventListener) return () => undefined
  media.addEventListener('devicechange', onChange)
  return () => media.removeEventListener('devicechange', onChange)
}
