'use client'

/**
 * PRD-266 — the Automatos Go device routes for the web app, called only through apiClient.
 *
 * US-005b, "Pair this browser" (§4.3): the discovery document (GET /.well-known/automatos.json
 * says whether this is a local install that takes paired devices), POST /api/v1/mobile/pair for
 * this browser (platform "web", its install id) and useBrowserPairing, the state the pairing gate
 * renders (components/settings/phones-devices/pair-browser-gate.tsx). The gate sits in the auth
 * boundary, above React Query's provider, so these are plain calls and React state.
 *
 * Only the local edition asks anything: in SaaS useBrowserPairing never reads discovery and never
 * needs pairing. On a local install that is not in paired mode (off, open) a token left from an
 * earlier paired run is cleared, so every request is the same as before pairing existed. A
 * discovery read that fails shows the app: the server stays the authority, and its first
 * `device_pairing_required` brings the screen.
 */
import { useCallback, useEffect, useState } from 'react'

import { apiClient, type ApiHttpError } from '@/lib/api-client'
import { onUnauthorized } from '@/lib/api-unauthorized'
import { isLocal } from '@/lib/auth-edition'
import {
  DEVICE_PAIRING_REQUIRED,
  DEVICE_REVOKED,
  browserInstallId,
  clearDeviceToken,
  deviceRefusalCode,
  readDeviceToken,
  restoreDeviceCookie,
  storeDevicePairing,
} from '@/lib/device-token'

export const DISCOVERY_PATH = '/.well-known/automatos.json'
export const PAIR_PATH = '/api/v1/mobile/pair'
export const WEB_PLATFORM = 'web'
const LOCAL_EDITION = 'local'
const INVALID_PAIRING_CODE = 'invalid_pairing_code'
const NOT_FOUND = 404
const TOO_MANY_ATTEMPTS = 429

export const PAIRING_MESSAGES = {
  invalidCode: 'That code didn’t work. A code works once, for ten minutes: check it, or mint a new one.',
  closed: 'This install isn’t taking paired devices right now.',
  failed: 'Pairing didn’t go through. Try again in a moment.',
} as const

/** The parts of the §4.1 discovery document the web app reads. */
export interface DiscoveryDocument {
  readonly edition: string
  readonly pairing?: { readonly enabled?: boolean }
}

/** POST /pair's answer: the device token is in it exactly once. */
export interface PairedBrowser {
  readonly device_id: string
  readonly workspace_id: string
  readonly token: string
  readonly rotate_after: string
}

export interface BrowserPairing {
  /** Local, paired mode and no device token: the browser must pair before the app shows. */
  readonly needsPairing: boolean
  /** Keep the pairing POST /pair answered with and show the app. Throws when the browser cannot keep it. */
  readonly keepPairing: (answer: PairedBrowser) => void
}

export function readDiscovery(): Promise<DiscoveryDocument> {
  return apiClient.get<DiscoveryDocument>(DISCOVERY_PATH)
}

/** Exchange a one-time code for this browser's device token; the server canonicalises the typed code. */
export function pairThisBrowser(code: string, name: string): Promise<PairedBrowser> {
  return apiClient.post<PairedBrowser>(PAIR_PATH, {
    code: code.trim(),
    name: name.trim() || null,
    platform: WEB_PLATFORM,
    install_id: browserInstallId(),
  })
}

/** What the pairing screen says about a failed POST /pair: never the server's internals. */
export function pairingErrorMessage(failure: unknown): string {
  const { status, body } = (failure ?? {}) as ApiHttpError
  const detail = (body as { detail?: { code?: unknown } } | undefined)?.detail
  if (typeof detail === 'object' && detail?.code === INVALID_PAIRING_CODE) return PAIRING_MESSAGES.invalidCode
  if (status === NOT_FOUND) return PAIRING_MESSAGES.closed
  if (status === TOO_MANY_ATTEMPTS && failure instanceof Error) return failure.message
  return PAIRING_MESSAGES.failed
}

type Setter<T> = (value: T) => void

/** Follow the discovery document once: paired mode on or off, and a stale token cleared when off. */
function followDiscovery(setToken: Setter<string | null>, setPairingOn: Setter<boolean>): () => void {
  let live = true
  readDiscovery()
    .then((discovery) => {
      if (!live) return
      const on = discovery?.edition === LOCAL_EDITION && discovery.pairing?.enabled === true
      if (on) restoreDeviceCookie()
      else clearDeviceToken()
      setToken(readDeviceToken())
      setPairingOn(on)
    })
    .catch((error: unknown) => console.error('[pairing] could not read the discovery document', error))
  return () => {
    live = false
  }
}

/**
 * Hear the API's device refusals. `device_revoked` answers this browser's token: it is cleared.
 * `device_pairing_required` answers a request sent with no token, so it means paired mode is on;
 * while a token is held it can only be the late answer to a request sent before pairing, and the
 * token stays.
 */
function followRefusals(setToken: Setter<string | null>, setPairingOn: Setter<boolean>): () => void {
  return onUnauthorized((body) => {
    const code = deviceRefusalCode(body)
    if (code === DEVICE_REVOKED) {
      clearDeviceToken()
      setToken(null)
    }
    if (code === DEVICE_PAIRING_REQUIRED) setPairingOn(true)
  })
}

/** The pairing gate's state: local edition only; SaaS never needs pairing. */
export function useBrowserPairing(): BrowserPairing {
  const [token, setToken] = useState<string | null>(() => (isLocal ? readDeviceToken() : null))
  const [pairingOn, setPairingOn] = useState<boolean | null>(null)
  useEffect(() => (isLocal ? followRefusals(setToken, setPairingOn) : undefined), [])
  useEffect(() => (isLocal ? followDiscovery(setToken, setPairingOn) : undefined), [])
  const keepPairing = useCallback((answer: PairedBrowser) => {
    storeDevicePairing(answer)
    setToken(answer.token)
  }, [])
  return { needsPairing: isLocal && pairingOn === true && token === null, keepPairing }
}
