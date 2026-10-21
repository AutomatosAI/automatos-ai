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
 *
 * US-005c, Settings → Phones & devices (§4.3 "Web"), below React Query's provider: the device
 * list (GET /api/v1/mobile/devices: the caller's own, and the workspace's for an owner or admin),
 * Revoke (DELETE /devices/{id}), a one-time code for a phone or another browser (POST
 * /pairing-codes, local edition only) and whether this install is in paired mode, from the same
 * discovery document.
 */
import { useCallback, useEffect, useState } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'

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
export const DEVICES_PATH = '/api/v1/mobile/devices'
export const PAIRING_CODES_PATH = '/api/v1/mobile/pairing-codes'
export const MOBILE_DEVICES_QUERY_KEY = ['mobile-devices'] as const
const DISCOVERY_QUERY_KEY = ['mobile-discovery'] as const
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
  readonly web_url?: string
  readonly pairing?: { readonly enabled?: boolean }
}

/** A device as GET /devices lists it (MobileDevice.to_dict: the fields Settings shows). */
export interface MobileDevice {
  readonly id: string
  /** paired (a device token) or clerk (a signed-in phone). */
  readonly kind: string
  readonly name: string | null
  readonly platform: string | null
  readonly status: string
  readonly push_enabled: boolean
  readonly last_seen_at: string | null
}

/** POST /pairing-codes's answer: the code, shown once; when it stops working; its universal link. */
export interface PairingCode {
  readonly code: string
  readonly expires_at: string
  readonly qr_payload: string
}

/** What Settings reads from discovery: paired mode, and the web address another browser opens. */
export interface PairingMode {
  readonly paired: boolean
  readonly webUrl: string | null
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

/** Does the discovery document say this is a local install that takes paired devices? */
export function takesPairedDevices(discovery: DiscoveryDocument | null | undefined): boolean {
  return discovery?.edition === LOCAL_EDITION && discovery.pairing?.enabled === true
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
      const on = takesPairedDevices(discovery)
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

/** Settings: is this install in paired mode? SaaS never reads discovery and is never paired. */
export function usePairingMode(): PairingMode {
  const { data } = useQuery({ queryKey: DISCOVERY_QUERY_KEY, queryFn: readDiscovery, enabled: isLocal })
  const paired = isLocal && takesPairedDevices(data)
  return { paired, webUrl: paired ? data?.web_url ?? null : null }
}

/** The devices the caller may see, read again every `refreshMs` when given (a live pairing code). */
export function useMobileDevices(refreshMs?: number) {
  return useQuery({
    queryKey: MOBILE_DEVICES_QUERY_KEY,
    queryFn: async () => (await apiClient.get<{ devices: MobileDevice[] }>(DEVICES_PATH)).devices,
    refetchInterval: refreshMs ?? false,
  })
}

/** Revoke a device (a paired one's next call is refused: device_revoked), then read the list again. */
export function useRevokeDevice() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: (deviceId: string) => apiClient.delete<MobileDevice>(`${DEVICES_PATH}/${encodeURIComponent(deviceId)}`),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: MOBILE_DEVICES_QUERY_KEY }),
  })
}

/** Mint a one-time code that pairs a phone or a browser to the caller: ten minutes, once. */
export function useMintPairingCode() {
  return useMutation({ mutationFn: () => apiClient.post<PairingCode>(PAIRING_CODES_PATH) })
}
