/**
 * PRD-264 EUAI-10 — the browser half of the detection check.
 *
 * The file is fingerprinted on the device with Web Crypto (SHA-256, checked here against
 * the FIPS 180-2 test vectors through Node's own Web Crypto), and only the 64-character
 * hex fingerprint goes to GET /api/public/provenance/{sha256}, through apiClient.
 */
import { webcrypto } from 'node:crypto'
import { beforeEach, describe, expect, it, vi } from 'vitest'

const get = vi.hoisted(() => vi.fn())
vi.mock('@/lib/api-client', () => ({ apiClient: { get } }))

import {
  MAX_HASH_BYTES,
  PROVENANCE_ROUTE,
  ProvenanceCheckError,
  isSha256Hex,
  lookupProvenance,
  sha256Hex,
  type HashableBytes,
} from '../provenance'

const ABC_SHA256 = 'ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad'
const EMPTY_SHA256 = 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855'
const subtle = webcrypto.subtle as unknown as SubtleCrypto

function bytesOf(values: readonly number[]): HashableBytes {
  const data = Uint8Array.from(values)
  return { size: data.byteLength, arrayBuffer: async () => data.buffer }
}

beforeEach(() => {
  get.mockReset()
})

describe('sha256Hex — the fingerprint is computed on this device', () => {
  it('matches the SHA-256 test vectors', async () => {
    expect(await sha256Hex(bytesOf([0x61, 0x62, 0x63]), subtle)).toBe(ABC_SHA256)
    expect(await sha256Hex(bytesOf([]), subtle)).toBe(EMPTY_SHA256)
  })

  it('refuses a file too large for one browser buffer without reading it', async () => {
    const arrayBuffer = vi.fn()
    const huge = { size: MAX_HASH_BYTES + 1, arrayBuffer } as unknown as HashableBytes
    await expect(sha256Hex(huge, subtle)).rejects.toBeInstanceOf(ProvenanceCheckError)
    expect(arrayBuffer).not.toHaveBeenCalled()
  })
})

describe('isSha256Hex — the one form the route takes', () => {
  it('accepts 64 lower-case hex characters only', () => {
    expect(isSha256Hex(ABC_SHA256)).toBe(true)
    expect(isSha256Hex(ABC_SHA256.toUpperCase())).toBe(false)
    expect(isSha256Hex(ABC_SHA256.slice(1))).toBe(false)
    expect(isSha256Hex(`${ABC_SHA256}0`)).toBe(false)
    expect(isSha256Hex(`${ABC_SHA256.slice(1)}g`)).toBe(false)
    expect(isSha256Hex(`${ABC_SHA256}\n`)).toBe(false)
    expect(isSha256Hex('')).toBe(false)
  })
})

describe('lookupProvenance — only the hash is sent', () => {
  it('asks the public route by hash and returns its answer', async () => {
    const answer = { made_by_automatos_ai: true, kind: 'video', date: '2026-10-19' }
    get.mockResolvedValueOnce(answer)

    await expect(lookupProvenance(ABC_SHA256)).resolves.toEqual(answer)
    expect(get).toHaveBeenCalledTimes(1)
    expect(get.mock.calls[0]).toEqual([`${PROVENANCE_ROUTE}/${ABC_SHA256}`])
    expect(PROVENANCE_ROUTE).toBe('/api/public/provenance')
  })

  it('never calls the backend with anything but a valid hash', async () => {
    await expect(lookupProvenance('../workspaces/current')).rejects.toBeInstanceOf(ProvenanceCheckError)
    await expect(lookupProvenance(ABC_SHA256.toUpperCase())).rejects.toBeInstanceOf(ProvenanceCheckError)
    expect(get).not.toHaveBeenCalled()
  })
})
