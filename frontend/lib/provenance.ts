/**
 * PRD-264 EUAI-10 — check a file for an Automatos AI mark without uploading it.
 *
 * The /verify page hashes the file in the browser with Web Crypto (SHA-256) and sends
 * only the 64-character hex digest to GET /api/public/provenance/{sha256}. The file
 * never leaves the visitor's machine. The answer is yes (with the media kind and the
 * date it was made) or no; it never names a workspace or a person.
 */
import { apiClient } from '@/lib/api-client'

export const PROVENANCE_ROUTE = '/api/public/provenance'
/** Reads C2PA Content Credentials, for files signed by any tool. */
export const CONTENT_CREDENTIALS_VERIFY_URL = 'https://contentcredentials.org/verify'
/** Web Crypto digests a whole buffer at once, so a file must fit in one browser buffer. */
export const MAX_HASH_BYTES = 2 * 1024 * 1024 * 1024

const SHA256_HEX = /^[0-9a-f]{64}$/
const HEX_RADIX = 16
const HEX_BYTE_WIDTH = 2

export type ProvenanceKind = 'video' | 'image' | 'audio' | 'document'

export type ProvenanceAnswer =
  | { readonly made_by_automatos_ai: true; readonly kind: ProvenanceKind; readonly date: string }
  | { readonly made_by_automatos_ai: false }

/** What sha256Hex reads: a File or a Blob. */
export interface HashableBytes {
  readonly size: number
  arrayBuffer(): Promise<ArrayBuffer>
}

export class ProvenanceCheckError extends Error {
  constructor(message: string) {
    super(message)
    this.name = 'ProvenanceCheckError'
  }
}

/** True only for 64 lower-case hex characters, the form the route takes. */
export function isSha256Hex(value: string): boolean {
  return SHA256_HEX.test(value)
}

function browserSubtle(): SubtleCrypto {
  const subtle = globalThis.crypto?.subtle
  if (!subtle) {
    throw new ProvenanceCheckError(
      'This browser cannot fingerprint files here. Open the page over HTTPS or on localhost.',
    )
  }
  return subtle
}

/** The file's SHA-256 as 64 lower-case hex characters, computed on this device. */
export async function sha256Hex(file: HashableBytes, subtle: SubtleCrypto = browserSubtle()): Promise<string> {
  if (file.size > MAX_HASH_BYTES) {
    throw new ProvenanceCheckError('This file is too large to fingerprint in the browser (the limit is 2 GB).')
  }
  const digest = await subtle.digest('SHA-256', await file.arrayBuffer())
  return Array.from(new Uint8Array(digest), (byte) => byte.toString(HEX_RADIX).padStart(HEX_BYTE_WIDTH, '0')).join('')
}

/** Ask Automatos whether a file with this hash is one it made. Only the hash is sent. */
export async function lookupProvenance(sha256: string): Promise<ProvenanceAnswer> {
  if (!isSha256Hex(sha256)) {
    throw new ProvenanceCheckError('A SHA-256 is 64 lower-case hex characters.')
  }
  return apiClient.get<ProvenanceAnswer>(`${PROVENANCE_ROUTE}/${sha256}`)
}
