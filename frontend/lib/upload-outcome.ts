/**
 * F087: what an upload did, in the server's own words. A file uploaded again
 * under the same name replaces the earlier document ("Replaced … (version N)"),
 * identical content is already stored, and a failed pipeline says so — the
 * fixed "Document uploaded successfully" toast hid all three.
 */
export interface UploadResponse {
  status?: string
  message?: string
}

export const UPLOAD_OK_FALLBACK = 'Document uploaded successfully'

export function uploadOutcome(response: UploadResponse | null | undefined): { ok: boolean; text: string } {
  return {
    ok: response?.status !== 'failed',
    text: response?.message || UPLOAD_OK_FALLBACK,
  }
}
