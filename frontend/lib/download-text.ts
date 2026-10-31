/**
 * Saving text the page already holds as a file from the browser: a Blob URL on
 * a throwaway link (PRD-260 P260-FX-27, M88), behind a knowledge document's
 * markdown download (DocumentWidget's downloadMarkdownFile), ported from
 * Automatos Play's components/looks/download.ts. Play's look Export uses its
 * own copy since PRD-261 A-14d (components/looks/download-text.ts: Play imports
 * no core module), which components/looks/__tests__/copied-helpers.test.tsx
 * holds to this one. The URL is let go a moment later, once the browser has
 * started the download: Safari needs it to live that long.
 *
 * PRD-265 W2: `downloadBlob` saves bytes the page fetched the same way (an
 * invoice PDF from the Billing tab, fetched through apiClient).
 */

/** How long the Blob URL outlives the click. */
export const DOWNLOAD_REVOKE_AFTER_MS = 1000

/**
 * Starts a download of `blob` as `fileName`. It throws only when the Blob URL
 * can't be made: a download the browser blocks, or the user cancels, can't be
 * seen from here, so callers say it started, never that it arrived.
 */
export function downloadBlob(fileName: string, blob: Blob): void {
  const url = URL.createObjectURL(blob)
  const link = document.createElement('a')
  link.href = url
  link.download = fileName
  document.body.append(link)
  try {
    link.click()
  } finally {
    link.remove()
    window.setTimeout(() => URL.revokeObjectURL(url), DOWNLOAD_REVOKE_AFTER_MS)
  }
}

/** Starts a download of `text` as `fileName` (see downloadBlob). */
export function downloadText(fileName: string, text: string, type: string): void {
  downloadBlob(fileName, new Blob([text], { type }))
}
