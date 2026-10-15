/**
 * PRD-260 — saving text as a file from the browser (a look's Export): a Blob
 * URL on a throwaway link, ported from Automatos Play's
 * components/looks/download.ts. The URL is let go a moment later, once the
 * browser has started the download (Safari needs it to live that long).
 */

const REVOKE_AFTER_MS = 1000

/**
 * Starts a download of `text` as `fileName`. It throws only when the Blob or
 * its URL can't be made: a download the browser blocks, or the user cancels,
 * can't be seen from here, so callers say it started, never that it arrived.
 */
export function downloadText(fileName: string, text: string, type: string): void {
  const url = URL.createObjectURL(new Blob([text], { type }))
  const link = document.createElement('a')
  link.href = url
  link.download = fileName
  document.body.append(link)
  try {
    link.click()
  } finally {
    link.remove()
    window.setTimeout(() => URL.revokeObjectURL(url), REVOKE_AFTER_MS)
  }
}
