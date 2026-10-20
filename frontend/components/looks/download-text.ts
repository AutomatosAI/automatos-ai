/**
 * PRD-260 P260-FX-27 (M88), PRD-261 A-14d — saving text the page already holds as a file from the
 * browser: a Blob URL on a throwaway link, behind a look's Export (./look-manager.tsx). Copied into
 * Play from the core's lib/download-text.ts (docs/theme-host/sdk-surface-map.md: copied into Play),
 * since Play imports no core module; components/looks/__tests__/copied-helpers.test.tsx holds the
 * copy to the core's behaviour. The URL is let go a moment later, once the browser has started the
 * download: Safari needs it to live that long.
 */

/** How long the Blob URL outlives the click. */
export const DOWNLOAD_REVOKE_AFTER_MS = 1000

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
    window.setTimeout(() => URL.revokeObjectURL(url), DOWNLOAD_REVOKE_AFTER_MS)
  }
}
