/**
 * PRD-261 A-14d — puts text on the clipboard, behind a look's Copy JSON (./look-manager.tsx).
 * Copied into Play from the core's lib/utils.ts copyToClipboard (docs/theme-host/sdk-surface-map.md:
 * copied into Play), since Play imports no core module;
 * components/looks/__tests__/copied-helpers.test.tsx holds the copy to the core's behaviour.
 */

/** True once the text is on the clipboard; false (and logged) when the browser refused it. */
export async function copyToClipboard(text: string): Promise<boolean> {
  try {
    await navigator.clipboard.writeText(text)
    return true
  } catch (error) {
    console.error('Failed to copy to clipboard:', error)
    return false
  }
}
