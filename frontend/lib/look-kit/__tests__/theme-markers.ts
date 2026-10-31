/**
 * PRD-261 A-13a, A-19f2 — what a theme's own code or names look like, for the scans that hold the
 * look runtimes to naming no theme: lib/look-kit/__tests__/theme-neutral.test.ts (lib/look-kit) and
 * lib/theme-host/looks/__tests__/theme-neutral.test.ts (the generic look runtime). One list for both,
 * so the two scans can never drift: no theme's variable prefix, quoted interface name, storage keys or
 * look format, no import of a theme's folders or of components.
 *
 * The names builtin Play left are built by parts, so this file's own text never reads as a theme's
 * code to the scan of the whole frontend (lib/__tests__/prd261-no-theme-in-core.test.ts).
 *
 * Plain .ts: a helper, never a test file.
 */

export const THEME_MARKERS: readonly RegExp[] = [
  new RegExp(['--', 'pl-'].join('')),
  /['"`]play['"`]/,
  new RegExp(['automatos', 'play'].join('-')),
  new RegExp(['LOOK', 'INTERFACE'].join('_')),
  /@\/lib\/(looks|play|mascot)(\/|['"])/,
  /@\/components\//,
]

/** The markers `source` holds, each as its pattern's text. */
export function markersIn(source: string): string[] {
  return THEME_MARKERS.filter((marker) => marker.test(source)).map(String)
}
