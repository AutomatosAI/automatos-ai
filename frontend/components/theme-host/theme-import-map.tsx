import { IMPORT_MAP_SCRIPT_BODY } from '@/lib/theme-host/import-map'

/**
 * PRD-261 A-01 — the theme import map, a server component. A browser applies an import map only
 * if it is parsed before any module script or import() starts loading, so the root layout
 * renders this as the first child of the document head, and only in a build with the spike on.
 * Next's own chunk scripts are classic scripts, so they do not count; the one module load on the
 * page is the theme loader's import(), which runs after hydration. The browser lane checks the
 * order (e2e/browser/specs/theme-host.spec.ts).
 */
export function ThemeImportMap() {
  return <script type="importmap" dangerouslySetInnerHTML={{ __html: IMPORT_MAP_SCRIPT_BODY }} />
}
