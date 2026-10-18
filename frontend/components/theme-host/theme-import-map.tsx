import { importMapScriptBody } from '@/lib/theme-host/import-map'

interface ThemeImportMapProps {
  /** The boot's integrity block (PA-13): module URL → 'sha256-<base64>'; checked before it is written. */
  readonly integrity?: Readonly<Record<string, string>> | null
}

/**
 * PRD-261 A-01, A-03 — the theme import map, a server component. A browser applies an import map
 * only if it is parsed before any module script or import() starts loading, so the root layout
 * renders this as the first child of the document head, on every page (PA-13). Next's own chunk
 * scripts are classic scripts, so they do not count; the one module load on the page is the theme
 * loader's import(), which runs after hydration. The browser lane checks the order
 * (e2e/browser/specs/first-paint.spec.ts).
 */
export function ThemeImportMap({ integrity = null }: ThemeImportMapProps) {
  return <script type="importmap" dangerouslySetInnerHTML={{ __html: importMapScriptBody(integrity) }} />
}
