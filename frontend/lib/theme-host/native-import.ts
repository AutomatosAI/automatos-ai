/**
 * PRD-261 A-01 — the browser's own import(), not the bundler's. The comments tell webpack (and
 * Turbopack and Vite, for vitest) to leave the call as it is, so the browser fetches the theme's
 * module and resolves its bare imports through the page's import map. Kept in its own file so
 * unit tests can replace it.
 */
export function importThemeModule(url: string): Promise<unknown> {
  return import(/* webpackIgnore: true */ /* turbopackIgnore: true */ /* @vite-ignore */ url)
}
