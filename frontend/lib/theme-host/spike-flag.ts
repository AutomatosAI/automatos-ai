/**
 * PRD-261 A-01 (D3) — whether this build carries the theme-host spike.
 *
 * next.config.js `env` inlines AUTOMATOS_THEME_HOST_SPIKE into the server and client bundles as
 * '1' or '0' when the app is built, so nothing a request carries can turn the spike on in a
 * running app. Only the browser lane's spike job builds with '1'; every shipped image is built
 * with '0'. A function, not a constant, so a test can stub the variable.
 */
export function isThemeHostSpikeOn(): boolean {
  return process.env.AUTOMATOS_THEME_HOST_SPIKE === '1'
}
