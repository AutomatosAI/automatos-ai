/**
 * PRD-261 A-01 — whether a theme's `engines.sdk` range accepts the host's SDK version.
 *
 * Two forms only: an exact `MAJOR.MINOR.PATCH`, or a caret `^MAJOR.MINOR.PATCH` (the same major,
 * and a minor.patch no newer than the host's). Anything else (`*`, `x`, `>=…`, empty, missing,
 * not a string) is refused: a theme states what it was built against, and a mismatch stops the
 * load before its code is requested.
 */
const VERSION = /^(\d+)\.(\d+)\.(\d+)$/
const CARET_RANGE = /^\^(\d+)\.(\d+)\.(\d+)$/

type Version = readonly [number, number, number]

function versionOf(match: RegExpExecArray): Version {
  return [Number(match[1]), Number(match[2]), Number(match[3])]
}

function notNewer(wanted: Version, host: Version): boolean {
  if (wanted[1] !== host[1]) return wanted[1] < host[1]
  return wanted[2] <= host[2]
}

export function satisfiesSdkRange(range: unknown, hostVersion: string): boolean {
  const hostMatch = VERSION.exec(hostVersion)
  if (typeof range !== 'string' || hostMatch === null) return false
  const host = versionOf(hostMatch)
  const exact = VERSION.exec(range)
  if (exact !== null) return versionOf(exact).every((part, index) => part === host[index])
  const caret = CARET_RANGE.exec(range)
  if (caret === null) return false
  const wanted = versionOf(caret)
  return wanted[0] === host[0] && notNewer(wanted, host)
}
