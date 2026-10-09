/**
 * PRD-256 O6b (#847): the web app's OpenTelemetry settings, read the way the API
 * reads them (orchestrator/core/observability/otel.py), with no OpenTelemetry
 * import here: `instrumentation.ts` checks these before it loads anything.
 */

export const DEFAULT_SERVICE_NAME = 'automatos-web'
export const DEFAULT_SAMPLER_RATIO = 1.0
export const REDACTED = 'REDACTED'

/** URL-shaped span attributes whose query values never leave (Principle 5). Next.js
 * also names its request span after the full target ("POST /api/chat?x=1"), and puts
 * that name in `next.span_name`, so the name and that attribute are redacted too. */
export const URL_ATTRIBUTES = ['http.url', 'http.target', 'url.full', 'url.query', 'next.span_name'] as const

const ON = new Set(['true', '1', 'yes', 'on'])

/** `OTEL_ENABLED`: true for true/1/yes/on (any case), as config.py reads it. */
export function otelEnabled(raw: string | undefined): boolean {
  return ON.has((raw ?? '').trim().toLowerCase())
}

/** `OTEL_TRACES_SAMPLER_RATIO`: a share in [0, 1]; anything else keeps every trace, never throws. */
export function samplerRatio(raw: string | undefined): number {
  const text = (raw ?? '').trim()
  const ratio = text === '' ? NaN : Number(text)
  if (!Number.isFinite(ratio)) return DEFAULT_SAMPLER_RATIO
  return Math.min(Math.max(ratio, 0), 1)
}

/** The URL with every query value replaced, keys kept: `?code=REDACTED&page=REDACTED`. */
export function redactQuery(url: string): string {
  const start = url.indexOf('?')
  if (start === -1) return url
  const hash = url.indexOf('#', start)
  const end = hash === -1 ? url.length : hash
  const query = url
    .slice(start + 1, end)
    .split('&')
    .map((pair) => (pair.includes('=') ? `${pair.slice(0, pair.indexOf('='))}=${REDACTED}` : pair))
    .join('&')
  return `${url.slice(0, start + 1)}${query}${url.slice(end)}`
}

/** A span's attributes with the query values on its URL attributes redacted. */
export function redactedAttributes<T extends Record<string, unknown>>(attributes: T): T {
  const copy: Record<string, unknown> = { ...attributes }
  for (const key of URL_ATTRIBUTES) {
    const value = copy[key]
    if (typeof value === 'string') copy[key] = key === 'url.query' ? redactQuery(`?${value}`).slice(1) : redactQuery(value)
  }
  return copy as T
}
