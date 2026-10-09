/**
 * Next.js calls register() once when the server starts (PRD-256 O6b, #847).
 *
 * With OTEL_ENABLED on, the Node runtime installs a tracer provider
 * (lib/otel/node.ts) and Next.js records its server spans into it. Off, or in
 * the Edge runtime (the middleware), nothing from OpenTelemetry is imported.
 */
import { otelEnabled } from './lib/otel/settings'

export async function register(): Promise<void> {
  // This shape (the import inside the nodejs branch) lets Next drop it from the Edge bundle.
  if (process.env.NEXT_RUNTIME === 'nodejs' && otelEnabled(process.env.OTEL_ENABLED)) {
    const { startTracing } = await import('./lib/otel/node')
    startTracing()
  }
}
