/**
 * PRD-256 O6b (#847): the web app's tracer provider, Node runtime only (imported by
 * `instrumentation.ts` when OTEL_ENABLED is on). Next.js then records its own server
 * spans (each request, server render and Node-runtime route handler) into it, and they
 * go over OTLP/HTTP to the same collector as the API's. Node's `fetch` (undici) is
 * instrumented so a request to the API carries `traceparent` and joins the API's
 * trace: Next's own fetch span doesn't send it. The Edge-runtime routes (the chat and
 * workflow-stream proxies) run where this SDK can't: #1094.
 *
 * Upstream OpenTelemetry packages only (vendor-neutral, decided on #847). The OTLP
 * exporter reads OTEL_EXPORTER_OTLP_ENDPOINT and OTEL_EXPORTER_OTLP_HEADERS itself;
 * OTEL_RESOURCE_ATTRIBUTES (the chart's pod and namespace) joins the resource.
 * Principle 5: every query value on a URL attribute is redacted before export, and an
 * error leaves as its type only, never its message or stack trace (#1107), as the API's do.
 */
import { OTLPTraceExporter } from '@opentelemetry/exporter-trace-otlp-http'
import { UndiciInstrumentation } from '@opentelemetry/instrumentation-undici'
import { detectResources, envDetector, resourceFromAttributes } from '@opentelemetry/resources'
import {
  BatchSpanProcessor,
  NodeTracerProvider,
  ParentBasedSampler,
  TraceIdRatioBasedSampler,
  type ReadableSpan,
  type SpanExporter,
} from '@opentelemetry/sdk-trace-node'

import { DEFAULT_SERVICE_NAME, EXCEPTION_KEPT, redactQuery, redactedAttributes, samplerRatio } from './settings'

/** The environment the settings are read from (process.env, or a test's own). */
type Env = Record<string, string | undefined>

type SpanEvent = ReadableSpan['events'][number]

/** An `exception` event keeps its type and drops its text: Next.js records a failure
 * with the error's message and stack trace, which can hold a URL or a user's data. */
function typeOnlyEvent(event: SpanEvent): SpanEvent {
  if (event.name !== 'exception') return event
  const attributes = Object.fromEntries(Object.entries(event.attributes ?? {})
    .filter(([key]) => (EXCEPTION_KEPT as readonly string[]).includes(key)))
  return { ...event, attributes }
}

/** What an ERROR status may say: the recorded exception's type, else the message when it
 * is the span's own `error.type`; never the error's text (Next.js copies its message there),
 * nor a word that merely looks like a type ("report_q3.xlsx"). */
function failedType(events: SpanEvent[], attributes: ReadableSpan['attributes'], message: string): string | undefined {
  const typed = events.find((event) => event.name === 'exception' && event.attributes?.['exception.type'])
  if (typed) return String(typed.attributes?.['exception.type'])
  const errorType = attributes['error.type']
  return errorType !== undefined && message === String(errorType) ? message : undefined
}

/** The span as exported: no query value in its name or URL attributes, and an error as
 * its type only. The span itself stays the prototype, so spanContext() still answers. */
function redactedSpan(span: ReadableSpan): ReadableSpan {
  const events = span.events.map(typeOnlyEvent)
  const status = span.status.message ? { ...span.status, message: failedType(events, span.attributes, span.status.message) } : span.status
  return Object.create(span, {
    name: { value: redactQuery(span.name), enumerable: true },
    attributes: { value: redactedAttributes({ ...span.attributes }), enumerable: true },
    status: { value: status, enumerable: true },
    events: { value: events, enumerable: true },
  })
}

/** The span redacted, or none: a span that can't be redacted is dropped, never sent as it was. */
function exportable(span: ReadableSpan): ReadableSpan[] {
  try {
    return [redactedSpan(span)]
  } catch (error) {
    console.warn('[otel] a span could not be redacted and was dropped', error)
    return []
  }
}

/** Wraps an exporter: each span goes out redacted (redactedSpan), or not at all. */
export class RedactingExporter implements SpanExporter {
  constructor(private readonly inner: SpanExporter) {}

  export(spans: ReadableSpan[], done: Parameters<SpanExporter['export']>[1]): void {
    this.inner.export(spans.flatMap(exportable), done)
  }

  shutdown(): Promise<void> {
    return this.inner.shutdown()
  }

  forceFlush(): Promise<void> {
    return this.inner.forceFlush ? this.inner.forceFlush() : Promise.resolve()
  }
}

/** This process's provider: the web app's resource, a parent-based ratio sampler, and
 * batched export through `exporter` (the OTLP one unless a test passes its own). */
export function buildProvider(env: Env, exporter?: SpanExporter): NodeTracerProvider {
  const resource = resourceFromAttributes({
    'service.name': env.OTEL_SERVICE_NAME?.trim() || DEFAULT_SERVICE_NAME,
    'automatos.edition': env.NEXT_PUBLIC_AUTH_EDITION || 'unknown',
  }).merge(detectResources({ detectors: [envDetector] }))
  const sampler = new ParentBasedSampler({ root: new TraceIdRatioBasedSampler(samplerRatio(env.OTEL_TRACES_SAMPLER_RATIO)) })
  const processor = new BatchSpanProcessor(new RedactingExporter(exporter ?? new OTLPTraceExporter()))
  return new NodeTracerProvider({ resource, sampler, spanProcessors: [processor] })
}

const installed: { provider?: NodeTracerProvider; fetch?: UndiciInstrumentation } = {}

/** Installs the provider globally (with W3C trace context), and traces Node's `fetch`,
 * once. Never stops the boot. `exporter` replaces the OTLP one (tests). */
export function startTracing(env: Env = process.env, exporter?: SpanExporter): boolean {
  if (installed.provider) return true
  try {
    installed.provider = buildProvider(env, exporter)
    installed.provider.register()
    installed.fetch = new UndiciInstrumentation()   // enabled on construction
    console.info(`[otel] tracing the web app to ${env.OTEL_EXPORTER_OTLP_ENDPOINT || 'http://localhost:4318'}`)
    return true
  } catch (error) {
    console.error('[otel] tracing not started; serving without traces', error)
    return false
  }
}

/** Undoes startTracing: flushes and stops the provider, and unpatches `fetch` (tests). */
export async function stopTracing(): Promise<void> {
  installed.fetch?.disable()
  await installed.provider?.shutdown()
  installed.fetch = installed.provider = undefined
}
