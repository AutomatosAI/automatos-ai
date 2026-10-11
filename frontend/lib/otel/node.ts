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
 * Principle 5: every query value on a URL attribute is redacted before export.
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

import { DEFAULT_SERVICE_NAME, redactQuery, redactQueryValuesIn, redactedAttributes, samplerRatio } from './settings'

/** The environment the settings are read from (process.env, or a test's own). */
type Env = Record<string, string | undefined>

/** An event's string attributes with their query values redacted: Next.js records a
 * failure as an `exception` event whose message and stack trace can hold a URL. */
function redactedEvent(event: ReadableSpan['events'][number]): ReadableSpan['events'][number] {
  const attributes = Object.fromEntries(Object.entries(event.attributes ?? {}).map(([key, value]) =>
    [key, typeof value === 'string' ? redactQueryValuesIn(value) : value]))
  return { ...event, attributes }
}

/** The span as exported: no query value in its name, URL attributes, status message or
 * events. The span itself stays the prototype, so spanContext() and the rest still answer. */
function redactedSpan(span: ReadableSpan): ReadableSpan {
  const status = span.status.message ? { ...span.status, message: redactQueryValuesIn(span.status.message) } : span.status
  return Object.create(span, {
    name: { value: redactQuery(span.name), enumerable: true },
    attributes: { value: redactedAttributes({ ...span.attributes }), enumerable: true },
    status: { value: status, enumerable: true },
    events: { value: span.events.map(redactedEvent), enumerable: true },
  })
}

/** Wraps an exporter: each span goes out with its query values redacted (redactedSpan). */
export class RedactingExporter implements SpanExporter {
  constructor(private readonly inner: SpanExporter) {}

  export(spans: ReadableSpan[], done: Parameters<SpanExporter['export']>[1]): void {
    this.inner.export(spans.map(redactedSpan), done)
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
