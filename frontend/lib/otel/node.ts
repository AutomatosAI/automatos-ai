/**
 * PRD-256 O6b (#847): the web app's tracer provider, Node runtime only (imported by
 * `instrumentation.ts` when OTEL_ENABLED is on). Next.js then records its own server
 * spans (each request, render and route handler, and each `fetch` the server makes)
 * into it, and they go over OTLP/HTTP to the same collector as the API's.
 *
 * Upstream OpenTelemetry packages only (vendor-neutral, decided on #847). The OTLP
 * exporter reads OTEL_EXPORTER_OTLP_ENDPOINT and OTEL_EXPORTER_OTLP_HEADERS itself;
 * OTEL_RESOURCE_ATTRIBUTES (the chart's pod and namespace) joins the resource.
 * Principle 5: every query value on a URL attribute is redacted before export.
 */
import { OTLPTraceExporter } from '@opentelemetry/exporter-trace-otlp-http'
import { detectResources, envDetector, resourceFromAttributes } from '@opentelemetry/resources'
import {
  BatchSpanProcessor,
  NodeTracerProvider,
  ParentBasedSampler,
  TraceIdRatioBasedSampler,
  type ReadableSpan,
  type SpanExporter,
} from '@opentelemetry/sdk-trace-node'

import { DEFAULT_SERVICE_NAME, redactedAttributes, samplerRatio } from './settings'

/** Wraps an exporter: each span goes out with its URL attributes' query values redacted. */
export class RedactingExporter implements SpanExporter {
  constructor(private readonly inner: SpanExporter) {}

  export(spans: ReadableSpan[], done: Parameters<SpanExporter['export']>[1]): void {
    // The span itself stays the prototype, so spanContext() and the rest still answer.
    const redacted = spans.map((span) =>
      Object.create(span, { attributes: { value: redactedAttributes({ ...span.attributes }), enumerable: true } }),
    )
    this.inner.export(redacted, done)
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
export function buildProvider(env: NodeJS.ProcessEnv, exporter?: SpanExporter): NodeTracerProvider {
  const resource = resourceFromAttributes({
    'service.name': env.OTEL_SERVICE_NAME?.trim() || DEFAULT_SERVICE_NAME,
    'automatos.edition': env.NEXT_PUBLIC_AUTH_EDITION || 'unknown',
  }).merge(detectResources({ detectors: [envDetector] }))
  const sampler = new ParentBasedSampler({ root: new TraceIdRatioBasedSampler(samplerRatio(env.OTEL_TRACES_SAMPLER_RATIO)) })
  const processor = new BatchSpanProcessor(new RedactingExporter(exporter ?? new OTLPTraceExporter()))
  return new NodeTracerProvider({ resource, sampler, spanProcessors: [processor] })
}

/** Installs the provider globally (with W3C trace context), once. Never stops the boot. */
export function startTracing(env: NodeJS.ProcessEnv = process.env): boolean {
  try {
    buildProvider(env).register()
    console.info(`[otel] tracing the web app to ${env.OTEL_EXPORTER_OTLP_ENDPOINT || 'http://localhost:4318'}`)
    return true
  } catch (error) {
    console.error('[otel] tracing not started; serving without traces', error)
    return false
  }
}
