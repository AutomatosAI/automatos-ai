// @vitest-environment node
/**
 * PRD-256 O6b (#847): the web app's server-side spans.
 *
 * Off (the default) or in the Edge runtime, register() loads nothing from
 * OpenTelemetry. On, the provider names the web app, follows a sampled caller,
 * keeps new traces by OTEL_TRACES_SAMPLER_RATIO, and never exports a query value.
 * Spans go to an in-memory exporter here.
 */
import { afterEach, describe, expect, it, vi } from 'vitest'
import { InMemorySpanExporter } from '@opentelemetry/sdk-trace-node'
import { context, trace, TraceFlags, type Span } from '@opentelemetry/api'

import { otelEnabled, redactQuery, redactedAttributes, samplerRatio } from '../otel/settings'
import { buildProvider } from '../otel/node'

const { startTracing } = vi.hoisted(() => ({ startTracing: vi.fn() }))
vi.mock('../otel/node', async (importOriginal) => ({ ...(await importOriginal<object>()), startTracing }))

afterEach(() => {
  vi.unstubAllEnvs()
  startTracing.mockClear()
})

describe('settings, read as the API reads them', () => {
  it.each([
    ['true', true], ['1', true], [' YES ', true], ['on', true],
    ['false', false], ['0', false], ['', false], [undefined, false],
  ])('OTEL_ENABLED=%s is %s', (raw, expected) => {
    expect(otelEnabled(raw)).toBe(expected)
  })

  it.each([
    ['0.25', 0.25], [' 1 ', 1], ['0', 0], ['5', 1], ['-1', 0], ['abc', 1], ['', 1], [undefined, 1],
  ])('OTEL_TRACES_SAMPLER_RATIO=%s is %s, and never throws', (raw, expected) => {
    expect(samplerRatio(raw)).toBe(expected)
  })

  it.each([
    ['http://h/x?code=abc&token=t%20u&page=2', 'http://h/x?code=REDACTED&token=REDACTED&page=REDACTED'],
    ['/x?flag&code=abc#frag', '/x?flag&code=REDACTED#frag'],
    ['/x', '/x'],
    ['/x?', '/x?'],
  ])('a query value never leaves on a URL: %s', (url, expected) => {
    expect(redactQuery(url)).toBe(expected)
  })

  it('redacts every URL attribute and leaves the rest', () => {
    const attributes = { 'http.target': '/chat?id=SECRET', 'url.query': 'q=SECRET&p=1', 'next.route': '/chat', n: 3 }
    expect(redactedAttributes(attributes)).toEqual({
      'http.target': '/chat?id=REDACTED', 'url.query': 'q=REDACTED&p=REDACTED', 'next.route': '/chat', n: 3,
    })
    expect(attributes['http.target']).toBe('/chat?id=SECRET')   // a copy, never the span's own
  })
})

describe('the provider', () => {
  async function exported(env: Record<string, string>, start: (tracer: ReturnType<typeof trace.getTracer>) => Span) {
    const memory = new InMemorySpanExporter()
    const provider = buildProvider(env, memory)
    start(provider.getTracer('t')).end()
    await provider.forceFlush()
    const spans = memory.getFinishedSpans()   // before shutdown, which clears the in-memory exporter
    await provider.shutdown()
    return spans
  }

  it('names the web app and joins OTEL_RESOURCE_ATTRIBUTES', async () => {
    vi.stubEnv('OTEL_RESOURCE_ATTRIBUTES', 'k8s.pod.name=web-0,k8s.namespace.name=automatos')
    const [span] = await exported({}, (tracer) => tracer.startSpan('GET /chat'))
    expect(span.resource.attributes['service.name']).toBe('automatos-web')
    expect(span.resource.attributes['k8s.pod.name']).toBe('web-0')
    const [named] = await exported({ OTEL_SERVICE_NAME: 'web-eu' }, (tracer) => tracer.startSpan('GET /chat'))
    expect(named.resource.attributes['service.name']).toBe('web-eu')
  })

  it('exports no query value, and the span still answers for its context', async () => {
    // As Next.js names its request span (seen live): the full target, query and all.
    const [span] = await exported({}, (tracer) =>
      tracer.startSpan('POST /api/chat?id=SECRET&page=2', { attributes: {
        'next.span_name': 'POST /api/chat?id=SECRET&page=2', 'http.target': '/api/chat?id=SECRET&page=2',
        'next.route': '/api/chat',
      } }))
    expect(span.name).toBe('POST /api/chat?id=REDACTED&page=REDACTED')
    expect(span.attributes['next.span_name']).toBe('POST /api/chat?id=REDACTED&page=REDACTED')
    expect(span.attributes['http.target']).toBe('/api/chat?id=REDACTED&page=REDACTED')
    expect(span.attributes['next.route']).toBe('/api/chat')
    expect(span.spanContext().traceId).toMatch(/^[0-9a-f]{32}$/)
    expect(JSON.stringify(span.attributes)).not.toContain('SECRET')
  })

  it('keeps no new trace at ratio 0, but follows a sampled caller', async () => {
    const ratioZero = { OTEL_TRACES_SAMPLER_RATIO: '0' }
    expect(await exported(ratioZero, (tracer) => tracer.startSpan('GET /chat'))).toHaveLength(0)
    const caller = trace.setSpanContext(context.active(), {
      traceId: '0af7651916cd43dd8448eb211c80319c', spanId: 'b7ad6b7169203331', traceFlags: TraceFlags.SAMPLED,
    })
    const [span] = await exported(ratioZero, (tracer) => tracer.startSpan('GET /chat', {}, caller))
    expect(span.spanContext().traceId).toBe('0af7651916cd43dd8448eb211c80319c')
  })
})

describe('register() (instrumentation.ts)', () => {
  async function registered(env: Record<string, string>) {
    for (const [key, value] of Object.entries(env)) vi.stubEnv(key, value)
    const { register } = await import('../../instrumentation')
    await register()
  }

  it('starts nothing when OTEL_ENABLED is off', async () => {
    await registered({ NEXT_RUNTIME: 'nodejs', OTEL_ENABLED: '' })
    expect(startTracing).not.toHaveBeenCalled()
  })

  it('starts nothing in the Edge runtime', async () => {
    await registered({ NEXT_RUNTIME: 'edge', OTEL_ENABLED: 'true' })
    expect(startTracing).not.toHaveBeenCalled()
  })

  it('starts tracing once in the Node runtime with OTEL_ENABLED on', async () => {
    await registered({ NEXT_RUNTIME: 'nodejs', OTEL_ENABLED: 'true' })
    expect(startTracing).toHaveBeenCalledTimes(1)
  })
})
