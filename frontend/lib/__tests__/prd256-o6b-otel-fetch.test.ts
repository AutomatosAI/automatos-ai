// @vitest-environment node
/**
 * PRD-256 O6b (#847): a request the web app's server makes with `fetch` (a Node-runtime
 * route proxying to the API) carries `traceparent`, so the API continues the web app's
 * trace. Seen live: Next's own fetch span records the call but sends no header, and
 * the API started a trace of its own. A local server here records what it was sent.
 */
import { createServer, type Server } from 'node:http'
import type { AddressInfo } from 'node:net'

import { afterAll, beforeAll, expect, it } from 'vitest'
import { trace, type ProxyTracerProvider } from '@opentelemetry/api'
import { InMemorySpanExporter, type NodeTracerProvider } from '@opentelemetry/sdk-trace-node'

import { startTracing, stopTracing } from '../otel/node'

const memory = new InMemorySpanExporter()
const received: (string | undefined)[] = []
let server: Server
let base = ''

beforeAll(async () => {
  server = createServer((request, response) => {
    received.push(request.headers.traceparent as string | undefined)
    response.end('{}')
  })
  await new Promise<void>((resolve) => server.listen(0, '127.0.0.1', resolve))
  base = `http://127.0.0.1:${(server.address() as AddressInfo).port}`
  expect(startTracing({}, memory)).toBe(true)
})

afterAll(async () => {
  await stopTracing()
  trace.disable()
  server.close()
})

it('a fetch to the API carries the trace, and its span keeps no query value', async () => {
  const tracer = trace.getTracer('t')
  const traceId = await tracer.startActiveSpan('GET /api/generated-images/[id]', async (span) => {
    await (await fetch(`${base}/api/generated-images/1?token=SECRET`)).text()
    span.end()
    return span.spanContext().traceId
  })
  expect(received).toHaveLength(1)
  expect(received[0]?.split('-')[1]).toBe(traceId)   // the API continues this trace

  const installed = (trace.getTracerProvider() as ProxyTracerProvider).getDelegate() as NodeTracerProvider
  await installed.forceFlush()
  const client = memory.getFinishedSpans().find((span) => span.attributes['url.full'])
  expect(client?.spanContext().traceId).toBe(traceId)
  expect(client?.attributes['url.full']).toBe(`${base}/api/generated-images/1?token=REDACTED`)
  expect(JSON.stringify(memory.getFinishedSpans().map((span) => [span.name, span.attributes]))).not.toContain('SECRET')
})
