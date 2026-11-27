/**
 * #1664: the workflow stream proxy refused every real execution id (it checked for a UUID; the ids
 * are integers), and it must still keep anything but an id out of the forwarded URL's path.
 */
import { afterEach, describe, expect, it, vi } from 'vitest'

import { executionIdPath } from '@/lib/workflow-execution-id'

describe('executionIdPath', () => {
  it.each([
    [123, '123'],
    ['123', '123'],
    [1, '1'],
    [Number.MAX_SAFE_INTEGER, String(Number.MAX_SAFE_INTEGER)],
  ])('takes the integer id %p as %p', (sent, path) => {
    expect(executionIdPath(sent)).toBe(path)
  })

  it.each([
    0, -1, 12.5, Number.NaN, Number.POSITIVE_INFINITY, 1e21, '0', '007', '-1', '1.0', '1e3', ' 1', '1 ',
    '../admin', '1/../../x', '1?x=1', '1#x', '%2e%2e', '11111111-2222-4333-8444-555555555555',
    String(Number.MAX_SAFE_INTEGER + 2), '', null, undefined, true, {}, [1],
  ])('refuses %p', (sent) => {
    expect(executionIdPath(sent)).toBeNull()
  })
})

describe('the workflow stream proxy', () => {
  afterEach(() => vi.unstubAllGlobals())

  function backend() {
    const seen: string[] = []
    vi.stubGlobal('fetch', vi.fn(async (url: string) => {
      seen.push(String(url))
      return new Response('0:"ok"\n', { status: 200 })
    }))
    return seen
  }

  it('forwards an integer id from the viewer\'s POST body, as the API takes it', async () => {
    const seen = backend()
    const { POST } = await import('@/app/api/workflows/stream/route')
    const response = await POST(new Request('http://localhost:3000/api/workflows/stream', {
      method: 'POST', body: JSON.stringify({ executionId: 123 }), headers: { 'Content-Type': 'application/json' },
    }) as never)
    expect(response.status).toBe(200)
    expect(seen).toEqual([expect.stringMatching(/\/api\/workflows\/executions\/123\/stream\/aisdk$/)])
  })

  it('forwards an integer id from a GET query', async () => {
    const seen = backend()
    const { GET } = await import('@/app/api/workflows/stream/route')
    const response = await GET(new Request('http://localhost:3000/api/workflows/stream?executionId=123') as never)
    expect(response.status).toBe(200)
    expect(seen).toEqual([expect.stringMatching(/\/api\/workflows\/executions\/123\/stream\/aisdk$/)])
  })

  it('answers 400 and calls nothing for an id that is not a positive whole number', async () => {
    const seen = backend()
    const { GET, POST } = await import('@/app/api/workflows/stream/route')
    const get = await GET(new Request('http://localhost:3000/api/workflows/stream?executionId=1%2F..%2Fx') as never)
    const post = await POST(new Request('http://localhost:3000/api/workflows/stream', {
      method: 'POST', body: JSON.stringify({ executionId: '../admin' }), headers: { 'Content-Type': 'application/json' },
    }) as never)
    expect([get.status, post.status]).toEqual([400, 400])
    expect(seen).toEqual([])
  })
})
