// @vitest-environment node
/**
 * Theme voice, web quick-ask — the AI notice (PRD-264 EUAI-04, D73): the greeting goes ahead of the first spoken
 * sentence of the browser session, once, remembered in this tab's sessionStorage, and never again; the answering
 * agent's name stands in for Auto.
 */
import { afterEach, describe, expect, it, vi } from 'vitest'

import { noticeHeard, noticeText, queueWithNotice, type NoticeStorage } from '@/lib/voice/quick-ask/ai-notice'
import { AI_NOTICE_STORAGE_KEY } from '@/lib/voice/quick-ask/config'

const GREETING = 'Hi, this is Auto, your AI assistant.'

function memoryStorage(): NoticeStorage & { readonly values: Map<string, string> } {
  const values = new Map<string, string>()
  return {
    values,
    getItem: (key: string) => values.get(key) ?? null,
    setItem: (key: string, value: string) => {
      values.set(key, value)
    },
  }
}

afterEach(() => {
  vi.restoreAllMocks()
})

describe('queueWithNotice', () => {
  it('puts the greeting ahead of the session’s first sentences and remembers it in the tab', () => {
    const storage = memoryStorage()

    const first = queueWithNotice(false, storage, ['Two meetings today.'])

    expect(first).toEqual({ heard: true, speak: [GREETING, 'Two meetings today.'] })
    expect(storage.values.get(AI_NOTICE_STORAGE_KEY)).toBe('1')
  })

  it('never greets again in the same tab session, even from a new engine', () => {
    const storage = memoryStorage()
    queueWithNotice(false, storage, ['First.'])

    expect(queueWithNotice(true, storage, ['Second.']).speak).toEqual(['Second.'])
    expect(queueWithNotice(false, storage, ['A new engine.']).speak).toEqual(['A new engine.'])
  })

  it('waits for the first sentence: nothing to say queues no greeting', () => {
    const storage = memoryStorage()

    expect(queueWithNotice(false, storage, [])).toEqual({ heard: false, speak: [] })
    expect(noticeHeard(storage)).toBe(false)
  })

  it('falls back to the engine’s memory when there is no storage', () => {
    const first = queueWithNotice(false, null, ['One.'])
    const second = queueWithNotice(first.heard, null, ['Two.'])

    expect(first.speak).toEqual([GREETING, 'One.'])
    expect(second.speak).toEqual(['Two.'])
  })

  it('says the greeting when the storage throws, and says why in the console', () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => undefined)
    const broken: NoticeStorage = {
      getItem: () => {
        throw new Error('blocked')
      },
      setItem: () => {
        throw new Error('blocked')
      },
    }

    expect(queueWithNotice(false, broken, ['Hello.']).speak).toEqual([GREETING, 'Hello.'])
    expect(warn).toHaveBeenCalled()
  })
})

describe('noticeText', () => {
  it('is the server’s greeting with Auto, or the answering agent’s name', () => {
    expect(noticeText()).toBe(GREETING)
    expect(noticeText('  Research   Agent ')).toBe('Hi, this is Research Agent, your AI assistant.')
    expect(noticeText('Cash $& Co')).toBe('Hi, this is Cash $& Co, your AI assistant.')
  })
})
