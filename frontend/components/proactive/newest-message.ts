'use client'

/**
 * PRD-268 §6.6, §6.12.4, K4(7) (US-W03a) — which message is a chat's newest, read through useSyncExternalStore.
 *
 * Reply chips show only under the chat's newest message, so they are gone once a newer one renders. chat.tsx, which
 * holds the list, is not touched (K4): message-with-receipts.tsx registers each message it renders, by chat and id,
 * from mount until it unmounts or its chat or id changes, in a presence store (presence.ts), whose marks keep the
 * order they were made in. The list only grows at its end (a send, the reply being streamed, a background post merged
 * in by id), and React registers a list's messages in the list's order, on a first render and whenever the whole list
 * remounts or moves to its new chat id, so the chat's newest message is the one registered last.
 *
 * A sent reply chip is the person's message, so a newer message follows it at once: `newerMessageWithin` says whether
 * one did, so a send the chat passed over (it was still finishing a turn) gives the chips back.
 */
import { useEffect } from 'react'
import { createPresence, usePresence, type Marks } from './presence'

export interface RenderedMessage {
  readonly chatId: string
  readonly messageId: string
}

const rendered = createPresence<RenderedMessage>('rendered-message')

/** The id of chat `chatId`'s newest rendered message among `marks`, or null when none is rendered. */
export function newestMessageId(marks: Marks<RenderedMessage>, chatId: string): string | null {
  const ids = Array.from(marks.values())
    .filter((message) => message.chatId === chatId)
    .map((message) => message.messageId)
  return ids.length > 0 ? ids[ids.length - 1] : null
}

/** Registers a rendered message of chat `chatId` while it is on screen. */
export function useRenderedMessage(chatId: string, messageId: string): void {
  useEffect(() => rendered.mark({ chatId, messageId }), [chatId, messageId])
}

/** Whether `messageId` is chat `chatId`'s newest rendered message, re-read on every change; false on the server. */
export function useIsNewestMessage(chatId: string, messageId: string): boolean {
  return newestMessageId(usePresence(rendered), chatId) === messageId
}

/**
 * Resolves once `messageId` is no longer chat `chatId`'s newest rendered message (a newer one rendered, or it left the
 * screen); rejects when it still is after `ms`.
 */
export function newerMessageWithin(chatId: string, messageId: string, ms: number): Promise<void> {
  const followed = () => newestMessageId(rendered.snapshot(), chatId) !== messageId
  return new Promise<void>((resolve, reject) => {
    let unsubscribe: () => void = () => undefined
    const timer = setTimeout(() => {
      unsubscribe()
      reject(new Error(`no newer message followed within ${ms} ms`))
    }, ms)
    const settleIfFollowed = () => {
      if (!followed()) return
      clearTimeout(timer)
      unsubscribe()
      resolve()
    }
    unsubscribe = rendered.subscribe(settleIfFollowed)
    settleIfFollowed()
  })
}
