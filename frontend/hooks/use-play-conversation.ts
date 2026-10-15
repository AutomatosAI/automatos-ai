'use client'

import { useCallback, useEffect, useRef, useState } from 'react'
import { useRouter, useSearchParams } from 'next/navigation'
import { toast } from 'sonner'

import { useChatSessionHydration } from '@/hooks/use-chat-session'
import { getChat, getChatHistory, getChatMessages } from '@/lib/chat/api'
import { type ComposerFill, DRAFT_PARAM, draftFrom } from '@/lib/play/draft'
import { useChatSessionStore } from '@/stores/chat-session-store'
import { usePlayChatStore } from '@/stores/play-chat-store'
import type { Chat, ChatMessage } from '@/types'

/**
 * PRD-260 D5 — the conversation Play's chat shows, from the core's session
 * store and lib/chat/api.ts, loaded the way app/chat/page.tsx loads it: a
 * `?chatId=` deep link opens that conversation (once per value, then leaves the
 * URL); a cold start on this device lands on the newest conversation, else a
 * new chat; the active pointer's conversation comes from getChat and
 * getChatMessages, and one that can't be loaded is closed with a toast. A
 * `?draft=` link (a Deliverable's "Ask Auto about this") opens a new chat with
 * its words waiting in the message box, then leaves the URL.
 *
 * One difference, for Auto: when the backend names a new chat mid-turn, the
 * screen keeps it (`adopt`) instead of reloading it, so the stream and Auto's
 * state carry on.
 */

export const CHAT_ID_PARAM = 'chatId'
export const CHAT_ROUTE = '/chat'
export const CONVERSATION_GONE = 'That conversation is no longer available'

export interface PlayConversationView {
  /** The chat screen's key: it changes only when another conversation is shown. */
  readonly key: string
  /** null for a new chat until the backend names it. */
  readonly chatId: string | null
  readonly title: string | null
  readonly messages: ChatMessage[]
  /** The server is still producing a reply for it (the page reloaded mid-turn). */
  readonly awaitingReply: boolean
}

export interface PlayConversation {
  /** null while the session hydrates, a link or a cold start settles, or a conversation loads. */
  readonly view: PlayConversationView | null
  /** The backend named the new chat on screen: keep showing it. Call before the session store hears. */
  readonly adopt: (chatId: string) => void
  /** A link's words for the new chat's message box, until the chat screen takes them. */
  readonly draft: ComposerFill | null
  /** The chat screen put the draft in its box: later new chats start empty. */
  readonly takeDraft: () => void
}

/** `/chat` with the query as it is, less the named parameters. */
export function chatUrlWithout(params: URLSearchParams | null, ...names: readonly string[]): string {
  const next = new URLSearchParams(params?.toString() ?? '')
  names.forEach((name) => next.delete(name))
  const query = next.toString()
  return query ? `${CHAT_ROUTE}?${query}` : CHAT_ROUTE
}

export function draftView(serial: number): PlayConversationView {
  return { key: `draft-${serial}`, chatId: null, title: null, messages: [], awaitingReply: false }
}

export function storedView(chatId: string, chat: Chat, messages: ChatMessage[]): PlayConversationView {
  return { key: chatId, chatId, title: chat.title ?? null, messages, awaitingReply: Boolean(chat.turnInFlight) }
}

/**
 * A deep link wins over the stored pointer, once per appearance in the URL (it
 * leaves the URL at once, and the same link clicked again later opens again);
 * pending until that conversation is the active one.
 */
function useDeepLink(hydrated: boolean, activeChatId: string | null): { linked: string | null; pending: boolean } {
  const params = useSearchParams()
  const router = useRouter()
  const openChat = useChatSessionStore((s) => s.openChat)
  const linked = params?.get(CHAT_ID_PARAM) ?? null
  const handled = useRef<string | null>(null)

  useEffect(() => {
    if (!linked) handled.current = null
    if (!hydrated || !linked || handled.current === linked) return
    handled.current = linked
    openChat(linked)
    // A draft's words riding the same link are dropped with it: the conversation wins.
    router.replace(chatUrlWithout(params, CHAT_ID_PARAM, DRAFT_PARAM))
  }, [hydrated, linked, openChat, router, params])

  return { linked, pending: linked !== null && activeChatId !== linked }
}

/**
 * A `?draft=` link, once per appearance in the URL: a new chat (the session's
 * draft), its words handed to the chat screen, the parameter dropped. Pending
 * until the draft is the active one. A `?chatId=` link wins over it (and drops
 * it with itself, useDeepLink).
 */
function useDraftLink(hydrated: boolean, linked: string | null, activeChatId: string | null) {
  const params = useSearchParams()
  const router = useRouter()
  const newDraft = useChatSessionStore((s) => s.newDraft)
  const text = linked ? null : draftFrom(params)
  const handled = useRef<string | null>(null)
  const [draft, setDraft] = useState<ComposerFill | null>(null)

  useEffect(() => {
    if (text === null) handled.current = null
    if (!hydrated || text === null || handled.current === text) return
    handled.current = text
    newDraft()
    setDraft((prev) => ({ text, serial: (prev?.serial ?? 0) + 1 }))
    router.replace(chatUrlWithout(params, DRAFT_PARAM))
  }, [hydrated, text, newDraft, router, params])

  const takeDraft = useCallback(() => setDraft(null), [])
  return { draft, takeDraft, held: text !== null, pending: text !== null && activeChatId !== null }
}

/** Nothing stored on this device and no link: the newest conversation, else a new chat. True while it settles. */
function useColdStart(hydrated: boolean, linkHeld: boolean): boolean {
  const nothingOpen = useChatSessionStore((s) => s.session.openChatIds.length === 0 && !s.session.draftOpen)
  const openChat = useChatSessionStore((s) => s.openChat)
  const newDraft = useChatSessionStore((s) => s.newDraft)
  const started = useRef(false)

  useEffect(() => {
    if (!hydrated || linkHeld || started.current || !nothingOpen) return
    started.current = true
    getChatHistory(1)
      .then((rows) => (rows[0] ? openChat(rows[0].id) : newDraft()))
      .catch((error: unknown) => {
        console.warn('[play-chat] the newest conversation could not be read, so a new chat opens:', error)
        newDraft()
      })
  }, [hydrated, linkHeld, nothingOpen, openChat, newDraft])

  return hydrated && !linkHeld && nothingOpen
}

interface LoadHandlers {
  readonly loaded: (chat: Chat, messages: ChatMessage[]) => void
  readonly failed: (error: unknown) => void
}

/** getChat and getChatMessages together; the returned cancel drops a late answer. */
function loadConversation(chatId: string, handlers: LoadHandlers): () => void {
  let cancelled = false
  Promise.all([getChat(chatId), getChatMessages(chatId)])
    .then(([chat, messages]) => {
      if (!cancelled) handlers.loaded(chat, messages)
    })
    .catch((error: unknown) => {
      if (!cancelled) handlers.failed(error)
    })
  return () => {
    cancelled = true
  }
}

/**
 * New chat pressed while a new chat is already on screen (its first send
 * failed, so it is still unnamed): the session store doesn't change, so the
 * count in play-chat-store says so, and a fresh new chat is shown. Only
 * presses after this mount count.
 */
function useFreshDraft(enabled: boolean, activeChatId: string | null, showDraft: () => void) {
  const freshDrafts = usePlayChatStore((s) => s.freshDrafts)
  const seen = useRef(freshDrafts)
  useEffect(() => {
    if (seen.current === freshDrafts) return
    seen.current = freshDrafts
    if (enabled && activeChatId === null) showDraft()
  }, [freshDrafts, enabled, activeChatId, showDraft])
}

/** The active pointer's conversation on screen: a new chat, an adopted one, or one loaded from the server. */
function useActiveView(enabled: boolean, activeChatId: string | null) {
  const closeTab = useChatSessionStore((s) => s.closeTab)
  const setTitles = useChatSessionStore((s) => s.setTitles)
  const [view, setView] = useState<PlayConversationView | null>(null)
  // The chat id on screen (undefined: nothing yet), the id the new chat on screen was named, the new chats shown so far.
  const shown = useRef<string | null | undefined>(undefined)
  const adopted = useRef<string | null>(null)
  const drafts = useRef(0)

  const show = useCallback((next: PlayConversationView | null) => {
    shown.current = next ? next.chatId : undefined
    adopted.current = null
    setView(next)
  }, [])
  const showDraft = useCallback(() => show(draftView(++drafts.current)), [show])
  useFreshDraft(enabled, activeChatId, showDraft)

  useEffect(() => {
    if (!enabled || activeChatId === shown.current) return
    if (activeChatId === null) {
      showDraft()
      return
    }
    const chatId = activeChatId
    if (shown.current === null && chatId === adopted.current) {
      // Once: any later move to this id loads it.
      shown.current = chatId
      adopted.current = null
      setView((current) => (current ? { ...current, chatId } : current))
      return
    }
    show(null)
    return loadConversation(chatId, {
      loaded: (chat, messages) => {
        show(storedView(chatId, chat, messages))
        setTitles({ [chatId]: chat.title })
      },
      failed: (error) => {
        console.error('[play-chat] the conversation could not be loaded:', error)
        toast.error(CONVERSATION_GONE)
        closeTab(chatId)
      },
    })
  }, [enabled, activeChatId, show, showDraft, setTitles, closeTab])

  // Only the new chat on screen is adopted: a stored conversation's later turns name it again.
  const adopt = useCallback((chatId: string) => {
    if (shown.current === null) adopted.current = chatId
  }, [])

  return { view, adopt }
}

export function usePlayConversation(): PlayConversation {
  const { hydrated } = useChatSessionHydration()
  const activeChatId = useChatSessionStore((s) => s.session.activeChatId)
  const { linked, pending } = useDeepLink(hydrated, activeChatId)
  const drafting = useDraftLink(hydrated, linked, activeChatId)
  const coldStarting = useColdStart(hydrated, linked !== null || drafting.held)
  const ready = hydrated && !pending && !drafting.pending && !coldStarting
  const { view, adopt } = useActiveView(ready, activeChatId)
  return { view: ready ? view : null, adopt, draft: drafting.draft, takeDraft: drafting.takeDraft }
}
