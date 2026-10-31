'use client'

import { useCallback, useEffect, useRef, useState } from 'react'
import { useRouter, useSearchParams } from 'next/navigation'
import { toast } from 'sonner'

import { useChatSessionHydration } from '@/hooks/use-chat-session'
import { getChat, getChatHistory, getChatMessages } from '@/lib/chat/api'
import { isNothingOpen } from '@/lib/chat/chat-session'
import { useChatSessionStore } from '@/stores/chat-session-store'
import type { Chat, ChatMessage } from '@/types'

import {
  CHAT_ID_PARAM,
  type ChatConversation,
  type ChatConversationOptions,
  type ChatConversationView,
  type ChatDraft,
  type ChatDraftLink,
  chatUrlWithout,
  CONVERSATION_GONE,
  draftView,
  storedView,
} from './view-model'

/**
 * PRD-260 D5, PRD-261 A-11a (M28) — the conversation a theme's chat shows,
 * from the core's session store and lib/chat/api.ts, loaded the way
 * app/chat/page.tsx loads it: a `?chatId=` deep link opens that conversation
 * (once per value, then leaves the URL); a cold start on this device lands on
 * the newest conversation, else a new chat; the active pointer's conversation
 * comes from getChat and getChatMessages, and one that can't be loaded is
 * closed with a toast. A theme's draft link (ChatDraftLink, a Deliverable's
 * "Ask Auto about this") opens a new chat with its words waiting in the
 * message box, then leaves the URL.
 *
 * One difference, for Auto: when the backend names a new chat mid-turn, the
 * screen keeps it (`adopt`) instead of reloading it, so the stream and Auto's
 * state carry on.
 */

/**
 * A deep link wins over the stored pointer, once per appearance in the URL (it
 * leaves the URL at once, and the same link clicked again later opens again);
 * pending until that conversation is the active one. The id is normalised once
 * here (the draft link and the cold start read this one): a blank `?chatId=` is
 * no link, so nothing waits on it, and it leaves the URL.
 */
function useDeepLink(hydrated: boolean, activeChatId: string | null, draftParam: string | undefined) {
  const params = useSearchParams()
  const router = useRouter()
  const openChat = useChatSessionStore((s) => s.openChat)
  const raw = params?.get(CHAT_ID_PARAM) ?? null
  const linked = raw?.trim() || null
  const blank = raw !== null && linked === null
  const handled = useRef<string | null>(null)

  useEffect(() => {
    if (blank) router.replace(chatUrlWithout(params, CHAT_ID_PARAM))
  }, [blank, router, params])

  useEffect(() => {
    if (!linked) handled.current = null
    if (!hydrated || !linked || handled.current === linked) return
    handled.current = linked
    openChat(linked)
    // A draft's words riding the same link are dropped with it: the conversation wins.
    const dropped = draftParam === undefined ? [CHAT_ID_PARAM] : [CHAT_ID_PARAM, draftParam]
    router.replace(chatUrlWithout(params, ...dropped))
  }, [hydrated, linked, openChat, router, params, draftParam])

  return { linked, pending: linked !== null && activeChatId !== linked }
}

/**
 * A draft link, once per appearance in the URL: a new chat (the session's
 * draft), its words handed to the chat screen, the parameter dropped. Pending
 * until the draft is the active one. A `?chatId=` link wins over it (and drops
 * it with itself, useDeepLink). No draft link: nothing is read.
 */
function useDraftLink(
  hydrated: boolean,
  linked: string | null,
  activeChatId: string | null,
  link: ChatDraftLink | undefined,
) {
  const params = useSearchParams()
  const router = useRouter()
  const newDraft = useChatSessionStore((s) => s.newDraft)
  const text = linked || !link ? null : link.read(params)
  const handled = useRef<string | null>(null)
  const [draft, setDraft] = useState<ChatDraft | null>(null)

  useEffect(() => {
    if (text === null) handled.current = null
    if (!hydrated || text === null || handled.current === text || !link) return
    handled.current = text
    newDraft()
    setDraft((prev) => ({ text, serial: (prev?.serial ?? 0) + 1 }))
    router.replace(chatUrlWithout(params, link.param))
  }, [hydrated, text, newDraft, router, params, link])

  const takeDraft = useCallback(() => setDraft(null), [])
  return { draft, takeDraft, held: text !== null, pending: text !== null && activeChatId !== null }
}

/**
 * Nothing stored on this device and no link: the newest conversation, else a
 * new chat. True while it settles. Its answer acts only while nothing is open
 * yet (the store as it is then, as app/chat/page.tsx's cancelled guard does): a
 * late one never replaces a new chat or a conversation opened meanwhile.
 */
function useColdStart(hydrated: boolean, linkHeld: boolean): boolean {
  const nothingOpen = useChatSessionStore((s) => isNothingOpen(s.session))
  const openChat = useChatSessionStore((s) => s.openChat)
  const newDraft = useChatSessionStore((s) => s.newDraft)
  const started = useRef(false)

  useEffect(() => {
    if (!hydrated || linkHeld || started.current || !nothingOpen) return
    started.current = true
    const stillNothingOpen = () => isNothingOpen(useChatSessionStore.getState().session)
    getChatHistory(1)
      .then((rows) => {
        if (!stillNothingOpen()) return
        if (rows[0]) openChat(rows[0].id)
        else newDraft()
      })
      .catch((error: unknown) => {
        console.warn('[chat] the newest conversation could not be read, so a new chat opens:', error)
        if (stillNothingOpen()) newDraft()
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
 * caller's count says so, and a fresh new chat is shown. Only presses after
 * this mount count.
 */
function useFreshDraft(freshDrafts: number, enabled: boolean, activeChatId: string | null, showDraft: () => void) {
  const seen = useRef(freshDrafts)
  useEffect(() => {
    if (seen.current === freshDrafts) return
    seen.current = freshDrafts
    if (enabled && activeChatId === null) showDraft()
  }, [freshDrafts, enabled, activeChatId, showDraft])
}

/** The stored conversation's load, on screen when it lands; one that can't be loaded is closed with a toast. */
function useLoadInto(show: (next: ChatConversationView | null) => void) {
  const closeTab = useChatSessionStore((s) => s.closeTab)
  const setTitles = useChatSessionStore((s) => s.setTitles)
  return useCallback(
    (chatId: string) =>
      loadConversation(chatId, {
        loaded: (chat, messages) => {
          show(storedView(chatId, chat, messages))
          setTitles({ [chatId]: chat.title })
        },
        failed: (error) => {
          console.error('[chat] the conversation could not be loaded:', error)
          toast.error(CONVERSATION_GONE)
          closeTab(chatId)
        },
      }),
    [show, setTitles, closeTab],
  )
}

/** The active pointer's conversation on screen: a new chat, an adopted one, or one loaded from the server. */
function useActiveView(enabled: boolean, activeChatId: string | null, freshDrafts: number) {
  const [view, setView] = useState<ChatConversationView | null>(null)
  // The chat id on screen (undefined: nothing yet), the id the new chat on screen was named, the new chats shown so far.
  const shown = useRef<string | null | undefined>(undefined)
  const adopted = useRef<string | null>(null)
  const drafts = useRef(0)

  const show = useCallback((next: ChatConversationView | null) => {
    shown.current = next ? next.chatId : undefined
    adopted.current = null
    setView(next)
  }, [])
  const showDraft = useCallback(() => show(draftView(++drafts.current)), [show])
  useFreshDraft(freshDrafts, enabled, activeChatId, showDraft)
  const loadInto = useLoadInto(show)

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
    return loadInto(chatId)
  }, [enabled, activeChatId, show, showDraft, loadInto])

  // Only the new chat on screen is adopted: a stored conversation's later turns name it again.
  const adopt = useCallback((chatId: string) => {
    if (shown.current === null) adopted.current = chatId
  }, [])

  // The active pointer's conversation only (a Recent link never shows the previous one for a render); the new
  // chat on screen stays while it takes the name the backend gave it, before and after the effect above adopts it.
  const naming = view?.chatId === null && activeChatId !== null && (activeChatId === adopted.current || activeChatId === shown.current)
  return { view: view?.chatId === activeChatId || naming ? view : null, adopt }
}

const NO_FRESH_DRAFTS = 0

export function useChatConversation(options: ChatConversationOptions = {}): ChatConversation {
  const { freshDrafts = NO_FRESH_DRAFTS, draftLink } = options
  const { hydrated } = useChatSessionHydration()
  const activeChatId = useChatSessionStore((s) => s.session.activeChatId)
  const { linked, pending } = useDeepLink(hydrated, activeChatId, draftLink?.param)
  const drafting = useDraftLink(hydrated, linked, activeChatId, draftLink)
  const coldStarting = useColdStart(hydrated, linked !== null || drafting.held)
  const ready = hydrated && !pending && !drafting.pending && !coldStarting
  const { view, adopt } = useActiveView(ready, activeChatId, freshDrafts)
  return { view: ready ? view : null, adopt, draft: drafting.draft, takeDraft: drafting.takeDraft }
}
