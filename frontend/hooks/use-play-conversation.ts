'use client'

import { useChatConversation } from '@/lib/chat-controller/conversation'
import type { ChatConversation, ChatDraftLink } from '@/lib/chat-controller/view-model'
import { DRAFT_PARAM, draftFrom } from '@/lib/play/draft'
import { usePlayChatStore } from '@/stores/play-chat-store'

/**
 * PRD-260 D5, PRD-261 A-11a (M28) — the conversation Play's chat shows: the
 * chat controller's (lib/chat-controller/conversation.ts: a `?chatId=` deep
 * link, a cold start, the active pointer's conversation loaded, a new chat
 * the backend names kept on screen), with Play's two additions: a `?draft=`
 * link (lib/play/draft.ts, a Deliverable's "Ask Auto about this") opens a new
 * chat with its words waiting in the message box, and New chat pressed over a
 * new chat still unnamed shows a fresh one (play-chat-store's count).
 */

const DRAFT_LINK: ChatDraftLink = { param: DRAFT_PARAM, read: draftFrom }

export function usePlayConversation(): ChatConversation {
  const freshDrafts = usePlayChatStore((s) => s.freshDrafts)
  return useChatConversation({ freshDrafts, draftLink: DRAFT_LINK })
}
