'use client'

import { type HostChatDraftLink, type HostChatOnScreen, useChatController } from '@automatos/theme-sdk'

import { DRAFT_PARAM, draftFrom } from '../lib/play/draft'

/**
 * PRD-260 D5, PRD-261 A-11a (M28), A-14a — the conversation Play's chat shows:
 * the SDK's (`useChatController()`, lib/theme-sdk/chat.ts, on the chat
 * controller's lib/chat-controller/conversation.ts: a `?chatId=` deep link, a
 * cold start, the active pointer's conversation loaded, a new chat the backend
 * names kept on screen), with Play's two additions: a `?draft=` link
 * (lib/play/draft.ts, a Deliverable's "Ask Auto about this") opens a new chat
 * with its words waiting in the message box, and New chat pressed over a new
 * chat still unnamed shows a fresh one (play-chat-store's count, which the SDK
 * follows). The conversation it hands out is the one Play's chat screen, keyed
 * by `conversation.key`, opens with usePlayTurn.
 */

const DRAFT_LINK: HostChatDraftLink = { param: DRAFT_PARAM, read: draftFrom }

export function usePlayConversation(): HostChatOnScreen {
  return useChatController({ draftLink: DRAFT_LINK })
}
