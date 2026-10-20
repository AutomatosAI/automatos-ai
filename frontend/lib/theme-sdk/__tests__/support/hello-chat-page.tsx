/**
 * PRD-261 A-19e1 — the hello theme's chat page (e2e/browser/fixtures/hello-a11b/entry.js) for the
 * SDK chat's vitest cases (chat-controller-cases.test.tsx, chat-controller-cards-cases.test.tsx),
 * drawn from the SDK alone: useChatController() for the conversation on screen, then
 * useChatController(conversation) in a child keyed by it, the host's own ToolApproval and
 * MissionApproval for each card a reply raised or that outlived its reply, and the host's
 * MissionSuggestionCard for the turn's suggestion. With it, the session pointers and the server's
 * answers the cases set, on the real session store and on the chat API the test file mocks
 * (vi.mock('@/lib/chat/api') in each test file). A helper, never a test file.
 */
import { act, fireEvent, render, screen, waitFor } from '@testing-library/react'
import type { ReactElement } from 'react'
import { expect, vi } from 'vitest'

import { getChat, getChatMessages } from '@/lib/chat/api'
import { chatSessionKey, type ChatSession, EMPTY_CHAT_SESSION } from '@/lib/chat/chat-session'
import { messageText } from '@/lib/chat/message-text'
import { useChatSessionStore } from '@/stores/chat-session-store'
import type { Chat, ChatMessage } from '@/types'

import {
  type HostChatApproval,
  type HostChatController,
  type HostChatConversation,
  type HostChatDraftLink,
  type HostChatOnScreen,
  useChatController,
  useStartChat,
} from '../../chat'
import { MissionApproval, type MissionApprovalData, ToolApproval, type ToolApprovalData } from '../../host-cards/approvals'
import { MissionSuggestionCard } from '../../host-cards/chat-cards'
import { inspectableClient } from './query-cache'

export const OPENING = 'Opening the conversation'
export const NEW_CHAT_LABEL = 'New chat'
const CARD_TEST_ID = 'hello-approval'

/** What the page last drew: the conversation on screen, and that conversation's chat. */
export const seen: { onScreen: HostChatOnScreen | null; chat: HostChatController | null } = { onScreen: null, chat: null }

/** A theme's own draft link (`/chat?draft=<words>`), a module constant as the SDK asks. */
export const DRAFT_LINK: HostChatDraftLink = { param: 'draft', read: (params) => params?.get('draft')?.trim() || null }

/** The host's own card for an approval, told when its decision went through and when it closes. */
function HelloApproval({ approval }: { readonly approval: HostChatApproval }) {
  const origin = { conversationId: approval.chatId || undefined, createdAt: approval.filedAt }
  return (
    <div data-testid={CARD_TEST_ID} data-status={approval.status}>
      {approval.kind === 'mission' ? (
        <MissionApproval data={approval.data as MissionApprovalData} onClose={approval.closed} {...origin} />
      ) : (
        <ToolApproval data={approval.data as ToolApprovalData} onDecided={() => approval.answered()} onClose={approval.closed} {...origin} />
      )}
    </div>
  )
}

/** One conversation's chat: mounted afresh (keyed) for each conversation, as the SDK asks. */
function HelloConversation({ conversation }: { readonly conversation: HostChatConversation }) {
  const hello = useChatController(conversation)
  seen.chat = hello
  const { suggestion } = hello
  return (
    <section aria-label="Hello chat">
      <h1>{hello.title}</h1>
      <ol aria-label="Conversation">
        {hello.messages.map((message) => (
          <li key={message.id}>
            <p>{message.text}</p>
            {message.approvals.map((approval) => <HelloApproval key={approval.key} approval={approval} />)}
          </li>
        ))}
      </ol>
      {hello.looseApprovals.map((approval) => <HelloApproval key={approval.key} approval={approval} />)}
      {suggestion ? (
        <MissionSuggestionCard
          goal={suggestion.goal}
          complexity={suggestion.complexity}
          chatId={hello.conversation.chatId ?? ''}
          recentMessages={hello.messages.map((message) => ({ role: message.role, content: message.text }))}
        />
      ) : null}
    </section>
  )
}

/** The chat page: the conversation on screen, or a line while it opens. */
export function HelloChatPage({ draftLink }: { readonly draftLink?: HostChatDraftLink }) {
  const onScreen = useChatController({ draftLink })
  seen.onScreen = onScreen
  if (!onScreen.conversation) return <p role="status">{OPENING}</p>
  return <HelloConversation key={onScreen.conversation.key} conversation={onScreen.conversation} />
}

/** New chat, as a theme's sidebar draws it. */
export function NewChatButton() {
  const { startChat, ready } = useStartChat()
  return (
    <button type="button" onClick={startChat} disabled={!ready}>
      {NEW_CHAT_LABEL}
    </button>
  )
}

/** The page under a react-query client, as the root providers give it. */
export function renderPage(ui: ReactElement = <HelloChatPage />) {
  return render(ui, { wrapper: inspectableClient().wrapper })
}

// --- core fixtures: types/chat.ts' shapes, as getChat and getChatMessages hand them over -------------------------------

export function message(id: string, role: 'user' | 'assistant', text: string, extra: Record<string, unknown> = {}): ChatMessage {
  return { id, role, content: text, parts: [{ type: 'text', text }], ...extra } as unknown as ChatMessage
}

export function storedChat(id: string, title: string): Chat {
  return { id, title, userId: 'u1', createdAt: '', updatedAt: '', visibility: 'private', turnInFlight: false }
}

/** A reply as getChatMessages hands it over: its parts and its persisted source, no content. */
export function storedReply(id: string, text: string, source: { origin: string; label: string }): ChatMessage {
  return { id, role: 'assistant', parts: [{ type: 'text', text }], source, metadata: { source: source.label } } as unknown as ChatMessage
}

/** A message's words, as the page shows them (the host's own reading, lib/chat/message-text.ts). */
export function firstWords(shown: ChatMessage): string {
  return messageText(shown)
}

/** getChat and getChatMessages answer this conversation (the test file mocks the chat API). */
export function serve(chatId: string, title: string, messages: ChatMessage[]): void {
  vi.mocked(getChat).mockResolvedValue(storedChat(chatId, title))
  vi.mocked(getChatMessages).mockResolvedValue(messages)
}

// --- the session the page reads: the signed-in stand-in's key, already hydrated -----------------------------------------

/** The stand-in identity's session key (the test files' useUser gives `u1`, with no workspace). */
export const SESSION_KEY = chatSessionKey(null, 'u1')

export type Pointer = Pick<ChatSession, 'activeChatId' | 'draftOpen' | 'openChatIds'>
export const NEW_CHAT: Pointer = { activeChatId: null, draftOpen: true, openChatIds: [] }
export const NOTHING_OPEN: Pointer = { activeChatId: null, draftOpen: false, openChatIds: [] }

export function on(chatId: string): Pointer {
  return { activeChatId: chatId, draftOpen: false, openChatIds: [chatId] }
}

export function setSession(pointer: Pointer): void {
  useChatSessionStore.setState({ key: SESSION_KEY, hydrated: true, session: { ...EMPTY_CHAT_SESSION, ...pointer } })
}

export function sessionNow(): ChatSession {
  return useChatSessionStore.getState().session
}

// --- reading the page ----------------------------------------------------------------------------------------------------

/** The page's chat once it shows `chatId` (null: a new chat). */
export async function shownChat(chatId: string | null): Promise<HostChatController> {
  await waitFor(() => expect(seen.chat?.conversation.chatId).toBe(chatId))
  return seen.chat as HostChatController
}

/** The host card whose title is `title`, as the page placed it. */
export function cardOf(title: string): HTMLElement {
  return screen.getByText(title).closest(`[data-testid="${CARD_TEST_ID}"]`) as HTMLElement
}

export function click(button: HTMLElement): Promise<void> {
  return act(async () => {
    fireEvent.click(button)
  })
}

/** `cards` as `key:status`, in order. */
export function statusesOf(cards: readonly HostChatApproval[] | undefined): string[] {
  return (cards ?? []).map((card) => `${card.key}:${card.status}`)
}

export function forgetPage(): void {
  seen.onScreen = null
  seen.chat = null
}
