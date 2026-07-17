/**
 * API client for chat operations
 * Uses apiClient for proper Clerk auth + workspace headers.
 */

import type { Chat, ChatMessage, Vote } from '@/types'
import { apiClient } from '@/lib/api-client'
import { attachPersistedSource, rememberViewerUserId } from './live-events'

/**
 * Fetch chat history
 */
export async function getChatHistory(
  limit: number = 20,
  starting_after?: string
): Promise<Chat[]> {
  const params = new URLSearchParams()
  params.set('limit', limit.toString())
  if (starting_after) {
    params.set('starting_after', starting_after)
  }

  const chats = await apiClient.request<Chat[]>(`/api/chat/history?${params.toString()}`)
  // PRD-205 S7: history rows are the viewer's own -- their userId feeds the
  // SSE chat_changed per-user filter.
  if (chats.length > 0) rememberViewerUserId(chats[0].userId)
  return chats
}

/**
 * Get a specific chat by ID
 */
export async function getChat(id: string): Promise<Chat> {
  const chat = await apiClient.request<Chat>(`/api/chat/${id}`)
  // PRD-205 S7: the endpoint 403s on foreign chats, so userId is the viewer's.
  rememberViewerUserId(chat.userId)
  return chat
}

/**
 * Delete a chat
 */
export async function deleteChat(id: string): Promise<void> {
  await apiClient.request(`/api/chat/${id}`, { method: 'DELETE' })
}

/**
 * Vote on a message
 */
export async function voteMessage(
  chatId: string,
  messageId: string,
  isUpvoted: boolean
): Promise<void> {
  await apiClient.request('/api/chat/vote', {
    method: 'PATCH',
    body: { chatId, messageId, isUpvoted } as any,
  })
}

/**
 * Get chat messages
 *
 * PRD-205 S7: rows carry a persisted `source` for background-authored
 * messages -- lift it into `metadata.source` so the existing badge slot
 * renders it (and so it survives reload).
 */
export async function getChatMessages(chatId: string): Promise<ChatMessage[]> {
  const rows = await apiClient.request<ChatMessage[]>(`/api/chat/${chatId}/messages`)
  return rows.map(attachPersistedSource)
}

/**
 * Update chat title
 */
export async function updateChatTitle(id: string, title: string): Promise<void> {
  await apiClient.request(`/api/chat/${id}`, {
    method: 'PATCH',
    body: { title } as any,
  })
}
