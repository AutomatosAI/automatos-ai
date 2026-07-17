/**
 * API client for chat operations
 * Uses apiClient for proper Clerk auth + workspace headers.
 */

import type { Chat, ChatMessage, Vote } from '@/types'
import { apiClient } from '@/lib/api-client'

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

  return apiClient.request<Chat[]>(`/api/chat/history?${params.toString()}`)
}

/**
 * Get a specific chat by ID
 */
export async function getChat(id: string): Promise<Chat> {
  return apiClient.request<Chat>(`/api/chat/${id}`)
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
 */
export async function getChatMessages(chatId: string): Promise<ChatMessage[]> {
  const rows = await apiClient.request<Array<ChatMessage & { source?: { label?: string } | null }>>(
    `/api/chat/${chatId}/messages`,
  )
  // PRD-205 S7: persisted background provenance (messages.source) → the
  // existing metadata badge slot, so "Auto · background" survives reload.
  return rows.map((row) =>
    row.source?.label
      ? { ...row, metadata: { ...(row.metadata ?? {}), source: row.source.label } }
      : row,
  )
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
