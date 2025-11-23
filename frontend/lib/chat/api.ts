/**
 * API client for chat operations
 */

import type { Chat, ChatMessage, Vote } from '@/types'

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || ''

/**
 * Get auth headers
 */
function getAuthHeaders(): HeadersInit {
  const token = typeof window !== 'undefined' ? localStorage.getItem('auth_token') : null
  const headers: HeadersInit = {
    'Content-Type': 'application/json',
  }
  if (token) {
    headers['Authorization'] = `Bearer ${token}`
  }
  return headers
}

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
  
  const url = `${API_BASE_URL}/api/chat/history?${params.toString()}`

  const response = await fetch(url, {
    headers: getAuthHeaders()
  })
  
  if (!response.ok) {
    throw new Error('Failed to fetch chat history')
  }

  return response.json()
}

/**
 * Get a specific chat by ID
 */
export async function getChat(id: string): Promise<Chat> {
  const response = await fetch(`${API_BASE_URL}/api/chat/${id}`, {
    headers: getAuthHeaders()
  })
  
  if (!response.ok) {
    throw new Error('Failed to fetch chat')
  }

  return response.json()
}

/**
 * Delete a chat
 */
export async function deleteChat(id: string): Promise<void> {
  const response = await fetch(`${API_BASE_URL}/api/chat/${id}`, {
    method: 'DELETE',
    headers: getAuthHeaders()
  })

  if (!response.ok) {
    throw new Error('Failed to delete chat')
  }
}

/**
 * Vote on a message
 */
export async function voteMessage(
  chatId: string,
  messageId: string,
  isUpvoted: boolean
): Promise<void> {
  const response = await fetch(`${API_BASE_URL}/api/chat/vote`, {
    method: 'PATCH',
    headers: getAuthHeaders(),
    body: JSON.stringify({
      chatId,
      messageId,
      isUpvoted,
    }),
  })

  if (!response.ok) {
    throw new Error('Failed to vote on message')
  }
}

/**
 * Get chat messages
 */
export async function getChatMessages(chatId: string): Promise<ChatMessage[]> {
  const response = await fetch(`${API_BASE_URL}/api/chat/${chatId}/messages`, {
    headers: getAuthHeaders()
  })
  
  if (!response.ok) {
    throw new Error('Failed to fetch messages')
  }

  return response.json()
}

/**
 * Update chat title
 */
export async function updateChatTitle(id: string, title: string): Promise<void> {
  const response = await fetch(`${API_BASE_URL}/api/chat/${id}`, {
    method: 'PATCH',
    headers: getAuthHeaders(),
    body: JSON.stringify({ title }),
  })

  if (!response.ok) {
    throw new Error('Failed to update chat title')
  }
}

