const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || ''

export interface VoiceChatResponse {
  conversation_id: string
  message_id: string
  transcript: string
  response_text: string
  audio_url: string | null
  audio_format: string
  stt_latency_ms: number
  tts_latency_ms: number
  voice_metadata: {
    stt_model: string
    tts_model: string
    tts_voice: string
    audio_s3_key: string | null
  }
}

export interface VoiceHealthResponse {
  voice_enabled: boolean
  voice_service_healthy: boolean
  voice_service_url: string
}

function getAuthToken(): string | null {
  if (typeof window === 'undefined') return null
  return localStorage.getItem('auth_token')
}

function getWorkspaceId(): string | null {
  if (typeof window === 'undefined') return null
  return localStorage.getItem('last_active_workspace')
}

function buildHeaders(includeContentType = false): HeadersInit {
  const headers: Record<string, string> = {}
  const token = getAuthToken()
  const workspaceId = getWorkspaceId()

  if (token) {
    headers['Authorization'] = `Bearer ${token}`
  }
  if (workspaceId) {
    headers['X-Workspace-ID'] = workspaceId
  }
  if (includeContentType) {
    headers['Content-Type'] = 'application/json'
  }
  return headers
}

export async function sendVoiceMessage(
  audio: Blob,
  conversationId: string,
  options?: {
    agentId?: number
    responseFormat?: 'audio' | 'text' | 'both'
    language?: string
    voice?: string
    authToken?: string | null
  }
): Promise<VoiceChatResponse> {
  const formData = new FormData()
  formData.append('audio', audio, 'recording.webm')
  formData.append('conversation_id', conversationId)

  if (options?.agentId !== undefined) {
    formData.append('agent_id', String(options.agentId))
  }
  if (options?.responseFormat) {
    formData.append('response_format', options.responseFormat)
  }
  if (options?.language) {
    formData.append('language', options.language)
  }
  if (options?.voice) {
    formData.append('voice', options.voice)
  }

  // Don't set Content-Type for FormData — browser sets it with boundary
  const headers: Record<string, string> = {}
  const token = options?.authToken ?? getAuthToken()
  const workspaceId = getWorkspaceId()
  if (token) headers['Authorization'] = `Bearer ${token}`
  if (workspaceId) headers['X-Workspace-ID'] = workspaceId

  const response = await fetch(`${API_BASE_URL}/api/chat/voice`, {
    method: 'POST',
    headers,
    body: formData,
  })

  if (!response.ok) {
    const errorText = await response.text().catch(() => '')
    throw new Error(
      `Voice request failed (${response.status})${errorText ? `: ${errorText}` : ''}`
    )
  }

  return response.json()
}

export function getVoiceAudioUrl(messageId: string): string {
  return `${API_BASE_URL}/api/chat/voice/audio/${messageId}`
}

export async function checkVoiceHealth(): Promise<VoiceHealthResponse> {
  const response = await fetch(`${API_BASE_URL}/api/voice/health`, {
    headers: buildHeaders(),
  })

  if (!response.ok) {
    throw new Error(`Voice health check failed (${response.status})`)
  }

  return response.json()
}
