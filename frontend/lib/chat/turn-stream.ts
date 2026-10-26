/**
 * Theme voice — one chat turn sent and read outside React: the web quick-ask voice's send. The body useChat
 * (hooks.ts) sends, by the same way: the app's own POST /api/chat (app/api/chat/route.ts), which forwards it to the
 * backend's with the server's API key where the member has no sign-in token, and streams the AI SDK data stream back.
 * So the turn is a normal chat turn in the conversation (saved, titled, answered by the agent the backend routes to),
 * and a voice turn reaches Auto wherever a typed one does. `spokenMode` (PRD-266 §4.7, D32, chat_models.py
 * ChatRequest.spokenMode) tells the backend the reply is heard, so it answers in short spoken sentences and
 * speechifies the text. The auth and workspace headers are apiClient's (apiClient.request parses JSON and cannot
 * hand over a stream).
 *
 * useChat keeps its own inline copy of this request: a changed line inside its body fails the code-shape gate until it
 * is split under the 50-line limit (the PRD-265 note in hooks.ts), so it adopts this module then.
 *
 * Never throws: a refused request is a failed turn, a request that never reached the server a network failure, and a
 * request the caller aborted (Stop) says so.
 */
import { apiClient } from '@/lib/api-client'
import { withoutNarration } from '@/lib/chat/narration'
import { parseChatLine, type ChatFrame } from '@/lib/chat/turn-frames'

export interface ChatTurnRequest {
  /** The conversation; null starts a new one, whose id arrives through onChatId. */
  readonly chatId: string | null
  readonly text: string
  /** The reply is heard, not read. */
  readonly spokenMode: boolean
}

export interface ChatTurnHandlers {
  readonly onChatId: (chatId: string) => void
  /** Auto's answer so far, the whole text each time, narration already taken out. */
  readonly onText: (textSoFar: string) => void
  /** The answering agent's name, when the turn names it. */
  readonly onAgent: (name: string) => void
}

export type ChatTurnOutcome =
  | { readonly kind: 'finished' }
  /** message: the turn's own sentence (an `e:` frame), or null when only the reason is known. */
  | { readonly kind: 'failed'; readonly reason: 'turn' | 'network'; readonly message: string | null }
  | { readonly kind: 'aborted' }

export type ChatTurnSender = (
  request: ChatTurnRequest,
  handlers: ChatTurnHandlers,
  signal: AbortSignal,
) => Promise<ChatTurnOutcome>

interface ReadState {
  readonly answer: string
  readonly errorMessage: string | null
}

/** The app's own chat route (app/api/chat/route.ts), not the backend's: useChat posts to it as well. */
const CHAT_PROXY_PATH = '/api/chat'
const LOG_PREFIX = '[chat-turn]'
const FINISHED: ChatTurnOutcome = Object.freeze({ kind: 'finished' })
const ABORTED: ChatTurnOutcome = Object.freeze({ kind: 'aborted' })

/** The JSON body of POST /api/chat for one text turn, as useChat sends it, with spokenMode when asked. */
export function chatTurnBody(request: ChatTurnRequest): Readonly<Record<string, unknown>> {
  return Object.freeze({
    id: request.chatId ?? '',
    message: { role: 'user', parts: [{ type: 'text', text: request.text }] },
    selectedVisibilityType: 'private',
    ...(request.spokenMode ? { spokenMode: true } : {}),
  })
}

function foldFrame(state: ReadState, frame: ChatFrame, handlers: ChatTurnHandlers): ReadState {
  switch (frame.kind) {
    case 'text':
    case 'narration': {
      const answer = frame.kind === 'text' ? state.answer + frame.text : withoutNarration(state.answer, frame.text)
      handlers.onText(answer)
      return { ...state, answer }
    }
    case 'chat-id':
      handlers.onChatId(frame.chatId)
      return state
    case 'agent':
      handlers.onAgent(frame.name)
      return state
    default:
      return { ...state, errorMessage: frame.message }
  }
}

function foldLines(state: ReadState, lines: readonly string[], handlers: ChatTurnHandlers): ReadState {
  return lines.reduce((current, line) => {
    const frame = parseChatLine(line)
    return frame === null ? current : foldFrame(current, frame, handlers)
  }, state)
}

async function readTurn(body: ReadableStream<Uint8Array>, handlers: ChatTurnHandlers): Promise<ChatTurnOutcome> {
  const reader = body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''
  let state: ReadState = { answer: '', errorMessage: null }
  for (;;) {
    const { done, value } = await reader.read()
    if (done) break
    const lines = (buffer + decoder.decode(value, { stream: true })).split('\n')
    buffer = lines.pop() ?? ''
    state = foldLines(state, lines, handlers)
  }
  state = foldLines(state, [buffer + decoder.decode()], handlers)
  if (state.errorMessage === null) return FINISHED
  return { kind: 'failed', reason: 'turn', message: state.errorMessage }
}

/** Sends one turn and reads its reply to the end, telling the handlers as it streams. */
export async function streamChatTurn(
  request: ChatTurnRequest,
  handlers: ChatTurnHandlers,
  signal: AbortSignal,
): Promise<ChatTurnOutcome> {
  try {
    const headers = await apiClient.getAuthHeaders()
    const response = await fetch(CHAT_PROXY_PATH, {
      method: 'POST',
      headers: { ...headers, 'Content-Type': 'application/json' },
      body: JSON.stringify(chatTurnBody(request)),
      signal,
    })
    if (!response.ok || !response.body) {
      console.warn(LOG_PREFIX, 'the chat turn was refused', response.status)
      return { kind: 'failed', reason: 'turn', message: null }
    }
    return await readTurn(response.body, handlers)
  } catch (error) {
    if (signal.aborted) return ABORTED
    console.warn(LOG_PREFIX, 'the chat turn did not reach Auto', error)
    return { kind: 'failed', reason: 'network', message: null }
  }
}
