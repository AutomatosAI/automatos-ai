import fs from 'fs'
import path from 'path'
import { expect, type APIRequestContext, type Locator, type Page } from '@playwright/test'
import { lineOf, noStoredConversations, stubChatStream, type ChatScript, type Frame, type ServedTurn } from './chat-stream'
import { openPlay, THEME_LOAD_TIMEOUT_MS, type AvailablePlay, type PlayVariant } from './play-parity'

// PRD-261 A-17b: Play's chat with recorded turns (fixtures/play-parity/chat-turns.json), the same
// way under builtin Play and the installed package. The page's POST /api/chat is answered in the
// page (support/chat-stream.ts) and the conversations list is empty, so every test starts on a new
// chat; nothing reaches a model.

const TURNS_FILE = path.resolve(__dirname, '..', 'fixtures', 'play-parity', 'chat-turns.json')
const CHAT_PATH = '/chat'
const CHAT_STREAM_PATH = '/api/chat'
const POSTS_KEY = '__playParityChatPosts'
/** Play's message box (components/play/chat/message-box.tsx PLACEHOLDER, its accessible name). */
export const MESSAGE_BOX = 'Message Auto'
/** Each reply of Auto's (components/play/chat/auto-reply.tsx: `${SPEAKER} said`). */
const REPLY_NAME = 'Auto said'
/** Auto's column beside the conversation (components/play/shell/page-columns.tsx MascotColumn). */
const AUTO_COLUMN = 'Auto'

export type PlayTurnName = 'answer' | 'partial' | 'approvals' | 'refused'

interface RecordedTurn {
  readonly frames: readonly Frame[]
  readonly hold?: boolean
  readonly status?: number
}

export interface PlayTurns {
  readonly chatId: string
  readonly turns: Readonly<Record<PlayTurnName, RecordedTurn>>
}

export function readPlayTurns(): PlayTurns {
  const turns = JSON.parse(fs.readFileSync(TURNS_FILE, 'utf8')) as PlayTurns
  if (typeof turns.chatId !== 'string' || typeof turns.turns !== 'object') throw new Error(`${TURNS_FILE} is not a set of recorded turns`)
  return turns
}

function served(recorded: PlayTurns, name: PlayTurnName): ServedTurn {
  const turn = recorded.turns[name]
  return { lines: turn.frames.map(lineOf), hold: turn.hold === true, status: turn.status }
}

/** Opens Play's chat on a new conversation, every POST /api/chat answered with `turns` in order. */
export async function openPlayChat(
  page: Page,
  request: APIRequestContext,
  variant: PlayVariant,
  turns: readonly PlayTurnName[],
): Promise<AvailablePlay> {
  const recorded = readPlayTurns()
  const script: ChatScript = { path: CHAT_STREAM_PATH, postsKey: POSTS_KEY, turns: turns.map((name) => served(recorded, name)) }
  await page.addInitScript(stubChatStream, script)
  await noStoredConversations(page)
  const play = await openPlay(page, request, variant, CHAT_PATH)
  await expect(messageBox(page)).toBeVisible({ timeout: THEME_LOAD_TIMEOUT_MS })
  return play
}

export function messageBox(page: Page): Locator {
  return page.getByRole('textbox', { name: MESSAGE_BOX, exact: true })
}

/** Types `words` into the message box and sends them. */
export async function ask(page: Page, words: string): Promise<void> {
  await messageBox(page).fill(words)
  await page.getByRole('button', { name: 'Send', exact: true }).click()
}

/** Auto's replies, oldest first. */
export function autoReplies(page: Page): Locator {
  return page.getByRole('article', { name: REPLY_NAME, exact: true })
}

/** Auto beside the conversation: his stage (an image named by his caption) and the caption. */
export function autoColumn(page: Page): Locator {
  return page.getByRole('complementary', { name: AUTO_COLUMN, exact: true })
}
