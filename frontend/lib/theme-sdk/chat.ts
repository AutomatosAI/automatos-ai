/**
 * PRD-261 A-10a — SDK area `chat`: the chat controller's view model, the conversation lists, New chat and
 * the receipt lines (all but the first planned by A-10c, docs/theme-host/sdk-surface-map.md). Stubbed here; A-11b fills the bodies.
 * Until then every name throws "not in this SDK build" (internal/stub.ts), so a theme built
 * against a later SDK fails loudly. The names are the plan's (surface/chat.json): the
 * area's lane replaces bodies, never names, unless the A-10c surface map changes first.
 */
import { notInThisBuild } from './internal/stub'

const AREA = 'chat'

export const useChatController = notInThisBuild(AREA, 'useChatController')
export const useConversationList = notInThisBuild(AREA, 'useConversationList')
export const useRecentChats = notInThisBuild(AREA, 'useRecentChats')
export const useStartChat = notInThisBuild(AREA, 'useStartChat')
export const receiptLine = notInThisBuild(AREA, 'receiptLine')
export const receiptLink = notInThisBuild(AREA, 'receiptLink')
export const movedSomethingOnTheBoard = notInThisBuild(AREA, 'movedSomethingOnTheBoard')
