/**
 * PRD-261 A-10b — SDK area `chat` (frontend/lib/theme-sdk/chat.ts): planned, not in this SDK build.
 * Every name the area will export is declared, and calling one throws "not in this SDK build".
 * A-11b fills the runtime and replaces each declaration here with its real type; the names never
 * change (A-10c).
 */
import type { NotInThisBuild } from '../internal/stub'

export declare const movedSomethingOnTheBoard: NotInThisBuild
export declare const receiptLine: NotInThisBuild
export declare const receiptLink: NotInThisBuild
export declare const useChatController: NotInThisBuild
export declare const useConversationList: NotInThisBuild
export declare const useRecentChats: NotInThisBuild
export declare const useStartChat: NotInThisBuild
