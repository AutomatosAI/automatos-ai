/**
 * PRD-261 A-10b — SDK area `pages/command-center` (frontend/lib/theme-sdk/pages/command-center.ts): planned, not in this SDK build.
 * Every name the area will export is declared, and calling one throws "not in this SDK build".
 * A-12h fills the runtime and replaces each declaration here with its real type; the names never
 * change (A-10c).
 */
import type { NotInThisBuild } from '../internal/stub'

export declare const useActivityFeed: NotInThisBuild
export declare const useActivitySchedule: NotInThisBuild
export declare const useActivityStats: NotInThisBuild
export declare const useBoardEventStream: NotInThisBuild
export declare const useNeedsYou: NotInThisBuild
export declare const useQuestions: NotInThisBuild
export declare const useWatches: NotInThisBuild
