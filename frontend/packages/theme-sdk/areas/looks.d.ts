/**
 * PRD-261 A-10b — SDK area `looks` (frontend/lib/theme-sdk/looks.ts): planned, not in this SDK build.
 * Every name the area will export is declared, and calling one throws "not in this SDK build".
 * A-13c fills the runtime and replaces each declaration here with its real type; the names never
 * change (A-10c).
 */
import type { NotInThisBuild } from '../internal/stub'

export declare const exportLook: NotInThisBuild
export declare const importLook: NotInThisBuild
export declare const listLooks: NotInThisBuild
export declare const saveLook: NotInThisBuild
export declare const useLook: NotInThisBuild
