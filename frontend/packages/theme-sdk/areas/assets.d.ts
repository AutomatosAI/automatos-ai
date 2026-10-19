/**
 * PRD-261 A-10b — SDK area `assets` (frontend/lib/theme-sdk/assets.ts): planned, not in this SDK build.
 * Every name the area will export is declared, and calling one throws "not in this SDK build".
 * A-13c fills the runtime and replaces each declaration here with its real type; the names never
 * change (A-10c).
 */
import type { NotInThisBuild } from '../internal/stub'

export declare const assetUrl: NotInThisBuild
