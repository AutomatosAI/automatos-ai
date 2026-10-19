/**
 * PRD-261 A-10b — the type of a planned SDK name whose area has not landed in this SDK build
 * (frontend/lib/theme-sdk/internal/stub.ts). Calling it throws "not in this SDK build". When the area's lane
 * fills the runtime it replaces each stub declaration in its areas/<area>.d.ts with the real type.
 *
 * Not re-exported by index.d.ts: a theme never names it.
 */
export type NotInThisBuild = (...args: readonly unknown[]) => never
