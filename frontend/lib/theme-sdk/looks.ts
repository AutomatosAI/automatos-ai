/**
 * PRD-261 A-10a — SDK area `looks`: the active look and the saved looks. Stubbed here; A-13c fills the bodies.
 * Until then every name throws "not in this SDK build" (internal/stub.ts), so a theme built
 * against a later SDK fails loudly. The names are the plan's (surface/looks.json): the
 * area's lane replaces bodies, never names, unless the A-10c surface map changes first.
 */
import { notInThisBuild } from './internal/stub'

const AREA = 'looks'

export const useLook = notInThisBuild(AREA, 'useLook')
export const saveLook = notInThisBuild(AREA, 'saveLook')
export const listLooks = notInThisBuild(AREA, 'listLooks')
export const exportLook = notInThisBuild(AREA, 'exportLook')
export const importLook = notInThisBuild(AREA, 'importLook')
