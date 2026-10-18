/**
 * PRD-261 A-10a — SDK area `assets`: the URL of a file inside the theme's package. Stubbed here; A-13c fills the bodies.
 * Until then every name throws "not in this SDK build" (internal/stub.ts), so a theme built
 * against a later SDK fails loudly. The names are the plan's (surface/assets.json): the
 * area's lane replaces bodies, never names, unless the A-10c surface map changes first.
 */
import { notInThisBuild } from './internal/stub'

const AREA = 'assets'

export const assetUrl = notInThisBuild(AREA, 'assetUrl')
