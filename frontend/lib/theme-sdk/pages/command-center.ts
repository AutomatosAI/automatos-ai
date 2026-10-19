/**
 * PRD-261 A-10a — SDK area `pages/command-center`: the data the /command-center page reads. Stubbed here; A-12h fills the bodies.
 * Until then every name throws "not in this SDK build" (internal/stub.ts), so a theme built
 * against a later SDK fails loudly. The names are the plan's (surface/pages-command-center.json): the
 * area's lane replaces bodies, never names, unless the A-10c surface map changes first.
 */
import { notInThisBuild } from '../internal/stub'

const AREA = 'pages/command-center'

export const useActivityFeed = notInThisBuild(AREA, 'useActivityFeed')
export const useActivityStats = notInThisBuild(AREA, 'useActivityStats')
export const useActivitySchedule = notInThisBuild(AREA, 'useActivitySchedule')
export const useQuestions = notInThisBuild(AREA, 'useQuestions')
export const useBoardEventStream = notInThisBuild(AREA, 'useBoardEventStream')
export const useNeedsYou = notInThisBuild(AREA, 'useNeedsYou')
export const useWatches = notInThisBuild(AREA, 'useWatches')
