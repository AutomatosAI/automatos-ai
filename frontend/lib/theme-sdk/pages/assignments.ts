/**
 * PRD-261 A-10a — SDK area `pages/assignments`: the data the /assignments missions tab and /missions/[id] read. Stubbed here; A-12e fills the bodies.
 * Until then every name throws "not in this SDK build" (internal/stub.ts), so a theme built
 * against a later SDK fails loudly. The names are the plan's (surface/pages-assignments.json): the
 * area's lane replaces bodies, never names, unless the A-10c surface map changes first.
 */
import { notInThisBuild } from '../internal/stub'

const AREA = 'pages/assignments'

export const useMissions = notInThisBuild(AREA, 'useMissions')
export const useMission = notInThisBuild(AREA, 'useMission')
export const useApproveMission = notInThisBuild(AREA, 'useApproveMission')
export const useRejectMission = notInThisBuild(AREA, 'useRejectMission')
export const usePauseMission = notInThisBuild(AREA, 'usePauseMission')
export const useResumeMission = notInThisBuild(AREA, 'useResumeMission')
export const useCancelMission = notInThisBuild(AREA, 'useCancelMission')
export const useReplanMission = notInThisBuild(AREA, 'useReplanMission')
export const useRerunMission = notInThisBuild(AREA, 'useRerunMission')
export const useSaveAsRoutine = notInThisBuild(AREA, 'useSaveAsRoutine')
export const useRecommendedAssignments = notInThisBuild(AREA, 'useRecommendedAssignments')
