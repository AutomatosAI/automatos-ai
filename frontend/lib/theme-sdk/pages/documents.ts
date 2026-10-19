/**
 * PRD-261 A-10a — SDK area `pages/documents`: the data the /documents page reads. Stubbed here; A-12f fills the bodies.
 * Until then every name throws "not in this SDK build" (internal/stub.ts), so a theme built
 * against a later SDK fails loudly. The names are the plan's (surface/pages-documents.json): the
 * area's lane replaces bodies, never names, unless the A-10c surface map changes first.
 */
import { notInThisBuild } from '../internal/stub'

const AREA = 'pages/documents'

export const useDocuments = notInThisBuild(AREA, 'useDocuments')
export const useDocumentStats = notInThisBuild(AREA, 'useDocumentStats')
export const useUploadDocument = notInThisBuild(AREA, 'useUploadDocument')
export const useDeleteDocument = notInThisBuild(AREA, 'useDeleteDocument')
export const useTeams = notInThisBuild(AREA, 'useTeams')
export const useDocumentTeamCounts = notInThisBuild(AREA, 'useDocumentTeamCounts')
export const useDatabaseKnowledge = notInThisBuild(AREA, 'useDatabaseKnowledge')
export const useCloudConnections = notInThisBuild(AREA, 'useCloudConnections')
export const useTriggerSync = notInThisBuild(AREA, 'useTriggerSync')
export const useSelectRootFolder = notInThisBuild(AREA, 'useSelectRootFolder')
