/**
 * PRD-261 A-10b — SDK area `pages/documents` (frontend/lib/theme-sdk/pages/documents.ts): planned, not in this SDK build.
 * Every name the area will export is declared, and calling one throws "not in this SDK build".
 * A-12f fills the runtime and replaces each declaration here with its real type; the names never
 * change (A-10c).
 */
import type { NotInThisBuild } from '../internal/stub'

export declare const useCloudConnections: NotInThisBuild
export declare const useDatabaseKnowledge: NotInThisBuild
export declare const useDeleteDocument: NotInThisBuild
export declare const useDocumentStats: NotInThisBuild
export declare const useDocumentTeamCounts: NotInThisBuild
export declare const useDocuments: NotInThisBuild
export declare const useSelectRootFolder: NotInThisBuild
export declare const useTeams: NotInThisBuild
export declare const useTriggerSync: NotInThisBuild
export declare const useUploadDocument: NotInThisBuild
