/**
 * PRD-261 A-10b — SDK area `pages/playbooks` (frontend/lib/theme-sdk/pages/playbooks.ts): planned, not in this SDK build.
 * Every name the area will export is declared, and calling one throws "not in this SDK build".
 * A-12g fills the runtime and replaces each declaration here with its real type; the names never
 * change (A-10c).
 */
import type { NotInThisBuild } from '../internal/stub'

export declare const useCreatePlaybook: NotInThisBuild
export declare const useDeletePlaybook: NotInThisBuild
export declare const useExecutePlaybook: NotInThisBuild
export declare const usePlaybookExecutions: NotInThisBuild
export declare const useUpdatePlaybook: NotInThisBuild
export declare const useWorkflowPlaybook: NotInThisBuild
export declare const useWorkflowPlaybooks: NotInThisBuild
