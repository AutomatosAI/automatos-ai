/**
 * PRD-261 A-10a — SDK area `pages/playbooks`: the data the /playbooks page reads. Stubbed here; A-12g fills the bodies.
 * Until then every name throws "not in this SDK build" (internal/stub.ts), so a theme built
 * against a later SDK fails loudly. The names are the plan's (surface/pages-playbooks.json): the
 * area's lane replaces bodies, never names, unless the A-10c surface map changes first.
 */
import { notInThisBuild } from '../internal/stub'

const AREA = 'pages/playbooks'

export const useWorkflowPlaybooks = notInThisBuild(AREA, 'useWorkflowPlaybooks')
export const useWorkflowPlaybook = notInThisBuild(AREA, 'useWorkflowPlaybook')
export const useCreatePlaybook = notInThisBuild(AREA, 'useCreatePlaybook')
export const useUpdatePlaybook = notInThisBuild(AREA, 'useUpdatePlaybook')
export const useDeletePlaybook = notInThisBuild(AREA, 'useDeletePlaybook')
export const useExecutePlaybook = notInThisBuild(AREA, 'useExecutePlaybook')
export const usePlaybookExecutions = notInThisBuild(AREA, 'usePlaybookExecutions')
