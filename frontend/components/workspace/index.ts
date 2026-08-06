/**
 * Workspace Components for PRD-38.1 Widget Architecture
 */

// Import widget components to trigger auto-registration
// These imports have side-effects: each widget registers itself with the registry
import '@/components/widgets/CodeWidget'
import '@/components/widgets/DataWidget'
import '@/components/widgets/DocumentWidget'
import '@/components/widgets/ImageWidget'
// Phase 2 widgets (PRD-38.2)
import '@/components/widgets/EmailWidget'
import '@/components/widgets/TerminalWidget'
import '@/components/widgets/WorkflowWidget'
import '@/components/widgets/MemoryWidget'
import '@/components/widgets/FileWidget'
// PRD-66: Coding Canvas Widget (workspace file browser + Monaco editor)
import '@/components/widgets/CodingCanvasWidget'
// PRD-163 S4: Mission plan approval card
// PRD-193 S3: Tool-call approval card (confirmation-gated actions)
//
// Both cards self-register on import like every widget above — but neither
// was ever added to this manifest, so neither had EVER rendered in
// production: chat.tsx addWidget({type: 'tool_approval'}) hit the registry
// with nothing registered and the user saw "Unknown widget type" instead of
// Approve/Deny. First surfaced 2026-08-06 by the first confirmation-gated
// destructive action a client pushed through chat (delete agent).
// registration-manifest.test.ts now sweeps the widgets directory so a new
// widget cannot be built-but-unregistered again.
import '@/components/widgets/MissionApprovalWidget'
import '@/components/widgets/ToolApprovalWidget'

export { Canvas } from './Canvas'
export { WidgetTray } from './WidgetTray'
export { WorkspaceSaveDialog } from './WorkspaceSaveDialog'
export { LayoutPresets } from './LayoutPresets'
export { WorkspaceManager } from './WorkspaceManager'
export { WorkspaceShareDialog } from './WorkspaceShareDialog'
export { WorkspaceSelector } from './WorkspaceSelector'
