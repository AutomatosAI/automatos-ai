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

export { Canvas } from './Canvas'
export { WidgetTray } from './WidgetTray'
