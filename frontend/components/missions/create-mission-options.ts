/**
 * The New mission form's fixed options: power modes, mission templates and the attachment
 * types it accepts. Data only, moved out of create-mission-modal.tsx so the modal stays small.
 */

import {
  Target, Pen, Search, BarChart3, Database, Briefcase, Sparkles,
  Zap, Rocket,
} from 'lucide-react'

export type PowerMode = 'light' | 'standard' | 'max'

export const DEFAULT_POWER_MODE: PowerMode = 'standard'

export const POWER_MODES: { id: PowerMode; name: string; description: string; icon: typeof Target; tooltip: string }[] = [
  {
    id: 'light',
    name: 'Light',
    description: 'Quick & cheap',
    icon: Zap,
    tooltip: 'Uses the background model with 2K token cap and 5 tool iterations. Good for blog posts, quick research.',
  },
  {
    id: 'standard',
    name: 'Standard',
    description: 'Balanced (default)',
    icon: Target,
    tooltip: 'Each agent uses its own model with 4K token cap and 10 tool iterations. Default behavior.',
  },
  {
    id: 'max',
    name: 'Max',
    description: 'Full power',
    icon: Rocket,
    tooltip: 'All agents use your best model with 16K tokens and 50 tool iterations. Decomposition limited to 1-2 focused agents for deep output.',
  },
]

// PRD-127: Ephemeral attachment metadata
export interface MissionAttachment {
  attachment_id: string
  filename: string
  mime: string
  media_type: 'image' | 'document'
}

export interface UploadingFile {
  file: File
  status: 'uploading' | 'done' | 'error'
  attachment?: MissionAttachment
  error?: string
}

// PRD-127: Extended to include images
const ALLOWED_TYPES: Record<string, string[]> = {
  // Images
  'image/jpeg': ['.jpg', '.jpeg'],
  'image/png': ['.png'],
  'image/gif': ['.gif'],
  'image/webp': ['.webp'],
  // Documents
  'application/pdf': ['.pdf'],
  'text/plain': ['.txt'],
  'text/markdown': ['.md'],
  'application/json': ['.json'],
  'text/csv': ['.csv'],
  'application/vnd.openxmlformats-officedocument.wordprocessingml.document': ['.docx'],
  'application/msword': ['.doc'],
  'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': ['.xlsx'],
}

export const ACCEPT_MAP = Object.fromEntries(
  Object.entries(ALLOWED_TYPES).map(([mime, exts]) => [mime, exts]),
)

export const MAX_FILE_SIZE = 20 * 1024 * 1024 // 20MB

export interface MissionTemplateOption {
  id: string | null // null = custom goal (no template)
  name: string
  description: string
  icon: typeof Target
  estimatedCost: string
}

export const BUSINESS_PLAN_TEMPLATE_ID = 'business_plan'

export const MISSION_TEMPLATES: MissionTemplateOption[] = [
  {
    id: null,
    name: 'Custom Goal',
    description: 'Freeform — describe anything',
    icon: Sparkles,
    estimatedCost: 'varies',
  },
  {
    id: BUSINESS_PLAN_TEMPLATE_ID,
    name: 'Business Plan',
    description: 'Research, financials, and full plan',
    icon: Briefcase,
    estimatedCost: '~500K tokens',
  },
  {
    id: 'research_and_report',
    name: 'Research Report',
    description: 'Research a topic and produce a report',
    icon: Search,
    estimatedCost: '~200K tokens',
  },
  {
    id: 'content_pipeline',
    name: 'Content Pipeline',
    description: 'Write, edit, and publish content',
    icon: Pen,
    estimatedCost: '~150K tokens',
  },
  {
    id: 'competitive_analysis',
    name: 'Competitive Analysis',
    description: 'Analyze competitors and market position',
    icon: BarChart3,
    estimatedCost: '~200K tokens',
  },
  {
    id: 'data_investigation',
    name: 'Data Investigation',
    description: 'Investigate, diagnose, and report on data',
    icon: Database,
    estimatedCost: '~150K tokens',
  },
]

/** A file size as the attachment list shows it: B, KB or MB. Pure. */
export function formatSize(bytes: number): string {
  if (bytes < 1024) return `${bytes}B`
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)}KB`
  return `${(bytes / (1024 * 1024)).toFixed(1)}MB`
}
