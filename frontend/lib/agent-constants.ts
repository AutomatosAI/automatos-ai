import {
  Bot,
  UserCircle,
  Headphones,
  Terminal,
  Share2,
  Calculator,
  ShoppingBag,
  PenTool,
  Users,
  BarChart3,
} from 'lucide-react'

/**
 * Shared agent category definitions.
 * Used by both create-agent-modal and agent-configuration-modal.
 */
export const AGENT_CATEGORIES = [
  { id: 'Personal Assistant', name: 'Personal Assistant', icon: UserCircle, color: 'text-blue-500' },
  { id: 'Customer Support', name: 'Customer Support', icon: Headphones, color: 'text-green-500' },
  { id: 'DevOps', name: 'DevOps', icon: Terminal, color: 'text-purple-500' },
  { id: 'Social Media', name: 'Social Media', icon: Share2, color: 'text-pink-500' },
  { id: 'Accounting', name: 'Accounting', icon: Calculator, color: 'text-amber-500' },
  { id: 'E-commerce', name: 'E-commerce', icon: ShoppingBag, color: 'text-cyan-500' },
  { id: 'Content Creation', name: 'Content Creation', icon: PenTool, color: 'text-indigo-500' },
  { id: 'HR', name: 'HR', icon: Users, color: 'text-teal-500' },
  { id: 'Data Analysis', name: 'Data Analysis', icon: BarChart3, color: 'text-rose-500' },
  { id: 'Custom', name: 'Custom', icon: Bot, color: 'text-orange-500' }
] as const

/**
 * Maps UI category names to database agent_type values.
 * Many categories map to 'custom' — use `marketplace_category` to preserve the original selection.
 */
export const CATEGORY_TO_DB_MAP: Record<string, string> = {
  'Personal Assistant': 'assistant',
  'Customer Support': 'support',
  'DevOps': 'devops',
  'Social Media': 'custom',
  'Accounting': 'custom',
  'E-commerce': 'custom',
  'Content Creation': 'custom',
  'HR': 'custom',
  'Data Analysis': 'data_analyst',
  'Custom': 'custom'
}

/**
 * Maps database agent_type values back to UI category names.
 * Specialized types that don't have their own UI category map to 'Custom'.
 */
export const DB_TO_CATEGORY_MAP: Record<string, string> = {
  'assistant': 'Personal Assistant',
  'support': 'Customer Support',
  'devops': 'DevOps',
  'data_analyst': 'Data Analysis',
  'code_architect': 'DevOps',
  'security_expert': 'Custom',
  'performance_optimizer': 'Custom',
  'infrastructure_manager': 'DevOps',
  'custom': 'Custom'
}
