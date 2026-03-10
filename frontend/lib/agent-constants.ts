import {
  Bot,
  BarChart3,
  Briefcase,
  MessageCircle,
  Palette,
  Code,
  GraduationCap,
  Globe,
  Users,
  Scale,
  Share2,
  Zap,
  Search,
  TrendingUp,
  Headphones,
  PenTool,
} from 'lucide-react'

/**
 * Shared agent category definitions — sourced from distinct persona categories in the DB.
 * IDs are lowercase to match database values exactly.
 * Used by both create-agent-modal and agent-configuration-modal.
 */
export const AGENT_CATEGORIES = [
  { id: 'analytics', name: 'Analytics', icon: BarChart3, color: 'text-rose-500' },
  { id: 'business', name: 'Business', icon: Briefcase, color: 'text-amber-500' },
  { id: 'communication', name: 'Communication', icon: MessageCircle, color: 'text-sky-500' },
  { id: 'design', name: 'Design', icon: Palette, color: 'text-fuchsia-500' },
  { id: 'development', name: 'Development', icon: Code, color: 'text-purple-500' },
  { id: 'education', name: 'Education', icon: GraduationCap, color: 'text-emerald-500' },
  { id: 'general', name: 'General', icon: Globe, color: 'text-slate-400' },
  { id: 'hr', name: 'HR', icon: Users, color: 'text-teal-500' },
  { id: 'legal', name: 'Legal', icon: Scale, color: 'text-yellow-500' },
  { id: 'marketing', name: 'Marketing', icon: Share2, color: 'text-pink-500' },
  { id: 'productivity', name: 'Productivity', icon: Zap, color: 'text-orange-500' },
  { id: 'research', name: 'Research', icon: Search, color: 'text-indigo-500' },
  { id: 'sales', name: 'Sales', icon: TrendingUp, color: 'text-green-500' },
  { id: 'support', name: 'Support', icon: Headphones, color: 'text-blue-500' },
  { id: 'writing', name: 'Writing', icon: PenTool, color: 'text-cyan-500' },
  { id: 'custom', name: 'Custom', icon: Bot, color: 'text-orange-400' },
] as const

/**
 * Maps UI category to database agent_type.
 * Most categories map to 'custom' — the original is preserved in marketplace_category.
 */
export const CATEGORY_TO_DB_MAP: Record<string, string> = {
  analytics: 'data_analyst',
  business: 'custom',
  communication: 'custom',
  design: 'custom',
  development: 'custom',
  education: 'custom',
  general: 'assistant',
  hr: 'custom',
  legal: 'custom',
  marketing: 'custom',
  productivity: 'custom',
  research: 'custom',
  sales: 'custom',
  support: 'support',
  writing: 'custom',
  custom: 'custom',
}

/**
 * Maps database agent_type values back to UI category names.
 */
export const DB_TO_CATEGORY_MAP: Record<string, string> = {
  assistant: 'general',
  support: 'support',
  devops: 'development',
  data_analyst: 'analytics',
  code_architect: 'development',
  security_expert: 'development',
  performance_optimizer: 'development',
  infrastructure_manager: 'development',
  researcher: 'research',
  worker: 'productivity',
  specialized: 'custom',
  'document generator': 'writing',
  custom: 'custom',
}
