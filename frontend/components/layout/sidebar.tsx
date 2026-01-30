
'use client'

import Link from 'next/link'
import { usePathname } from 'next/navigation'
import { motion } from 'framer-motion'
import {
  LayoutDashboard,
  Users,
  FileText,
  GitBranch,
  Brain,
  BarChart3,
  Settings,
  ChevronLeft,
  Bot,
  MessageCircle,
  PanelLeft,
  Wrench,
  Database,
  Lightbulb,
  Store
} from 'lucide-react'
import { cn } from '@/lib/utils'
import { useSystemRole } from '@/contexts/role-context'

interface SidebarProps {
  collapsed: boolean
  onToggle: (collapsed: boolean) => void
}

const navigationItems = [
  {
    name: 'Chat',
    href: '/chat',
    icon: MessageCircle,
    iconColor: 'text-orange-400',
    description: 'Your AI workspace'
  },
  {
    name: 'Workflow Management',
    href: '/workflows',
    icon: GitBranch,
    iconColor: 'text-purple-400',
    description: 'Create and monitor workflows'
  },
  {
    name: 'Agent Management',
    href: '/agents',
    icon: Bot,
    iconColor: 'text-orange-400',
    description: 'Manage AI agents and skills'
  },
  {
    name: 'My Tools',
    href: '/tools',
    icon: Wrench,
    iconColor: 'text-yellow-400',
    description: 'Manage connected integrations'
  },
  {
    name: 'Community Marketplace',
    href: '/marketplace',
    icon: Store,
    iconColor: 'text-orange-400',
    description: 'Discover agents, recipes & tools'
  },
  {
    name: 'Knowledge Bases',
    href: '/documents',
    icon: Database,
    iconColor: 'text-green-400',
    description: 'Documents, databases & code-graph'
  },
  {
    name: 'Team Management',
    href: '/team',
    icon: Users,
    iconColor: 'text-blue-400',
    description: 'Manage workspace members',
    requiredRole: 'admin' as const, // Admin only
  },
  {
    name: 'Context Engineering',
    href: '/context',
    icon: Brain,
    iconColor: 'text-pink-400',
    description: 'RAG system and field theory',
    requiredRole: 'admin' as const,  // Admin only
  },
  {
    name: 'Intelligence & Learning',
    href: '/analytics',
    icon: Lightbulb,
    iconColor: 'text-cyan-400',
    description: 'AI insights and system learning',
    requiredRole: 'admin' as const,  // Admin only
  },
  // Move dashboard to bottom (near Settings)
  {
    name: 'System Dashboard',
    href: '/dashboard',
    icon: LayoutDashboard,
    iconColor: 'text-blue-400',
    description: 'System overview and metrics',
    requiredRole: 'admin' as const,  // Admin only
  },
]

export function Sidebar({ collapsed, onToggle }: SidebarProps) {
  const pathname = usePathname()
  const isChatPage = pathname?.startsWith('/chat') ?? false
  const { systemRole, isAdmin } = useSystemRole()

  // Filter navigation items based on user's system role
  const filteredNavItems = navigationItems.filter(item => {
    if (!item.requiredRole) return true  // No role required, show to everyone
    return item.requiredRole === 'admin' && isAdmin
  })

  return (
    <motion.div
      className={cn(
        'fixed left-0 top-0 z-40 h-screen glass-card border-r border-orange-500/15 bg-background/25 backdrop-blur-xl shadow-[0_0_80px_rgba(249,115,22,0.06)] transition-all duration-300',
        collapsed ? 'w-16' : 'w-64'
      )}
      initial={false}
      animate={{ width: collapsed ? 64 : 256 }}
    >
      {/* Header */}
      <div className="flex items-center justify-between p-4 border-b border-border/50">
        {/* Keep header minimal (branding is in top banner/header) */}
        {!collapsed && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="flex items-center"
          >
            <div className="w-9 h-9 rounded-xl bg-secondary/30 border border-orange-500/15 flex items-center justify-center shadow-[0_0_18px_rgba(249,115,22,0.10)]">
              <MessageCircle className="w-5 h-5 text-orange-400" />
            </div>
          </motion.div>
        )}

        <button
          onClick={() => onToggle(!collapsed)}
          className="p-1 rounded-md hover:bg-secondary/50 transition-colors"
        >
          <ChevronLeft
            className={cn(
              'w-5 h-5 transition-transform duration-300',
              collapsed && 'rotate-180'
            )}
          />
        </button>
      </div>

      {/* Navigation */}
      <nav className="p-4 space-y-2">
        {/* Chat-only: toggle chat history panel */}
        {isChatPage && (
          <motion.div
            key="chat-history-toggle"
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.05 }}
          >
            <button
              type="button"
              onClick={() => {
                if (typeof window !== 'undefined') {
                  window.dispatchEvent(new CustomEvent('automatos:chat-history-toggle'))
                }
              }}
              className={cn('sidebar-item group relative')}
              aria-label="Toggle chat history"
            >
              <div
                className={cn(
                  'w-10 h-10 rounded-lg flex items-center justify-center transition-all duration-200',
                  'bg-secondary/30 group-hover:bg-secondary/50'
                )}
              >
                <PanelLeft className={cn('w-5 h-5 transition-colors', collapsed ? 'text-orange-300' : 'text-orange-400')} />
              </div>

              {!collapsed && (
                <div className="ml-3 flex-1">
                  <p className="text-sm font-medium">Chat history</p>
                  <p className="text-xs text-muted-foreground">Toggle previous chats</p>
                </div>
              )}

              {collapsed && (
                <div className="absolute left-full ml-2 px-2 py-1 bg-popover border border-border rounded-md text-sm whitespace-nowrap opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none z-50">
                  Chat history
                </div>
              )}
            </button>
          </motion.div>
        )}

        {filteredNavItems.map((item, index) => {
          if (!item || !(item as any).href) return null
          const isActive = pathname === item.href
          const Icon = item.icon

          return (
            <motion.div
              key={item.name}
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: index * 0.1 }}
            >
              <Link
                href={item.href}
                onClick={() => onToggle(true)}
                className={cn(
                  'sidebar-item group relative',
                  isActive && 'active'
                )}
              >
                <div
                  className={cn(
                    'w-10 h-10 rounded-lg flex items-center justify-center transition-all duration-200',
                    isActive ? 'icon-gradient' : 'bg-secondary/30 group-hover:bg-secondary/50'
                  )}
                >
                  <Icon
                    className={cn(
                      'w-5 h-5 transition-colors',
                      isActive ? 'text-foreground' : item.iconColor || 'text-muted-foreground group-hover:text-foreground'
                    )}
                  />
                </div>

                {!collapsed && (
                  <div className="ml-3 flex-1">
                    {/* Icon inline with title (same row) */}
                    <div className="flex items-center gap-3">
                      <p className="text-sm font-medium">{item.name}</p>
                    </div>
                    {/* Description indented under the title */}
                    <p className="text-xs text-muted-foreground">{item.description}</p>
                  </div>
                )}

                {/* Tooltip for collapsed state */}
                {collapsed && (
                  <div className="absolute left-full ml-2 px-2 py-1 bg-popover border border-border rounded-md text-sm whitespace-nowrap opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none z-50">
                    {item.name}
                  </div>
                )}
              </Link>
            </motion.div>
          )
        })}
      </nav>

      {/* Settings at bottom - Admin only */}
      {isAdmin && (
        <div className="absolute bottom-4 left-4 right-4">
          <Link
            href="/settings"
            className="sidebar-item group"
          >
            <div className="w-10 h-10 rounded-lg bg-secondary/30 group-hover:bg-secondary/50 flex items-center justify-center transition-all duration-200">
              <Settings className="w-5 h-5 text-muted-foreground group-hover:text-foreground" />
            </div>

            {!collapsed && (
              <div className="ml-3">
                <p className="text-sm font-medium">Settings</p>
                <p className="text-xs text-muted-foreground">System configuration</p>
              </div>
            )}

            {collapsed && (
              <div className="absolute left-full ml-2 px-2 py-1 bg-popover border border-border rounded-md text-sm whitespace-nowrap opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none z-50">
                Settings
              </div>
            )}
          </Link>
        </div>
      )}
    </motion.div>
  )
}
