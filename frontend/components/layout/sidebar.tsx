
'use client'

import Link from 'next/link'
import { usePathname } from 'next/navigation'
import { motion } from 'framer-motion'
import {
  Users,
  GitBranch,
  Brain,
  Settings,
  ChevronLeft,
  Bot,
  MessageCircle,
  PanelLeft,
  Wrench,
  Database,
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
    iconColor: 'text-primary',
    description: 'Your AI workspace'
  },
  {
    name: 'Workflow Management',
    href: '/workflows',
    icon: GitBranch,
    iconColor: 'text-[hsl(var(--agent))]',
    description: 'Create and monitor workflows'
  },
  {
    name: 'Agent Management',
    href: '/agents',
    icon: Bot,
    iconColor: 'text-primary',
    description: 'Manage AI agents and skills'
  },
  {
    name: 'Tools & Integrations',
    href: '/tools',
    icon: Wrench,
    iconColor: 'text-[hsl(var(--warning))]',
    description: 'Development and utility tools'
  },
  {
    name: 'Community Marketplace',
    href: '/marketplace',
    icon: Store,
    iconColor: 'text-primary',
    description: 'Discover agents, recipes & tools'
  },
  {
    name: 'Knowledge Bases',
    href: '/documents',
    icon: Database,
    iconColor: 'text-[hsl(var(--success))]',
    description: 'Documents, databases & code-graph'
  },
  {
    name: 'Team Management',
    href: '/team',
    icon: Users,
    iconColor: 'text-[hsl(var(--info))]',
    description: 'Manage workspace members',
    requiredRole: 'admin' as const, // Admin only
  },
  {
    name: 'Context Engineering',
    href: '/context',
    icon: Brain,
    iconColor: 'text-[hsl(var(--chart-4))]',
    description: 'RAG system and field theory',
    requiredRole: 'admin' as const,  // Admin only
  },
  {
    name: 'Analytics',
    href: '/analytics',
    icon: BarChart3,
    iconColor: 'text-cyan-400',
    description: 'Performance, costs & insights',
  },
  // Dashboard removed — analytics consolidated into /analytics
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
      data-tour="sidebar"
      className={cn(
        'fixed left-0 top-0 z-40 h-screen glass-card border-r border-primary/15 bg-background/25 backdrop-blur-xl shadow-[0_0_80px_hsla(var(--primary)/0.06)] transition-all duration-300',
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
            <div className="w-9 h-9 rounded-xl bg-secondary/30 border border-primary/15 flex items-center justify-center shadow-[0_0_18px_hsla(var(--primary)/0.10)]">
              <MessageCircle className="w-5 h-5 text-primary" />
            </div>
          </motion.div>
        )}

        <button
          onClick={() => onToggle(!collapsed)}
          className="p-1 rounded-full hover:bg-secondary/50 transition-colors"
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
      <nav className="px-3 py-3 space-y-0.5">
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
              className={cn(
                'flex items-center gap-3 w-full px-3 py-2 rounded-xl transition-all duration-200 group relative',
                'hover:bg-secondary/40'
              )}
              aria-label="Toggle chat history"
            >
              <PanelLeft className={cn('w-[18px] h-[18px] shrink-0 transition-colors', collapsed ? 'text-primary/70' : 'text-primary')} />

              {!collapsed && (
                <span className="text-sm font-medium truncate">Chat History</span>
              )}

              {collapsed && (
                <div className="absolute left-full ml-2 px-2 py-1 bg-popover/90 border border-border/50 rounded-xl backdrop-blur-lg text-sm whitespace-nowrap opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none z-50">
                  Chat History
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
              transition={{ delay: index * 0.05 }}
            >
              <Link
                href={item.href}
                onClick={() => onToggle(true)}
                data-tour={item.href === '/agents' ? 'nav-agents' : undefined}
                className={cn(
                  'flex items-center gap-3 w-full px-3 py-2 rounded-xl transition-all duration-200 group relative',
                  isActive
                    ? 'bg-primary/10 border border-primary/20'
                    : 'hover:bg-secondary/40'
                )}
              >
                <Icon
                  className={cn(
                    'w-[18px] h-[18px] shrink-0 transition-colors',
                    isActive ? 'text-primary' : item.iconColor || 'text-muted-foreground group-hover:text-foreground'
                  )}
                />

                {!collapsed && (
                  <div className="min-w-0">
                    <p className={cn(
                      'text-sm truncate',
                      isActive ? 'font-semibold text-foreground' : 'font-medium text-muted-foreground group-hover:text-foreground'
                    )}>
                      {item.name}
                    </p>
                    <p className="text-xs text-muted-foreground truncate">{item.description}</p>
                  </div>
                )}

                {/* Tooltip for collapsed state */}
                {collapsed && (
                  <div className="absolute left-full ml-2 px-2 py-1 bg-popover/90 border border-border/50 rounded-xl backdrop-blur-lg text-sm whitespace-nowrap opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none z-50">
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
        <div className="absolute bottom-4 left-3 right-3">
          <Link
            href="/settings"
            className={cn(
              'flex items-center gap-3 w-full px-3 py-2 rounded-xl transition-all duration-200 group relative',
              pathname === '/settings'
                ? 'bg-primary/10 border border-primary/20'
                : 'hover:bg-secondary/40'
            )}
          >
            <Settings className="w-[18px] h-[18px] shrink-0 text-muted-foreground group-hover:text-foreground" />

            {!collapsed && (
              <div className="min-w-0">
                <p className="text-sm font-medium text-muted-foreground group-hover:text-foreground truncate">Settings</p>
                <p className="text-xs text-muted-foreground truncate">System configuration</p>
              </div>
            )}

            {collapsed && (
              <div className="absolute left-full ml-2 px-2 py-1 bg-popover/90 border border-border/50 rounded-xl backdrop-blur-lg text-sm whitespace-nowrap opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none z-50">
                Settings
              </div>
            )}
          </Link>
        </div>
      )}
    </motion.div>
  )
}
