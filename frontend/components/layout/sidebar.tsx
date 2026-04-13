
'use client'

import Link from 'next/link'
import { usePathname } from 'next/navigation'
import { motion } from 'framer-motion'
import {
  Users,
  Activity,
  Brain,
  Settings,
  ChevronLeft,
  Bot,
  MessageCircle,
  PanelLeft,
  Wrench,
  Database,
  Store,
  BarChart3,
  LayoutDashboard,
  HardDrive,
  BookOpen,
  ExternalLink,
} from 'lucide-react'
import { cn } from '@/lib/utils'
import { PremiumIcon } from '@/components/shared'
import { useSystemIcons } from '@/hooks/use-system-config-api'
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
    navIconKey: 'nav_chat',
    description: 'Your AI workspace'
  },
  {
    name: 'Workspace',
    href: '/workspace',
    icon: HardDrive,
    iconColor: 'text-[hsl(var(--chart-3))]',
    navIconKey: 'nav_workspace',
    description: 'Files, code & agent output'
  },
  {
    name: 'Activity',
    href: '/activity',
    icon: Activity,
    iconColor: 'text-[hsl(var(--info))]',
    navIconKey: 'nav_activity',
    description: 'Your AI workforce at a glance'
  },
  {
    name: 'Agent Management',
    href: '/agents',
    icon: Bot,
    iconColor: 'text-primary',
    navIconKey: 'nav_agents',
    description: 'Manage AI agents and skills'
  },
  {
    name: 'Tools & Integrations',
    href: '/tools',
    icon: Wrench,
    iconColor: 'text-[hsl(var(--warning))]',
    navIconKey: 'nav_tools',
    description: 'Development and utility tools'
  },
  {
    name: 'Community Marketplace',
    href: '/marketplace',
    icon: Store,
    iconColor: 'text-primary',
    navIconKey: 'nav_marketplace',
    description: 'Discover agents, recipes & tools'
  },
  {
    name: 'Knowledge Bases',
    href: '/documents',
    icon: Database,
    iconColor: 'text-[hsl(var(--success))]',
    navIconKey: 'nav_knowledge',
    description: 'Documents, databases & code-graph'
  },
  {
    name: 'Team Management',
    href: '/team',
    icon: Users,
    iconColor: 'text-[hsl(var(--info))]',
    navIconKey: 'nav_team',
    description: 'Manage workspace members',
    requiredRole: 'admin' as const,
  },
  {
    name: 'Context Engineering',
    href: '/context',
    icon: Brain,
    iconColor: 'text-[hsl(var(--chart-4))]',
    navIconKey: 'nav_context',
    description: 'RAG system and field theory',
    requiredRole: 'admin' as const,
  },
  {
    name: 'Dashboard',
    href: '/dashboard',
    icon: LayoutDashboard,
    iconColor: 'text-emerald-400',
    navIconKey: 'nav_dashboard',
    description: 'System metrics & health',
  },
  {
    name: 'Analytics',
    href: '/analytics',
    icon: BarChart3,
    iconColor: 'text-cyan-400',
    navIconKey: 'nav_analytics',
    description: 'Performance, costs & insights',
  },
]

export function Sidebar({ collapsed, onToggle }: SidebarProps) {
  const pathname = usePathname()
  const isChatPage = pathname?.startsWith('/chat') ?? false
  const { systemRole, isAdmin } = useSystemRole()
  const { data: iconMappings = {} } = useSystemIcons()

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
          const premiumNavIcon = item.navIconKey ? iconMappings[item.navIconKey] : null

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
                data-tour={`nav-${item.href.replace('/', '')}`}
                className={cn(
                  'flex items-center gap-3 w-full px-3 py-2 rounded-xl transition-all duration-200 group relative',
                  isActive
                    ? 'bg-primary/10 border border-primary/20'
                    : 'hover:bg-secondary/40'
                )}
              >
                {premiumNavIcon ? (
                  <PremiumIcon name={premiumNavIcon} size={18} className="shrink-0" />
                ) : (
                  <Icon
                    className={cn(
                      'w-[18px] h-[18px] shrink-0 transition-colors',
                      isActive ? 'text-primary' : item.iconColor || 'text-muted-foreground group-hover:text-foreground'
                    )}
                  />
                )}

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

      {/* Docs + Settings at bottom */}
      <div className="absolute bottom-4 left-3 right-3 space-y-0.5">
        {/* Documentation link */}
        <a
          href="https://automatos.gitbook.io/automatos-ai"
          target="_blank"
          rel="noopener noreferrer"
          className={cn(
            'flex items-center gap-3 w-full px-3 py-2 rounded-xl transition-all duration-200 group relative',
            'hover:bg-secondary/40'
          )}
        >
          <BookOpen className="w-[18px] h-[18px] shrink-0 text-muted-foreground group-hover:text-foreground" />

          {!collapsed && (
            <div className="min-w-0 flex items-center gap-2">
              <p className="text-sm font-medium text-muted-foreground group-hover:text-foreground truncate">Docs</p>
              <ExternalLink className="w-3 h-3 text-muted-foreground/50 shrink-0" />
            </div>
          )}

          {collapsed && (
            <div className="absolute left-full ml-2 px-2 py-1 bg-popover/90 border border-border/50 rounded-xl backdrop-blur-lg text-sm whitespace-nowrap opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none z-50">
              Documentation
            </div>
          )}
        </a>

        {/* Settings */}
        <Link
          href="/settings"
          data-tour="nav-settings"
          className={cn(
            'flex items-center gap-3 w-full px-3 py-2 rounded-xl transition-all duration-200 group relative',
            pathname === '/settings'
              ? 'bg-primary/10 border border-primary/20'
              : 'hover:bg-secondary/40'
          )}
        >
          {iconMappings['nav_settings'] ? (
            <PremiumIcon name={iconMappings['nav_settings']} size={18} className="shrink-0" />
          ) : (
            <Settings className="w-[18px] h-[18px] shrink-0 text-muted-foreground group-hover:text-foreground" />
          )}

          {!collapsed && (
            <div className="min-w-0">
              <p className="text-sm font-medium text-muted-foreground group-hover:text-foreground truncate">Settings</p>
              <p className="text-xs text-muted-foreground truncate">Profile, API keys, preferences</p>
            </div>
          )}

          {collapsed && (
            <div className="absolute left-full ml-2 px-2 py-1 bg-popover/90 border border-border/50 rounded-xl backdrop-blur-lg text-sm whitespace-nowrap opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none z-50">
              Settings
            </div>
          )}
        </Link>
      </div>
    </motion.div>
  )
}
