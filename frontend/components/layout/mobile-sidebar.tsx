'use client'

import Link from 'next/link'
import { usePathname } from 'next/navigation'
import { motion } from 'framer-motion'
import {
  Users,
  Settings,
  Bot,
  MessageCircle,
  Wrench,
  Database,
  Store,
  BarChart3,
  Package,
  X,
  BookOpen,
  ExternalLink,
  Radar,
  ClipboardList,
} from 'lucide-react'
import { cn } from '@/lib/utils'
import { PremiumIcon } from '@/components/shared'
import { useSystemIcons } from '@/hooks/use-system-config-api'
import { useSystemRole } from '@/contexts/role-context'

interface MobileSidebarProps {
  onNavigate: () => void
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
    name: 'Assignments',
    href: '/assignments',
    icon: ClipboardList,
    iconColor: 'text-[hsl(var(--chart-4))]',
    navIconKey: 'nav_assignments',
    description: 'Plan, schedule & orchestrate work'
  },
  {
    name: 'Deliverables',
    href: '/deliverables',
    icon: Package,
    iconColor: 'text-[hsl(var(--chart-3))]',
    navIconKey: 'nav_workspace',
    description: 'Files, code & agent output'
  },
  {
    name: 'Command Center',
    href: '/command-center',
    icon: Radar,
    iconColor: 'text-[hsl(var(--info))]',
    navIconKey: 'nav_activity',
    description: 'Your AI workforce at a glance'
  },
  {
    name: 'Agents',
    href: '/agents',
    icon: Bot,
    iconColor: 'text-primary',
    navIconKey: 'nav_agents',
    description: 'Manage AI agents and skills'
  },
  {
    name: 'Tools',
    href: '/tools',
    icon: Wrench,
    iconColor: 'text-[hsl(var(--warning))]',
    navIconKey: 'nav_tools',
    description: 'Development and utility tools'
  },
  {
    name: 'Marketplace',
    href: '/marketplace',
    icon: Store,
    iconColor: 'text-primary',
    navIconKey: 'nav_marketplace',
    description: 'Discover agents, playbooks & tools'
  },
  {
    name: 'Knowledge',
    href: '/documents',
    icon: Database,
    iconColor: 'text-[hsl(var(--success))]',
    navIconKey: 'nav_knowledge',
    description: 'Documents, databases & code-graph'
  },
  {
    name: 'Team',
    href: '/team',
    icon: Users,
    iconColor: 'text-[hsl(var(--info))]',
    navIconKey: 'nav_team',
    description: 'Manage workspace members',
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

export function MobileSidebar({ onNavigate }: MobileSidebarProps) {
  const pathname = usePathname()
  const { isAdmin } = useSystemRole()
  const { data: iconMappings = {} } = useSystemIcons()

  const filteredNavItems = navigationItems.filter(item => {
    if (!(item as any).requiredRole) return true
    return (item as any).requiredRole === 'admin' && isAdmin
  })

  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      <div className="flex items-center gap-3 px-5 py-4 border-b border-border/50">
        <img
          src="/brand/automatos-mark-hi.png"
          alt="Automatos"
          className="h-8 w-auto opacity-95"
          draggable={false}
        />
        <span className="text-lg font-semibold tracking-wide text-foreground">
          AUTOMATOS A.I.
        </span>
      </div>

      {/* Navigation */}
      <nav className="flex-1 overflow-y-auto px-3 py-3 space-y-1">
        {filteredNavItems.map((item, index) => {
          const isActive = pathname === item.href
          const Icon = item.icon
          const premiumNavIcon = item.navIconKey ? iconMappings[item.navIconKey] : null

          return (
            <motion.div
              key={item.name}
              initial={{ opacity: 0, x: -12 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: index * 0.03 }}
            >
              <Link
                href={item.href}
                onClick={onNavigate}
                className={cn(
                  'flex items-center gap-3 w-full px-4 py-3 rounded-xl transition-all duration-200',
                  'min-h-[48px]',
                  isActive
                    ? 'bg-primary/10 border border-primary/20'
                    : 'hover:bg-secondary/40 active:bg-secondary/60'
                )}
              >
                {premiumNavIcon ? (
                  <PremiumIcon name={premiumNavIcon} size={20} className="shrink-0" />
                ) : (
                  <Icon
                    className={cn(
                      'w-5 h-5 shrink-0',
                      isActive ? 'text-primary' : item.iconColor || 'text-muted-foreground'
                    )}
                  />
                )}
                <div className="min-w-0">
                  <p className={cn(
                    'text-sm',
                    isActive ? 'font-semibold text-foreground' : 'font-medium text-muted-foreground'
                  )}>
                    {item.name}
                  </p>
                  <p className="text-xs text-muted-foreground/70 truncate">{item.description}</p>
                </div>
              </Link>
            </motion.div>
          )
        })}
      </nav>

      {/* Docs + Settings at bottom */}
      <div className="border-t border-border/50 px-3 py-3 space-y-1">
        {/* Documentation link */}
        <a
          href="https://docs.automatos.app"
          target="_blank"
          rel="noopener noreferrer"
          className="flex items-center gap-3 w-full px-4 py-3 rounded-xl transition-all duration-200 min-h-[48px] hover:bg-secondary/40 active:bg-secondary/60"
        >
          <BookOpen className="w-5 h-5 shrink-0 text-muted-foreground" />
          <p className="text-sm font-medium text-muted-foreground">Docs</p>
          <ExternalLink className="w-3 h-3 text-muted-foreground/50 ml-auto" />
        </a>

        {/* Settings — Admin only */}
        {isAdmin && (
          <Link
            href="/settings"
            onClick={onNavigate}
            className={cn(
              'flex items-center gap-3 w-full px-4 py-3 rounded-xl transition-all duration-200 min-h-[48px]',
              pathname === '/settings'
                ? 'bg-primary/10 border border-primary/20'
                : 'hover:bg-secondary/40 active:bg-secondary/60'
            )}
          >
            {iconMappings['nav_settings'] ? (
              <PremiumIcon name={iconMappings['nav_settings']} size={20} className="shrink-0" />
            ) : (
              <Settings className="w-5 h-5 shrink-0 text-muted-foreground" />
            )}
            <p className="text-sm font-medium text-muted-foreground">Settings</p>
          </Link>
        )}
      </div>
    </div>
  )
}
