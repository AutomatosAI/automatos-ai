
'use client'

import { useState } from 'react'
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
  Bot
} from 'lucide-react'
import { cn } from '@/lib/utils'

interface SidebarProps {
  collapsed: boolean
  onToggle: (collapsed: boolean) => void
}

const navigationItems = [
  {
    name: 'Dashboard',
    href: '/',
    icon: LayoutDashboard,
    description: 'System overview and metrics'
  },
  {
    name: 'Agent Management',
    href: '/agents',
    icon: Bot,
    description: 'Manage AI agents and skills'
  },
  {
    name: 'Document Management',
    href: '/documents',
    icon: FileText,
    description: 'Knowledge base and documents'
  },
  {
    name: 'Workflow Management',
    href: '/workflows',
    icon: GitBranch,
    description: 'Create and monitor workflows'
  },
  {
    name: 'Context Engineering',
    href: '/context',
    icon: Brain,
    description: 'RAG system and context patterns'
  },
  {
    name: 'Performance Analytics',
    href: '/analytics',
    icon: BarChart3,
    description: 'Metrics and system performance'
  },
  {
    name: 'Playbooks',
    href: '/playbooks',
    icon: FileText,
    description: 'Mine and manage playbooks'
  }
]

export function Sidebar({ collapsed, onToggle }: SidebarProps) {
  const pathname = usePathname()

  return (
    <motion.div
      className={cn(
        'fixed left-0 top-0 z-40 h-screen glass-card border-r border-border/50 transition-all duration-300',
        collapsed ? 'w-16' : 'w-64'
      )}
      initial={false}
      animate={{ width: collapsed ? 64 : 256 }}
    >
      {/* Header */}
      <div className="flex items-center justify-between p-4 border-b border-border/50">
        {!collapsed && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="flex items-center space-x-2"
          >
            <div className="w-8 h-8 rounded-lg icon-gradient flex items-center justify-center">
              <Bot className="w-5 h-5 text-white" />
            </div>
            <span className="font-bold text-xl gradient-text">Automotas AI</span>
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
        {navigationItems.map((item, index) => {
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
                className={cn(
                  'sidebar-item group relative',
                  isActive && 'active'
                )}
              >
                <div className={cn(
                  'w-10 h-10 rounded-lg flex items-center justify-center transition-all duration-200',
                  isActive 
                    ? 'icon-gradient' 
                    : 'bg-secondary/30 group-hover:bg-secondary/50'
                )}>
                  <Icon className={cn(
                    'w-5 h-5 transition-colors',
                    isActive ? 'text-white' : 'text-muted-foreground group-hover:text-foreground'
                  )} />
                </div>
                
                {!collapsed && (
                  <div className="ml-3 flex-1">
                    <p className="text-sm font-medium">{item.name}</p>
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

      {/* Settings at bottom */}
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
    </motion.div>
  )
}
