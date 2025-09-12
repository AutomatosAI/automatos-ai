
'use client'

import { useState } from 'react'
import { usePathname } from 'next/navigation'
import Link from 'next/link'
import { motion } from 'framer-motion'
import { 
  LayoutDashboard,
  Bot,
  FileText,
  GitBranch,
  BarChart3,
  Settings,
  ChevronLeft,
  ChevronRight,
  Activity,
  Users,
  Database,
  Zap,
  BookOpen,
  Brain,
  Target,
  Workflow
} from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Separator } from '@/components/ui/separator'
import { Badge } from '@/components/ui/badge'

interface EnhancedSidebarProps {
  collapsed: boolean
  onToggle: (collapsed: boolean) => void
}

export function EnhancedSidebar({ collapsed, onToggle }: EnhancedSidebarProps) {
  const pathname = usePathname()

  const navigationItems = [
    {
      title: 'Overview',
      items: [
        { 
          name: 'Dashboard', 
          href: '/', 
          icon: LayoutDashboard,
          badge: null,
          description: 'System overview and metrics'
        },
        { 
          name: 'Analytics', 
          href: '/analytics', 
          icon: BarChart3,
          badge: null,
          description: 'Performance analytics and insights'
        },
      ]
    },
    {
      title: 'Core Platform',
      items: [
        { 
          name: 'Agent Management', 
          href: '/agents', 
          icon: Bot,
          badge: '904',
          description: 'Manage AI agents and orchestration'
        },
        { 
          name: 'Document Hub', 
          href: '/documents', 
          icon: FileText,
          badge: null,
          description: 'Document processing and management'
        },
        { 
          name: 'Workflows', 
          href: '/workflows', 
          icon: GitBranch,
          badge: 'Beta',
          description: 'Automated workflow management'
        },
      ]
    },
    {
      title: 'Advanced Features',
      items: [
        { 
          name: 'Context Engineering', 
          href: '/context', 
          icon: Brain,
          badge: 'New',
          description: 'Advanced context optimization'
        },
        { 
          name: 'Field Theory', 
          href: '/field-theory', 
          icon: Zap,
          badge: null,
          description: 'Neural field operations'
        },
        { 
          name: 'Playbooks', 
          href: '/playbooks', 
          icon: BookOpen,
          badge: null,
          description: 'Automated playbook execution'
        },
      ]
    },
    {
      title: 'System',
      items: [
        { 
          name: 'Settings', 
          href: '/settings', 
          icon: Settings,
          badge: null,
          description: 'System configuration'
        },
      ]
    },
  ]

  return (
    <motion.aside
      initial={{ x: -20, opacity: 0 }}
      animate={{ x: 0, opacity: 1 }}
      className={`fixed left-0 top-0 z-50 h-screen bg-card/80 backdrop-blur-md border-r border-border/50 transition-all duration-300 ${
        collapsed ? 'w-16' : 'w-64'
      }`}
    >
      {/* Sidebar Header */}
      <div className="flex items-center justify-between p-4 border-b border-border/50">
        {!collapsed && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="flex items-center gap-3"
          >
            <div className="flex items-center justify-center w-8 h-8 rounded-lg bg-gradient-to-r from-brand-primary to-brand-secondary">
              <Activity className="w-4 h-4 text-white" />
            </div>
            <div className="flex flex-col">
              <span className="font-bold gradient-text">Automotas AI</span>
              <span className="text-xs text-muted-foreground">Platform</span>
            </div>
          </motion.div>
        )}
        <Button
          variant="ghost"
          size="icon"
          onClick={() => onToggle(!collapsed)}
          className="h-8 w-8"
        >
          {collapsed ? (
            <ChevronRight className="h-4 w-4" />
          ) : (
            <ChevronLeft className="h-4 w-4" />
          )}
        </Button>
      </div>

      {/* Navigation */}
      <div className="flex-1 overflow-y-auto py-4">
        <nav className="space-y-6 px-3">
          {navigationItems.map((section) => (
            <div key={section.title}>
              {!collapsed && (
                <h3 className="px-3 text-xs font-semibold text-muted-foreground uppercase tracking-wider mb-2">
                  {section.title}
                </h3>
              )}
              <ul className="space-y-1">
                {section.items.map((item) => {
                  const isActive = pathname === item.href
                  return (
                    <li key={item.name}>
                      <Link href={item.href}>
                        <Button
                          variant={isActive ? 'secondary' : 'ghost'}
                          className={`w-full justify-start h-10 px-3 ${
                            isActive 
                              ? 'bg-brand-primary/10 text-brand-primary border-brand-primary/20' 
                              : 'hover:bg-accent/50'
                          } ${collapsed ? 'px-2' : ''}`}
                          title={collapsed ? item.name : undefined}
                        >
                          <item.icon className={`h-4 w-4 ${
                            isActive ? 'text-brand-primary' : ''
                          } ${collapsed ? '' : 'mr-3'}`} />
                          {!collapsed && (
                            <>
                              <span className="flex-1 text-left">{item.name}</span>
                              {item.badge && (
                                <Badge 
                                  variant={item.badge === '904' ? 'default' : 'secondary'}
                                  className="text-xs px-2 py-0"
                                >
                                  {item.badge}
                                </Badge>
                              )}
                            </>
                          )}
                        </Button>
                      </Link>
                    </li>
                  )
                })}
              </ul>
              {!collapsed && <Separator className="my-4" />}
            </div>
          ))}
        </nav>
      </div>

      {/* System Status */}
      {!collapsed && (
        <div className="p-4 border-t border-border/50">
          <div className="glass-card p-3 rounded-lg">
            <div className="flex items-center justify-between mb-2">
              <span className="text-sm font-medium">System Status</span>
              <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse" />
            </div>
            <div className="space-y-1 text-xs text-muted-foreground">
              <div className="flex justify-between">
                <span>Agents Active</span>
                <span className="font-mono text-brand-primary">904</span>
              </div>
              <div className="flex justify-between">
                <span>CPU Usage</span>
                <span className="font-mono">23%</span>
              </div>
              <div className="flex justify-between">
                <span>Memory</span>
                <span className="font-mono">2.1GB</span>
              </div>
            </div>
          </div>
        </div>
      )}
    </motion.aside>
  )
}
