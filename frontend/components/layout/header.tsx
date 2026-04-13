'use client'

import { motion } from 'framer-motion'
import { Menu, BookOpen, ExternalLink, Code2, Bug } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { ThemeToggle } from '@/components/ui/theme-toggle'
import { ProfileMenu } from '@/components/auth/profile-menu'
import { NotificationBell } from '@/components/notifications/notification-bell'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu'

interface HeaderProps {
  onMenuClick: () => void
}

export function Header({ onMenuClick }: HeaderProps) {
  return (
    <header className="sticky top-0 z-30 glass-card border-b border-border/50 px-3 py-2 md:px-6 md:py-4">
      <div className="flex items-center justify-between">
        {/* Left side */}
        <div className="flex items-center space-x-3 md:space-x-4">
          <Button
            variant="ghost"
            size="icon"
            onClick={onMenuClick}
            className="text-orange-400 hover:text-orange-300 hover:bg-orange-500/5"
          >
            <Menu className="w-5 h-5" />
          </Button>

          {/* Brand (replaces non-functional search) */}
          <div className="flex items-center">
            {/* Brand lockup: orange ship always, wordmark color follows theme */}
            <img
              src="/brand/automatos-mark-hi.png"
              alt="Automatos ship mark"
              className="h-7 w-auto md:h-8 opacity-95"
              draggable={false}
            />
            <span className="hidden sm:inline ml-3 text-lg font-semibold tracking-wide text-foreground dark:text-white">
              AUTOMATOS A.I.
            </span>
          </div>
        </div>

        {/* Right side */}
        <div className="flex items-center space-x-2 md:space-x-4">
          {/* Help & Docs */}
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button
                variant="ghost"
                size="icon"
                className="text-muted-foreground hover:text-foreground"
                aria-label="Help & documentation"
              >
                <BookOpen className="w-5 h-5" />
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="end" className="w-48">
              <DropdownMenuItem asChild>
                <a
                  href="https://automatos.gitbook.io/automatos-ai"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="flex items-center justify-between"
                >
                  Documentation
                  <ExternalLink className="w-3 h-3 opacity-50" />
                </a>
              </DropdownMenuItem>
              <DropdownMenuItem asChild>
                <a
                  href="https://automatos.gitbook.io/automatos-ai/api-reference"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="flex items-center justify-between"
                >
                  API Reference
                  <Code2 className="w-3 h-3 opacity-50" />
                </a>
              </DropdownMenuItem>
              <DropdownMenuSeparator />
              <DropdownMenuItem asChild>
                <a
                  href="https://github.com/AutomatosAI/automatos-ai/issues"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="flex items-center justify-between"
                >
                  Report a Bug
                  <Bug className="w-3 h-3 opacity-50" />
                </a>
              </DropdownMenuItem>
            </DropdownMenuContent>
          </DropdownMenu>

          {/* Theme Toggle */}
          <ThemeToggle />

          {/* Notifications */}
          <NotificationBell />

          {/* User Menu */}
          <ProfileMenu />
        </div>
      </div>
    </header>
  )
}
