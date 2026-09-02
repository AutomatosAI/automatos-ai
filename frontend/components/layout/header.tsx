'use client'

import { Menu, BookOpen, ExternalLink, Code2 } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { ThemeToggle } from '@/components/ui/theme-toggle'
import { ProfileMenu } from '@/components/auth/profile-menu'
import { UserProfileButton } from '@/components/auth/user-profile-button'
import { NotificationBell } from '@/components/notifications/notification-bell'
import { isSaaS } from '@/lib/auth-edition'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
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
            className="text-primary hover:text-primary/80 hover:bg-primary/5"
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
                  href="https://docs.automatos.app"
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
                  href="https://docs.automatos.app/automatos-ai-docs/api-reference"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="flex items-center justify-between"
                >
                  API Reference
                  <Code2 className="w-3 h-3 opacity-50" />
                </a>
              </DropdownMenuItem>
            </DropdownMenuContent>
          </DropdownMenu>

          {/* Theme Toggle */}
          <ThemeToggle />

          {/* Notifications */}
          <NotificationBell />

          {/* User Menu — hosted edition only (PRD-233 S7). ProfileMenu is the
              account menu, and signed-out it is a sign-in affordance; the local
              edition has no accounts. Settings stays reachable from the rail. */}
          {isSaaS ? <ProfileMenu /> : <UserProfileButton />}
        </div>
      </div>
    </header>
  )
}
