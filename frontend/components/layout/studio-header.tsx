'use client';

import * as React from 'react';
import { Search, BookOpen, HelpCircle, ExternalLink, Code2 } from 'lucide-react';
import { ThemeToggle } from '@/components/ui/theme-toggle';
import { NotificationBell } from '@/components/notifications/notification-bell';
import { ProfileMenu } from '@/components/auth/profile-menu';
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu';

/**
 * StudioHeader — HEAD-A (editorial) per CD's shell delivery.
 *
 * Brand + utilities cluster only. Page titles live in the page (we use the
 * editorial PageHeader for that). Reuses the live ProfileMenu and
 * NotificationBell components so authentication, notifications, and theme
 * actions all behave consistently with the classic layout.
 */

export interface StudioHeaderProps {
  /** Search trigger handler (cmdK). If absent, the search input is non-interactive visual stub. */
  onSearchClick?: () => void;
}

export function StudioHeader({ onSearchClick }: StudioHeaderProps) {
  return (
    <header className="sh-headbar">
      {/* Search cmdK */}
      <button
        type="button"
        className="sh-cmdk"
        onClick={onSearchClick}
        aria-label="Search the platform"
      >
        <Search style={{ width: 12, height: 12 }} />
        <span>Jump to mission, agent, run…</span>
        <span className="sh-kbd">⌘K</span>
      </button>

      {/* Spacer */}
      <div style={{ flex: 1 }} />

      {/* Utilities cluster */}
      <div className="sh-utils">
        {/* Docs + Help dropdown */}
        <DropdownMenu>
          <DropdownMenuTrigger asChild>
            <button
              type="button"
              className="sh-icon-btn"
              aria-label="Help & documentation"
              title="Help"
            >
              <HelpCircle style={{ width: 15, height: 15, strokeWidth: 1.6 }} />
            </button>
          </DropdownMenuTrigger>
          <DropdownMenuContent align="end" className="w-48">
            <DropdownMenuItem asChild>
              <a
                href="https://docs.automatos.app"
                target="_blank"
                rel="noopener noreferrer"
                className="flex items-center justify-between"
              >
                <span className="flex items-center gap-2">
                  <BookOpen className="w-4 h-4" />
                  Documentation
                </span>
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
                <span className="flex items-center gap-2">
                  <Code2 className="w-4 h-4" />
                  API Reference
                </span>
                <ExternalLink className="w-3 h-3 opacity-50" />
              </a>
            </DropdownMenuItem>
          </DropdownMenuContent>
        </DropdownMenu>

        {/* Notifications */}
        <NotificationBell />

        <span className="sh-sep" />

        {/* Theme toggle */}
        <ThemeToggle />

        {/* Profile menu */}
        <ProfileMenu />
      </div>
    </header>
  );
}
