'use client';

import * as React from 'react';
import { Search, BookOpen, HelpCircle, ExternalLink, Code2 } from 'lucide-react';
import { ThemeToggle } from '@/components/ui/theme-toggle';
import { NotificationBell } from '@/components/notifications/notification-bell';
import { ProfileMenu } from '@/components/auth/profile-menu';
import { UserProfileButton } from '@/components/auth/user-profile-button';
import { CreditBalancePill } from '@/components/billing/credit-balance-pill';
import { isSaaS } from '@/lib/auth-edition';
import { openGlobalSearch } from '@/lib/global-search-event';
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

export function StudioHeader() {
  return (
    <header className="sh-headbar">
      {/* Search cmdK — dispatches the global search open event */}
      <button
        type="button"
        className="sh-cmdk"
        onClick={openGlobalSearch}
        aria-label="Open search (⌘K)"
      >
        <Search style={{ width: 12, height: 12 }} />
        <span>Search</span>
        <span className="sh-kbd">⌘K</span>
      </button>

      {/* Spacer */}
      <div style={{ flex: 1 }} />

      <StudioHeaderUtilities />
    </header>
  );
}

/**
 * The header's utilities cluster — the credit balance (hosted edition), Help,
 * notifications, the theme menu and the profile control by edition. Play's frame
 * (PRD-260 T-004) keeps the same one.
 */
export function StudioHeaderUtilities() {
  return (
    <div className="sh-utils">
      {/* Credit balance — hosted edition only (PRD-265 US-008); nothing is metered locally. */}
      {isSaaS && <CreditBalancePill className="mr-2" />}

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

      {/* Profile menu — hosted edition only (PRD-233 S7): signed-out it is a
          sign-in affordance, and the local edition has no accounts. */}
      {isSaaS ? <ProfileMenu /> : <UserProfileButton />}
    </div>
  );
}
