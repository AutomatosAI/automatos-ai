'use client';

import * as React from 'react';
import { useUser } from '@clerk/nextjs';
import { Search, BookOpen, HelpCircle, Bell, ChevronDown } from 'lucide-react';
import { ThemeToggle } from '@/components/ui/theme-toggle';

/**
 * StudioHeader — HEAD-A (editorial) per CD's shell delivery.
 *
 * Brand + utilities cluster only. Page titles live in the page (we use the
 * editorial PageHeader for that). Order: cmdK search · Docs · Help · Alerts
 * (with 9+ badge) · sep · ThemeToggle · Profile pill.
 *
 * PRD shell rollout reference.
 */

export interface StudioHeaderProps {
  /** Search trigger handler (cmdK). If absent, the search input is non-interactive visual stub. */
  onSearchClick?: () => void;
  /** Number of unread alerts. Renders 9+ badge if >9. */
  alertCount?: number;
  /** Profile dropdown handler. If absent, profile pill is non-interactive visual. */
  onProfileClick?: () => void;
}

export function StudioHeader({
  onSearchClick,
  alertCount = 0,
  onProfileClick,
}: StudioHeaderProps) {
  const { user, isLoaded } = useUser();

  const displayName = isLoaded && user
    ? user.firstName ?? user.username ?? user.primaryEmailAddress?.emailAddress?.split('@')[0] ?? 'You'
    : 'Loading';
  const initial = displayName.slice(0, 1).toUpperCase();

  const renderAlertBadge = () => {
    if (alertCount <= 0) return null;
    return <span className="sh-badge">{alertCount > 9 ? '9+' : alertCount}</span>;
  };

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
        <a
          className="sh-icon-btn"
          href="https://docs.automatos.app"
          target="_blank"
          rel="noopener noreferrer"
          title="Docs"
          aria-label="Open documentation"
        >
          <BookOpen style={{ width: 15, height: 15, strokeWidth: 1.6 }} />
        </a>
        <button
          type="button"
          className="sh-icon-btn"
          title="Help"
          aria-label="Help"
        >
          <HelpCircle style={{ width: 15, height: 15, strokeWidth: 1.6 }} />
        </button>
        <button
          type="button"
          className="sh-icon-btn"
          title="Alerts"
          aria-label={`Alerts${alertCount > 0 ? ` (${alertCount > 9 ? '9 or more' : alertCount})` : ''}`}
        >
          <Bell style={{ width: 15, height: 15, strokeWidth: 1.6 }} />
          {renderAlertBadge()}
        </button>
        <span className="sh-sep" />

        {/* Theme toggle slot — uses the existing component which already handles Studio */}
        <ThemeToggle />

        {/* Profile pill */}
        <button
          type="button"
          className="sh-profile"
          onClick={onProfileClick}
          title={displayName}
          aria-label={`Profile menu for ${displayName}`}
        >
          <span className="sh-av">{initial}</span>
          <span style={{ fontSize: 12.5, fontWeight: 500, color: 'hsl(var(--foreground))' }}>
            {displayName}
          </span>
          <ChevronDown
            style={{
              width: 12,
              height: 12,
              color: 'hsl(var(--muted-foreground))',
              strokeWidth: 1.8,
            }}
          />
        </button>
      </div>
    </header>
  );
}
