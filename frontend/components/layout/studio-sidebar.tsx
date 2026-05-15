'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { ExternalLink, PanelLeftClose, PanelLeftOpen } from 'lucide-react';
import {
  STUDIO_MENU_PRIMARY,
  STUDIO_MENU_FOOTER,
  resolveActiveMenuId,
  type StudioMenuGroup,
} from '@/lib/studio-menu';

/**
 * StudioSidebar — SIDE-B (labelled rail, 232px) per CD's shell delivery.
 *
 * 13 menu items in 3 groups (OPERATIONS / WORKFORCE / WORKSPACE) +
 * workspace switcher + footer (Docs external + Settings) + mini-stats foot.
 *
 * Supports collapse to 56px icon-rail via `collapsed` + `onToggle`. State is
 * owned by MainLayout (persisted to localStorage).
 *
 * Renders under `.studio` scope only (gated by MainLayout). Wire workspace
 * data + sign-out handlers in via props when integrating with Clerk.
 */

const GROUP_ORDER: StudioMenuGroup[] = ['OPERATIONS', 'WORKFORCE', 'WORKSPACE'];

export interface StudioSidebarProps {
  workspaceName?: string;
  workspaceMeta?: string;
  workspaceMark?: string;
  alerts?: Record<string, string | number>;
  showStats?: boolean;
  /** Collapse to icon-rail (60px). Defaults false. */
  collapsed?: boolean;
  /** Toggle handler. If absent, the toggle button is hidden. */
  onToggle?: () => void;
}

// NOTE: Multi-workspace switching is OOS for the Studio rebrand (one
// workspace per user). The workspace pill is intentionally non-interactive
// — kept as a static identity card showing the current workspace.
export function StudioSidebar({
  workspaceName = 'Automatos AI',
  workspaceMeta = 'pilot · 11 op',
  workspaceMark,
  alerts = {},
  showStats = true,
  collapsed = false,
  onToggle,
}: StudioSidebarProps) {
  const pathname = usePathname();
  const activeId = pathname ? resolveActiveMenuId(pathname) : null;
  const mark = workspaceMark ?? workspaceName.slice(0, 1).toUpperCase();

  return (
    <aside
      className={`sh-side${collapsed ? ' collapsed' : ''}`}
      data-tour="sidebar-studio"
    >
      {/* Brand row */}
      <div className="sh-brand-row">
        <Link
          href="/"
          className="sh-brand"
          style={{ fontSize: 17 }}
          aria-label="Automatos home"
        >
          <span className="sh-glyph" />
          {!collapsed && <span>automatos</span>}
        </Link>
        {!collapsed && (
          <span
            style={{
              marginLeft: 'auto',
              fontFamily: 'var(--font-geist-mono, monospace)',
              fontSize: 9,
              color: 'hsl(var(--muted-foreground))',
              letterSpacing: '0.08em',
            }}
          >
            v0.11
          </span>
        )}
        {onToggle && (
          <button
            type="button"
            className="sh-toggle"
            onClick={onToggle}
            aria-label={collapsed ? 'Expand sidebar' : 'Collapse sidebar'}
            title={collapsed ? 'Expand sidebar' : 'Collapse sidebar'}
          >
            {collapsed ? (
              <PanelLeftOpen style={{ width: 14, height: 14, strokeWidth: 1.6 }} />
            ) : (
              <PanelLeftClose style={{ width: 14, height: 14, strokeWidth: 1.6 }} />
            )}
          </button>
        )}
      </div>

      {/* Workspace identity (static — multi-workspace OOS for the rebrand) */}
      <div className="sh-ws-card" role="group" aria-label="Current workspace">
        <span className="sh-mk">{mark}</span>
        {!collapsed && (
          <div style={{ flex: 1, minWidth: 0, textAlign: 'left' }}>
            <div className="sh-nm">{workspaceName}</div>
            <div className="sh-meta">{workspaceMeta}</div>
          </div>
        )}
      </div>

      {/* Primary menu — grouped */}
      <nav style={{ flex: 1, overflow: 'hidden auto' }} aria-label="Main">
        {GROUP_ORDER.map((group) => (
          <div key={group}>
            {!collapsed && <div className="sh-group">{group}</div>}
            {collapsed && <div className="sh-group-rule" aria-hidden />}
            {STUDIO_MENU_PRIMARY.filter((m) => m.group === group).map((m) => {
              const Icon = m.icon;
              const isActive = activeId === m.id;
              const alert = alerts[m.id];
              return (
                <Link
                  key={m.id}
                  href={m.href}
                  className={`sh-item${isActive ? ' active' : ''}`}
                  aria-current={isActive ? 'page' : undefined}
                  title={collapsed ? m.label : m.desc}
                >
                  <Icon className="sh-ic" strokeWidth={1.6} />
                  {!collapsed && <span>{m.label}</span>}
                  {!collapsed && alert != null && (
                    <span
                      className={`sh-ct${typeof alert === 'string' && alert === '!' ? ' alert' : ''}`}
                    >
                      {alert}
                    </span>
                  )}
                </Link>
              );
            })}
          </div>
        ))}
      </nav>

      {/* Footer — Docs external + Settings */}
      <div className="sh-footer">
        {STUDIO_MENU_FOOTER.map((m) => {
          const Icon = m.icon;
          const isActive = activeId === m.id;
          if (m.external) {
            return (
              <a
                key={m.id}
                href={m.href}
                target="_blank"
                rel="noopener noreferrer"
                className="sh-item"
                title={collapsed ? m.label : m.desc}
              >
                <Icon className="sh-ic" strokeWidth={1.6} />
                {!collapsed && (
                  <>
                    <span>{m.label}</span>
                    <ExternalLink
                      style={{
                        width: 11,
                        height: 11,
                        color: 'hsl(var(--muted-foreground))',
                        marginLeft: 'auto',
                      }}
                    />
                  </>
                )}
              </a>
            );
          }
          return (
            <Link
              key={m.id}
              href={m.href}
              className={`sh-item${isActive ? ' active' : ''}`}
              aria-current={isActive ? 'page' : undefined}
              title={collapsed ? m.label : m.desc}
            >
              <Icon className="sh-ic" strokeWidth={1.6} />
              {!collapsed && <span>{m.label}</span>}
            </Link>
          );
        })}

        {showStats && !collapsed && (
          <div className="sh-mini-stats">
            <div className="sh-row">
              <span>tick</span>
              <span>5s</span>
            </div>
            <div className="sh-row">
              <span>$/dec</span>
              <span className="sh-ok">$0.0027</span>
            </div>
            <div className="sh-row">
              <span>cache</span>
              <span className="sh-ok">68%</span>
            </div>
          </div>
        )}
      </div>
    </aside>
  );
}
