'use client';

import Link from 'next/link';
import { usePathname, useSearchParams } from 'next/navigation';
import {
  STUDIO_PAGE_TABS,
  resolveActiveMenuId,
} from '@/lib/studio-menu';

/**
 * StudioPageTabs — sub-nav strip beneath the StudioHeader. Reads the active
 * menu id from the pathname and looks up STUDIO_PAGE_TABS for the [name,
 * count] pairs. Active tab is driven by `?tab=` (defaults to the first tab).
 *
 * Renders nothing when there are no tabs registered for the active route.
 */

function slugify(label: string): string {
  return label.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/(^-|-$)/g, '');
}

export function StudioPageTabs() {
  const pathname = usePathname() ?? '';
  const searchParams = useSearchParams();
  const menuId = resolveActiveMenuId(pathname);
  if (!menuId) return null;

  const tabs = STUDIO_PAGE_TABS[menuId];
  if (!tabs || tabs.length === 0) return null;

  const activeTab = searchParams?.get('tab') ?? slugify(tabs[0][0]);

  return (
    <nav className="sh-tabs" aria-label="Page sections">
      {tabs.map(([label, count]) => {
        const slug = slugify(label);
        const isActive = slug === activeTab;
        const href = `${pathname}?tab=${slug}`;
        return (
          <Link
            key={slug}
            href={href}
            className={`sh-tab${isActive ? ' active' : ''}`}
            aria-current={isActive ? 'page' : undefined}
          >
            <span>{label}</span>
            {count > 0 && <span className="sh-tab-ct">{count}</span>}
          </Link>
        );
      })}
    </nav>
  );
}
