'use client';

import * as React from 'react';
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from './tooltip';
import { GLOSSARY, type GlossaryTerm } from '@/lib/glossary';

const STORAGE_KEY = 'automatos-glossary-seen';
const SUPPRESS_AFTER = 3;

interface GlossaryCounts {
  [term: string]: number;
}

function readCounts(): GlossaryCounts {
  if (typeof window === 'undefined') return {};
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    return raw ? (JSON.parse(raw) as GlossaryCounts) : {};
  } catch {
    return {};
  }
}

function writeCounts(counts: GlossaryCounts): void {
  if (typeof window === 'undefined') return;
  try {
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify(counts));
  } catch {
    /* localStorage disabled — silent fail. */
  }
}

interface GlossaryTooltipProps {
  /** The term whose definition to surface on hover */
  term: GlossaryTerm;
  /** The visible text in the page. Usually the same as the term but can differ
   *  (e.g. show "msn_8f3a" but hover defines "mission"). */
  children: React.ReactNode;
  /** If true, never suppress regardless of sighting count. Use sparingly. */
  alwaysShow?: boolean;
  /** Optional className passed to the inline wrapper */
  className?: string;
  /** Optional override of the dotted-underline affordance */
  noUnderline?: boolean;
}

/**
 * GlossaryTooltip — Move 2 from PRD §1.
 *
 * Wraps an inline element with a plain-English definition tooltip from the
 * glossary. After SUPPRESS_AFTER sightings of a given term in the same browser,
 * the tooltip stops appearing — the user has learned the term.
 *
 * Used per phase to soften technical surfaces (mission IDs, event types,
 * model names) for non-techy users. See PRD §1 for the philosophy.
 */
export function GlossaryTooltip({
  term,
  children,
  alwaysShow = false,
  className,
  noUnderline = false,
}: GlossaryTooltipProps) {
  const entry = GLOSSARY[term];
  const [suppressed, setSuppressed] = React.useState(false);

  // Read suppression state on mount (client-side only)
  React.useEffect(() => {
    if (alwaysShow) return;
    const counts = readCounts();
    if ((counts[term] ?? 0) >= SUPPRESS_AFTER) {
      setSuppressed(true);
    }
  }, [term, alwaysShow]);

  const onOpenChange = React.useCallback(
    (open: boolean) => {
      if (!open || alwaysShow) return;
      const counts = readCounts();
      const next = (counts[term] ?? 0) + 1;
      writeCounts({ ...counts, [term]: next });
      if (next >= SUPPRESS_AFTER) {
        setSuppressed(true);
      }
    },
    [term, alwaysShow]
  );

  // After suppression kicks in, render children as plain text — no tooltip,
  // no underline. The user has graduated past this glossary term.
  if (suppressed) {
    return <span className={className}>{children}</span>;
  }

  return (
    <TooltipProvider delayDuration={300}>
      <Tooltip onOpenChange={onOpenChange}>
        <TooltipTrigger asChild>
          <span
            className={
              (noUnderline
                ? ''
                : 'cursor-help underline decoration-dotted decoration-muted-foreground/60 underline-offset-[3px]') +
              (className ? ' ' + className : '')
            }
            tabIndex={0}
          >
            {children}
          </span>
        </TooltipTrigger>
        <TooltipContent
          side="top"
          className="max-w-xs space-y-1 px-3 py-2 text-sm leading-snug"
        >
          <div className="font-medium">{entry.label}</div>
          <p className="text-xs text-muted-foreground">{entry.definition}</p>
          {entry.example && (
            <p className="text-[11px] text-muted-foreground/80 italic">
              {entry.example}
            </p>
          )}
        </TooltipContent>
      </Tooltip>
    </TooltipProvider>
  );
}

/**
 * Test helper — reset the local sighting counts. Useful for QA scripts or
 * a "show glossary again" admin action. Not wired into UI yet.
 */
export function resetGlossarySightings(): void {
  if (typeof window === 'undefined') return;
  try {
    window.localStorage.removeItem(STORAGE_KEY);
  } catch {
    /* noop */
  }
}
