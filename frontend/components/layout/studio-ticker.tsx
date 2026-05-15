'use client';

/**
 * StudioTicker — TICK-A (editorial slim 30px paper strip).
 *
 * Renders 7 KPIs in mono with semantic tones (olive ok, burnt orange err,
 * navy info), plus a live dot and a right-aligned clock. Lives directly
 * below the header. PRD shell rollout reference.
 *
 * Default data is the round-2 pilot snapshot. Wire real metrics via the
 * `cells` prop when the metrics API is online.
 */

export type TickerTone = 'ok' | 'err' | 'info' | null;

export interface TickerCell {
  /** Short uppercase label */
  label: string;
  /** Value (formatted) */
  value: string;
  /** Tone tint applied to the value */
  tone?: TickerTone;
  /** Optional delta footnote (e.g. "↑ vs 41%") */
  delta?: string;
}

export interface StudioTickerProps {
  cells?: TickerCell[];
  /** Right-aligned clock string. Pass an ISO string or pre-formatted text. */
  clock?: string;
  /** Whether to show the leading LIVE dot. Defaults true. */
  live?: boolean;
}

const DEFAULT_CELLS: TickerCell[] = [
  { label: 'UPTIME', value: '99.84%', tone: 'ok' },
  { label: 'CACHE', value: '68%', tone: 'ok', delta: '↑ vs 41%' },
  { label: '$/DEC', value: '$0.0027', tone: 'ok', delta: '↓ vs $0.0091' },
  { label: 'P50', value: '95ms', tone: 'ok', delta: '↓ vs 340ms' },
  { label: 'ERR/HR', value: '6', tone: 'err' },
  { label: 'T2.5', value: '988 hits' },
  { label: 'QUEUE', value: '14', tone: 'info' },
];

function formatClock(input?: string): string {
  if (input) return input;
  const now = new Date();
  const date = now.toISOString().slice(0, 10);
  const time = now.toLocaleTimeString('en-GB', {
    hour: '2-digit',
    minute: '2-digit',
    timeZone: 'Europe/Lisbon',
  });
  return `${date} · ${time} WET · tick 5s`;
}

export function StudioTicker({
  cells = DEFAULT_CELLS,
  clock,
  live = true,
}: StudioTickerProps) {
  return (
    <div className="sh-ticker" role="status" aria-label="Live operational metrics">
      {live && (
        <>
          <span className="sh-cell">
            <span
              style={{
                width: 7,
                height: 7,
                borderRadius: '50%',
                background: 'hsl(82 50% 22%)',
                display: 'inline-block',
                marginRight: 2,
              }}
            />
            <span className="sh-l" style={{ color: 'hsl(82 50% 22%)' }}>
              LIVE
            </span>
          </span>
          <span className="sh-divider" />
        </>
      )}
      {cells.map((cell) => (
        <span key={cell.label} className="sh-cell">
          <span className="sh-l">{cell.label}</span>
          <span className={`sh-v${cell.tone ? ' ' + cell.tone : ''}`}>{cell.value}</span>
          {cell.delta && <span className="sh-delta">{cell.delta}</span>}
        </span>
      ))}
      <span className="sh-clock">{formatClock(clock)}</span>
    </div>
  );
}
