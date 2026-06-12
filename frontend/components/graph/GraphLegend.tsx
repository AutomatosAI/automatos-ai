'use client'

import { Layers, ChevronDown } from 'lucide-react'

/**
 * Collapsible legend / filter-chip overlay for the GraphView shell
 * (PRD-165 S1, BINDING Q25: "chips collapsed by default"). The owner's
 * complaint was that filter chips ate the screen — so this defaults to a
 * single small "Legend" pill and only expands the chip rows on demand. The
 * collapsed state is persisted by the parent via useGraphPrefs.
 *
 * Engine-agnostic: any surface (KG types/relations, codegraph symbol kinds,
 * mission task states) passes its own sections of chips.
 */

export interface LegendChip {
  key: string
  label: string
  color: string
  count?: number
  /** Struck-through + greyed when true. */
  hidden?: boolean
}

export interface LegendSection {
  title: string
  chips: LegendChip[]
  /** Omit to render a non-interactive key (swatch + label only). */
  onToggle?: (key: string) => void
}

interface GraphLegendProps {
  sections: LegendSection[]
  collapsed: boolean
  onCollapsedChange: (collapsed: boolean) => void
  /** Override positioning; defaults to bottom-right overlay. */
  className?: string
}

const HIDDEN_SWATCH = '#52525b'

export function GraphLegend({
  sections,
  collapsed,
  onCollapsedChange,
  className,
}: GraphLegendProps) {
  const hasChips = sections.some((s) => s.chips.length > 0)
  if (!hasChips) return null

  const position = className ?? 'absolute bottom-3 right-3 z-10'

  if (collapsed) {
    return (
      <button
        type="button"
        onClick={() => onCollapsedChange(false)}
        className={`${position} flex items-center gap-1.5 px-2.5 py-1 rounded-md text-xs border border-white/15 bg-black/50 backdrop-blur-sm text-muted-foreground hover:text-foreground hover:bg-black/60 transition`}
        title="Show legend"
        aria-expanded={false}
      >
        <Layers className="w-3.5 h-3.5" />
        Legend
      </button>
    )
  }

  return (
    <div
      className={`${position} flex flex-col items-end gap-1.5 max-w-[65%] rounded-lg border border-white/10 bg-black/50 backdrop-blur-sm p-2`}
    >
      <button
        type="button"
        onClick={() => onCollapsedChange(true)}
        className="flex items-center gap-1 self-end text-[10px] uppercase tracking-wider text-muted-foreground/70 hover:text-foreground transition"
        title="Hide legend"
        aria-expanded={true}
      >
        Legend
        <ChevronDown className="w-3 h-3" />
      </button>

      {sections.map((section) =>
        section.chips.length === 0 ? null : (
          <div key={section.title} className="flex flex-wrap gap-1.5 justify-end">
            <span className="text-[10px] uppercase tracking-wider text-muted-foreground/70 self-center mr-1">
              {section.title}
            </span>
            {section.chips.map((chip) => {
              const cls = `flex items-center gap-1.5 px-2 py-1 rounded-md text-xs border transition ${
                chip.hidden
                  ? 'bg-black/40 border-white/10 text-muted-foreground/60 line-through'
                  : 'bg-white/10 border-white/20 text-foreground'
              }`
              const title =
                chip.count != null
                  ? `${chip.count.toLocaleString()} ${chip.label.toLowerCase()}`
                  : chip.label
              const inner = (
                <>
                  <span
                    className="w-2 h-2 rounded-full"
                    style={{ backgroundColor: chip.hidden ? HIDDEN_SWATCH : chip.color }}
                  />
                  {chip.label}
                  {chip.count != null && (
                    <span className="text-[10px] text-muted-foreground">
                      {chip.count.toLocaleString()}
                    </span>
                  )}
                </>
              )
              return section.onToggle ? (
                <button
                  key={chip.key}
                  type="button"
                  onClick={() => section.onToggle!(chip.key)}
                  className={cls}
                  title={title}
                >
                  {inner}
                </button>
              ) : (
                <span key={chip.key} className={cls} title={title}>
                  {inner}
                </span>
              )
            })}
          </div>
        ),
      )}
    </div>
  )
}
