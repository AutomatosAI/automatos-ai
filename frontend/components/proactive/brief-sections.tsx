'use client'

/**
 * PRD-268 §6.11.2, K16 (US-C21c) — the brief's five sections, in Brief@1's order: needs you,
 * suggestions, today, since your last brief, and what couldn't be checked.
 *
 * Every word is the server's (K16). Each section's label and the empty sentence come from `brief.copy`
 * (US-C23), else from `brief.labels` (a brief built before W3, or one a theme hands the card: the SDK's
 * Brief carries labels only, K17). A section draws the rows the server sent (it folds at 5, never at 6)
 * and, when its exact count is more than those rows, the server's "Show N more" for it from
 * `labels.more`: the server filled that line from the same template `copy` carries, with the number its
 * fold hid. The card never fills the template itself: `counts.needs_you` is the badge's total, and a
 * viewer's fold counts only the rows they may see. A section with no rows draws nothing, and a brief
 * with no rows anywhere says so in the server's sentence.
 *
 * A pressed row is remembered here, on this card, so it shows "✓ {done_label}" in place. A press the
 * surface refuses (its handler rejects) leaves the row as it was, pressable again; the surface that
 * refused it says why.
 */
import { useState } from 'react'
import { BRIEF_SECTIONS, type Brief, type BriefRow, type BriefSection } from '@/lib/proactive/types'
import { Eyebrow, TEXT_BUTTON_CLASS } from './card-parts'
import { BriefRowItem, type BriefRowHandlers } from './brief-row'

export interface BriefSectionsProps extends BriefRowHandlers {
  readonly brief: Brief
  /** "Show N more" was pressed for a folded section; without it, the line is drawn as text. */
  readonly onShowMore?: (section: BriefSection) => void
}

type Keys = ReadonlySet<string>

const keyOf = (section: BriefSection, row: BriefRow): string => `${section}:${row.id}`
const withKey = (key: string) => (keys: Keys): Keys => new Set([...Array.from(keys), key])
const withoutKey = (key: string) => (keys: Keys): Keys => new Set(Array.from(keys).filter((kept) => kept !== key))

function rowsOf(brief: Brief, section: BriefSection): readonly BriefRow[] {
  return brief.sections?.[section] ?? []
}

/** How many rows a section folded away: its exact count less the rows shown (`unchecked` has no count). */
export function foldedCount(brief: Brief, section: BriefSection): number {
  if (section === 'unchecked') return 0
  const count = brief.counts?.[section]
  return typeof count === 'number' ? Math.max(count - rowsOf(brief, section).length, 0) : 0
}

/** A section's label: the brief's copy, else its labels. */
function sectionLabel(brief: Brief, section: BriefSection): string | undefined {
  return brief.copy?.section_labels?.[section] ?? brief.labels?.sections?.[section]
}

/** The sentence a brief with no rows says: the copy's, else the labels'. */
function emptySentence(brief: Brief): string | null | undefined {
  return brief.copy?.empty_sentence ?? brief.labels?.empty
}

interface SectionProps extends BriefSectionsProps {
  readonly section: BriefSection
  readonly pressed: Keys
  readonly pending: Keys
  readonly onPress: (row: BriefRow, section: BriefSection) => void
}

function ShowMore({ section, text, onShowMore }: { readonly section: BriefSection; readonly text: string; readonly onShowMore?: (section: BriefSection) => void }) {
  if (!onShowMore) return <p className="text-sm text-muted-foreground">{text}</p>
  return (
    <button type="button" onClick={() => onShowMore(section)} className={TEXT_BUTTON_CLASS}>
      {text}
    </button>
  )
}

function Section({ brief, section, pressed, pending, onPress, onAction, onOpen, onShowMore }: SectionProps) {
  const rows = rowsOf(brief, section)
  if (rows.length === 0) return null
  const label = sectionLabel(brief, section)
  const count = section === 'unchecked' ? null : brief.counts?.[section]
  const more = foldedCount(brief, section) > 0 ? brief.labels?.more?.[section] : undefined
  return (
    <section aria-label={label || undefined} data-section={section} className="space-y-1">
      {label ? (
        <Eyebrow>
          {label}
          {typeof count === 'number' ? (
            <span data-section-count="">
              <span aria-hidden="true"> · </span>
              {count}
            </span>
          ) : null}
        </Eyebrow>
      ) : null}
      <ul className="divide-y divide-border">
        {rows.map((row) => (
          <BriefRowItem
            key={keyOf(section, row)}
            row={row}
            section={section}
            pressed={pressed.has(keyOf(section, row))}
            pending={pending.has(keyOf(section, row))}
            onPress={onPress}
            onAction={onAction}
            onOpen={onOpen}
          />
        ))}
      </ul>
      {more ? <ShowMore section={section} text={more} onShowMore={onShowMore} /> : null}
    </section>
  )
}

/** The brief's sections with their rows, or the server's empty sentence when there are none. */
export function BriefSections(props: BriefSectionsProps) {
  const { brief, onAction } = props
  const [pressed, setPressed] = useState<Keys>(() => new Set())
  const [pending, setPending] = useState<Keys>(() => new Set())

  const onPress = (row: BriefRow, section: BriefSection) => {
    if (!onAction) return
    const key = keyOf(section, row)
    setPending(withKey(key))
    Promise.resolve()
      .then(() => onAction(row, section))
      .then(
        () => {
          setPressed(withKey(key))
          setPending(withoutKey(key))
        },
        () => setPending(withoutKey(key)),
      )
  }

  if (BRIEF_SECTIONS.every((section) => rowsOf(brief, section).length === 0)) {
    const empty = emptySentence(brief)
    return empty ? (
      <p data-ai-generated="true" data-brief-empty="" className="text-sm text-muted-foreground">
        {empty}
      </p>
    ) : null
  }
  return (
    <div className="space-y-4">
      {BRIEF_SECTIONS.map((section) => (
        <Section key={section} {...props} section={section} pressed={pressed} pending={pending} onPress={onPress} />
      ))}
    </div>
  )
}
