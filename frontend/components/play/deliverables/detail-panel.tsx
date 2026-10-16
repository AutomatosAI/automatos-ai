'use client'

import Link from 'next/link'
import { useId } from 'react'

import { useApiFileDownload } from '@/hooks/use-api-file-download'
import type { Deliverable } from '@/hooks/use-deliverables-api'
import { deliverableDownloadUrl, deliverableFilename, DOWNLOAD_FAILED } from '@/lib/deliverables/download'
import { askAboutText, madeBy, sourceLabel, sourcesOf } from '@/lib/play/detail'
import { draftHref } from '@/lib/play/draft'
import { typeLabel } from '@/lib/play/shelf'

import page from './deliverable-detail.module.css'
import styles from './detail-panel.module.css'

/**
 * PRD-260 §4 — the side panel on a Deliverable's page (Ember mock 05), ported
 * from Automatos Play's components/deliverables/DetailPanel.tsx: Ask Auto about
 * this (a new chat with the Deliverable named in its message box: the core has
 * no way to attach one to a message, so the words alone, through
 * /chat?draft=, which Play's chat reads), Download when it has a file (the
 * core's authenticated download of the file's bytes, deliverableDownloadUrl),
 * About this Deliverable: its type, when it was made, by which agent (or the
 * core's word for an upload, a Templates document or a Socials post), and
 * where it came from (the core's source_type); and Sources · N when it carries
 * any (extra.sources, sourcesOf).
 */

export const ASK_AUTO = 'Ask Auto about this'
export const ABOUT = 'About this Deliverable'
export const DOWNLOAD = 'Download'
export const DOWNLOADING = 'Downloading…'
export const SOURCES = 'Sources'

export interface DetailPanelProps {
  readonly deliverable: Deliverable
  readonly kind: string
  readonly title: string
  /** "Today, 15:42", or "" when unknown. */
  readonly date: string
}

export function DetailPanel({ deliverable, kind, title, date }: DetailPanelProps) {
  const aboutId = useId()
  return (
    <aside className={styles.panel} aria-label={ABOUT}>
      <div className={styles.actions}>
        <Link href={draftHref(askAboutText(title))} className={page.primary}>
          {ASK_AUTO}
        </Link>
        <DownloadButton deliverable={deliverable} />
      </div>
      <section className={styles.panelSection} aria-labelledby={aboutId}>
        <h2 id={aboutId} className={styles.panelLabel}>
          {ABOUT}
        </h2>
        <Facts deliverable={deliverable} kind={kind} date={date} />
      </section>
      <Sources sources={sourcesOf(deliverable)} />
    </aside>
  )
}

/** "Sources · 3" and the list (a web address as its host); nothing when it carries none. */
function Sources({ sources }: { readonly sources: readonly string[] }) {
  const sourcesId = useId()
  if (sources.length === 0) return null
  return (
    <section className={styles.panelSection} aria-labelledby={sourcesId}>
      <h2 id={sourcesId} className={styles.panelLabel}>
        {`${SOURCES} · ${sources.length}`}
      </h2>
      <ul className={styles.sources}>
        {sources.map((source, index) => (
          <li key={`${index}-${source}`}>{source}</li>
        ))}
      </ul>
    </section>
  )
}

function DownloadButton({ deliverable }: { readonly deliverable: Deliverable }) {
  const { download, downloading } = useApiFileDownload(DOWNLOAD_FAILED)
  const url = deliverableDownloadUrl(deliverable)
  if (!url) return null
  return (
    <button
      type="button"
      className={page.secondary}
      onClick={() => void download(url, deliverableFilename(deliverable))}
      disabled={downloading}
    >
      {downloading ? DOWNLOADING : DOWNLOAD}
    </button>
  )
}

interface FactsProps {
  readonly deliverable: Deliverable
  readonly kind: string
  readonly date: string
}

function Facts({ deliverable, kind, date }: FactsProps) {
  const source = sourceLabel(deliverable.source_type)
  const facts = [
    { term: 'Type', value: typeLabel(kind) },
    { term: 'Made', value: date },
    { term: 'Made by', value: madeBy(deliverable.agent_name, deliverable.source_type) },
    { term: 'From', value: source },
  ].filter((fact) => fact.value)
  return (
    <dl className={styles.facts}>
      {facts.map((fact) => (
        <div key={fact.term} className={styles.fact}>
          <dt>{fact.term}</dt>
          <dd>{fact.value}</dd>
        </div>
      ))}
    </dl>
  )
}
