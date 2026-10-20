'use client'

import { useState } from 'react'
import { HostLink, LOOKS_HREF } from '@automatos/theme-sdk'

import { rootBlock } from '../../../lib/looks/style'
import { PLAY_ONLY_HEADING } from '../screen-heading'
import { type Told, useNoticeSeq } from '../use-notice-seq'
import { usePlayLook } from '../use-play-look'

import { StyleColour } from './style-colour'
import { StyleComponents } from './style-components'
import { StyleMascots } from './style-mascots'
import { StyleMotion } from './style-motion'
import { StyleScale } from './style-scale'
import { StyleType } from './style-type'
import styles from './style.module.css'

/**
 * PRD-260 §4 — Play's live style sheet at /looks/style (the Ember style sheet,
 * mocks ember-12 and -13), ported from Automatos Play's
 * components/looks/style/StyleScreen.tsx: the active look's colours with live
 * contrast, its type, the spacing and radii, the motion, the components and
 * Auto's 12 states. Everything is computed from usePlayLook().look, so it redraws
 * as soon as the look changes anywhere in the app. Copy tokens puts the look's
 * variables on the clipboard as a :root block. app/looks/style/page.tsx shows
 * it through PlayOnlyRoute: under Play at >= 1024 px only, full bleed.
 * Until the stored look is read the frame shows only the heading and is
 * aria-busy, so the sheet never describes Ember in place of the user's look
 * (M72). Copy tokens' block names each face itself (lib/looks/style.ts
 * rootBlock), so it works outside Automatos. Each copy's outcome is numbered
 * (useNoticeSeq), so copying twice, or the same refusal twice, is announced
 * again (M22).
 */

/** The sheet's heading, and the title of the note /looks/style shows outside Play (PlayOnlyRoute). */
export const STYLE_HEADING = 'Style sheet'
export const COPY_TOKENS = 'Copy tokens'
export const BACK_TO_LOOKS = 'Back to Looks'
export const COPY_FAILED = "The tokens weren't copied. The browser blocked the clipboard."
/** What /looks/style says under Classic or Studio, beside Switch to Play (K9). */
export const STYLE_SHEET_NOTE = 'The style sheet is part of the Play theme.'

/** What the last copy did, kept with the block it copied: a change of look hides it. */
type CopyOutcome =
  | { readonly kind: 'done'; readonly block: string; readonly text: string }
  | { readonly kind: 'failed'; readonly block: string }

type CopyNotice = Told<CopyOutcome>

function useCopyTokens() {
  const { look } = usePlayLook()
  const block = rootBlock(look)
  const [notice, setNotice] = useState<CopyNotice | null>(null)
  const stamp = useNoticeSeq()

  const copy = async (): Promise<void> => {
    try {
      await navigator.clipboard.writeText(block)
      setNotice(stamp<CopyOutcome>({ kind: 'done', block, text: `Copied ${look.name}'s tokens as a :root block.` }))
    } catch (error) {
      console.error('[looks] copying the tokens failed', error)
      setNotice(stamp<CopyOutcome>({ kind: 'failed', block }))
    }
  }

  return { notice: notice?.block === block ? notice : null, copy }
}

/**
 * A success goes to a polite live region that is always present, its words in a span keyed by
 * the copy's seq; a failure is an alert keyed the same way: a repeat is a new node, told again.
 */
function CopyResult({ notice }: { readonly notice: CopyNotice | null }) {
  return (
    <>
      <p className={styles.done} aria-live="polite">
        {notice?.kind === 'done' && <span key={notice.seq}>{notice.text}</span>}
      </p>
      {notice?.kind === 'failed' && (
        <p key={notice.seq} className={styles.failed} role="alert">
          <span aria-hidden="true">✕</span> {COPY_FAILED}
        </p>
      )}
    </>
  )
}

export function StyleScreen() {
  const { look, loaded } = usePlayLook()
  const { notice, copy } = useCopyTokens()

  return (
    <div className={styles.page} aria-busy={loaded ? undefined : true}>
      <header className={styles.header}>
        <div className={styles.headingText}>
          <h1 className={styles.heading} tabIndex={-1} {...PLAY_ONLY_HEADING}>
            {STYLE_HEADING}
          </h1>
          {loaded && (
            <p className={styles.lead}>
              Every value here comes from {look.name}, the active look. Change the look and this page redraws.
            </p>
          )}
        </div>
        {loaded && (
          <div className={styles.actions}>
            <HostLink href={LOOKS_HREF} className={styles.linkButton}>
              {BACK_TO_LOOKS}
            </HostLink>
            <button type="button" className={styles.copyButton} onClick={() => void copy()}>
              {COPY_TOKENS}
            </button>
          </div>
        )}
      </header>
      <CopyResult notice={notice} />
      {loaded && (
        <div className={styles.sheet}>
          <StyleColour look={look} />
          <div className={styles.columns}>
            <StyleType look={look} />
            <div className={styles.stack}>
              <StyleScale />
              <StyleMotion />
            </div>
          </div>
          <StyleComponents />
          <StyleMascots look={look} />
        </div>
      )}
    </div>
  )
}
