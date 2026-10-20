import chat from '../../play/chat/chat.module.css'
import composer from '../../play/chat/composer.module.css'
import { JUST_NOW, OPEN } from '../../play/chat/deliverable-card'
import { MARK_DONE, MARK_FAILED } from '../../play/chat/labels'
import { BADGE_DONE, RECEIPTS_TITLE } from '../../play/chat/receipts'
import turn from '../../play/chat/turn.module.css'

import gallery from './gallery.module.css'
import styles from './style.module.css'

/**
 * PRD-260 §4 — the style sheet's Components gallery, ported from Automatos
 * Play's components/looks/style/StyleComponents.tsx and drawn with Play's own
 * chat styles (components/play/chat/*.module.css), so it shows exactly what
 * the active look produces: your message and Auto's answer, tool lines
 * (running, done, failed), "What Auto did" (a write tagged Done), a
 * Deliverable card, filter chips (on, off, focus) and the message box with
 * Send and Stop. It is a picture: nothing here is a real button or link.
 */

export const COMPONENTS_LABEL = 'Components'
export const SAMPLE_TITLE = 'Laptop shortlist'
export const MESSAGE_PLACEHOLDER = 'Message Auto'
/**
 * The sample card's kind, in the words the core's Deliverable cards use (KIND_WORDS.document in
 * lib/chat/deliverables.ts): copied into Play, since Play imports no core module (PRD-261 A-14d);
 * components/looks/__tests__/copied-helpers.test.tsx holds it equal.
 */
export const SAMPLE_KIND = 'Document'

function Messages() {
  return (
    <figure className={gallery.component}>
      <figcaption className={gallery.componentLabel}>message · yours outlined, Auto&apos;s unboxed</figcaption>
      <div className={gallery.yours}>
        <p className={chat.bubble}>Make me a shortlist?</p>
      </div>
      <div className={chat.answer}>
        <div className={chat.markdown}>
          <p>
            Three make the cut. My pick is the <strong>Vivobook</strong>.
          </p>
        </div>
      </div>
    </figure>
  )
}

function Tools() {
  return (
    <figure className={gallery.component}>
      <figcaption className={gallery.componentLabel}>tool line · running / done / failed</figcaption>
      <ul className={turn.toolLines}>
        <li className={turn.toolLine} data-tone="running">
          <span className={turn.spinner} aria-hidden="true" />
          <span>Reading 4 pages</span>
        </li>
        <li className={turn.toolLine} data-tone="done">
          <span className={turn.mark} aria-hidden="true">
            {MARK_DONE}
          </span>
          <span>Read 4 pages · 3.4 s</span>
        </li>
        <li className={turn.toolLine} data-tone="failed">
          <span className={turn.mark} aria-hidden="true">
            {MARK_FAILED}
          </span>
          <span>Web search didn&apos;t answer</span>
        </li>
      </ul>
    </figure>
  )
}

function ReceiptsBox() {
  return (
    <figure className={gallery.component}>
      <figcaption className={gallery.componentLabel}>receipts</figcaption>
      <div className={turn.receipts}>
        <p className={turn.receiptsTitle}>{RECEIPTS_TITLE}</p>
        <ul className={turn.receiptList}>
          <li className={turn.receipt} data-tone="done">
            <span className={turn.mark} aria-hidden="true">
              {MARK_DONE}
            </span>
            <span>Searched the web · 5 results</span>
          </li>
          <li className={turn.receipt} data-tone="done">
            <span className={turn.mark} aria-hidden="true">
              {MARK_DONE}
            </span>
            <span>Made a report: {SAMPLE_TITLE}</span>
            <span className={turn.badge}>{BADGE_DONE}</span>
          </li>
        </ul>
      </div>
    </figure>
  )
}

function DeliverableSample() {
  return (
    <figure className={gallery.component}>
      <figcaption className={gallery.componentLabel}>Deliverable card</figcaption>
      <div className={turn.deliverable}>
        <span className={turn.deliverableTile} data-kind="document" aria-hidden="true" />
        <div className={turn.deliverableText}>
          <p className={turn.deliverableTitle}>{SAMPLE_TITLE}</p>
          <p className={turn.deliverableMeta}>
            {SAMPLE_KIND} · {JUST_NOW}
          </p>
        </div>
        <span className={turn.openButton}>{OPEN}</span>
      </div>
    </figure>
  )
}

function Chips() {
  return (
    <figure className={gallery.component}>
      <figcaption className={gallery.componentLabel}>filter chip · on / off / focus</figcaption>
      <div className={gallery.chips}>
        <span className={gallery.chip} data-state="on">
          All
        </span>
        <span className={gallery.chip} data-state="off">
          Reports
        </span>
        <span className={gallery.chip} data-state="focus">
          Images
        </span>
      </div>
    </figure>
  )
}

function MessageBoxes() {
  return (
    <figure className={gallery.component}>
      <figcaption className={gallery.componentLabel}>message box · Send / Stop</figcaption>
      <div className={composer.box}>
        <span className={gallery.placeholder}>{MESSAGE_PLACEHOLDER}</span>
        <span className={composer.send}>Send</span>
      </div>
      <div className={composer.box}>
        <span className={gallery.placeholder}>{MESSAGE_PLACEHOLDER}</span>
        <span className={composer.stop}>
          <span className={composer.stopIcon} aria-hidden="true" />
          Stop
        </span>
      </div>
    </figure>
  )
}

export function StyleComponents() {
  return (
    <section className={styles.section} aria-label={COMPONENTS_LABEL}>
      <h2 className={styles.label}>{COMPONENTS_LABEL}</h2>
      <div className={gallery.gallery}>
        <Messages />
        <Tools />
        <ReceiptsBox />
        <DeliverableSample />
        <Chips />
        <MessageBoxes />
      </div>
    </section>
  )
}
