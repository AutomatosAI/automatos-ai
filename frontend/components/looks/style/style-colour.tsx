import { colourSwatches, contrastChecks, formatRatio, type ContrastCheck } from '../../../lib/looks/style'
import type { Look } from '../../../lib/looks/types'
import { colourStyle } from '../swatch-style'

import styles from './style.module.css'

/**
 * PRD-260 §4 — the style sheet's Colour section, ported from Automatos Play's
 * components/looks/style/StyleColour.tsx: a swatch for every colour the look
 * sets (name, value, role, and why a readable variant differs from its raw
 * colour), then the live contrast ratios with an AA verdict in a glyph and a
 * word, never colour alone. A chip's colour goes through swatch-style.ts's
 * colourStyle, the helper Make it yours draws its swatches with.
 */

export const COLOUR_LABEL = 'Colour'
export const CONTRAST_CAPTION = 'Contrast, live'
export const AA_PASS = 'AA pass'
export const AA_FAIL = 'AA fail'

function Verdict({ check }: { readonly check: ContrastCheck }) {
  return (
    <span className={styles.verdict} data-passes={check.passes}>
      <span aria-hidden="true">{check.passes ? '✓' : '✕'}</span> {check.passes ? AA_PASS : AA_FAIL}
    </span>
  )
}

function ContrastTable({ look }: { readonly look: Look }) {
  return (
    <table className={styles.contrast}>
      <caption className={styles.caption}>{CONTRAST_CAPTION}</caption>
      <thead>
        <tr>
          <th scope="col">Pair</th>
          <th scope="col">Colours</th>
          <th scope="col">Ratio</th>
          <th scope="col">WCAG AA</th>
        </tr>
      </thead>
      <tbody>
        {contrastChecks(look).map((check) => (
          <tr key={check.label}>
            <th scope="row">{check.label}</th>
            <td>
              {check.foreground} on {check.background}
            </td>
            <td>{formatRatio(check.ratio)}</td>
            <td>
              <Verdict check={check} />
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  )
}

export function StyleColour({ look }: { readonly look: Look }) {
  return (
    <section className={styles.section} aria-label={COLOUR_LABEL}>
      <h2 className={styles.label}>{COLOUR_LABEL}</h2>
      <ul className={styles.swatches}>
        {colourSwatches(look).map((swatch) => (
          <li key={swatch.name} className={styles.swatch}>
            <span className={styles.swatchChip} style={colourStyle(swatch.value)} aria-hidden="true" />
            <span className={styles.tokenName}>{swatch.name}</span>
            <span className={styles.tokenValue}>{swatch.value}</span>
            <span className={styles.tokenRole}>{swatch.role}</span>
            {swatch.note && <span className={styles.tokenNote}>{swatch.note}</span>}
          </li>
        ))}
      </ul>
      <ContrastTable look={look} />
    </section>
  )
}
