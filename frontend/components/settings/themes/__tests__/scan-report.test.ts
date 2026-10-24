/**
 * PRD-261 A-07d / M-03 — the scan report of a private upload, as `POST /api/themes/import` sends it
 * (`review_stage.member_view` of the stored report).
 *
 * - M-03: the preset line for a theme painted from its looks. The contrast check
 *   (services/themes/mobile_contrast.py) measures the prepaint table's defaults first, as id
 *   `boot/prepaint.json#defaults` and kind `defaults`; they are no preset, so the line leaves them
 *   out and keeps a preset that is itself named `defaults` (a valid look id).
 * - The review stage's report (`services/themes/review_stage.py`, v2): each check is shown by its
 *   `level` with its own label, never by a raw key or "unknown"; what it found is listed under it;
 *   the verdict is in words; and when the verdict is worse than every check shown, one line says
 *   the AI code read (which member_view leaves out) decided it.
 * - A report stored before the review stage (`scan.py`, v1) is still read by its `status`.
 *
 * The keys, levels and verdicts are read from the Python sources, so a check the server adds
 * without a label here fails this file.
 */
import { readFileSync } from 'fs'
import path from 'path'

import { render, screen, within } from '@testing-library/react'
import { createElement } from 'react'
import { describe, expect, it } from 'vitest'

import { ScanReport, hiddenCheckNote, scanChecksOf, verdictLabel } from '../scan-report'
import {
  CHECK_LABELS,
  HIDDEN_AI_BLOCKED,
  HIDDEN_AI_REVIEW,
  LEVEL_LABELS,
  STATUS_LABELS,
  VERDICT_LABELS,
} from '../theme-labels'

const THEMES_PY = path.resolve(__dirname, '..', '..', '..', '..', '..', 'orchestrator', 'services', 'themes')

function pySource(name: string): string {
  return readFileSync(path.join(THEMES_PY, name), 'utf8')
}

function quoted(text: string): string[] {
  return [...text.matchAll(/"(\w+)"/g)].map((match) => match[1])
}

/** The values of a Python tuple assignment such as `A, B = "x", "y"`, by its first name. */
function assigned(source: string, firstName: string): string[] {
  const line = new RegExp(`^${firstName}\\b[^=\\n]*=([^\\n]*)$`, 'm').exec(source)
  return line ? quoted(line[1]) : []
}

/** Every check key the review stage (v2) or the early scan (v1) stores. */
function storedCheckKeys(): string[] {
  const stage = pySource('review_stage.py')
  const reader = assigned(stage, 'READER_CHECKS')
  const added = [...stage.matchAll(/"(\w+)": \w+_check\(/g)].map((match) => match[1])
  const early = [...pySource('scan.py').matchAll(/^ {12}"(\w+)": /gm)].map((match) => match[1])
  return [...new Set([...reader, ...added, ...early])]
}

// sdk_uses.py UNKNOWN_NAMES and review_stage.py STALE_CLONES, as the server formats them.
const UNKNOWN_NAMES = "The theme imports names this host's SDK doesn't export, so its code can't load: useSecretThing."
const STALE_CLONES =
  '1 cloned page(s) copied a core file this host has changed since, so the clone may miss fixes and features the core page has now.'
const CONTRAST_NOT_RUN =
  'Not measured: the package declares no text and ground colour pairs for its looks, so the server has nothing to check AA contrast against.'

const subject = (id: string, kind: string) => ({ id, kind, status: 'pass', missing: [], pairs: [] })

const REPORT = {
  schema: 'automatos-theme-scan/v2',
  verdict: 'pass',
  checks: {
    looks: { level: 'pass', status: 'pass', presets: ['defaults', 'night'] },
    contrast: {
      source: 'fromLooks',
      status: 'pass',
      level: 'pass',
      problems: [],
      presets: [
        subject('boot/prepaint.json#defaults', 'defaults'),
        subject('defaults', 'preset'),
        subject('night', 'preset'),
      ],
    },
  },
}

/** A local-edition upload's member view: every check shown passes, the hidden AI code read did not run. */
function memberView(verdict: string, overrides: Record<string, unknown> = {}) {
  return {
    schema: 'automatos-theme-scan/v2',
    stage: 'C-02a',
    verdict,
    checks: {
      sizes: { status: 'pass', packageBytes: 20480, codeBytes: 10240, entryBytes: 2048, largestClipBytes: 0, limits: {}, level: 'pass' },
      cssScope: { status: 'pass', path: 'dist/theme.css', level: 'pass' },
      looks: { status: 'pass', presets: ['default', 'night'], level: 'pass' },
      contrast: {
        status: 'not_run', reason: CONTRAST_NOT_RUN, level: 'not_measured',
        presets: [{ id: 'default', status: 'not_run' }, { id: 'night', status: 'not_run' }],
      },
      uses: {
        level: 'pass', status: 'checked', problems: [], unknownNames: [], usedAreas: ['chat', 'looks'],
        usedNames: ['useChatController', 'useLook'], undeclaredAreas: [], notAreas: [], declaredUnused: [], wideImport: false,
      },
      clonedFrom: { level: 'pass', clones: [], stale: [], reason: null },
      ...overrides,
    },
  }
}

const USES_ERROR = {
  level: 'error', status: 'checked', problems: [UNKNOWN_NAMES], unknownNames: ['useSecretThing'], usedAreas: ['chat'],
  usedNames: ['useChatController'], undeclaredAreas: [], notAreas: [], declaredUnused: [], wideImport: false,
}
const CLONES_STALE = {
  level: 'warning',
  clones: ['/agents', '/legacy'],
  stale: [
    { route: '/agents', path: 'frontend/app/agents/page.tsx', sha: 'a'.repeat(64), releaseSha: 'b'.repeat(64) },
    { route: '/legacy', path: 'frontend/app/legacy/page.tsx', sha: 'c'.repeat(64), releaseSha: null },
  ],
  reason: STALE_CLONES,
}

function checkOf(report: unknown, key: string) {
  const check = scanChecksOf(report).find((item) => item.key === key)
  if (!check) throw new Error(`no ${key} check`)
  return check
}

function renderReport(report: unknown, verdict: string | null) {
  render(createElement(ScanReport, { report, verdict }))
  return screen.getByTestId('theme-scan-report')
}

function badgeOf(key: string): HTMLElement {
  const badge = screen.getByTestId(`theme-scan-check-${key}`).querySelector('[data-result]')
  if (!(badge instanceof HTMLElement)) throw new Error(`no ${key} badge`)
  return badge
}

describe('PRD-261 M-03 — the scan report lists presets, never the table defaults', () => {
  it('leaves the table defaults out of the contrast line and keeps a preset named defaults', () => {
    expect(checkOf(REPORT, 'contrast').detail).toBe('Presets: defaults, night')
  })

  it('lists the looks check presets as they are', () => {
    expect(checkOf(REPORT, 'looks').detail).toBe('Presets: defaults, night')
  })

  it("lists a mobile block's schemes when the contrast check measured its tokens", () => {
    const tokens = { level: 'pass', status: 'pass', source: 'tokens', problems: [], schemes: [subject('light', 'scheme'), subject('dark', 'scheme')] }
    expect(checkOf(memberView('pass', { contrast: tokens }), 'contrast').detail).toBe('Schemes: light, dark')
  })
})

describe('the words match the server (review_stage.py, scan.py)', () => {
  it('names every check the review stage or the early scan can store', () => {
    const keys = storedCheckKeys()
    expect(keys).toEqual(
      expect.arrayContaining(['sizes', 'cssScope', 'looks', 'contrast', 'uses', 'clonedFrom', 'llm', 'automated']),
    )
    for (const key of keys) {
      expect(CHECK_LABELS[key], key).toBeTruthy()
      expect(CHECK_LABELS[key]).not.toBe(key)
    }
    expect(CHECK_LABELS.uses).toBe('SDK areas')
    expect(CHECK_LABELS.clonedFrom).toBe('Cloned core pages')
  })

  it('labels every level and verdict the review stage gives, and the not_run verdict of an older row', () => {
    const stage = pySource('review_stage.py')
    const levels = assigned(stage, 'LEVEL_PASS')
    expect(levels).toEqual(['pass', 'warning', 'error', 'not_measured'])
    expect(Object.keys(LEVEL_LABELS).sort()).toEqual([...levels].sort())
    const verdicts = [...assigned(stage, 'VERDICT_PASS'), ...assigned(pySource('scan.py'), 'VERDICT_NOT_RUN')]
    expect(Object.keys(VERDICT_LABELS).sort()).toEqual([...verdicts].sort())
  })
})

describe("the review stage's report (v2): each check by its level", () => {
  it('labels every check and never shows a raw key, "unknown" or "checked"', () => {
    renderReport(memberView('review_required'), 'review_required')
    for (const key of Object.keys(memberView('pass').checks)) {
      expect(within(screen.getByTestId(`theme-scan-check-${key}`)).getByText(CHECK_LABELS[key])).toBeInTheDocument()
    }
    const report = screen.getByTestId('theme-scan-report')
    expect(report).not.toHaveTextContent(/\bunknown\b/i)
    expect(report).not.toHaveTextContent(/\bchecked\b/)
    expect(report).not.toHaveTextContent('clonedFrom')
  })

  it('clonedFrom (a level, no status) reads Passed, and uses reads by its level, not its status', () => {
    const report = memberView('review_required')
    expect(checkOf(report, 'clonedFrom')).toMatchObject({ result: 'pass', resultLabel: 'Passed', label: 'Cloned core pages' })
    expect(checkOf(report, 'uses')).toMatchObject({ result: 'pass', resultLabel: 'Passed', label: 'SDK areas' })
    expect(checkOf(report, 'uses').detail).toBe('Areas used: chat, looks')
  })

  it('a badge per level: Passed solid, Not measured muted, Warning amber, Error red, each distinct', () => {
    renderReport(memberView('blocked', { uses: USES_ERROR, clonedFrom: CLONES_STALE }), 'blocked')

    expect(badgeOf('sizes')).toHaveTextContent(LEVEL_LABELS.pass)
    expect(badgeOf('sizes').className).toContain('bg-primary')
    expect(badgeOf('contrast')).toHaveTextContent(LEVEL_LABELS.not_measured)
    expect(badgeOf('contrast').className).toContain('bg-secondary')
    expect(badgeOf('clonedFrom')).toHaveTextContent(LEVEL_LABELS.warning)
    expect(badgeOf('clonedFrom').className).toContain('bg-warning')
    expect(badgeOf('uses')).toHaveTextContent(LEVEL_LABELS.error)
    expect(badgeOf('uses').className).toContain('bg-destructive')
    expect(badgeOf('uses').className).not.toBe(badgeOf('clonedFrom').className)
    expect(screen.getByTestId('theme-scan-report').innerHTML).not.toContain('dark:')
  })

  it("shows what a check found: uses' problems with the unknown names, and each stale clone", () => {
    renderReport(memberView('blocked', { uses: USES_ERROR, clonedFrom: CLONES_STALE }), 'blocked')

    expect(screen.getByTestId('theme-scan-problems-uses')).toHaveTextContent(UNKNOWN_NAMES)
    expect(screen.getByTestId('theme-scan-problems-uses')).toHaveTextContent('useSecretThing')
    const clones = within(screen.getByTestId('theme-scan-check-clonedFrom'))
    expect(clones.getByText(STALE_CLONES)).toBeInTheDocument()
    expect(clones.getByText('Cloned pages: /agents, /legacy')).toBeInTheDocument()
    const stale = screen.getByTestId('theme-scan-problems-clonedFrom')
    expect(stale).toHaveTextContent('/agents copies frontend/app/agents/page.tsx, which this release has changed since.')
    expect(stale).toHaveTextContent('/legacy copies frontend/app/legacy/page.tsx, which this release no longer ships.')
  })

  it("keeps the server's reason for an unmeasured check, and says when a level-carrying cssScope found no style sheet", () => {
    renderReport(memberView('pass', { cssScope: { status: 'absent', path: 'dist/theme.css', level: 'pass' } }), 'pass')
    expect(within(screen.getByTestId('theme-scan-check-contrast')).getByText(CONTRAST_NOT_RUN)).toBeInTheDocument()
    const css = within(screen.getByTestId('theme-scan-check-cssScope'))
    expect(css.getByText(LEVEL_LABELS.pass)).toBeInTheDocument()
    expect(css.getByText(STATUS_LABELS.absent)).toBeInTheDocument()
  })
})

describe('the verdict in words', () => {
  it.each([
    ['pass', 'Passed'],
    ['review_required', 'Needs a reviewer'],
    ['blocked', 'Blocked'],
    ['not_run', 'Not run'],
  ])('%s reads "%s"', (verdict, words) => {
    expect(verdictLabel(verdict)).toBe(words)
  })

  it('a missing verdict reads Not run, an unknown one as it is', () => {
    expect(verdictLabel(null)).toBe('Not run')
    expect(verdictLabel('later')).toBe('later')
  })

  it('the report prints the words, never the raw verdict', () => {
    renderReport(memberView('review_required'), 'review_required')
    expect(screen.getByTestId('theme-scan-verdict')).toHaveTextContent('Verdict: Needs a reviewer')
    expect(screen.getByTestId('theme-scan-report')).not.toHaveTextContent('review_required')
  })
})

describe('the AI code read member_view leaves out', () => {
  it('says it decided a review_required verdict when every check shown passed or was not measured', () => {
    renderReport(memberView('review_required'), 'review_required')
    expect(screen.getByTestId('theme-scan-hidden-check')).toHaveTextContent(HIDDEN_AI_REVIEW)
  })

  it('says nothing when a check shown has the warning that explains it', () => {
    const report = memberView('review_required', { clonedFrom: CLONES_STALE })
    expect(hiddenCheckNote('review_required', scanChecksOf(report))).toBeNull()
    renderReport(report, 'review_required')
    expect(screen.queryByTestId('theme-scan-hidden-check')).toBeNull()
  })

  it('says it decided a blocked verdict when no check shown has an error', () => {
    const shownWarning = scanChecksOf(memberView('blocked', { clonedFrom: CLONES_STALE }))
    expect(hiddenCheckNote('blocked', shownWarning)).toBe(HIDDEN_AI_BLOCKED)
    expect(hiddenCheckNote('blocked', scanChecksOf(memberView('blocked', { uses: USES_ERROR })))).toBeNull()
  })

  it('says nothing for a pass, a not_run verdict or a report with no checks', () => {
    expect(hiddenCheckNote('pass', scanChecksOf(memberView('pass')))).toBeNull()
    expect(hiddenCheckNote('not_run', scanChecksOf(memberView('not_run')))).toBeNull()
    expect(hiddenCheckNote('review_required', scanChecksOf(null))).toBeNull()
  })

  it('names the AI code read and why it may not have run, and that a private upload installs anyway', () => {
    expect(HIDDEN_AI_REVIEW).toContain('AI code read')
    expect(HIDDEN_AI_REVIEW).toContain('Anthropic API key')
    expect(HIDDEN_AI_REVIEW).toContain('a private upload installs here whatever the verdict')
    // member_view hides exactly the llm check (review_stage.ADMIN_ONLY_CHECKS), which the line stands for.
    expect(pySource('review_stage.py')).toContain('ADMIN_ONLY_CHECKS = frozenset({"llm"})')
  })
})

describe('a report stored before the review stage (scan.py, v1): each check by its status', () => {
  const EARLY = {
    schema: 'automatos-theme-scan/v1',
    verdict: 'not_run',
    checks: {
      sizes: { status: 'pass', packageBytes: 20480, codeBytes: 10240, entryBytes: 2048 },
      cssScope: { status: 'absent', path: 'dist/theme.css' },
      contrast: { status: 'not_run', reason: CONTRAST_NOT_RUN, presets: [{ id: 'default', status: 'not_run' }] },
      automated: { status: 'not_run', reason: 'The automated review stage is not part of this release yet, so it did not run.' },
    },
  }

  it('reads the status labels, and a check with neither level nor status as having no result', () => {
    expect(checkOf(EARLY, 'sizes')).toMatchObject({ result: 'pass', resultLabel: STATUS_LABELS.pass })
    expect(checkOf(EARLY, 'cssScope')).toMatchObject({ result: 'absent', resultLabel: STATUS_LABELS.absent, detail: null })
    expect(checkOf(EARLY, 'automated')).toMatchObject({ label: 'Automated review', resultLabel: STATUS_LABELS.not_run })
    expect(checkOf({ checks: { later: {} } }, 'later')).toMatchObject({ label: 'later', result: 'unknown', resultLabel: 'No result' })
  })

  it('renders them muted, with no hidden-check line', () => {
    renderReport(EARLY, 'not_run')
    expect(badgeOf('automated')).toHaveTextContent('Not run')
    expect(badgeOf('automated').className).toContain('bg-secondary')
    expect(screen.getByTestId('theme-scan-verdict')).toHaveTextContent('Verdict: Not run')
    expect(screen.queryByTestId('theme-scan-hidden-check')).toBeNull()
  })
})
