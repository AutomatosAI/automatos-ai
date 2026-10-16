// PRD-261 A-01: prints the theme-host records (<dir>/*.json, written by specs/record.spec.ts) as
// one Markdown table, one row per Playwright project, for the job summary. An empty or missing
// folder prints "no records".
//
// Usage (from e2e/browser): node scripts/summarise-records.mjs records >> "$GITHUB_STEP_SUMMARY"

import fs from 'node:fs'
import path from 'node:path'

const DEFAULT_DIR = 'records'
const RECORD_FILE = /^(.+)-(app|integrity)\.json$/
const COLUMNS = ['Project', 'Browser version', 'Import map supported', 'Integrity', 'React live', 'React expected', 'Next']
const WEBKIT_NOTE = "Playwright's WebKit is a WebKit build, not shipping Safari."

function readRecords(dir) {
  if (!fs.existsSync(dir)) return new Map()
  const byProject = new Map()
  for (const name of fs.readdirSync(dir).sort()) {
    const match = RECORD_FILE.exec(name)
    if (!match) continue
    const record = JSON.parse(fs.readFileSync(path.join(dir, name), 'utf8'))
    byProject.set(match[1], { ...(byProject.get(match[1]) ?? {}), [match[2]]: record })
  }
  return byProject
}

function cell(value) {
  if (value === undefined || value === null || value === '') return 'n/a'
  // Backslashes first, so a value's own backslash cannot cancel the escape on a pipe after it.
  return String(value).replace(/\\/g, '\\\\').replace(/\|/g, '\\|').replace(/\r?\n/g, ' ')
}

function row(project, { app = {}, integrity = {} }) {
  const values = [
    project,
    app.browserVersion ?? integrity.browserVersion,
    app.importMapSupported,
    integrity.integrity,
    app.reactLive,
    app.reactExpected,
    app.next,
  ]
  return `| ${values.map(cell).join(' | ')} |`
}

function main() {
  const dir = process.argv[2] ?? DEFAULT_DIR
  const byProject = readRecords(dir)
  if (byProject.size === 0) {
    console.log('no records')
    return
  }
  const lines = [
    '### Theme host record (PRD-261 A-01)',
    '',
    `| ${COLUMNS.join(' | ')} |`,
    `| ${COLUMNS.map(() => '---').join(' | ')} |`,
    ...Array.from(byProject.entries(), ([project, records]) => row(project, records)),
    '',
    WEBKIT_NOTE,
  ]
  console.log(lines.join('\n'))
}

main()
