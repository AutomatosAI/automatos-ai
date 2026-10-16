import fs from 'fs'
import path from 'path'
import type { TestInfo } from '@playwright/test'

// PRD-261 A-01: the measured facts a run records (React version, import-map support) as one JSON
// file per project and kind under RECORD_DIR, also attached to the test. The workflow summarises
// them (scripts/summarise-records.mjs) and uploads the folder.

const DEFAULT_RECORD_DIR = 'records'
const JSON_INDENT = 2

export type RecordKind = 'app' | 'integrity'

export async function writeRecord(testInfo: TestInfo, kind: RecordKind, record: Readonly<Record<string, unknown>>): Promise<void> {
  const dir = path.resolve(process.env.RECORD_DIR ?? DEFAULT_RECORD_DIR)
  const body = `${JSON.stringify(record, null, JSON_INDENT)}\n`
  fs.mkdirSync(dir, { recursive: true })
  fs.writeFileSync(path.join(dir, `${testInfo.project.name}-${kind}.json`), body)
  await testInfo.attach(`${kind}-record`, { body, contentType: 'application/json' })
}
