/**
 * PRD-260 T-007b — one Deliverable's page as pure data (lib/play/detail.ts),
 * ported from Automatos Play's tests/deliverables/detail.test.ts onto the
 * core's Deliverable: how the content shows, the fence, the date, "Made by",
 * where it came from and the words "Ask Auto about this" puts in the box. Also
 * the file rule the core's preview shares (lib/deliverables/download.ts).
 */
import { describe, expect, it } from 'vitest'

import type { Deliverable } from '@/hooks/use-deliverables-api'
import { deliverableFilename, deliverableFileUrl } from '@/lib/deliverables/download'
import {
  ASK_TITLE_CHARS,
  askAboutText,
  contentView,
  detailDate,
  detailTitle,
  fenced,
  fileTypeOf,
  madeBy,
  madeLine,
  sourceLabel,
  unsupportedMessage,
} from '@/lib/play/detail'

const RAW = '/api/workspaces/ws-1/files/raw?path=outputs%2Fa%20b.pdf'
const CONTENT = '/api/workspaces/ws-1/files/content?path=notes.md'

function deliverable(fields: Partial<Deliverable>): Deliverable {
  return {
    id: 'd1',
    workspace_id: 'ws-1',
    source_type: 'chat',
    source_id: null,
    agent_id: null,
    agent_name: null,
    artifact_type: 'document',
    title: 'Plan',
    summary: null,
    storage_type: 'workspace',
    file_path: 'outputs/plan.md',
    file_name: null,
    file_type: null,
    file_size_bytes: null,
    preview_url: null,
    preview_type: null,
    extra: {},
    status: 'active',
    created_at: '2026-10-14T09:00:00+00:00',
    updated_at: '2026-10-14T09:00:00+00:00',
    ...fields,
  }
}

describe('contentView', () => {
  it('reads markdown and plain text as markdown, and a report or blog post whatever its file says', () => {
    expect(contentView(deliverable({ file_type: 'md', content: '# Hi' }))).toEqual({ kind: 'text', markdown: '# Hi', truncated: false })
    expect(contentView(deliverable({ file_type: null, file_path: 'notes', content: 'Hello' }))).toMatchObject({ kind: 'text', markdown: 'Hello' })
    expect(contentView(deliverable({ artifact_type: 'report', file_type: 'html', content: '<h1>x</h1>' }))).toMatchObject({
      kind: 'text',
      markdown: '<h1>x</h1>',
    })
    expect(contentView(deliverable({ artifact_type: 'blog_post', file_type: null, file_path: '', content: 'Post' }))).toMatchObject({
      kind: 'text',
      markdown: 'Post',
    })
  })

  it('wraps any other text file in a code block, and says when it was cut', () => {
    expect(contentView(deliverable({ file_type: 'csv', content: 'a,b', content_truncated: true }))).toEqual({
      kind: 'text',
      markdown: '```csv\na,b\n```',
      truncated: true,
    })
  })

  it('shows an image or a video from its file', () => {
    expect(contentView(deliverable({ artifact_type: 'image', content: null, content_url: RAW }))).toEqual({ kind: 'image', url: RAW })
    expect(contentView(deliverable({ artifact_type: 'video', preview_url: RAW }))).toEqual({ kind: 'video', url: RAW })
  })

  it("never shows a binary file's bytes as text", () => {
    expect(contentView(deliverable({ file_type: 'pdf', content_url: RAW, content: '%PDF-1.7 �' }))).toEqual({
      kind: 'file',
      reason: 'unsupported',
      url: RAW,
    })
    expect(contentView(deliverable({ file_type: '.XLSX', content: 'PK' }))).toMatchObject({ kind: 'file', reason: 'unsupported' })
  })

  it("can't preview an image or a video it has no file for", () => {
    expect(contentView(deliverable({ artifact_type: 'image' }))).toEqual({ kind: 'file', reason: 'unsupported', url: null })
    expect(contentView(deliverable({ artifact_type: 'video' }))).toEqual({ kind: 'file', reason: 'unsupported', url: null })
  })

  it("says when Automatos couldn't read a text file, and when it is empty", () => {
    expect(contentView(deliverable({ file_type: 'md', content: null, content_error: 'File not found', content_url: CONTENT }))).toEqual({
      kind: 'file',
      reason: 'unreadable',
      url: CONTENT,
    })
    expect(contentView(deliverable({ file_type: 'md', content: '  \n' }))).toEqual({ kind: 'empty' })
  })

  it('takes the file type from the name when the row has none', () => {
    expect(fileTypeOf({ file_type: null, file_name: 'Budget.XLSX', file_path: 'x' })).toBe('xlsx')
    expect(fileTypeOf({ file_type: '.Md', file_name: null, file_path: 'x' })).toBe('md')
    expect(fileTypeOf({ file_type: null, file_name: null, file_path: 'outputs.v2/README' })).toBe('')
  })
})

describe('fenced', () => {
  it("uses a fence longer than any backtick run inside, so the content can't close it", () => {
    expect(fenced('a ```` b', 'md')).toBe('`````md\na ```` b\n`````')
  })

  it("drops a file type that isn't a plain language name", () => {
    expect(fenced('x', 'c++ <b>')).toBe('```\nx\n```')
  })
})

describe("the page's words", () => {
  const now = new Date(2026, 9, 14, 16, 0)

  it('dates it with the day and the time', () => {
    expect(detailDate(new Date(2026, 9, 14, 15, 42).toISOString(), now)).toBe('Today, 15:42')
    expect(detailDate(new Date(2026, 9, 4, 9, 5).toISOString(), now)).toBe('4 Oct, 09:05')
    expect(detailDate(null, now)).toBe('')
    expect(detailDate('not a date', now)).toBe('')
  })

  it('credits the agent with Auto, or Auto alone', () => {
    expect(madeLine('Researcher')).toBe('Made by Auto with Researcher')
    expect(madeLine(null)).toBe('Made by Auto')
    expect(madeLine('auto')).toBe('Made by Auto')
    expect(madeBy('Researcher')).toBe('Researcher')
    expect(madeBy('  ')).toBe('Auto')
  })

  it('names where it came from in the canonical words', () => {
    expect(sourceLabel('chat')).toBe('A chat')
    expect(sourceLabel('mission')).toBe('A Mission')
    expect(sourceLabel('playbook')).toBe('A Playbook')
    expect(sourceLabel('TASK')).toBe('A Task')
    expect(sourceLabel('agent_run')).toBe('Agent run')
    expect(sourceLabel(null)).toBe('')
    expect(sourceLabel('constructor')).toBe('Constructor')
  })

  it('asks Auto about it by title, and names an untitled one', () => {
    expect(askAboutText('Laptop shortlist')).toBe('About my Deliverable "Laptop shortlist": ')
    const long = askAboutText('x'.repeat(ASK_TITLE_CHARS + 10))
    expect(long).toBe(`About my Deliverable "${'x'.repeat(ASK_TITLE_CHARS - 1)}…": `)
    expect(detailTitle({ title: '  ' })).toBe('Your Deliverable')
  })

  it("says which type can't be previewed", () => {
    expect(unsupportedMessage('spreadsheet')).toBe("Play can't preview this spreadsheet here.")
    expect(unsupportedMessage('slide')).toBe("Play can't preview this slide deck here.")
    expect(unsupportedMessage('toString')).toBe("Play can't preview this toString here.")
  })
})

describe("the file a Deliverable hands over (shared with the core's preview)", () => {
  it('is the content route, else the preview, else none', () => {
    expect(deliverableFileUrl({ content_url: RAW, preview_url: CONTENT })).toBe(RAW)
    expect(deliverableFileUrl({ content_url: null, preview_url: CONTENT })).toBe(CONTENT)
    expect(deliverableFileUrl({ content_url: undefined, preview_url: null })).toBeNull()
  })

  it("saves under the file's own name, else the last segment of its path", () => {
    expect(deliverableFilename({ file_name: 'budget.xlsx', file_path: 'outputs/x.xlsx' })).toBe('budget.xlsx')
    expect(deliverableFilename({ file_name: null, file_path: 'outputs/x.xlsx' })).toBe('x.xlsx')
    expect(deliverableFilename({ file_name: null, file_path: '' })).toBe('download')
  })
})
