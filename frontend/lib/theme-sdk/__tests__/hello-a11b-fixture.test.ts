/**
 * PRD-261 A-11b — the SDK-chat hello theme the browser lane serves (e2e/browser/fixtures/hello-a11b,
 * specs/sdk-chat.spec.ts) is valid as files, and is a theme the way the PRD means it: its entry imports only
 * the shared modules, takes from the SDK only names a built area exports, draws /chat from useChatController
 * (both calls) and places the host's ToolApproval card; its stylesheet stays under its own class and never
 * names the approval card; its recorded turns are well-formed test data in the stream format the chat reads.
 */
import { readdirSync, readFileSync } from 'fs'
import path from 'path'
import ts from 'typescript'
import { describe, expect, it } from 'vitest'

import sharedModules from '@/lib/theme-host/shared-modules.json'

interface SurfaceFile {
  readonly status: 'built' | 'stub'
  readonly names: readonly string[]
}

interface Frame {
  readonly text?: unknown
  readonly data?: unknown
}

interface RecordedTurns {
  readonly chatId: string
  readonly turns: Readonly<Record<string, { readonly frames: readonly Frame[]; readonly hold?: boolean }>>
}

const FIXTURE = path.resolve(__dirname, '../../../../e2e/browser/fixtures/hello-a11b')
const SURFACE_DIR = path.resolve(__dirname, '../surface')
const SDK_SPECIFIER = '@automatos/theme-sdk'
const HTML_CLASS = 'hello-a11b'
const ENTRY = readFileSync(path.join(FIXTURE, 'entry.js'), 'utf8')
const STYLESHEET = readFileSync(path.join(FIXTURE, 'theme.css'), 'utf8')
const TURNS = JSON.parse(readFileSync(path.join(FIXTURE, 'chat-turns.json'), 'utf8')) as RecordedTurns
const IMPORT = /^import\s+(?:\{([^}]*)\}|[\w$]+)\s+from\s+'([^']+)'/gm
/** Each rule's selector list: the text before a `{`, comments removed. */
const SELECTOR = /([^{}]+)\{/g
const COMMENT = /\/\*[\s\S]*?\*\//g
const TURN_NAMES = ['again', 'answer', 'approval', 'partial']

function syntaxErrors(source: string, fileName: string): string[] {
  const output = ts.transpileModule(source, {
    reportDiagnostics: true,
    fileName,
    compilerOptions: { module: ts.ModuleKind.ESNext, target: ts.ScriptTarget.ES2022, allowJs: true },
  })
  return (output.diagnostics ?? []).map((diagnostic) => ts.flattenDiagnosticMessageText(diagnostic.messageText, '\n'))
}

function importsOf(source: string): { specifier: string; names: string[] }[] {
  return Array.from(source.matchAll(IMPORT), (match) => ({
    specifier: match[2],
    names: (match[1] ?? '').split(',').map((name) => name.trim()).filter((name) => name.length > 0),
  }))
}

function builtNames(): string[] {
  return readdirSync(SURFACE_DIR)
    .filter((file) => file.endsWith('.json'))
    .map((file) => JSON.parse(readFileSync(path.join(SURFACE_DIR, file), 'utf8')) as SurfaceFile)
    .filter((surface) => surface.status === 'built')
    .flatMap((surface) => surface.names)
}

function selectors(css: string): string[] {
  return Array.from(css.replace(COMMENT, '').matchAll(SELECTOR), (match) => match[1])
    .flatMap((list) => list.split(','))
    .map((selector) => selector.trim())
    .filter((selector) => selector.length > 0)
}

function isFrame(frame: Frame): boolean {
  const text = typeof frame.text === 'string'
  const data = typeof frame.data === 'object' && frame.data !== null && typeof (frame.data as { type?: unknown }).type === 'string'
  return text !== data
}

describe('PRD-261 A-11b — the SDK-chat hello theme', () => {
  it('has its entry, manifest, stylesheet, preset, recorded turns and README', () => {
    expect(readdirSync(FIXTURE).sort()).toEqual(['README.md', 'chat-turns.json', 'entry.js', 'presets', 'theme.css', 'theme.json'])
    const manifest = JSON.parse(readFileSync(path.join(FIXTURE, 'theme.json'), 'utf8')) as Record<string, unknown>
    expect(manifest).toMatchObject({ entry: 'entry.js', styleSheet: 'theme.css', preset: 'presets/default.look.json', htmlClass: HTML_CLASS })
    expect(() => JSON.parse(readFileSync(path.join(FIXTURE, 'presets', 'default.look.json'), 'utf8'))).not.toThrow()
  })

  it('parses as an ES module', () => {
    expect(syntaxErrors(ENTRY, 'entry.js')).toEqual([])
  })

  it('imports only modules the host shares, and never sonner, next/* or an @/ path', () => {
    const shared = sharedModules.modules.map((entry) => entry.specifier)
    const imported = importsOf(ENTRY).map((entry) => entry.specifier)
    expect(imported.filter((specifier) => !shared.includes(specifier))).toEqual([])
    expect(ENTRY).not.toMatch(/['"](sonner|next\/[^'"]*|@\/[^'"]*)['"]/)
  })

  it('takes from the SDK only names a built area exports: useChatController and the host ToolApproval card', () => {
    const fromSdk = importsOf(ENTRY).filter((entry) => entry.specifier === SDK_SPECIFIER).flatMap((entry) => entry.names)
    expect(fromSdk).toEqual(expect.arrayContaining(['defineTheme', 'useChatController', 'ToolApproval']))
    expect(fromSdk.filter((name) => !builtNames().includes(name))).toEqual([])
  })

  it('draws /chat from both calls of useChatController, the conversation screen keyed by its conversation', () => {
    expect(ENTRY).toMatch(/registerPage\('\/chat', HelloChatPage\)/)
    expect(ENTRY).toMatch(/useChatController\(\)/)
    expect(ENTRY).toMatch(/useChatController\(conversation\)/)
    expect(ENTRY).toMatch(/jsx\(HelloConversation, \{ conversation \}, conversation\.key\)/)
  })

  it('keeps every rule under its own class and never names the approval card', () => {
    const all = selectors(STYLESHEET)
    expect(all.length).toBeGreaterThan(0)
    expect(all.filter((selector) => !selector.startsWith(`html.${HTML_CLASS} `))).toEqual([])
    expect(STYLESHEET).not.toMatch(/protected-surface|!important/)
  })
})

describe('PRD-261 A-11b — the recorded turns', () => {
  it('has the four turns the spec plays, each frame a text line or a typed data line', () => {
    expect(Object.keys(TURNS.turns).sort()).toEqual(TURN_NAMES)
    for (const turn of Object.values(TURNS.turns)) {
      expect(turn.frames.length).toBeGreaterThan(0)
      expect(turn.frames.filter((frame) => !isFrame(frame))).toEqual([])
    }
  })

  it('holds only the partial turn open, and names the new chat the way the backend does', () => {
    const held = Object.entries(TURNS.turns).filter(([, turn]) => turn.hold === true).map(([name]) => name)
    expect(held).toEqual(['partial'])
    expect(TURNS.turns.partial.frames[0]).toEqual({ data: { type: 'chat-id', chatId: TURNS.chatId } })
    expect(TURNS.turns.answer.frames[0]).toEqual({ data: { type: 'chat-id', chatId: TURNS.chatId } })
  })

  it("carries a tool approval with a whole-number grant id, as the host's chat checks it", () => {
    const toolData = TURNS.turns.approval.frames
      .map((frame) => frame.data as { type?: string; data?: { tool_approval?: { grant_id?: unknown } } } | undefined)
      .find((data) => data?.type === 'tool-data')
    expect(Number.isInteger(toolData?.data?.tool_approval?.grant_id)).toBe(true)
  })
})
