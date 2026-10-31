/**
 * PRD-270 W2 US-270-14c (§5.6, K18, K19, K30) — the disclosure dialog: what an Automatos Cloud item will
 * do and use, read before "Add to my workspace".
 *
 * - each kind's sections from a mocked disclosure answer (agent, Playbook, package, plugin), with what
 *   didn't resolve as "Needs …" lines;
 * - a prompt holding markup renders as text: no element is made from it;
 * - Add calls onAdd once per press (and stays pressed while its install is pending), Close calls onClose;
 *   a loading read, a failed one, an answer it can't show whole and the hosted edition offer no Add;
 * - the Studio rules over the dialog's files: tokens only (no hex), no rounded-3xl, no orange-*, no
 *   innerHTML, no markdown renderer, and one primary in the view.
 */
import { readFileSync } from 'fs'
import path from 'path'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { act, fireEvent, render, screen, within } from '@testing-library/react'
import { beforeEach, describe, expect, it, vi } from 'vitest'

const api = vi.hoisted(() => ({ get: vi.fn() }))
const edition = vi.hoisted(() => ({ local: true }))

vi.mock('@/lib/api-client', () => ({ apiClient: api }))
vi.mock('@/lib/auth-edition', () => ({
  get isLocal() {
    return edition.local
  },
  get isSaaS() {
    return !edition.local
  },
}))

import type {
  AgentDisclosure, CloudDisclosure, CloudItemKey, CloudItemKind, PackageDisclosure, PlaybookDisclosure,
  PluginDisclosure,
} from '@/lib/marketplace/cloud-items'

import { CloudDisclosureDialog } from '../disclosure-dialog'
import {
  ADDING, ADD_TO_WORKSPACE, CLOUD_BADGE, DISCLOSURE_UNREADABLE, NEEDS_A_HELPER, NEEDS_CONNECTION, NO_AGENT_STEP,
  NO_CUSTOM_PROMPT, NOT_ON_THIS_INSTALL, READING_DISCLOSURE, REVIEWED_BADGE, UPDATES_IN_PLACE,
} from '../item-copy'

const PLUGIN_ID = '3f2a8c1e-0b4d-4e6f-9a7b-1c2d3e4f5a6b'
const MARKUP = '<img src=x onerror=alert(1)>'
const CLOUD_DIR = path.resolve(__dirname, '..')

const HEADER = {
  version: '1.2.0', syncedAt: '2026-10-30T10:00:00+00:00', trust: 'reviewed' as const, publisher: 'Cloud Publisher',
}

const AGENT: AgentDisclosure = {
  ...HEADER, kind: 'agent', name: 'Ledger Helper',
  unresolved: [{ kind: 'skill', name: 'Ledger Lookup' }, { kind: 'model', name: 'cloud-model-9' }],
  prompt: { source: 'custom', text: 'You keep the books.\nAsk before paying anything.', personaName: null },
  model: { modelId: 'cloud-model-9', provider: 'openai', here: false },
  apps: [{ name: 'GMAIL', needsConnection: true }, { name: 'SLACK', needsConnection: false }],
  skills: [{ name: 'Reconcile', here: true }, { name: 'Ledger Lookup', here: false }],
  plugins: [{ slug: 'invoice-tools', here: false }],
}

const PLAYBOOK: PlaybookDisclosure = {
  ...HEADER, kind: 'playbook', name: 'Weekly Close', unresolved: [{ kind: 'helper', name: 'Closer' }],
  steps: [
    { order: 1, helper: 'Ledger Helper', needsHelper: false, prompt: 'Gather the week’s invoices.' },
    { order: 2, helper: 'Closer', needsHelper: true, prompt: 'Close the books.' },
  ],
  agents: ['Ledger Helper', 'Closer'],
  apps: ['GMAIL'],
}

const PACKAGE: PackageDisclosure = {
  ...HEADER, kind: 'package', name: 'Shop Starter', publisher: 'Automatos', unresolved: [],
  members: [
    { type: 'agent', name: 'Shop Clerk', here: true },
    { type: 'skill', name: 'Stock Count', here: false },
  ],
  connects: [{ app: 'SHOPIFY', optional: false, note: 'For orders' }],
  agents: [{ name: 'Shop Clerk', prompt: { source: 'custom', text: 'Answer shoppers politely.', personaName: null } }],
  playbooks: [{ name: 'Restock', steps: 3 }],
}

const PLUGIN: PluginDisclosure = {
  ...HEADER, kind: 'plugin', name: 'Invoice Tools', unresolved: [],
  scan: {
    verdict: 'safe', status: 'passed', riskScore: 12, summary: 'Reads invoices and writes summaries.',
    findings: [{ severity: 'low', description: 'Calls a shell helper' }],
  },
  localCheck: {
    status: 'passed',
    findings: [{ type: 'pattern', severity: 'medium', file: 'scripts/run.sh', line: 12, description: 'Runs curl' }],
  },
  contents: { skills: 2, commands: 1, agents: 0, hooks: 0 },
  skills: ['invoice-summary', 'invoice-extract'],
  updatesInPlace: true,
}

function newClient() {
  return new QueryClient({ defaultOptions: { queries: { retry: false } } })
}

interface Handlers {
  onAdd: () => void | Promise<unknown>
  onClose: () => void
}

function open(kind: CloudItemKind, id: CloudItemKey, handlers: Handlers = { onAdd: vi.fn(), onClose: vi.fn() }) {
  render(
    <QueryClientProvider client={newClient()}>
      <CloudDisclosureDialog kind={kind} id={id} open onAdd={handlers.onAdd} onClose={handlers.onClose} />
    </QueryClientProvider>,
  )
  return handlers
}

async function shown(answer: CloudDisclosure, id: CloudItemKey, handlers?: Handlers) {
  api.get.mockResolvedValue(answer)
  const opened = open(answer.kind, id, handlers)
  const dialog = await screen.findByTestId('cloud-disclosure-sections')
  return { ...opened, dialog: screen.getByTestId('cloud-disclosure'), sections: dialog }
}

function lineOf(container: HTMLElement, text: string): HTMLElement {
  return within(container).getByText(text).closest('li') as HTMLElement
}

beforeEach(() => {
  vi.clearAllMocks()
  edition.local = true
})

describe('the header', () => {
  it('names the item, its publisher, version and Sync, with both badges and no Official', async () => {
    const { dialog } = await shown(AGENT, 7)

    expect(api.get).toHaveBeenCalledWith('/api/marketplace/cloud-sync/items/agent/7/disclosure')
    expect(within(dialog).getByRole('heading', { name: 'Ledger Helper' })).toBeInTheDocument()
    expect(within(dialog).getByText(/^by Cloud Publisher · Version 1\.2\.0 · Synced /)).toBeInTheDocument()
    expect(within(dialog).getByText(CLOUD_BADGE)).toBeInTheDocument()
    expect(within(dialog).getByText(REVIEWED_BADGE)).toBeInTheDocument()
    expect(within(dialog).queryByText(/official/i)).not.toBeInTheDocument()
  })
})

describe('each kind’s sections', () => {
  it('an agent: its prompt, model, apps, skills, plugins and what it needs', async () => {
    const { sections } = await shown(AGENT, 7)

    const prompt = within(sections).getByText(/You keep the books\./)
    expect(prompt.tagName).toBe('PRE')
    expect(prompt).toHaveClass('whitespace-pre-wrap')
    expect(prompt.textContent).toBe(AGENT.prompt.text)
    expect(within(lineOf(sections, 'cloud-model-9 (openai)')).getByText(NOT_ON_THIS_INSTALL)).toBeInTheDocument()
    expect(within(lineOf(sections, 'GMAIL')).getByText(NEEDS_CONNECTION)).toBeInTheDocument()
    expect(within(lineOf(sections, 'SLACK')).queryByText(NEEDS_CONNECTION)).not.toBeInTheDocument()
    expect(within(lineOf(sections, 'Reconcile')).queryByText(NOT_ON_THIS_INSTALL)).not.toBeInTheDocument()
    expect(within(lineOf(sections, 'Ledger Lookup')).getByText(NOT_ON_THIS_INSTALL)).toBeInTheDocument()
    expect(within(lineOf(sections, 'invoice-tools')).getByText(NOT_ON_THIS_INSTALL)).toBeInTheDocument()
    expect(within(sections).getByText('Needs the skill Ledger Lookup')).toBeInTheDocument()
    expect(within(sections).getByText('Needs the model cloud-model-9')).toBeInTheDocument()
  })

  it('an agent with no custom prompt says it uses the default for its type', async () => {
    const { sections } = await shown({ ...AGENT, prompt: { source: null, text: null, personaName: null } }, 7)

    expect(within(sections).getByText(NO_CUSTOM_PROMPT)).toBeInTheDocument()
    expect(sections.querySelector('pre')).toBeNull()
  })

  it('a Playbook: each step’s helper or "needs a helper", its prompt, and the agents it adds', async () => {
    const { sections } = await shown(PLAYBOOK, 12)

    const steps = within(within(sections).getByRole('region', { name: 'Steps' })).getAllByRole('listitem')
    expect(steps).toHaveLength(2)
    expect(within(steps[0]).getByText('Step 1')).toBeInTheDocument()
    expect(within(steps[0]).getByText('Ledger Helper')).toBeInTheDocument()
    expect(within(steps[0]).queryByText(NEEDS_A_HELPER)).not.toBeInTheDocument()
    expect(within(steps[0]).getByText('Gather the week’s invoices.').tagName).toBe('PRE')
    expect(within(steps[1]).getByText('Closer')).toBeInTheDocument()
    expect(within(steps[1]).getByText(NEEDS_A_HELPER)).toBeInTheDocument()
    const adds = within(sections).getByRole('region', { name: 'Agents it adds' })
    expect(within(adds).getByText('Ledger Helper')).toBeInTheDocument()
    expect(within(adds).getByText('Closer')).toBeInTheDocument()
    expect(within(sections).getByText('Needs the helper agent Closer')).toBeInTheDocument()
  })

  it('a step that names no agent runs none: no "needs a helper"', async () => {
    const reportStep = { order: 3, helper: null, needsHelper: false, prompt: 'Write the report.' }
    const { sections } = await shown({ ...PLAYBOOK, steps: [reportStep], unresolved: [] }, 12)

    const step = within(within(sections).getByRole('region', { name: 'Steps' })).getByRole('listitem')
    expect(within(step).getByText('Step 3')).toBeInTheDocument()
    expect(within(step).getByText(NO_AGENT_STEP)).toBeInTheDocument()
    expect(within(step).queryByText(NEEDS_A_HELPER)).not.toBeInTheDocument()
  })

  it('a package: its members, connections and each agent’s prompt', async () => {
    const { sections } = await shown(PACKAGE, 'shop-starter')

    expect(api.get).toHaveBeenCalledWith('/api/marketplace/cloud-sync/items/package/shop-starter/disclosure')
    expect(within(lineOf(sections, 'Shop Clerk (agent)')).queryByText(NOT_ON_THIS_INSTALL)).not.toBeInTheDocument()
    expect(within(lineOf(sections, 'Stock Count (skill)')).getByText(NOT_ON_THIS_INSTALL)).toBeInTheDocument()
    expect(within(lineOf(sections, 'SHOPIFY: For orders')).getByText(NEEDS_CONNECTION)).toBeInTheDocument()
    const clerk = within(sections).getByRole('region', { name: 'Shop Clerk: prompt' })
    expect(within(clerk).getByText('Answer shoppers politely.').tagName).toBe('PRE')
    expect(within(lineOf(sections, 'Restock')).getByText('3 steps')).toBeInTheDocument()
  })

  it('a plugin: Automatos Cloud’s report, this install’s check, contents, skills, and updates in place', async () => {
    const { sections } = await shown(PLUGIN, PLUGIN_ID)

    expect(api.get).toHaveBeenCalledWith(`/api/marketplace/cloud-sync/items/plugin/${PLUGIN_ID}/disclosure`)
    const report = within(sections).getByRole('region', { name: "Automatos Cloud's scan report" })
    expect(within(report).getByText('Verdict: safe · Review: passed · Risk score: 12 of 100')).toBeInTheDocument()
    expect(within(report).getByText('Reads invoices and writes summaries.').tagName).toBe('PRE')
    expect(within(report).getByText('low: Calls a shell helper')).toBeInTheDocument()
    const check = within(sections).getByRole('region', { name: "This install's check" })
    expect(within(check).getByText('Result: passed')).toBeInTheDocument()
    expect(within(check).getByText('medium: Runs curl (scripts/run.sh:12)')).toBeInTheDocument()
    expect(within(sections).getByText('2 skills · 1 command · 0 agents · 0 hooks')).toBeInTheDocument()
    expect(within(sections).getByText('invoice-summary')).toBeInTheDocument()
    expect(within(sections).getByText(UPDATES_IN_PLACE)).toBeInTheDocument()
  })
})

const MARKED: Array<[string, CloudDisclosure, CloudItemKey]> = [
  ['an agent’s prompt', { ...AGENT, prompt: { source: 'custom', text: MARKUP, personaName: null } }, 7],
  ['a step’s prompt', { ...PLAYBOOK, steps: [{ order: 1, helper: 'Ledger Helper', needsHelper: false, prompt: MARKUP }] }, 12],
  ['a plugin’s report', { ...PLUGIN, scan: { ...PLUGIN.scan!, summary: MARKUP } }, PLUGIN_ID],
]

describe('server text is text', () => {
  it.each(MARKED)('%s holding markup makes no element', async (_label, answer, id) => {
    await shown(answer, id)

    expect(document.body.querySelector('img')).toBeNull()
    expect(screen.getByText(MARKUP)).toBeInTheDocument()
  })

  it('an item name holding markup is its title as written', async () => {
    await shown({ ...AGENT, name: MARKUP }, 7)

    expect(document.body.querySelector('img')).toBeNull()
    expect(screen.getByRole('heading', { name: MARKUP })).toBeInTheDocument()
  })
})

describe('Add and Close', () => {
  it('Add calls onAdd once, however often it is pressed, and never onClose', async () => {
    const { onAdd, onClose } = await shown(AGENT, 7)
    const add = screen.getByRole('button', { name: ADD_TO_WORKSPACE })

    fireEvent.click(add)
    fireEvent.click(add)

    expect(onAdd).toHaveBeenCalledTimes(1)
    expect(onClose).not.toHaveBeenCalled()
  })

  it('Close calls onClose and adds nothing', async () => {
    const { onAdd, onClose } = await shown(AGENT, 7)

    fireEvent.click(screen.getByTestId('cloud-disclosure-close'))

    expect(onClose).toHaveBeenCalledTimes(1)
    expect(onAdd).not.toHaveBeenCalled()
  })

  it('shows a loading line, and no Add, while it reads', () => {
    api.get.mockReturnValue(new Promise(() => {}))
    open('agent', 7)

    expect(screen.getByText(READING_DISCLOSURE)).toBeInTheDocument()
    expect(screen.queryByRole('button', { name: ADD_TO_WORKSPACE })).not.toBeInTheDocument()
  })

  it('a failed read says so and offers no Add', async () => {
    api.get.mockRejectedValue(new Error('Not Found'))
    const { onAdd } = open('agent', 7)

    expect(await screen.findByText(DISCLOSURE_UNREADABLE)).toBeInTheDocument()
    expect(screen.queryByRole('button', { name: ADD_TO_WORKSPACE })).not.toBeInTheDocument()
    expect(screen.getByTestId('cloud-disclosure-close')).toBeInTheDocument()
    expect(onAdd).not.toHaveBeenCalled()
  })

  it.each([
    ['another kind’s answer', 'agent', PLUGIN],
    ['an agent with no apps list', 'agent', { ...AGENT, apps: undefined }],
    ['a Playbook whose steps are not rows', 'playbook', { ...PLAYBOOK, steps: ['Step one'] }],
    ['a package agent with no prompt', 'package', { ...PACKAGE, agents: [{ name: 'Shop Clerk' }] }],
    ['a plugin report with no findings', 'plugin', { ...PLUGIN, scan: { ...PLUGIN.scan, findings: null } }],
    ['an answer with no name', 'agent', { ...AGENT, name: null }],
  ] as Array<[string, CloudItemKind, unknown]>)('%s is not shown and offers no Add', async (_label, kind, answer) => {
    api.get.mockResolvedValue(answer)
    open(kind, PLUGIN_ID)

    expect(await screen.findByText(DISCLOSURE_UNREADABLE)).toBeInTheDocument()
    expect(screen.queryByRole('button', { name: ADD_TO_WORKSPACE })).not.toBeInTheDocument()
    expect(screen.queryByTestId('cloud-disclosure-sections')).not.toBeInTheDocument()
  })

  it('Add stays pressed while the install it starts is pending, and can be pressed again once it fails', async () => {
    let fail: (reason: Error) => void = () => {}
    const onAdd = vi.fn(() => new Promise<void>((_resolve, reject) => { fail = reject }))
    await shown(AGENT, 7, { onAdd, onClose: vi.fn() })

    fireEvent.click(screen.getByRole('button', { name: ADD_TO_WORKSPACE }))

    expect(screen.getByRole('button', { name: ADDING })).toBeDisabled()
    await act(async () => fail(new Error('Not added')))
    fireEvent.click(screen.getByRole('button', { name: ADD_TO_WORKSPACE }))
    expect(onAdd).toHaveBeenCalledTimes(2)
  })

  it('on the hosted edition it asks nothing and offers no Add', async () => {
    edition.local = false
    open('agent', 7)

    expect(await screen.findByText(DISCLOSURE_UNREADABLE)).toBeInTheDocument()
    expect(screen.queryByText(READING_DISCLOSURE)).not.toBeInTheDocument()
    expect(screen.queryByRole('button', { name: ADD_TO_WORKSPACE })).not.toBeInTheDocument()
    expect(api.get).not.toHaveBeenCalled()
  })

  it('asks nothing while closed', () => {
    render(
      <QueryClientProvider client={newClient()}>
        <CloudDisclosureDialog kind="agent" id={7} open={false} onAdd={vi.fn()} onClose={vi.fn()} />
      </QueryClientProvider>,
    )

    expect(api.get).not.toHaveBeenCalled()
    expect(screen.queryByTestId('cloud-disclosure')).not.toBeInTheDocument()
  })
})

describe('the Studio rules', () => {
  const FILES = ['disclosure-dialog.tsx', 'disclosure-sections.tsx', 'disclosure-plugin.tsx', 'disclosure-parts.tsx']
  const sources = FILES.map((name) => [name, readFileSync(path.join(CLOUD_DIR, name), 'utf8')] as const)

  it.each(sources)('%s uses tokens only and no forbidden surface', (_name, source) => {
    expect(source).not.toMatch(/#[0-9a-fA-F]{3,8}\b/)
    expect(source).not.toMatch(/rounded-3xl/)
    expect(source).not.toMatch(/orange-/)
    expect(source).not.toMatch(/dark:/)
    expect(source).not.toMatch(/innerHTML/)
    expect(source).not.toMatch(/from ['"][^'"]*markdown[^'"]*['"]/i)
  })

  it('the view has one primary: Add to my workspace', async () => {
    const { dialog } = await shown(AGENT, 7)

    const primaries = within(dialog).getAllByRole('button').filter((button) => button.className.includes('bg-primary '))
    expect(primaries).toHaveLength(1)
    expect(primaries[0]).toHaveTextContent(ADD_TO_WORKSPACE)
  })
})
