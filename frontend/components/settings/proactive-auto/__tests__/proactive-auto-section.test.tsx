/**
 * PRD-268 §6.9 — Settings → Orchestrator's "Proactive Auto" section.
 *
 * - An owner or admin flips the workspace switch: PUT /api/proactive/settings `{enabled}`, shown at once, and the
 *   open suggestions and the day's brief are asked for again once it lands.
 * - Anyone else sees the switch read-only, from the workspace they load, with who can change it; the settings route
 *   (owners' and admins') is never asked, and there are no caps to edit.
 * - The caps edit within the bounds the route accepts (none looser than `platform_caps`, the shortest gap no longer
 *   than the longest), and Save sends only the caps that changed.
 * - Pause suggestions writes the person's own settings, PUT /api/proactive/me `{paused}`, for every role.
 * - A refused change goes back to what it was, with the server's sentence.
 * - One line says what Auto is doing here, from the reads' `effective_mode`: live, shadow, or off while the switch is
 *   on; nothing when it is off and so is the switch. A flip reads the mode again rather than show the old one.
 *
 * The routes are a small fake (`server`): a read answers the state and its mode (the platform's where the switch is
 * on, else off), and a write applies its body and answers without a mode, as api/proactive.py does.
 */
import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { beforeEach, describe, expect, it, vi } from 'vitest'

const requestMock = vi.hoisted(() => vi.fn())
const putMock = vi.hoisted(() => vi.fn())
const current = vi.hoisted(() => ({ workspace: null as Record<string, unknown> | null }))

vi.mock('@/lib/api-client', () => ({
  apiClient: {
    request: (...args: unknown[]) => requestMock(...args),
    put: (...args: unknown[]) => putMock(...args),
  },
}))
vi.mock('@/components/workspace-provider', () => ({
  useWorkspaceOptional: () => ({ workspace: current.workspace }),
}))

import { briefQueryKeys } from '@/hooks/use-brief'
import { suggestionQueryKeys } from '@/hooks/use-suggestions'
import type { CapName } from '@/lib/proactive/settings'
import {
  ASK_AN_OWNER_OR_ADMIN,
  CAP_COPY,
  MODE_LINES,
  PAUSE_LABEL,
  PAUSE_LINE,
  RESET_CAPS,
  SAVE_CAPS,
  WORKSPACE_OFF,
  WORKSPACE_ON,
  WORKSPACE_SWITCH_LABEL,
} from '../copy'
import { ProactiveAutoSection } from '../proactive-auto-section'

const SETTINGS_PATH = '/api/proactive/settings'
const ME_PATH = '/api/proactive/me'

/** The config defaults (services/proactive/limits.py `default_caps`), answered as `platform_caps`. */
const PLATFORM = {
  max_model_runs_per_day: 48,
  max_suggestions_per_run: 3,
  max_new_per_person_per_day: 10,
  max_open_per_person: 20,
  max_open_per_agent: 3,
  min_interval_minutes: 30,
  max_interval_minutes: 1440,
}
const ME = { paused: false, quiet_start: null, quiet_end: null, timezone: null }
type CapKey = keyof typeof PLATFORM
type Mode = 'off' | 'shadow' | 'live'

function settings(enabled: boolean, caps: Partial<typeof PLATFORM> = {}) {
  return { enabled, caps: { ...PLATFORM, ...caps }, platform_caps: PLATFORM }
}

function workspace(role: string, proactive?: Record<string, unknown>) {
  const proactiveSettings = proactive ? { proactive } : {}
  return { id: 'ws-1', name: 'Acme', slug: 'acme', plan: 'pro', role, planLimits: {}, settings: { integrations: {}, ...proactiveSettings } }
}

/** apiClient's ApiHttpError: the status, and the route's `{code, detail}` body. */
function refusal(status: number, code: string, detail: string): Error {
  return Object.assign(new Error(detail), { status, body: { code, detail } })
}

function deferred<T>() {
  let resolve: (value: T) => void = () => undefined
  let reject: (reason: unknown) => void = () => undefined
  const promise = new Promise<T>((res, rej) => {
    resolve = res
    reject = rej
  })
  return { promise, resolve, reject }
}

interface FakeServer {
  platform: Mode
  settings: ReturnType<typeof settings>
  me: Record<string, unknown>
  settingsRefusal: Error | null
}

interface WriteBody {
  readonly enabled?: boolean
  readonly caps?: Readonly<Record<string, number | null>>
  readonly paused?: boolean
}

let server: FakeServer

/** The workspace's effective mode: the platform's where the switch is on, else off. */
const modeOf = (): Mode => (server.settings.enabled ? server.platform : 'off')

/** A PUT as the routes take it: applied, and answered without a mode. */
function write(path: string, body: WriteBody) {
  if (path === ME_PATH) {
    server.me = { ...server.me, ...body }
    return server.me
  }
  const asked = Object.entries(body.caps ?? {}).map(([name, value]) => [name, value ?? PLATFORM[name as CapKey]])
  server.settingsRefusal = null
  server.settings = {
    ...server.settings,
    enabled: body.enabled ?? server.settings.enabled,
    caps: { ...server.settings.caps, ...Object.fromEntries(asked) },
  }
  return server.settings
}

function serve() {
  requestMock.mockImplementation(async (path: string) => {
    if (path === SETTINGS_PATH) {
      if (server.settingsRefusal) throw server.settingsRefusal
      return { ...server.settings, effective_mode: modeOf() }
    }
    if (path === ME_PATH) return { ...server.me, effective_mode: modeOf() }
    throw new Error(`unexpected read of ${path}`)
  })
  putMock.mockImplementation(async (path: string, body: WriteBody) => write(path, body))
}

function renderSection() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false }, mutations: { retry: false } } })
  const invalidate = vi.spyOn(client, 'invalidateQueries')
  render(
    <QueryClientProvider client={client}>
      <ProactiveAutoSection />
    </QueryClientProvider>,
  )
  const refreshed = (key: readonly unknown[]) =>
    invalidate.mock.calls.some(([filters]) => JSON.stringify(filters) === JSON.stringify({ queryKey: key }))
  return { refreshed }
}

const workspaceSwitch = () => screen.findByRole('switch', { name: WORKSPACE_SWITCH_LABEL })
const pauseSwitch = () => screen.findByRole('switch', { name: PAUSE_LABEL })
const capInput = (name: CapName) => screen.findByLabelText(CAP_COPY[name].label)
const typeInto = (input: HTMLElement, value: string) => fireEvent.change(input, { target: { value } })

beforeEach(() => {
  requestMock.mockReset()
  putMock.mockReset()
  current.workspace = workspace('owner')
  server = { platform: 'off', settings: settings(false), me: { ...ME }, settingsRefusal: null }
  serve()
})

describe('the workspace switch', () => {
  it.each(['owner', 'admin'])('an %s switches Proactive Auto on: shown at once, then the suggestions and brief refresh', async (role) => {
    current.workspace = workspace(role)
    const gate = deferred<void>()
    putMock.mockImplementation(async (path: string, body: WriteBody) => {
      await gate.promise
      return write(path, body)
    })
    const { refreshed } = renderSection()

    const toggle = await workspaceSwitch()
    expect(toggle).toHaveAttribute('aria-checked', 'false')
    expect(toggle).toBeEnabled()
    expect(screen.getByText(WORKSPACE_OFF)).toBeInTheDocument()

    fireEvent.click(toggle)
    await waitFor(() => expect(putMock).toHaveBeenCalledWith(SETTINGS_PATH, { enabled: true }))
    await waitFor(() => expect(toggle).toHaveAttribute('aria-checked', 'true'))
    expect(refreshed(suggestionQueryKeys.all)).toBe(false)

    gate.resolve()
    await waitFor(() => expect(refreshed(suggestionQueryKeys.all)).toBe(true))
    expect(refreshed(briefQueryKeys.all)).toBe(true)
    // The platform is off here, so the mode read again after the flip says so.
    expect(await screen.findByText(MODE_LINES.offButSwitchedOn)).toBeInTheDocument()
    expect(toggle).toHaveAttribute('aria-checked', 'true')
    expect(screen.getByText(WORKSPACE_ON)).toBeInTheDocument()
    expect(screen.queryByText(ASK_AN_OWNER_OR_ADMIN)).not.toBeInTheDocument()
  })

  it.each(['editor', 'viewer', 'member'])('a %s sees it read-only from the workspace, with who can change it', async (role) => {
    current.workspace = workspace(role, { enabled: true })
    server.settings = settings(true)
    renderSection()

    const toggle = await workspaceSwitch()
    expect(toggle).toHaveAttribute('aria-checked', 'true')
    expect(toggle).toBeDisabled()
    expect(screen.getByText(WORKSPACE_ON)).toBeInTheDocument()
    expect(screen.getByText(ASK_AN_OWNER_OR_ADMIN)).toBeInTheDocument()

    fireEvent.click(toggle)
    expect(await pauseSwitch()).toBeEnabled()
    expect(putMock).not.toHaveBeenCalled()
    expect(requestMock).not.toHaveBeenCalledWith(SETTINGS_PATH)
    expect(screen.queryByRole('spinbutton')).not.toBeInTheDocument()
    expect(screen.queryByRole('button', { name: SAVE_CAPS })).not.toBeInTheDocument()
  })

  it('reads off for anyone else unless the workspace has it exactly true', async () => {
    current.workspace = workspace('viewer', { enabled: 'true' })
    renderSection()

    expect(await workspaceSwitch()).toHaveAttribute('aria-checked', 'false')
    expect(screen.getByText(WORKSPACE_OFF)).toBeInTheDocument()
  })

  it('a refused switch goes back to what it was, with the server’s sentence', async () => {
    const sentence = 'The check-in settings could not be saved right now. Try again in a minute.'
    const put = deferred<unknown>()
    putMock.mockReturnValue(put.promise)
    const { refreshed } = renderSection()

    const toggle = await workspaceSwitch()
    fireEvent.click(toggle)
    await waitFor(() => expect(toggle).toHaveAttribute('aria-checked', 'true'))

    put.reject(refusal(503, 'settings_unavailable', sentence))
    expect(await screen.findByRole('alert')).toHaveTextContent(sentence)
    expect(toggle).toHaveAttribute('aria-checked', 'false')
    expect(screen.getByText(WORKSPACE_OFF)).toBeInTheDocument()
    expect(refreshed(suggestionQueryKeys.all)).toBe(false)
  })
})

describe('the caps', () => {
  it('edit within the bounds the route accepts, and Save sends only the caps that changed', async () => {
    // The server's answer, not what was typed, shows once Save lands: 11 here, against the 12 typed.
    putMock.mockResolvedValue(settings(false, { max_model_runs_per_day: 11, min_interval_minutes: 90, max_interval_minutes: 600 }))
    renderSection()

    const runs = await capInput('max_model_runs_per_day')
    const shortest = await capInput('min_interval_minutes')
    const longest = await capInput('max_interval_minutes')
    const save = screen.getByRole('button', { name: SAVE_CAPS })
    expect(runs).toHaveValue(48)
    expect(save).toBeDisabled()

    typeInto(runs, '60')
    expect(screen.getByText('Enter a whole number from 0 to 48.')).toBeInTheDocument()
    expect(runs).toHaveAttribute('aria-invalid', 'true')
    expect(save).toBeDisabled()
    typeInto(runs, '12')
    expect(runs).toHaveAttribute('aria-invalid', 'false')

    typeInto(shortest, '20')
    expect(screen.getByText('Enter a whole number from 30 to 1440.')).toBeInTheDocument()
    expect(save).toBeDisabled()
    typeInto(shortest, '90')
    typeInto(longest, '60')
    expect(screen.getByText('Enter a whole number from 30 to 60.')).toBeInTheDocument()
    expect(screen.getByText('Enter a whole number from 90 to 1440.')).toBeInTheDocument()
    expect(save).toBeDisabled()
    typeInto(longest, '600')
    expect(save).toBeEnabled()

    fireEvent.click(save)
    await waitFor(() =>
      expect(putMock).toHaveBeenCalledWith(SETTINGS_PATH, {
        caps: { max_model_runs_per_day: 12, min_interval_minutes: 90, max_interval_minutes: 600 },
      }),
    )
    await waitFor(() => expect(runs).toHaveValue(11))
    expect(longest).toHaveValue(600)
    expect(screen.getByRole('button', { name: SAVE_CAPS })).toBeDisabled()
  })

  it('refused caps go back to the saved ones and say why; what was typed stays, to fix', async () => {
    const sentence = "A workspace can only tighten the platform's caps: none above the platform's. Nothing was saved."
    putMock.mockRejectedValue(refusal(422, 'caps_looser', sentence))
    renderSection()

    const runs = await capInput('max_model_runs_per_day')
    typeInto(runs, '12')
    fireEvent.click(screen.getByRole('button', { name: SAVE_CAPS }))

    expect(await screen.findByRole('alert')).toHaveTextContent(sentence)
    expect(runs).toHaveValue(12)
    // Save is back: the saved caps are the old ones again, so 12 is a change once more.
    expect(screen.getByRole('button', { name: SAVE_CAPS })).toBeEnabled()
  })

  it('caps stored wrong say so, and putting them back to their defaults sends every cap as null', async () => {
    server.settingsRefusal = refusal(409, 'caps_stored_wrong', "This workspace's caps are stored wrong.")
    renderSection()

    fireEvent.click(await screen.findByRole('button', { name: RESET_CAPS }))
    await waitFor(() =>
      expect(putMock).toHaveBeenCalledWith(SETTINGS_PATH, {
        caps: Object.fromEntries(Object.keys(PLATFORM).map((name) => [name, null])),
      }),
    )
    expect(await capInput('max_model_runs_per_day')).toHaveValue(48)
  })
})

describe('Pause suggestions', () => {
  it.each(['owner', 'admin', 'editor', 'viewer'])('writes the %s’s own settings, and says what a pause does', async (role) => {
    current.workspace = workspace(role)
    putMock.mockResolvedValue({ ...ME, paused: true })
    const { refreshed } = renderSection()

    const pause = await pauseSwitch()
    expect(pause).toHaveAttribute('aria-checked', 'false')
    expect(screen.getByText(PAUSE_LINE)).toBeInTheDocument()

    fireEvent.click(pause)
    await waitFor(() => expect(putMock).toHaveBeenCalledWith(ME_PATH, { paused: true }))
    await waitFor(() => expect(pause).toHaveAttribute('aria-checked', 'true'))
    await waitFor(() => expect(refreshed(suggestionQueryKeys.all)).toBe(true))
    expect(putMock).toHaveBeenCalledTimes(1)
  })

  it('a refused pause goes back to what it was, with the server’s sentence', async () => {
    const sentence = 'The check-in settings could not be saved right now. Try again in a minute.'
    server.me = { ...ME, paused: true }
    const put = deferred<unknown>()
    putMock.mockReturnValue(put.promise)
    renderSection()

    const pause = await pauseSwitch()
    expect(pause).toHaveAttribute('aria-checked', 'true')
    fireEvent.click(pause)
    await waitFor(() => expect(putMock).toHaveBeenCalledWith(ME_PATH, { paused: false }))
    await waitFor(() => expect(pause).toHaveAttribute('aria-checked', 'false'))

    put.reject(refusal(503, 'settings_unavailable', sentence))
    expect(await screen.findByRole('alert')).toHaveTextContent(sentence)
    expect(pause).toHaveAttribute('aria-checked', 'true')
  })
})

describe('the line on what Auto is doing here', () => {
  const CASES: [Mode, boolean, string | null][] = [
    ['live', true, MODE_LINES.live],
    ['shadow', true, MODE_LINES.shadow],
    ['off', true, MODE_LINES.offButSwitchedOn],
    ['live', false, null],
    ['off', false, null],
  ]

  function expectOnly(line: string | null) {
    Object.values(MODE_LINES)
      .filter((other) => other !== line)
      .forEach((other) => expect(screen.queryByText(other)).not.toBeInTheDocument())
  }

  it.each(CASES)('an owner, the platform %s and the switch on: %s, reads: %s', async (platform, enabled, line) => {
    server.platform = platform
    server.settings = settings(enabled)
    renderSection()

    await workspaceSwitch()
    if (line !== null) expect(await screen.findByText(line)).toBeInTheDocument()
    expectOnly(line)
  })

  it.each(CASES)('a viewer, the platform %s and the switch on: %s, reads: %s', async (platform, enabled, line) => {
    current.workspace = workspace('viewer', { enabled })
    server.platform = platform
    server.settings = settings(enabled)
    renderSection()

    await pauseSwitch()
    if (line !== null) expect(await screen.findByText(line)).toBeInTheDocument()
    expectOnly(line)
    expect(requestMock).not.toHaveBeenCalledWith(SETTINGS_PATH)
  })

  it('an owner who switches it on sees the mode the server reads next', async () => {
    server.platform = 'shadow'
    renderSection()

    const toggle = await workspaceSwitch()
    expectOnly(null)
    fireEvent.click(toggle)

    expect(await screen.findByText(MODE_LINES.shadow)).toBeInTheDocument()
    expectOnly(MODE_LINES.shadow)
    expect(putMock).toHaveBeenCalledWith(SETTINGS_PATH, { enabled: true })
  })
})
