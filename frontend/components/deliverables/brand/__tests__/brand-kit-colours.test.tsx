/**
 * PRD-255 Wave 1, US-007 — the colour roles on the Brand kit page, against a mocked apiClient
 * holding one v2 kit: each role's swatch, hex, contrast badge and source; "Reset to derived"
 * (PUT palette {role: ""}, then the kit read back); a contrast refusal on Save shown under
 * the role it names; an edited role sent as set; how far the accent goes; a viewer edits nothing.
 */
import { describe, it, expect, vi, afterEach, beforeEach } from 'vitest'
import { render, screen, cleanup, fireEvent, within, waitFor } from '@testing-library/react'

import { designKit } from './brand-kit-design-kit'

const server = vi.hoisted(() => ({
  kit: null as any,
  putError: null as Error | null,
  // What a role reads as once it is no longer set: derived from the kit's four colours.
  derived: { accent: '#c2410c', muted: '#5b6170' } as Record<string, string>,
}))

vi.mock('sonner', () => ({ toast: { success: vi.fn(), error: vi.fn() } }))
vi.mock('@/lib/api-client', () => {
  const apiClient = {
    get: vi.fn(async (path: string) => {
      if (path === '/api/documents/brand-kit') return server.kit
      if (path === '/api/documents/brand-kit/suggestions') return { suggestions: {} }
      throw new Error(`unexpected GET ${path}`)
    }),
    put: vi.fn(async (path: string, body: any) => {
      if (path !== '/api/documents/brand-kit') throw new Error(`unexpected PUT ${path}`)
      if (server.putError) throw server.putError
      const palette = { ...server.kit.palette }
      const sources = { ...server.kit.palette_source }
      for (const [role, hex] of Object.entries(body.palette ?? {})) {
        palette[role] = hex || server.derived[role]
        sources[role] = hex ? 'set' : 'derived'
      }
      server.kit = { ...server.kit, ...body, palette, palette_source: sources }
      return server.kit
    }),
    post: vi.fn(),
    delete: vi.fn(),
    getAuthHeaders: vi.fn(async () => ({})),
    getBaseUrl: vi.fn(() => ''),
  }
  return { apiClient, default: apiClient }
})

import { toast } from 'sonner'
import { apiClient } from '@/lib/api-client'
import { BrandKitDesign } from '../brand-kit-design'
import { ROLE_LABELS } from '../brand-kit-colours'
import { roleErrorsFrom, saveErrorMessage } from '../save-errors'
import { useBrandKitForm } from '../use-brand-kit-form'
import { withPaletteFrom } from '../use-brand-palette'

const api = apiClient as unknown as Record<string, ReturnType<typeof vi.fn>>

function Harness({ canEdit }: { canEdit: boolean }) {
  const form = useBrandKitForm()
  return <BrandKitDesign form={form} canEdit={canEdit} />
}

async function colours(canEdit = true) {
  render(<Harness canEdit={canEdit} />)
  const section = await screen.findByRole('region', { name: 'Colours' })
  return section
}

function role(section: HTMLElement, name: string) {
  return within(section).getByTestId(`role-${name}`)
}

beforeEach(() => {
  server.kit = designKit()
  server.putError = null
  Object.values(api).forEach((fn) => fn.mockClear())
  vi.mocked(toast.success).mockClear()
  vi.mocked(toast.error).mockClear()
})
afterEach(() => cleanup())

describe('the colour roles', () => {
  it('shows every role with its hex, its contrast badge and whether it is set or derived', async () => {
    const section = await colours()
    for (const name of ['ink', 'heading', 'paper', 'surface', 'surface_2', 'accent', 'accent_2', 'muted', 'rule']) {
      const label = ROLE_LABELS[name as keyof typeof ROLE_LABELS]
      expect(within(role(section, name)).getByLabelText(`${label} hex`)).toHaveValue(server.kit.palette[name])
    }
    expect(within(role(section, 'accent')).getByText('set')).toBeInTheDocument()
    expect(within(role(section, 'ink')).getByText('derived')).toBeInTheDocument()
    // #1a1a2e on #fbfaf7 is 16.3:1; the accent #9a3412 on it 7.0:1.
    expect(within(section).getByTestId('contrast-ink')).toHaveTextContent('16.3:1 on paper')
    expect(within(section).getByTestId('contrast-accent')).toHaveTextContent('7.0:1 on paper')
    expect(within(section).getByTestId('contrast-surface_2')).toHaveTextContent('ink on it')
  })

  it('recomputes the badge as a role is edited, and says when it is too low', async () => {
    const section = await colours()
    fireEvent.change(within(role(section, 'muted')).getByLabelText('Secondary text hex'), { target: { value: '#bbbbbb' } })
    expect(within(section).getByTestId('contrast-muted')).toHaveTextContent('1.8:1 on paper — too low')
    expect(within(role(section, 'muted')).getByText('set')).toBeInTheDocument()
  })

  it('offers "Reset to derived" only on a set role, stores it empty, and shows the colour it derives', async () => {
    const section = await colours()
    expect(within(role(section, 'ink')).queryByRole('button', { name: /Reset to derived/ })).toBeNull()
    fireEvent.click(within(role(section, 'accent')).getByRole('button', { name: /Reset to derived/ }))

    await waitFor(() => expect(within(role(section, 'accent')).getByText('derived')).toBeInTheDocument())
    expect(api.put).toHaveBeenCalledWith('/api/documents/brand-kit', { palette: { accent: '' } })
    expect(api.get.mock.calls.filter(([path]) => path === '/api/documents/brand-kit')).toHaveLength(2)
    expect(within(role(section, 'accent')).getByLabelText('Accent (highlights) hex')).toHaveValue('#c2410c')
    expect(within(role(section, 'accent')).queryByRole('button', { name: /Reset to derived/ })).toBeNull()
  })

  it('shows the save\'s contrast refusal under the role it names, and clears it once the role changes', async () => {
    server.putError = new Error(JSON.stringify({
      message: 'Invalid brand kit',
      errors: [{ loc: ['palette', 'muted'], msg: 'muted on paper is 1.8:1; text needs 4.5:1' }],
    }))
    const section = await colours()
    fireEvent.change(within(role(section, 'muted')).getByLabelText('Secondary text hex'), { target: { value: '#bbbbbb' } })
    fireEvent.click(screen.getByRole('button', { name: 'Save' }))

    const alert = await within(role(section, 'muted')).findByRole('alert')
    expect(alert).toHaveTextContent('muted on paper is 1.8:1; text needs 4.5:1')
    expect(within(role(section, 'ink')).queryByRole('alert')).toBeNull()
    expect(toast.error).toHaveBeenCalledWith('palette.muted: muted on paper is 1.8:1; text needs 4.5:1')
    fireEvent.change(within(role(section, 'muted')).getByLabelText('Secondary text hex'), { target: { value: '#3f4450' } })
    expect(within(role(section, 'muted')).queryByRole('alert')).toBeNull()
  })

  it('sends an edited role as set, and only the set roles, with how far the accent goes', async () => {
    const section = await colours()
    fireEvent.change(within(role(section, 'heading')).getByLabelText('Headings colour'), { target: { value: '#0b2545' } })
    fireEvent.change(within(section).getByLabelText('How far the accent goes'), { target: { value: 'bold' } })
    fireEvent.click(screen.getByRole('button', { name: 'Save' }))

    await waitFor(() => expect(api.put).toHaveBeenCalledTimes(1))
    const [, body] = api.put.mock.calls[0] as [string, any]
    expect(body.palette).toEqual({ accent: '#9a3412', heading: '#0b2545' })
    expect(body.accent_use).toBe('bold')
    expect(body).not.toHaveProperty('palette_source')
  })

  it('lets a viewer read the roles and change nothing', async () => {
    const section = await colours(false)
    expect(within(role(section, 'accent')).queryByRole('button', { name: /Reset to derived/ })).toBeNull()
    expect(within(role(section, 'ink')).getByLabelText('Body text hex')).toBeDisabled()
    expect(screen.queryByRole('button', { name: 'Save' })).toBeNull()
  })
})

describe('the palette helpers', () => {
  it('reads the roles a refusal names, from either shape, and ignores other fields', () => {
    const refusal = new Error(JSON.stringify({
      message: 'Invalid brand kit',
      errors: [
        { loc: ['palette', 'ink'], msg: 'ink on paper is 2.0:1' },
        { loc: ['body', 'palette', 'accent'], msg: 'Value error, must be a hex colour such as #1a1a2e or #abc' },
        { loc: ['currency'], msg: 'currency must be a three-letter ISO 4217 code' },
      ],
    }))
    expect(roleErrorsFrom(refusal)).toEqual({ ink: 'ink on paper is 2.0:1', accent: 'must be a hex colour such as #1a1a2e or #abc' })
    expect(roleErrorsFrom(new Error('Network down'))).toEqual({})
    expect(saveErrorMessage(new Error('Network down'))).toBe('Network down')
  })

  it('keeps an unsaved set role when another role is reset', () => {
    const local = designKit({
      palette: { ...designKit().palette, heading: '#0b2545' },
      palette_source: { ...designKit().palette_source, heading: 'set' },
    })
    const fresh = designKit({
      palette: { ...designKit().palette, accent: '#c2410c' },
      palette_source: { ...designKit().palette_source, accent: 'derived' },
    })
    const merged = withPaletteFrom(local, fresh, 'accent')
    expect(merged.palette).toMatchObject({ heading: '#0b2545', accent: '#c2410c' })
    expect(merged.palette_source).toMatchObject({ heading: 'set', accent: 'derived' })
  })
})
