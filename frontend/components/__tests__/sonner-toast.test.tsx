import { describe, it, expect, beforeAll } from 'vitest'
import { readFileSync } from 'fs'
import path from 'path'
import { render, screen } from '@testing-library/react'
import { toast } from 'sonner'
import { Toaster } from '../ui/sonner'

// jsdom has no matchMedia; sonner reads it for theme detection at mount.
beforeAll(() => {
  Object.defineProperty(window, 'matchMedia', {
    writable: true,
    value: (query: string) => ({
      matches: false,
      media: query,
      onchange: null,
      addListener: () => {},
      removeListener: () => {},
      addEventListener: () => {},
      removeEventListener: () => {},
      dispatchEvent: () => false,
    }),
  })
})

// PRD-154 S9 — one toast system (sonner). Deterministic proxy for the
// browser AC: the use-toast toasts NEVER rendered because the shadcn
// <Toaster> was mounted nowhere; sonner's react-hot-toast Toaster was the
// only one mounted. Now the single sonner Toaster is mounted, so the
// mission approve/reject and template-save-error feedback render.

describe('sonner toast system (PRD-154 S9)', () => {
  it('renders a success toast (mission approve/reject feedback) via the mounted Toaster', async () => {
    render(<Toaster />)
    toast.success('Mission approved')
    expect(await screen.findByText('Mission approved')).toBeInTheDocument()
  })

  it('renders an error toast (template-save-error path)', async () => {
    render(<Toaster />)
    toast.error('Failed to save template')
    expect(await screen.findByText('Failed to save template')).toBeInTheDocument()
  })

  it('renders a neutral toast with a description (converted use-toast object call)', async () => {
    render(<Toaster />)
    toast('Connected!', { description: 'App is now connected and ready to use.' })
    expect(await screen.findByText('Connected!')).toBeInTheDocument()
    expect(await screen.findByText('App is now connected and ready to use.')).toBeInTheDocument()
  })

  it('has removed react-hot-toast from package.json (single toast system)', () => {
    const pkg = JSON.parse(
      readFileSync(path.resolve(process.cwd(), 'package.json'), 'utf8'),
    )
    const deps = { ...(pkg.dependencies || {}), ...(pkg.devDependencies || {}) }
    expect(deps['react-hot-toast']).toBeUndefined()
    expect(deps['sonner']).toBeDefined()
  })
})
