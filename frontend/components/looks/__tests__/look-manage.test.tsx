/**
 * PRD-260 T-008a — 'This look' and 'Import a look' on Make it yours, ported
 * from Automatos Play's tests/looks/LookManage.test.tsx: Save, Rename and
 * Delete for the user's saved looks (a preset only duplicates), Export as a
 * .look.json download, Copy JSON, and Import from pasted text or a file
 * through lib/looks/transfer.ts with its messages shown. Imports are untrusted:
 * nothing invalid is kept, and imported text is never read as HTML.
 */
import React from 'react'
import { cleanup, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

vi.mock('@/components/layout/main-layout', () => ({
  MainLayout: ({ children }: { children: React.ReactNode }) => <div data-testid="main-layout">{children}</div>,
}))

import { myLook } from '@/lib/looks/__tests__/helpers'
import { EMBER, PRESETS } from '@/lib/looks/presets'
import { ACTIVE_LOOK_KEY, SAVED_LOOKS_KEY } from '@/lib/looks/storage'
import { LOOK_FILE_TYPE, NOT_JSON, lookJson } from '@/lib/looks/transfer'

import {
  BLOB_URL,
  type DownloadStub,
  blobText,
  freshBrowser,
  group,
  importer,
  lookOnPage,
  looksRow,
  manage,
  pressedCard,
  renderLooks,
  status,
  stored,
  stubDownloads,
} from './looks-harness'

type User = ReturnType<typeof userEvent.setup>

/** Starts on the user's own 'My look 1' (rose on Night). */
async function readyOnMyLook(): Promise<void> {
  window.localStorage.setItem(SAVED_LOOKS_KEY, JSON.stringify([myLook()]))
  window.localStorage.setItem(ACTIVE_LOOK_KEY, JSON.stringify(myLook()))
  renderLooks()
  await waitFor(() => expect(status()).toHaveTextContent(/^My look 1$/))
}

async function paste(user: User, text: string): Promise<void> {
  await user.click(importer().getByLabelText("Paste a Look's JSON"))
  await user.paste(text)
  await user.click(importer().getByRole('button', { name: 'Import pasted JSON' }))
}

function toolbar(): (string | null)[] {
  return manage().getAllByRole('button').map((button) => button.textContent)
}

beforeEach(() => {
  freshBrowser()
  // jsdom plays no media; refused imports and blocked copies are logged.
  vi.spyOn(console, 'warn').mockImplementation(() => undefined)
  vi.spyOn(console, 'error').mockImplementation(() => undefined)
})

afterEach(() => {
  cleanup()
  vi.restoreAllMocks()
  freshBrowser()
})

describe('This look: managing looks', () => {
  it('only offers Duplicate (plus Export and Copy JSON) for a preset, edited or not', async () => {
    const user = userEvent.setup()
    renderLooks()
    await lookOnPage(EMBER)
    expect(toolbar()).toEqual(['Duplicate', 'Export', 'Copy JSON'])
    await user.click(group(/^Accent/).getByRole('button', { name: 'Rose' }))
    expect(status()).toHaveTextContent(/^Ember · edited$/)
    expect(toolbar()).toEqual(['Duplicate', 'Export', 'Copy JSON'])
  })

  it('duplicates a preset into a saved look and picks it', async () => {
    const user = userEvent.setup()
    renderLooks()
    await lookOnPage(EMBER)
    await user.click(manage().getByRole('button', { name: 'Duplicate' }))

    const copy = { ...EMBER, id: 'look-1', name: 'Ember copy' }
    expect(pressedCard()).toHaveTextContent(`${copy.name} ${copy.sub}`)
    expect(status()).toHaveTextContent(/^Ember copy$/)
    expect(manage().getByText('Duplicated as Ember copy.')).toBeInTheDocument()
    expect(stored(SAVED_LOOKS_KEY)).toEqual([copy])
    expect(stored(ACTIVE_LOOK_KEY)).toEqual(copy)
    // The copy is the user's own: it can be renamed and deleted.
    expect(manage().getByRole('button', { name: 'Rename' })).toBeInTheDocument()
  })

  it('Save writes the edits over the saved look they were made to, and only then is offered', async () => {
    const user = userEvent.setup()
    await readyOnMyLook()
    expect(toolbar()).toEqual(['Rename', 'Duplicate', 'Delete', 'Export', 'Copy JSON'])

    await user.click(group(/^Accent/).getByRole('button', { name: 'Gold' }))
    expect(status()).toHaveTextContent(/^My look 1 · edited$/)
    expect(toolbar()).toEqual(['Save', 'Rename', 'Duplicate', 'Delete', 'Export', 'Copy JSON'])
    await user.click(manage().getByRole('button', { name: 'Save' }))

    expect(status()).toHaveTextContent(/^My look 1$/)
    expect(manage().getByText('Saved My look 1.')).toBeInTheDocument()
    expect(stored(SAVED_LOOKS_KEY)).toEqual([myLook({ accent: 'gold' })])
    expect(stored(ACTIVE_LOOK_KEY)).toEqual(myLook({ accent: 'gold' }))
    expect(looksRow().getAllByRole('button')).toHaveLength(PRESETS.length + 1)
    expect(manage().queryByRole('button', { name: 'Save' })).not.toBeInTheDocument()
    await lookOnPage(myLook({ accent: 'gold' }))
  })

  it('closes an open form when another look is picked', async () => {
    const user = userEvent.setup()
    await readyOnMyLook()
    await user.click(manage().getByRole('button', { name: 'Rename' }))
    expect(manage().getByRole('textbox', { name: 'New name' })).toBeInTheDocument()
    await user.click(looksRow().getByRole('button', { name: /^Aurora\b/ }))
    await user.click(looksRow().getByRole('button', { name: /^My look 1\b/ }))
    expect(manage().queryByRole('textbox')).not.toBeInTheDocument()
  })

  it('renames a saved look', async () => {
    const user = userEvent.setup()
    await readyOnMyLook()
    await user.click(manage().getByRole('button', { name: 'Rename' }))
    const field = manage().getByRole('textbox', { name: 'New name' })
    expect(field).toHaveValue('My look 1')
    expect(field).toHaveFocus()
    await user.clear(field)
    await user.type(field, 'Sunset')
    await user.click(manage().getByRole('button', { name: 'Save name' }))

    expect(status()).toHaveTextContent(/^Sunset$/)
    expect(pressedCard()).toHaveTextContent('Sunset')
    expect(manage().getByText('Renamed to Sunset.')).toBeInTheDocument()
    expect(manage().queryByRole('textbox')).not.toBeInTheDocument()
    expect(stored(SAVED_LOOKS_KEY)).toEqual([myLook({ name: 'Sunset' })])
    expect(stored(ACTIVE_LOOK_KEY)).toEqual(myLook({ name: 'Sunset' }))
  })

  it("says in words why a name wasn't saved, and keeps the old one", async () => {
    const user = userEvent.setup()
    await readyOnMyLook()
    await user.click(manage().getByRole('button', { name: 'Rename' }))
    await user.clear(manage().getByRole('textbox', { name: 'New name' }))
    await user.click(manage().getByRole('button', { name: 'Save name' }))

    const alert = manage().getByRole('alert')
    expect(alert).toHaveTextContent("That name wasn't saved.")
    expect(alert).toHaveTextContent(`"name" can't be empty.`)
    expect(status()).toHaveTextContent(/^My look 1$/)
    expect(stored(SAVED_LOOKS_KEY)).toEqual([myLook()])
  })

  it('deletes a saved look only once confirmed, then goes back to Ember', async () => {
    const user = userEvent.setup()
    await readyOnMyLook()
    await user.click(manage().getByRole('button', { name: 'Delete' }))
    const confirm = () => within(screen.getByRole('group', { name: /^Delete My look 1\?/ }))
    await user.click(confirm().getByRole('button', { name: 'Keep it' }))
    expect(screen.queryByRole('group', { name: /^Delete My look 1\?/ })).not.toBeInTheDocument()
    expect(stored(SAVED_LOOKS_KEY)).toEqual([myLook()])

    await user.click(manage().getByRole('button', { name: 'Delete' }))
    await user.click(confirm().getByRole('button', { name: 'Delete look' }))
    expect(status()).toHaveTextContent(/^Ember$/)
    expect(looksRow().getAllByRole('button')).toHaveLength(PRESETS.length)
    expect(manage().getByText('Deleted My look 1. Play is back to Ember.')).toBeInTheDocument()
    expect(stored(SAVED_LOOKS_KEY)).toEqual([])
    await lookOnPage(EMBER)
  })
})

describe('Export and Copy JSON', () => {
  let stub: DownloadStub

  beforeEach(() => {
    stub = stubDownloads()
  })

  afterEach(() => stub.restore())

  async function exportActive(user: User): Promise<string> {
    await user.click(manage().getByRole('button', { name: 'Export' }))
    expect(stub.downloads).toHaveLength(1)
    expect(stub.downloads[0]?.href).toBe(BLOB_URL)
    expect(stub.blobs).toHaveLength(1)
    expect(stub.blobs[0]?.type).toBe(LOOK_FILE_TYPE)
    // The URL is let go once the download has started.
    await waitFor(() => expect(stub.revoke).toHaveBeenCalledWith(BLOB_URL), { timeout: 2000 })
    return stub.downloads[0]?.download ?? ''
  }

  it('downloads <name>.look.json holding the active look, pretty-printed', async () => {
    const user = userEvent.setup()
    await readyOnMyLook()
    expect(await exportActive(user)).toBe('My look 1.look.json')
    expect(await blobText(stub.blobs[0] as Blob)).toBe(lookJson(myLook()))
    expect(manage().getByText('Downloaded My look 1.look.json.')).toBeInTheDocument()
  })

  it('round-trips: an exported look imports as an equal saved look with a new id', async () => {
    const user = userEvent.setup()
    renderLooks()
    await lookOnPage(EMBER)
    await exportActive(user)
    const text = await blobText(stub.blobs[0] as Blob)

    await paste(user, text)
    const imported = { ...EMBER, id: 'look-1' }
    expect(stored(SAVED_LOOKS_KEY)).toEqual([imported])
    expect(stored(ACTIVE_LOOK_KEY)).toEqual(imported)
    const cards = looksRow().getAllByRole('button')
    expect(cards).toHaveLength(PRESETS.length + 1)
    expect(cards[cards.length - 1]).toHaveAttribute('aria-pressed', 'true')
    expect(importer().getByText('Added Ember to your Looks.')).toBeInTheDocument()
    expect(importer().getByLabelText("Paste a Look's JSON")).toHaveValue('')
  })

  it("copies the active look's JSON", async () => {
    const user = userEvent.setup()
    renderLooks()
    await lookOnPage(EMBER)
    await user.click(manage().getByRole('button', { name: 'Copy JSON' }))
    expect(await navigator.clipboard.readText()).toBe(lookJson(EMBER))
    expect(await manage().findByText('Copied the JSON.')).toBeInTheDocument()
  })

  it('says so when the browser blocks the clipboard', async () => {
    const user = userEvent.setup()
    renderLooks()
    await lookOnPage(EMBER)
    const denied = new DOMException('Write permission denied.', 'NotAllowedError')
    vi.spyOn(navigator.clipboard, 'writeText').mockRejectedValue(denied)
    await user.click(manage().getByRole('button', { name: 'Copy JSON' }))
    expect(await manage().findByRole('alert')).toHaveTextContent("The JSON wasn't copied.")
    expect(console.error).toHaveBeenCalledWith('Failed to copy to clipboard:', denied)
  })
})

describe('Import a look', () => {
  function fileInput(): HTMLInputElement {
    return importer().getByLabelText<HTMLInputElement>('Or choose a .json file')
  }

  it('adds a valid .json file as a saved look with a fresh id, and picks it', async () => {
    const user = userEvent.setup()
    renderLooks()
    await lookOnPage(EMBER)
    const file = new File([lookJson(myLook({ id: 'ember' }))], 'mine.look.json', { type: LOOK_FILE_TYPE })
    await user.upload(fileInput(), file)

    const added = myLook({ id: 'look-1' })
    await waitFor(() => expect(stored(SAVED_LOOKS_KEY)).toEqual([added]))
    expect(status()).toHaveTextContent(/^My look 1$/)
    expect(pressedCard()).toHaveTextContent('My look 1')
    expect(importer().getByText('Added My look 1 to your Looks.')).toBeInTheDocument()
    await lookOnPage(added)
  })

  it("refuses pasted text that isn't JSON and keeps it in the box to fix", async () => {
    const user = userEvent.setup()
    renderLooks()
    await lookOnPage(EMBER)
    await paste(user, '{ name: Ember')
    expect(importer().getByRole('alert')).toHaveTextContent(NOT_JSON)
    expect(importer().getByLabelText("Paste a Look's JSON")).toHaveValue('{ name: Ember')
    expect(stored(SAVED_LOOKS_KEY)).toBeNull()
  })

  it('never renders imported text as HTML', async () => {
    const user = userEvent.setup()
    renderLooks()
    await lookOnPage(EMBER)
    await paste(user, JSON.stringify(myLook({ name: '<b>bold</b>', sub: '<img src=x onerror=alert(1)>' })))

    expect(pressedCard()).toHaveTextContent('<b>bold</b> <img src=x onerror=alert(1)>')
    expect(importer().getByText('Added <b>bold</b> to your Looks.')).toBeInTheDocument()
    const row = screen.getByRole('region', { name: 'Looks' })
    const imports = screen.getByRole('region', { name: 'Import a look' })
    expect([row, imports].map((region) => region.querySelector('b, img'))).toEqual([null, null])
    expect(status()).toHaveTextContent('<b>bold</b>')
  })
})
