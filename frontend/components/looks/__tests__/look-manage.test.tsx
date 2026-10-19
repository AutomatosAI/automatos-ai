/**
 * PRD-260 T-008a — 'This look' and 'Import a look' on Make it yours, ported
 * from Automatos Play's tests/looks/LookManage.test.tsx: Save, Rename and
 * Delete for the user's saved looks (a preset only duplicates), Export as a
 * .look.json download, Copy JSON, and Import from pasted text or a file
 * through lib/looks/transfer.ts with its messages shown. Imports are untrusted:
 * nothing invalid is kept, and imported text is never read as HTML. Export
 * says the download started, never that it arrived (M71), and a success says
 * so when this browser isn't keeping looks (M98).
 *
 * P260-FX-22 (M20): by keyboard, focus never falls to <body> when the control
 * just used goes (Save, Save name, Cancel, Keep it, Delete look), and Delete
 * moves it into its question, on Keep it.
 *
 * P260-FX-24 (M22): the same outcome twice is announced twice. A second
 * identical success is a new node in the polite region, and a second identical
 * failure puts the alert in again, so a screen reader never skips a repeat.
 */
import React from 'react'
import { cleanup, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

vi.mock('@/components/layout/main-layout', () => ({
  MainLayout: ({ children }: { children: React.ReactNode }) => <div data-testid="main-layout">{children}</div>,
}))

import { exportStarted } from '@/components/looks/look-manager'
import { LOOKS_NOT_KEPT } from '@/components/looks/look-notice'
import { blockedStorage, memoryStorage, myLook } from '@/lib/looks/__tests__/helpers'
import { EMBER, PRESETS } from '@/lib/looks/presets'
import { LOOK_FILE_TYPE, NOT_JSON, lookJson } from '@/lib/looks/transfer'

import {
  BLOB_URL,
  type DownloadStub,
  KEYS,
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
import { BUILTIN_PLAY_STYLE } from '@/lib/theme-host/builtin/play-entry'

type User = ReturnType<typeof userEvent.setup>

/** Starts on the user's own 'My look 1' (rose on Night). */
async function readyOnMyLook(): Promise<void> {
  window.localStorage.setItem(KEYS.saved, JSON.stringify([myLook()]))
  window.localStorage.setItem(KEYS.active, JSON.stringify(myLook()))
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
    expect(stored(KEYS.saved)).toEqual([copy])
    expect(stored(KEYS.active)).toEqual(copy)
    // The copy is the user's own: it can be renamed and deleted.
    expect(manage().getByRole('button', { name: 'Rename' })).toBeInTheDocument()
  })

  it('keeps unsaved edits in the copy', async () => {
    const user = userEvent.setup()
    renderLooks()
    await lookOnPage(EMBER)
    await user.click(group(/^Accent/).getByRole('button', { name: 'Rose' }))
    expect(status()).toHaveTextContent(/^Ember · edited$/)
    await user.click(manage().getByRole('button', { name: 'Duplicate' }))

    const copy = { ...EMBER, id: 'look-1', name: 'Ember copy', accent: 'rose' } as const
    expect(stored(KEYS.saved)).toEqual([copy])
    expect(stored(KEYS.active)).toEqual(copy)
    expect(status()).toHaveTextContent(/^Ember copy$/)
    expect(pressedCard()).toHaveTextContent(`${copy.name} ${copy.sub}`)
    await lookOnPage(copy)
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
    expect(stored(KEYS.saved)).toEqual([myLook({ accent: 'gold' })])
    expect(stored(KEYS.active)).toEqual(myLook({ accent: 'gold' }))
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
    expect(stored(KEYS.saved)).toEqual([myLook({ name: 'Sunset' })])
    expect(stored(KEYS.active)).toEqual(myLook({ name: 'Sunset' }))
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
    expect(stored(KEYS.saved)).toEqual([myLook()])
  })

  it('deletes a saved look only once confirmed, then goes back to Ember', async () => {
    const user = userEvent.setup()
    await readyOnMyLook()
    await user.click(manage().getByRole('button', { name: 'Delete' }))
    const confirm = () => within(screen.getByRole('group', { name: /^Delete My look 1\?/ }))
    await user.click(confirm().getByRole('button', { name: 'Keep it' }))
    expect(screen.queryByRole('group', { name: /^Delete My look 1\?/ })).not.toBeInTheDocument()
    expect(stored(KEYS.saved)).toEqual([myLook()])

    await user.click(manage().getByRole('button', { name: 'Delete' }))
    await user.click(confirm().getByRole('button', { name: 'Delete look' }))
    expect(status()).toHaveTextContent(/^Ember$/)
    expect(looksRow().getAllByRole('button')).toHaveLength(PRESETS.length)
    expect(manage().getByText('Deleted My look 1. Play is back to Ember.')).toBeInTheDocument()
    expect(stored(KEYS.saved)).toEqual([])
    await lookOnPage(EMBER)
  })
})

describe('This look by keyboard: focus stays on a button that is there (P260-FX-22, M20)', () => {
  const tool = (name: string) => manage().getByRole('button', { name })

  /** Focus a control and press Enter on it, as a keyboard user does. */
  async function press(user: User, control: HTMLElement): Promise<void> {
    control.focus()
    await user.keyboard('{Enter}')
  }

  function expectFocusOn(control: HTMLElement): void {
    expect(control).toHaveFocus()
    expect(document.activeElement).not.toBe(document.body)
  }

  it('Save hands focus to Rename, since Save is gone once the look is saved', async () => {
    const user = userEvent.setup()
    await readyOnMyLook()
    await user.click(group(/^Accent/).getByRole('button', { name: 'Gold' }))
    await press(user, tool('Save'))
    expect(status()).toHaveTextContent(/^My look 1$/)
    expect(manage().queryByRole('button', { name: 'Save' })).not.toBeInTheDocument()
    expectFocusOn(tool('Rename'))
  })

  it('Save name and Cancel hand focus back to Rename as the form closes', async () => {
    const user = userEvent.setup()
    await readyOnMyLook()
    await press(user, tool('Rename'))
    const field = manage().getByRole('textbox', { name: 'New name' })
    expect(field).toHaveFocus()
    await user.clear(field)
    await user.type(field, 'Sunset')
    await press(user, tool('Save name'))
    expect(status()).toHaveTextContent(/^Sunset$/)
    expect(manage().queryByRole('textbox')).not.toBeInTheDocument()
    expectFocusOn(tool('Rename'))

    await press(user, tool('Rename'))
    await press(user, tool('Cancel'))
    expect(manage().queryByRole('textbox')).not.toBeInTheDocument()
    expectFocusOn(tool('Rename'))
  })

  it('Delete moves focus into its question, on Keep it; Keep it hands it back to Delete', async () => {
    const user = userEvent.setup()
    await readyOnMyLook()
    await press(user, tool('Delete'))
    const question = screen.getByRole('group', { name: /^Delete My look 1\?/ })
    expectFocusOn(within(question).getByRole('button', { name: 'Keep it' }))

    await press(user, within(question).getByRole('button', { name: 'Keep it' }))
    expect(screen.queryByRole('group', { name: /^Delete My look 1\?/ })).not.toBeInTheDocument()
    expect(stored(KEYS.saved)).toEqual([myLook()])
    expectFocusOn(tool('Delete'))
  })

  it('Delete look hands focus to Duplicate, the one button every look has', async () => {
    const user = userEvent.setup()
    await readyOnMyLook()
    await press(user, tool('Delete'))
    await press(user, tool('Delete look'))
    expect(status()).toHaveTextContent(/^Ember$/)
    expect(manage().queryByRole('button', { name: 'Delete' })).not.toBeInTheDocument()
    expectFocusOn(tool('Duplicate'))
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
    expect(manage().getByText(exportStarted('My look 1.look.json'))).toBeInTheDocument()
  })

  it("says the download started, never that it arrived, and offers Copy JSON (M71)", async () => {
    const user = userEvent.setup()
    renderLooks()
    await lookOnPage(EMBER)
    await exportActive(user)
    const notice = manage().getByText(exportStarted('Ember.look.json'))
    expect(notice).toHaveTextContent("Your browser is saving Ember.look.json. If it didn't appear, use Copy JSON.")
    expect(manage().queryByText(/^Downloaded/)).not.toBeInTheDocument()
  })

  it('round-trips: an exported look imports as an equal saved look with a new id', async () => {
    const user = userEvent.setup()
    renderLooks()
    await lookOnPage(EMBER)
    await exportActive(user)
    const text = await blobText(stub.blobs[0] as Blob)

    await paste(user, text)
    const imported = { ...EMBER, id: 'look-1' }
    expect(stored(KEYS.saved)).toEqual([imported])
    expect(stored(KEYS.active)).toEqual(imported)
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
    await waitFor(() => expect(stored(KEYS.saved)).toEqual([added]))
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
    expect(stored(KEYS.saved)).toBeNull()
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

describe("When this browser isn't keeping looks (M98)", () => {
  function fullStorage(): Storage {
    const full = memoryStorage()
    full.setItem = () => {
      throw new DOMException('The quota has been exceeded.', 'QuotaExceededError')
    }
    return full
  }

  it('with storage blocked, Duplicate, Rename and Import say the look lasts until the tab closes', async () => {
    const user = userEvent.setup()
    renderLooks(BUILTIN_PLAY_STYLE, blockedStorage)
    await lookOnPage(EMBER)

    await user.click(manage().getByRole('button', { name: 'Duplicate' }))
    expect(manage().getByText(`Duplicated as Ember copy. ${LOOKS_NOT_KEPT}`)).toBeInTheDocument()

    await user.click(manage().getByRole('button', { name: 'Rename' }))
    await user.clear(manage().getByRole('textbox', { name: 'New name' }))
    await user.type(manage().getByRole('textbox', { name: 'New name' }), 'Sunset')
    await user.click(manage().getByRole('button', { name: 'Save name' }))
    expect(manage().getByText(`Renamed to Sunset. ${LOOKS_NOT_KEPT}`)).toBeInTheDocument()

    await paste(user, lookJson(myLook()))
    expect(importer().getByText(`Added My look 1 to your Looks. ${LOOKS_NOT_KEPT}`)).toBeInTheDocument()
    expect([stored(KEYS.saved), stored(KEYS.active)]).toEqual([null, null])
  })

  it('with storage full, the refused write after Duplicate is told in its notice', async () => {
    const user = userEvent.setup()
    const full = fullStorage()
    renderLooks(BUILTIN_PLAY_STYLE, () => full)
    await lookOnPage(EMBER)
    await user.click(manage().getByRole('button', { name: 'Duplicate' }))
    expect(await manage().findByText(`Duplicated as Ember copy. ${LOOKS_NOT_KEPT}`)).toBeInTheDocument()
    expect(full.length).toBe(0)
  })

  it('Export still says the download started and offers Copy JSON', async () => {
    const user = userEvent.setup()
    const stub = stubDownloads()
    try {
      renderLooks(BUILTIN_PLAY_STYLE, blockedStorage)
      await lookOnPage(EMBER)
      await user.click(manage().getByRole('button', { name: 'Export' }))
      // Exact text: Export changes no look, so it never adds LOOKS_NOT_KEPT.
      expect(manage().getByText(exportStarted('Ember.look.json'))).toHaveTextContent('Copy JSON')
      await waitFor(() => expect(stub.revoke).toHaveBeenCalledWith(BLOB_URL), { timeout: 2000 })
    } finally {
      stub.restore()
    }
  })
})

describe('A repeat is announced again (P260-FX-24, M22)', () => {
  const ADDED = 'Added My look 1 to your Looks.'

  it("pasting the same JSON twice puts a new 'Added …' node in the polite region", async () => {
    const user = userEvent.setup()
    renderLooks()
    await lookOnPage(EMBER)
    await paste(user, lookJson(myLook()))
    const first = importer().getByText(ADDED)

    await paste(user, lookJson(myLook()))
    const second = importer().getByText(ADDED)
    expect(second).not.toBe(first)
    expect(first.isConnected).toBe(false)
    expect(second.closest('[aria-live="polite"]')).not.toBeNull()
    expect(stored(KEYS.saved)).toEqual([myLook({ id: 'look-1' }), myLook({ id: 'look-2' })])
  })

  it('two identical bad pastes put the alert in again', async () => {
    const user = userEvent.setup()
    renderLooks()
    await lookOnPage(EMBER)
    await paste(user, '{ name: Ember')
    const first = importer().getByRole('alert')
    expect(first).toHaveTextContent(NOT_JSON)

    await user.clear(importer().getByLabelText("Paste a Look's JSON"))
    await paste(user, '{ name: Ember')
    const second = importer().getByRole('alert')
    expect(second).toHaveTextContent(NOT_JSON)
    expect(second).not.toBe(first)
    expect(first.isConnected).toBe(false)
  })

  it('Copy JSON twice says it twice: the second is a new node', async () => {
    const user = userEvent.setup()
    renderLooks()
    await lookOnPage(EMBER)
    await user.click(manage().getByRole('button', { name: 'Copy JSON' }))
    const first = await manage().findByText('Copied the JSON.')

    await user.click(manage().getByRole('button', { name: 'Copy JSON' }))
    await waitFor(() => expect(first.isConnected).toBe(false))
    expect(manage().getByText('Copied the JSON.').closest('[aria-live="polite"]')).not.toBeNull()
  })
})
