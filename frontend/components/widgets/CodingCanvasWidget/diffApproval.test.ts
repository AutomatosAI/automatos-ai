/**
 * PRD-170 S4 — diff approval model (vitest).
 *
 * The approve/deny decision flow + session-scoped auto-accept (default OFF,
 * per-turn, never applied to bash/exec permissions).
 */
import { describe, it, expect } from 'vitest'

import {
  initialDiffApprovalState,
  addEditCard,
  addPermissionCard,
  decide,
  setAutoAccept,
  pendingCards,
  languageForPath,
  type DiffApprovalState,
} from './diffApproval'

const edit = (requestId: string, path = 'src/app.py') => ({
  requestId,
  toolName: 'Write',
  path,
  oldContent: 'old',
  newContent: 'new',
})

describe('diff approval model', () => {
  it('defaults auto-accept OFF', () => {
    expect(initialDiffApprovalState.autoAcceptEdits).toBe(false)
  })

  it('an incoming edit lands pending when auto-accept is OFF', () => {
    const s = addEditCard(initialDiffApprovalState, edit('r1'))
    expect(s.cards).toHaveLength(1)
    expect(s.cards[0].status).toBe('pending')
    expect(s.cards[0].autoAccepted).toBe(false)
    expect(s.cards[0].diff?.path).toBe('src/app.py')
    expect(s.cards[0].diff?.language).toBe('python')
  })

  it('approve resolves a card to approved', () => {
    let s = addEditCard(initialDiffApprovalState, edit('r1'))
    s = decide(s, 'r1', 'approve')
    expect(s.cards[0].status).toBe('approved')
  })

  it('deny resolves a card to denied', () => {
    let s = addEditCard(initialDiffApprovalState, edit('r1'))
    s = decide(s, 'r1', 'deny')
    expect(s.cards[0].status).toBe('denied')
  })

  it('decide is a no-op on an already-resolved card', () => {
    let s = addEditCard(initialDiffApprovalState, edit('r1'))
    s = decide(s, 'r1', 'approve')
    const before = s.cards[0].status
    s = decide(s, 'r1', 'deny') // must not flip an approved card
    expect(s.cards[0].status).toBe(before)
  })

  it('auto-accept ON pre-approves incoming EDIT cards (visibly flagged)', () => {
    let s: DiffApprovalState = setAutoAccept(initialDiffApprovalState, true)
    s = addEditCard(s, edit('r1'))
    expect(s.cards[0].status).toBe('approved')
    expect(s.cards[0].autoAccepted).toBe(true)
    expect(pendingCards(s)).toHaveLength(0)
  })

  it('auto-accept NEVER applies to a bash/exec permission card', () => {
    let s = setAutoAccept(initialDiffApprovalState, true)
    s = addPermissionCard(s, 'r-bash', 'Bash')
    expect(s.cards[0].status).toBe('pending') // still requires explicit decision
    expect(s.cards[0].autoAccepted).toBe(false)
    expect(pendingCards(s)).toHaveLength(1)
  })

  it('toggling auto-accept does not retroactively resolve pending cards', () => {
    let s = addEditCard(initialDiffApprovalState, edit('r1')) // pending
    s = setAutoAccept(s, true)
    expect(s.cards[0].status).toBe('pending')
    // but the NEXT edit auto-accepts
    s = addEditCard(s, edit('r2'))
    expect(s.cards[1].status).toBe('approved')
  })

  it('new-file diff (empty old content) derives language and keeps both sides', () => {
    const s = addEditCard(initialDiffApprovalState, {
      requestId: 'r1',
      toolName: 'Write',
      path: 'README.md',
      oldContent: '',
      newContent: '# Title',
    })
    expect(s.cards[0].diff).toMatchObject({
      path: 'README.md',
      oldContent: '',
      newContent: '# Title',
      language: 'markdown',
    })
  })

  it('languageForPath maps common extensions and falls back to plaintext', () => {
    expect(languageForPath('a.ts')).toBe('typescript')
    expect(languageForPath('a.py')).toBe('python')
    expect(languageForPath('Makefile')).toBe('plaintext')
    expect(languageForPath('noext.')).toBe('plaintext')
  })
})
