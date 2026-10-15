/**
 * PRD-260 §4, K8 — which Deliverables a turn made (components/play/chat/deliverables.ts), ported
 * from Automatos Play's tests/chat/deliverables.test.ts onto the core's shapes. P260-FX-09 (M76):
 * a receipt link is a path in the app or the core's absolute open_url (deliverable_open_url in
 * generate_document_tool.py: FRONTEND_URL + /deliverables?tab=outputs&deliverable=<id>); only its
 * checked id is kept, and the card links the core's own page relatively. Receipts don't link a
 * Deliverable yet (receipts.py, decision F8): this is the frontend half.
 */
import { describe, expect, it } from 'vitest'

import {
  deliverableFromToolData,
  deliverableHref,
  deliverableIdFromLink,
  deliverablesFromReceipts,
  UNTITLED,
} from '@/components/play/chat/deliverables'
import type { Receipt } from '@/types'

const ID = '9b2f4c1e-7a3d-4e5f-8a90-1b2c3d4e5f60'
const OPEN_URL = 'https://app.example/deliverables?tab=outputs&deliverable=d-7'

function receipt(fields: Partial<Receipt>): Receipt {
  return {
    action: 'generate_document',
    kind: 'write',
    status: 'done',
    subject: 'Laptop shortlist',
    effect: 'document made',
    link: null,
    reason: null,
    ...fields,
  }
}

describe('deliverableIdFromLink: the Deliverable a receipt link names', () => {
  it("reads the core's absolute open_url, keeping only the id (M76)", () => {
    expect(deliverableIdFromLink(OPEN_URL)).toBe('d-7')
    expect(deliverableIdFromLink(`http://localhost:3000/deliverables?tab=outputs&deliverable=${ID}`)).toBe(ID)
  })

  it("links the core's page relatively, whatever origin the receipt named", () => {
    expect(deliverableHref(deliverableIdFromLink(OPEN_URL) ?? '')).toBe('/deliverables?deliverable=d-7')
  })

  it("reads a path in the app: the open_url without FRONTEND_URL, and Play's /deliverables/<id>", () => {
    expect(deliverableIdFromLink('/deliverables?tab=outputs&deliverable=d-7')).toBe('d-7')
    expect(deliverableIdFromLink(`/deliverables/${ID}`)).toBe(ID)
  })

  it('ignores card and agent pages, and ids that cannot go in a link', () => {
    expect(deliverableIdFromLink('/command-center?tab=board&task_id=12')).toBeNull()
    expect(deliverableIdFromLink('/agents?agent=7')).toBeNull()
    expect(deliverableIdFromLink('https://app.example/agents?deliverable=d-7')).toBeNull()
    expect(deliverableIdFromLink('/deliverables/a%2Fb')).toBeNull()
    expect(deliverableIdFromLink('https://app.example/deliverables?deliverable=..%2F..%2Fadmin')).toBeNull()
    expect(deliverableIdFromLink(null)).toBeNull()
  })

  it('ignores any other scheme, a protocol-relative link and a bare relative one', () => {
    expect(deliverableIdFromLink('javascript:alert(1)//deliverables?deliverable=d-7')).toBeNull()
    expect(deliverableIdFromLink('ftp://app.example/deliverables?deliverable=d-7')).toBeNull()
    expect(deliverableIdFromLink('//app.example/deliverables?deliverable=d-7')).toBeNull()
    expect(deliverableIdFromLink('deliverables?deliverable=d-7')).toBeNull()
  })
})

describe('deliverablesFromReceipts', () => {
  it('only done receipts count, titled by their subject; the absolute open_url works like a path', () => {
    const found = deliverablesFromReceipts([
      receipt({ link: OPEN_URL }),
      receipt({ link: OPEN_URL, status: 'refused' }),
      receipt({ link: `/deliverables/${ID}`, subject: '  ' }),
    ])
    expect(found).toEqual([
      { id: 'd-7', title: 'Laptop shortlist', kind: 'document' },
      { id: ID, title: UNTITLED, kind: 'document' },
    ])
    expect(deliverablesFromReceipts(undefined)).toEqual([])
  })
})

describe('deliverableFromToolData', () => {
  it("reads the generated document's id, title and kind; a bad id is none", () => {
    const made = { generated_document: { deliverable_id: ID, title: 'Laptop shortlist', format: 'png' } }
    expect(deliverableFromToolData(made)).toEqual({ id: ID, title: 'Laptop shortlist', kind: 'image' })
    expect(deliverableFromToolData({ generated_document: { deliverable_id: '../../admin' } })).toBeNull()
    expect(deliverableFromToolData({ receipts: { receipts: [] } })).toBeNull()
  })
})
