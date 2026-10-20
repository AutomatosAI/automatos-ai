/**
 * PRD-260 §4, K8, PRD-261 A-11a (M28) — which Deliverables a turn made, and the core's page for one
 * (lib/chat/deliverables.ts), tested in core. Until W-L1 the only tests of these functions were
 * builtin Play's (components/play/chat/__tests__/deliverables.test.ts) and the port guards that hold
 * Play's copies to them (lib/__tests__/prd261-port-{chat,deliverables}.test.ts), all of which W-A8
 * deletes. This file replaces them in the core test ledger (plan §2.2.8):
 *
 * - every assertion of components/play/chat/__tests__/deliverables.test.ts (all of it tests core);
 * - deliverableHref as the port guards pin it ('d-laptop', 'A_b-9', 'a b');
 * - isDeliverableId on the ids lib/__tests__/prd261-port-deliverables.test.ts holds Play's copy to,
 *   each one accepted or refused here by name, and kindOfFormat's format mapping;
 * - the card's link of components/play/chat/__tests__/play-chat-turn.test.tsx: a live tool-data
 *   part naming d-42 links '/deliverables?deliverable=d-42'.
 *
 * P260-FX-09 (M76): a receipt link is a path in the app or the core's absolute open_url
 * (deliverable_open_url in generate_document_tool.py: FRONTEND_URL +
 * /deliverables?tab=outputs&deliverable=<id>); only its checked id is kept, and the card links the
 * core's own page relatively. Receipts don't link a Deliverable yet (receipts.py, decision F8).
 */
import { describe, expect, it } from 'vitest'

import {
  deliverableFromToolData,
  deliverableHref,
  deliverableIdFromLink,
  deliverablesFromReceipts,
  deliverablesInPart,
  isDeliverableId,
  kindOfFormat,
  UNTITLED,
} from '@/lib/chat/deliverables'
import type { Receipt } from '@/types'

const ID = '9b2f4c1e-7a3d-4e5f-8a90-1b2c3d4e5f60'
const OPEN_URL = 'https://app.example/deliverables?tab=outputs&deliverable=d-7'
const LIVE_DOCUMENT = { deliverable_id: 'd-42', title: 'Laptop shortlist', format: 'pdf' }

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

describe("deliverableHref: the core's page for one Deliverable (K8)", () => {
  it("links the core's page relatively, whatever origin the receipt named", () => {
    expect(deliverableHref(deliverableIdFromLink(OPEN_URL) ?? '')).toBe('/deliverables?deliverable=d-7')
  })

  it('links the Deliverable a live tool-data part announces: d-42 opens /deliverables?deliverable=d-42', () => {
    const [made] = deliverablesInPart({ type: 'tool-data', data: { generated_document: LIVE_DOCUMENT } })
    expect(made).toEqual({ id: 'd-42', title: 'Laptop shortlist', kind: 'document' })
    expect(deliverableHref(made.id)).toBe('/deliverables?deliverable=d-42')
  })

  it('puts the id in the deliverable parameter, encoded', () => {
    expect(deliverableHref('d-laptop')).toBe('/deliverables?deliverable=d-laptop')
    expect(deliverableHref('A_b-9')).toBe('/deliverables?deliverable=A_b-9')
    expect(deliverableHref('a b')).toBe('/deliverables?deliverable=a%20b')
  })
})

describe('deliverableIdFromLink: the Deliverable a receipt link names', () => {
  it("reads the core's absolute open_url, keeping only the id (M76)", () => {
    expect(deliverableIdFromLink(OPEN_URL)).toBe('d-7')
    expect(deliverableIdFromLink(`http://localhost:3000/deliverables?tab=outputs&deliverable=${ID}`)).toBe(ID)
  })

  it('reads a path in the app: the open_url without FRONTEND_URL, and /deliverables/<id>', () => {
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

describe('isDeliverableId: an id the core links to or opens (K8)', () => {
  it('accepts letters, digits, - and _, from 1 to 128 of them', () => {
    const accepted = ['d-laptop', 'A_b-9', 'd-42', 'x', ID, 'x'.repeat(128)]
    expect(accepted.filter((id) => !isDeliverableId(id))).toEqual([])
  })

  it('refuses an empty id, spaces, a path, a query, a dot, a line break and anything over 128', () => {
    const refused = ['', ' ', '  d-1', 'd laptop', 'd/1', '../../admin?x=1', '..', 'd.pdf', 'd-1?x=1', 'd-1#top', 'd-1\n', 'x'.repeat(129)]
    expect(refused.filter(isDeliverableId)).toEqual([])
  })
})

describe("kindOfFormat: the generated document's format as a shelf kind", () => {
  it('reads the image and video formats as image and video, in any case', () => {
    const images = ['png', 'jpg', 'jpeg', 'gif', 'webp', 'svg', 'PNG', 'Jpeg']
    const videos = ['mp4', 'webm', 'mov', 'MP4', 'MoV']
    expect(images.map(kindOfFormat)).toEqual(images.map(() => 'image'))
    expect(videos.map(kindOfFormat)).toEqual(videos.map(() => 'video'))
  })

  it('files every other format, and none, as a document', () => {
    const documents = ['pdf', 'docx', 'md', 'xlsx', 'html', 'mp3', 'png.exe', '']
    expect(documents.map(kindOfFormat)).toEqual(documents.map(() => 'document'))
    expect(kindOfFormat(null)).toBe('document')
  })
})
