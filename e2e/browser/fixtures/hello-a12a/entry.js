// PRD-261 A-12a: the host-cards hello theme. Its frame renders the host's HeaderUtilities in its
// own header (PRD §4.4), and its route `hello-a12a` (served at /x/hello-a12a) places the host's
// ToolApproval card for one grant, from '@automatos/theme-sdk' alone: the theme never draws its own
// approval card, and Approve & run is the host's own call. The route shows what the card told it
// through onDecided. It reaches the browser only through the lane's route interception
// (e2e/browser/specs/host-cards.spec.ts, through support/hello-theme.ts), and its bare imports
// resolve through the host's import map to the host's own React and theme SDK.
import { jsx, jsxs } from 'react/jsx-runtime'
import { useState } from 'react'
import { defineTheme, HeaderUtilities, ToolApproval } from '@automatos/theme-sdk'

const ROUTE = 'hello-a12a'
const WAITING = 'waiting'

/** One confirmation-gated tool call, as the chat's tool_approval payload carries it. Test data only. */
const GRANT = {
  grant_id: 12101,
  action: 'platform_delete_document',
  message: 'This action (destructive) requires confirmation.',
  permission_level: 'destructive',
  params: { document_id: 'e2e-a12a-doc' },
  risk_tier: 'human_in_the_loop',
  oversight_rationale: 'Deletes a document. A human must approve before it runs.',
}

/** @param {{ children?: import('react').ReactNode }} props */
function CardsFrame({ children }) {
  const header = jsxs('header', {
    'data-testid': 'hello-a12a-header',
    className: 'hello-a12a-header',
    children: [jsx('strong', { children: 'Hello host cards' }), jsx(HeaderUtilities, {})],
  })
  return jsxs('div', {
    'data-testid': 'hello-a12a-frame',
    className: 'hello-a12a-frame',
    children: [header, jsx('div', { 'data-testid': 'hello-a12a-content', children })],
  })
}

function CardsRoute() {
  const [decision, setDecision] = useState(WAITING)
  return jsxs('section', {
    'data-testid': 'hello-a12a-route',
    className: 'hello-a12a-route',
    'aria-label': 'Host cards',
    children: [
      jsx(ToolApproval, { data: GRANT, conversationId: 'e2e-a12a-chat', onDecided: setDecision }),
      jsx('p', { 'data-testid': 'hello-a12a-decision', children: decision }),
    ],
  })
}

export default defineTheme({
  register(host) {
    host.registerSlot('frame@1', CardsFrame)
    host.registerRoute(ROUTE, CardsRoute)
  },
  unregister() {},
})
