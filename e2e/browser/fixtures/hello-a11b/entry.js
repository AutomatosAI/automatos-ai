// @ts-check
// PRD-261 A-11b: the SDK-chat hello theme. It registers its own page on /chat and draws the chat from
// the SDK alone: useChatController() for the conversation on screen, then useChatController(conversation)
// in a child keyed by the conversation for its stream (send, Stop, Try again), and the host's
// ToolApproval card for each approval a reply raised. It imports nothing of the host but React and the
// SDK, and reaches the browser only through the lane's route interception (e2e/browser/specs/
// sdk-chat.spec.ts, through support/hello-theme.ts); its bare imports resolve through the import map.
import { useState } from 'react'
import { jsx, jsxs } from 'react/jsx-runtime'
import { defineTheme, ToolApproval, useChatController } from '@automatos/theme-sdk'

const OPENING = 'Opening the conversation'

/** @param {{ children?: import('react').ReactNode }} props */
function ChatFrame({ children }) {
  return jsx('div', { 'data-testid': 'hello-a11b-frame', className: 'hello-a11b-frame', children })
}

/**
 * The host's own approval card, told when its decision went through and when it closes.
 * @param {{ approval: any }} props
 */
function HelloApproval({ approval }) {
  const card =
    approval.kind === 'tool'
      ? jsx(ToolApproval, {
          data: approval.data,
          conversationId: approval.chatId || undefined,
          createdAt: approval.filedAt,
          onDecided: () => approval.answered(),
          onClose: () => approval.closed(),
        })
      : null
  return jsx('div', { 'data-testid': 'hello-a11b-approval', 'data-status': approval.status, children: card })
}

/** @param {{ message: any }} props */
function HelloMessage({ message }) {
  return jsxs('li', {
    'data-testid': 'hello-a11b-message',
    className: 'hello-a11b-message',
    'data-role': message.role,
    'data-writing': String(message.writing),
    'data-stopped': String(message.stopped),
    children: [
      jsx('p', { 'data-testid': 'hello-a11b-text', children: message.text }),
      ...message.approvals.map((/** @type {any} */ approval) => jsx(HelloApproval, { approval }, approval.key)),
    ],
  })
}

/** @param {{ chat: any }} props */
function HelloComposer({ chat }) {
  const [text, setText] = useState('')
  /** @param {{ preventDefault: () => void }} event */
  const submit = (event) => {
    event.preventDefault()
    if (text.trim() !== '' && chat.send(text)) setText('')
  }
  return jsxs('form', {
    className: 'hello-a11b-composer',
    onSubmit: submit,
    children: [
      jsx('textarea', {
        'aria-label': 'Message',
        value: text,
        onChange: (/** @type {{ target: { value: string } }} */ event) => setText(event.target.value),
      }),
      jsx('button', { type: 'submit', disabled: chat.streaming, children: 'Send' }),
      jsx('button', { type: 'button', disabled: !chat.streaming, onClick: () => chat.stop(), children: 'Stop' }),
      jsx('button', { type: 'button', disabled: chat.streaming, onClick: () => chat.retry(), children: 'Try again' }),
    ],
  })
}

/**
 * One conversation's chat: mounted afresh (keyed) for each conversation, as the SDK asks.
 * @param {{ conversation: any }} props
 */
function HelloConversation({ conversation }) {
  const chat = useChatController(conversation)
  return jsxs('section', {
    'data-testid': 'hello-a11b-chat',
    className: 'hello-a11b-chat',
    'data-status': chat.status,
    'data-chat-id': chat.conversation.chatId || '',
    'aria-label': 'Hello chat',
    children: [
      jsx('h1', { children: chat.title }),
      jsx('ol', {
        className: 'hello-a11b-messages',
        children: chat.messages.map((/** @type {any} */ message) => jsx(HelloMessage, { message }, message.id)),
      }),
      ...chat.looseApprovals.map((/** @type {any} */ approval) => jsx(HelloApproval, { approval }, approval.key)),
      jsx(HelloComposer, { chat }),
    ],
  })
}

/** The chat page: the conversation on screen, or a line while it opens. */
function HelloChatPage() {
  const { conversation } = useChatController()
  if (!conversation) return jsx('p', { 'data-testid': 'hello-a11b-opening', role: 'status', children: OPENING })
  return jsx(HelloConversation, { conversation }, conversation.key)
}

export default defineTheme({
  register(host) {
    host.registerSlot('frame@1', ChatFrame)
    host.registerPage('/chat', HelloChatPage)
  },
  unregister() {},
})
