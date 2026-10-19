// PRD-261 A-10a: the SDK-context hello theme. Its frame reads the host through the SDK alone:
// useWorkspace() and useRole() (context), useCoreNav() and HostLink (navigation), and notify()
// (notices), whose toast the host's own sonner draws. It imports nothing else of the host, and
// never sonner. It reaches the browser only through the lane's route interception
// (e2e/browser/specs/sdk-context.spec.ts, through support/hello-theme.ts).
import { jsx, jsxs } from 'react/jsx-runtime'
import { defineTheme, HostLink, notify, useCoreNav, useRole, useWorkspace } from '@automatos/theme-sdk'

const NOTICE_TITLE = 'Hello from the SDK'

/** @param {{ children?: import('react').ReactNode }} props */
function HelloContextFrame({ children }) {
  const { workspace, isLoading } = useWorkspace()
  const role = useRole()
  const nav = useCoreNav()
  const first = nav.primary[0]
  const bar = jsxs('div', {
    className: 'hello-a10a-bar',
    children: [
      jsx('strong', { children: 'Hello SDK context' }),
      jsx('span', {
        'data-testid': 'hello-a10a-workspace',
        'data-workspace-id': workspace ? workspace.id : '',
        'data-loading': isLoading ? 'true' : 'false',
        children: workspace ? workspace.name : 'No workspace yet',
      }),
      jsx('span', { 'data-testid': 'hello-a10a-can-edit', children: role.canEdit ? 'can edit' : 'read only' }),
      first ? jsx(HostLink, { href: first.href, 'data-testid': 'hello-a10a-first-nav', children: first.label }) : null,
      jsx('button', {
        type: 'button',
        'data-testid': 'hello-a10a-notify',
        onClick: () => notify({ kind: 'success', title: NOTICE_TITLE }),
        children: 'Notify',
      }),
    ],
  })
  return jsxs('div', {
    'data-testid': 'hello-a10a-frame',
    className: 'hello-a10a-frame',
    children: [bar, jsx('div', { 'data-testid': 'hello-a10a-content', className: 'hello-a10a-content', children })],
  })
}

export default defineTheme({
  register(host) {
    host.registerSlot('frame@1', HelloContextFrame)
  },
  unregister() {},
})
