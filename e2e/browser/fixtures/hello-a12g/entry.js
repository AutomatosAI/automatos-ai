// PRD-261 A-12g: the playbooks-data hello theme. Besides a frame that only wraps the page, it
// registers its own page on /playbooks, which reads its data through the SDK's pages/playbooks
// area alone:
// - the list: useWorkflowPlaybooks(), each playbook with an Open button;
// - the open playbook: useWorkflowPlaybook() for its steps, usePlaybookExecutions() for its runs,
//   and a Run button that runs useExecutePlaybook() and shows the new run's id.
// It imports nothing of the host but the shared modules. It reaches the browser only through the
// lane's route interception (e2e/browser/specs/pages-playbooks.spec.ts, through
// support/hello-theme.ts), which also answers the Playbook endpoints from responses.json.
import { useState } from 'react'
import { jsx, jsxs } from 'react/jsx-runtime'
import {
  defineTheme,
  useExecutePlaybook,
  usePlaybookExecutions,
  useWorkflowPlaybook,
  useWorkflowPlaybooks,
} from '@automatos/theme-sdk'

const LIST_LIMIT = 10
const RUNS_LIMIT = 5
const LIST_SORT = 'name'

/** @param {{ children?: import('react').ReactNode }} props */
function PlaybooksFrame({ children }) {
  return jsx('div', { 'data-testid': 'hello-a12g-frame', className: 'hello-a12g-frame', children })
}

/** @param {{ playbook: { template_id: string, name: string }, onOpen: (id: string) => void }} props */
function PlaybookRow({ playbook, onOpen }) {
  return jsxs('li', {
    'data-testid': 'hello-a12g-playbook',
    'data-template-id': playbook.template_id,
    children: [
      jsx('span', { 'data-testid': 'hello-a12g-playbook-name', children: playbook.name }),
      jsx('button', { type: 'button', onClick: () => onOpen(playbook.template_id), children: `Open ${playbook.name}` }),
    ],
  })
}

/** @param {{ playbookId: string }} props */
function OpenPlaybook({ playbookId }) {
  const playbook = useWorkflowPlaybook(playbookId)
  const runs = usePlaybookExecutions(playbookId, { limit: RUNS_LIMIT })
  const execute = useExecutePlaybook()
  const detail = playbook.data
  const items = runs.data ? runs.data.items : []
  return jsxs('section', {
    'data-testid': 'hello-a12g-open',
    'aria-label': 'Open playbook',
    children: [
      jsx('h3', { 'data-testid': 'hello-a12g-open-name', children: detail ? detail.name : '' }),
      jsx('p', { 'data-testid': 'hello-a12g-steps', children: detail ? String(detail.steps.length) : '' }),
      jsx('ul', {
        'data-testid': 'hello-a12g-runs',
        children: items.map((run) =>
          jsx('li', { 'data-testid': 'hello-a12g-run', 'data-status': run.status, children: run.execution_id }, run.execution_id),
        ),
      }),
      jsx('button', {
        type: 'button',
        'data-testid': 'hello-a12g-run-now',
        disabled: detail === null || execute.isRunning,
        onClick: () => execute.run({ id: playbookId }),
        children: 'Run',
      }),
      jsx('p', { 'data-testid': 'hello-a12g-started', children: execute.result ? execute.result.recipe_execution_id : '' }),
    ],
  })
}

function HelloPlaybooksPage() {
  const [openId, setOpenId] = useState(/** @type {string | null} */ (null))
  const playbooks = useWorkflowPlaybooks({ limit: LIST_LIMIT, sort_by: LIST_SORT })
  const rows = playbooks.data ? playbooks.data.items : []
  return jsxs('section', {
    'data-testid': 'hello-a12g-page',
    className: 'hello-a12g-page',
    'aria-label': 'Hello playbooks',
    children: [
      jsx('h2', { children: 'Playbooks, read through the SDK' }),
      jsx('p', { 'data-testid': 'hello-a12g-total', children: playbooks.data ? String(playbooks.data.total) : '' }),
      jsx('ul', {
        'data-testid': 'hello-a12g-playbooks',
        children: rows.map((playbook) => jsx(PlaybookRow, { playbook, onOpen: setOpenId }, playbook.template_id)),
      }),
      openId === null ? null : jsx(OpenPlaybook, { playbookId: openId }),
    ],
  })
}

export default defineTheme({
  register(host) {
    host.registerSlot('frame@1', PlaybooksFrame)
    host.registerPage('/playbooks', HelloPlaybooksPage)
  },
  unregister() {},
})
