// PRD-261 A-12f: the knowledge-data hello theme. Besides a frame that only wraps the page, it
// registers its own page on /documents (Gerard's Q3 "knowledge", D0), and that page reads its data
// through the SDK's pages/documents area alone: useDocuments(), useDocumentStats(), useTeams(),
// useDocumentTeamCounts(), useCloudConnections() and useDatabaseKnowledge(), with a Delete button
// per document that runs useDeleteDocument(). It imports nothing of the host but the shared
// modules. It reaches the browser only through the lane's route interception
// (e2e/browser/specs/pages-documents.spec.ts, through support/hello-theme.ts), which also answers
// the Knowledge endpoints from responses.json.
import { jsx, jsxs } from 'react/jsx-runtime'
import {
  defineTheme,
  useCloudConnections,
  useDatabaseKnowledge,
  useDeleteDocument,
  useDocumentStats,
  useDocumentTeamCounts,
  useDocuments,
  useTeams,
} from '@automatos/theme-sdk'

const NO_ROWS = []

/** @param {{ children?: import('react').ReactNode }} props */
function KnowledgeFrame({ children }) {
  return jsx('div', { 'data-testid': 'hello-a12f-frame', className: 'hello-a12f-frame', children })
}

/** @param {{ document: { id: number, filename: string, status: string }, onDelete: (id: number) => void, busy: boolean }} props */
function DocumentRow({ document, onDelete, busy }) {
  return jsxs('li', {
    'data-testid': 'hello-a12f-document',
    'data-document-id': String(document.id),
    'data-status': document.status,
    children: [
      jsx('span', { 'data-testid': 'hello-a12f-filename', children: document.filename }),
      jsx('button', {
        type: 'button',
        'data-testid': 'hello-a12f-delete',
        disabled: busy,
        onClick: () => onDelete(document.id),
        children: `Delete ${document.filename}`,
      }),
    ],
  })
}

function DocumentList() {
  const documents = useDocuments()
  const remove = useDeleteDocument()
  const rows = documents.data || NO_ROWS
  return jsx('ul', {
    'data-testid': 'hello-a12f-documents',
    'data-loading': documents.isLoading ? 'true' : 'false',
    children: rows.map((document) =>
      jsx(DocumentRow, { document, onDelete: remove.run, busy: remove.isRunning }, String(document.id)),
    ),
  })
}

function TeamList() {
  const teams = useTeams()
  const counts = useDocumentTeamCounts()
  const perTeam = counts.data ? counts.data.counts : {}
  return jsx('ul', {
    'data-testid': 'hello-a12f-teams',
    children: (teams.data || NO_ROWS).map((team) =>
      jsx('li', { 'data-testid': 'hello-a12f-team', children: `${team.name} (${perTeam[team.normalized_name] || 0})` }, team.normalized_name),
    ),
  })
}

function SourceLists() {
  const connections = useCloudConnections()
  const database = useDatabaseKnowledge()
  return jsxs('div', {
    children: [
      jsx('ul', {
        'data-testid': 'hello-a12f-connections',
        children: (connections.data || NO_ROWS).map((connection) =>
          jsx('li', { 'data-testid': 'hello-a12f-connection', children: connection.app_name }, String(connection.id)),
        ),
      }),
      jsx('ul', {
        'data-testid': 'hello-a12f-sources',
        children: database.sources.map((source) =>
          jsx('li', { 'data-testid': 'hello-a12f-source', children: `${source.name} (${source.tables_count} tables)` }, String(source.id)),
        ),
      }),
    ],
  })
}

function HelloKnowledgePage() {
  const stats = useDocumentStats()
  return jsxs('section', {
    'data-testid': 'hello-a12f-page',
    className: 'hello-a12f-page',
    'aria-label': 'Hello knowledge',
    children: [
      jsx('h2', { children: 'Knowledge, read through the SDK' }),
      jsx('p', { 'data-testid': 'hello-a12f-total', children: stats.data ? String(stats.data.total_documents) : '' }),
      jsx(DocumentList, {}),
      jsx(TeamList, {}),
      jsx(SourceLists, {}),
    ],
  })
}

export default defineTheme({
  register(host) {
    host.registerSlot('frame@1', KnowledgeFrame)
    host.registerPage('/documents', HelloKnowledgePage)
  },
  unregister() {},
})
