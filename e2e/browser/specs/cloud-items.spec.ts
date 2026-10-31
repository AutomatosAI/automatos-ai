import { expect, test, type Locator, type Page, type Route, type TestInfo } from '@playwright/test'
import { fulfillBackendJson } from '../support/backend-route'
import { chooseCoreStyle, STUDIO_STYLE } from '../support/hello-theme'

// PRD-270 US-270-14f (@host, K31): an Automatos Cloud agent on the local edition, in a real browser.
//
// The backend's answers are fulfilled here (support/backend-route.ts fulfillBackendJson): the agent
// listing (GET /api/marketplace/items), the Cloud items (GET /api/marketplace/cloud-sync/items), the
// agent's disclosure (GET …/items/agent/<id>/disclosure), the update notices (GET …/updates), the
// agent install (POST /api/marketplace/items/<id>/install) and the Sync's status (GET …/status, so the
// header's Sync line never depends on the lane's database). A POST of the Sync is recorded and aborted,
// and every request the browser makes to Automatos Cloud's origin is recorded and aborted, so nothing
// reaches the real Automatos Cloud: the spec asserts neither happened.
//
// 1. /marketplace?tab=agents: the agent's card carries From Automatos Cloud and Reviewed on Automatos
//    Cloud. Add to Workspace opens the disclosure (lib/marketplace/cloud-install-gate.ts, called by
//    apiClient.request(), asks CloudItemsLayer) with the prompt, the model and the apps; Close sends no
//    install request and the card's toast says why; Add to Workspace again, then Add to my workspace,
//    sends exactly one.
// 2. /marketplace: a copy of the agent with a newer version on Automatos Cloud shows its update notice;
//    Review the new version opens the same disclosure; Close sends nothing; Review again, then Add, sends
//    exactly one install without asking a second time (the notice's allowance), and the notice clears.
//
// A screenshot of each is attached. Both browsers: the answers are this page's own, so nothing is shared
// through the lane's database. Studio is pinned with the style cookie and GET /api/themes/active answered
// with it (chooseCoreStyle), so another spec's stored theme choice never moves this page.
//
// The Playbook, package and plugin paths are not run here (their cards and dialogs sit in components the
// wave leaves untouched, K4, and each reaches the same gate and layer). Vitest covers them:
//   badges     frontend/components/marketplace/cloud/__tests__/cloud-item-badges.test.tsx
//              (the package card, the Playbook detail header, a plugin)
//   disclosure frontend/components/marketplace/cloud/__tests__/disclosure-dialog.test.tsx
//              (a Playbook, a package, a plugin)
//   the gate   frontend/lib/marketplace/__tests__/cloud-install-gate.test.ts (each of the four install
//              calls, Add and Close), frontend/lib/__tests__/api-client-cloud-gate.test.ts
//   the layer  frontend/components/marketplace/cloud/__tests__/cloud-items-layer.test.tsx
//   notices    frontend/components/marketplace/cloud/__tests__/update-notices.test.tsx
//              (a Playbook's new copy, an updated plugin's notice and its Dismiss)

const APP_URL = process.env.E2E_APP_URL ?? 'http://localhost:3000'
const MARKETPLACE_PATH = '/marketplace'
const AGENTS_TAB_PATH = `${MARKETPLACE_PATH}?tab=agents`
const OK = 200

const LISTING_PATH = '/api/marketplace/items'
const SYNC_PATH = '/api/marketplace/cloud-sync'
const STATUS_PATH = `${SYNC_PATH}/status`
const ITEMS_PATH = `${SYNC_PATH}/items`
const UPDATES_PATH = `${SYNC_PATH}/updates`

/** lib/marketplace/cloud-files.ts AUTOMATOS_CLOUD_ORIGIN. */
const AUTOMATOS_CLOUD_ORIGIN = 'https://api.automatos.app'

const AGENT_ID = 927014
const AGENT_NAME = 'Ledger Helper (14f)'
const COPY_ID = 927015
const CLOUD_VERSION = '1.2.0'
const REVISION = 'f14f'.repeat(16)
const LOCAL_VERSION = `${CLOUD_VERSION}+${REVISION.slice(0, 12)}`
const DISCLOSURE_PATH = `${ITEMS_PATH}/agent/${AGENT_ID}/disclosure`
const INSTALL_PATH = `${LISTING_PATH}/${AGENT_ID}/install`
const PROMPT = 'You keep the ledger.\nReconcile each invoice against the bank feed, and ask before you post.'
const MODEL_LINE = 'gpt-4o-mini (openai)'
const APP_NEEDING_CONNECTION = 'gmail'
const APP_CONNECTED = 'github'

// The words the page shows (components/marketplace/cloud/item-copy.ts, marketplace-agent-cards.tsx,
// lib/marketplace/cloud-install-gate.ts, hooks/use-marketplace-api.ts).
const CLOUD_BADGE = 'From Automatos Cloud'
const REVIEWED_BADGE = 'Reviewed on Automatos Cloud'
const ADD_TO_WORKSPACE_CARD = 'Add to Workspace'
const ADDED = 'Added'
const ADD_TO_MY_WORKSPACE = 'Add to my workspace'
const NEEDS_CONNECTION = 'needs a connection'
const ON_THIS_INSTALL = 'on this install'
const NOT_ON_THIS_INSTALL = 'not on this install'
const CLOSED_SENTENCE = 'Nothing was added: you closed the review.'
const INSTALLED_TOAST = 'Agent installed successfully!'
const NOTICE_LINE = `${AGENT_NAME} has a newer version on Automatos Cloud. Your copy stays as it is.`
const REVIEW_NEW_VERSION = 'Review the new version'

/** pull.sync_status before any Sync: on, nothing held. */
const NOT_SYNCED = {
  enabled: true,
  cloudUrl: AUTOMATOS_CLOUD_ORIGIN,
  lastAttemptAt: null,
  lastSuccessAt: null,
  lastError: null,
  generatedAt: null,
  counts: { new: 0, updated: 0, unchanged: 0, refused: 0 },
  themes: 0,
  stale: true,
}

/** The Cloud agent's marketplace row as GET /api/marketplace/items lists it (MarketplaceItemOut). */
const LISTED_AGENT = {
  id: AGENT_ID,
  type: 'agent',
  name: AGENT_NAME,
  description: 'Reconciles invoices against the bank feed.',
  creator_name: 'Cloud Publisher (14f)',
  icon: null,
  category: 'business',
  tags: [],
  install_count: 0,
  is_featured: false,
  is_approved: true,
  version: LOCAL_VERSION,
  metadata: { agent_type: 'custom', configuration: {}, model_config: {}, tool_names: [], tool_icons: [] },
  created_at: '2026-10-30T09:00:00Z',
  updated_at: '2026-10-30T09:00:00Z',
}

/** local_reads.cloud_items: this install holds one Cloud agent. */
const CLOUD_ITEMS = {
  agents: [{ id: AGENT_ID, externalId: 'a'.repeat(64), version: CLOUD_VERSION, revision: REVISION, name: AGENT_NAME }],
  playbooks: [],
  packages: [],
  plugins: [],
}

/** disclosure.disclosure for the agent: its own prompt, a model this install has, two apps. */
const DISCLOSURE = {
  kind: 'agent',
  name: AGENT_NAME,
  version: CLOUD_VERSION,
  syncedAt: '2026-10-30T09:00:00Z',
  trust: 'reviewed',
  publisher: 'Cloud Publisher (14f)',
  unresolved: [],
  prompt: { source: 'custom', text: PROMPT, personaName: null },
  model: { modelId: 'gpt-4o-mini', provider: 'openai', here: true },
  apps: [{ name: APP_NEEDING_CONNECTION, needsConnection: true }, { name: APP_CONNECTED, needsConnection: false }],
  skills: [],
  plugins: [],
}

/** updates.updates: the workspace holds a copy of the agent at the version before. */
const AGENT_UPDATE = {
  kind: 'agent',
  cloudId: AGENT_ID,
  name: AGENT_NAME,
  installed: [{ id: COPY_ID, name: AGENT_NAME, version: '1.1.0+0123456789ab' }],
  available: { version: CLOUD_VERSION, revision: REVISION, syncedAt: '2026-10-30T09:00:00Z' },
}

/** install_item's answer (InstallResponse). */
const INSTALLED = {
  success: true,
  message: `${AGENT_NAME} installed successfully`,
  cloned_items: [],
  installed_dependencies: [],
  warnings: [],
}

interface CloudStub {
  /** The path of every install POST the spec answered (none reached the backend). */
  readonly installs: () => readonly string[]
  /** How many POSTs of the Sync arrived (each aborted: the backend never pulls). */
  readonly syncs: () => number
  /** Every URL the browser asked of Automatos Cloud's origin (each aborted). */
  readonly cloud: () => readonly string[]
}

function onPath(pathname: string): (url: URL) => boolean {
  return (url) => url.pathname === pathname
}

function isInstallCall(url: URL): boolean {
  return url.pathname.startsWith(`${LISTING_PATH}/`) && url.pathname.endsWith('/install')
}

/** The agent listing holds the Cloud agent; every other listing (the rail's) holds nothing. */
async function answerListing(route: Route): Promise<void> {
  const type = new URL(route.request().url()).searchParams.get('type')
  await fulfillBackendJson(route, OK, type === 'agent' ? [LISTED_AGENT] : [])
}

/** The update notice holds until a new copy is installed (a copy at the current version clears it, K17). */
async function answerUpdates(route: Route, installs: readonly string[], notice: boolean): Promise<void> {
  await fulfillBackendJson(route, OK, { updates: notice && installs.length === 0 ? [AGENT_UPDATE] : [] })
}

/** Answers every read and the install this page makes about the Cloud agent; `notice` adds its update. */
async function serveCloudAgent(page: Page, notice: boolean): Promise<CloudStub> {
  const installs: string[] = []
  const cloud: string[] = []
  const state = { syncs: 0 }
  await page.route((url) => url.origin === AUTOMATOS_CLOUD_ORIGIN, async (route) => {
    cloud.push(route.request().url())
    await route.abort()
  })
  await page.route(onPath(STATUS_PATH), (route) => fulfillBackendJson(route, OK, NOT_SYNCED))
  await page.route(onPath(SYNC_PATH), async (route) => {
    if (route.request().method() === 'POST') state.syncs += 1
    await route.abort()
  })
  await page.route(onPath(LISTING_PATH), answerListing)
  await page.route(onPath(ITEMS_PATH), (route) => fulfillBackendJson(route, OK, CLOUD_ITEMS))
  await page.route(onPath(DISCLOSURE_PATH), (route) => fulfillBackendJson(route, OK, DISCLOSURE))
  await page.route(onPath(UPDATES_PATH), (route) => answerUpdates(route, installs, notice))
  await page.route(isInstallCall, async (route) => {
    if (route.request().method() === 'POST') installs.push(new URL(route.request().url()).pathname)
    await fulfillBackendJson(route, OK, INSTALLED)
  })
  return { installs: () => installs, syncs: () => state.syncs, cloud: () => cloud }
}

async function attachShot(testInfo: TestInfo, name: string, body: Buffer): Promise<void> {
  await testInfo.attach(name, { body, contentType: 'image/png' })
}

function toast(page: Page, text: string): Locator {
  return page.locator('[data-sonner-toast]').filter({ hasText: text })
}

/** The disclosure shows the agent: both badges, its prompt as written, its model, its apps. */
async function expectAgentDisclosure(dialog: Locator): Promise<void> {
  await expect(dialog).toBeVisible()
  await expect(dialog.getByRole('heading', { name: AGENT_NAME, exact: true })).toBeVisible()
  await expect(dialog.getByText(CLOUD_BADGE, { exact: true })).toBeVisible()
  await expect(dialog.getByText(REVIEWED_BADGE, { exact: true })).toBeVisible()
  await expect(dialog.getByRole('region', { name: 'Prompt', exact: true }).locator('pre')).toHaveText(PROMPT)
  const model = dialog.getByRole('region', { name: 'Model', exact: true })
  await expect(model).toContainText(MODEL_LINE)
  await expect(model.getByText(ON_THIS_INSTALL, { exact: true })).toBeVisible()
  await expect(model.getByText(NOT_ON_THIS_INSTALL, { exact: true })).toHaveCount(0)
  const apps = dialog.getByRole('region', { name: 'Apps', exact: true }).getByRole('listitem')
  await expect(apps).toHaveCount(2)
  await expect(apps.filter({ hasText: APP_NEEDING_CONNECTION })).toContainText(NEEDS_CONNECTION)
  await expect(apps.filter({ hasText: APP_CONNECTED })).not.toContainText(NEEDS_CONNECTION)
  await expect(dialog.getByRole('button', { name: ADD_TO_MY_WORKSPACE })).toBeEnabled()
}

/** Nothing reached Automatos Cloud: the browser asked it nothing, and no Sync was posted. */
function expectNothingAskedOfCloud(stub: CloudStub): void {
  expect(stub.cloud(), 'nothing asked of Automatos Cloud').toEqual([])
  expect(stub.syncs(), 'no Sync posted, so the backend never pulled').toBe(0)
}

test.describe('Automatos Cloud items (local edition)', () => {
  test.beforeEach(async ({ page }) => {
    await chooseCoreStyle(page, STUDIO_STYLE, APP_URL)
  })

  test('the agent card shows both badges; Add opens the disclosure; Close adds nothing, Add adds once', { tag: '@host' }, async ({ page }, testInfo) => {
    const stub = await serveCloudAgent(page, false)
    await page.goto(AGENTS_TAB_PATH)
    await expect(page.getByRole('tab', { name: 'Agents', exact: true })).toHaveAttribute('data-state', 'active')

    // In the tab's panel: the disclosure is portaled outside it, and its content is a glass card too.
    const panel = page.getByRole('tabpanel', { name: 'Agents', exact: true })
    const card = panel.locator('.glass-card').filter({ has: page.getByRole('heading', { name: AGENT_NAME, exact: true }) })
    await expect(card.getByText(CLOUD_BADGE, { exact: true })).toBeVisible()
    await expect(card.getByText(REVIEWED_BADGE, { exact: true })).toBeVisible()
    await expect(card.getByText('Official', { exact: true })).toHaveCount(0)
    await attachShot(testInfo, 'cloud-agent-card', await card.screenshot())

    const dialog = page.getByTestId('cloud-disclosure')
    await card.getByRole('button', { name: ADD_TO_WORKSPACE_CARD }).click()
    await expectAgentDisclosure(dialog)
    expect(stub.installs(), 'the install waits for the disclosure').toEqual([])
    await attachShot(testInfo, 'cloud-agent-disclosure', await page.screenshot())

    await dialog.getByTestId('cloud-disclosure-close').click()
    await expect(dialog).toBeHidden()
    await expect(toast(page, CLOSED_SENTENCE)).toBeVisible()
    await expect(card.getByRole('button', { name: ADD_TO_WORKSPACE_CARD })).toBeEnabled()
    expect(stub.installs(), 'Close sends no install request').toEqual([])
    await attachShot(testInfo, 'cloud-agent-closed', await page.screenshot())

    await card.getByRole('button', { name: ADD_TO_WORKSPACE_CARD }).click()
    await expectAgentDisclosure(dialog)
    await dialog.getByRole('button', { name: ADD_TO_MY_WORKSPACE }).click()
    await expect(dialog).toBeHidden()
    await expect(toast(page, INSTALLED_TOAST)).toBeVisible()
    await expect(card.getByRole('button', { name: ADDED })).toBeVisible()
    expect(stub.installs(), 'Add sends exactly one install request').toEqual([INSTALL_PATH])
    await attachShot(testInfo, 'cloud-agent-added', await page.screenshot())

    expectNothingAskedOfCloud(stub)
  })

  test('an update notice shows, and its Review opens the disclosure; its Add installs once', { tag: '@host' }, async ({ page }, testInfo) => {
    const stub = await serveCloudAgent(page, true)
    await page.goto(MARKETPLACE_PATH)

    const notice = page.getByTestId('cloud-update-notice')
    await expect(notice).toHaveCount(1)
    await expect(notice).toContainText(NOTICE_LINE)
    await attachShot(testInfo, 'cloud-update-notice', await page.getByTestId('cloud-update-notices').screenshot())

    const dialog = page.getByTestId('cloud-disclosure')
    await notice.getByRole('button', { name: REVIEW_NEW_VERSION }).click()
    await expectAgentDisclosure(dialog)
    await attachShot(testInfo, 'cloud-update-review', await page.screenshot())

    await dialog.getByTestId('cloud-disclosure-close').click()
    await expect(dialog).toBeHidden()
    await expect(notice).toContainText(NOTICE_LINE)
    expect(stub.installs(), 'Close sends no install request').toEqual([])

    await notice.getByRole('button', { name: REVIEW_NEW_VERSION }).click()
    await expectAgentDisclosure(dialog)
    await dialog.getByRole('button', { name: ADD_TO_MY_WORKSPACE }).click()
    await expect(toast(page, INSTALLED_TOAST)).toBeVisible()
    await expect(dialog).toBeHidden()
    await expect(page.getByTestId('cloud-update-notices')).toHaveCount(0)
    expect(stub.installs(), 'the notice\'s Add sends exactly one install, asked once').toEqual([INSTALL_PATH])
    await attachShot(testInfo, 'cloud-update-added', await page.screenshot())

    expectNothingAskedOfCloud(stub)
  })
})
