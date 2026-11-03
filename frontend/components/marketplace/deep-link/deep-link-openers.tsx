'use client'

/**
 * PRD-270 US-270-09b (K22) — one opener per tab: it reads the link's item (deep-link-loaders.ts)
 * and, once found, renders that tab's own detail modal on it, open. An item that isn't found renders
 * nothing; a failed read says so in a toast and renders nothing.
 *
 * The modals are the tabs' own, unchanged. Opening one installs nothing: an install, add or enable
 * runs only on the person's click, through the route, permission, who-installs and step-up rules it
 * runs through from its tab (the agent, Playbook, package and LLM installs inside their modals; the
 * application's add and the plugin's enable here, the tabs' one request each).
 *
 * The plugin modal opens for adding: its tab's disable and admin approval stay on the tab. The
 * Plugins tab keeps which plugins are enabled in its own state (read as it mounts), so a plugin
 * enabled here still offers Enable on that grid until the tab is next opened; enabling it again
 * answers Already Enabled.
 */
import { useState } from 'react'
import { useQuery, useQueryClient } from '@tanstack/react-query'
import type { QueryKey } from '@tanstack/react-query'
import { toast } from 'sonner'

import type { ComposioApp } from '@/hooks/use-composio-api'
import type { ApiHttpError } from '@/lib/api-client'

import { LLMModelDetailModal } from '../llm-model-detail-modal'
import { MarketplaceAppDetailsModal } from '../marketplace-app-details-modal'
import { MarketplaceItemModal } from '../marketplace-item-modal'
import { MarketplacePluginDetailModal } from '../marketplace-plugin-detail-modal'
import { PackageDetailModal } from '../package-detail-modal'
import {
  WORKSPACE_TOOLS_KEY,
  WORKSPACE_TOOLS_STALE_MS,
  addApplication,
  enablePlugin,
  enabledPluginIds,
  isConnectedApp,
  loadApplication,
  loadLlmRoute,
  loadMarketplaceItem,
  loadPackage,
  loadPlugin,
  workspaceTools,
} from './deep-link-loaders'
import type { DeepLinkItemType, FoundPlugin } from './deep-link-loaders'

export interface OpenerProps {
  /** The link's item: the key its tab's public read answers. */
  readonly item: string
  /** The workspace the Marketplace installs into (the homepage's, an admin's override first). */
  readonly workspaceId: string
  readonly onClose: () => void
}

const DEEP_LINK_KEY = 'marketplace-deep-link'
/** The LLMs tab's catalogue queries start with this key (marketplace-llms-tab.tsx): an install refreshes both. */
const LLM_CATALOG_KEY = 'marketplaceLlmCatalog'
const CONFLICT = 409
const LOG_PREFIX = '[marketplace]'
export const COULD_NOT_OPEN = 'That Marketplace item could not be opened. Try again from its tab.'

function reportUnopened(error: unknown) {
  console.error(LOG_PREFIX, 'a Marketplace link could not be opened', error)
  toast.error(COULD_NOT_OPEN)
}

/** The link's item, read once; null while it loads, when it isn't found and when the read fails. */
function useLinkedItem<T>(queryKey: QueryKey, load: () => Promise<T | null>): T | null {
  const found = useQuery({ queryKey, queryFn: load, retry: false, refetchOnWindowFocus: false, onError: reportUnopened })
  return found.data ?? null
}

function MarketplaceItemOpener({ item, type, onClose }: OpenerProps & { readonly type: DeepLinkItemType }) {
  const found = useLinkedItem([DEEP_LINK_KEY, type, item], () => loadMarketplaceItem(item, type))
  if (found === null) return null
  return <MarketplaceItemModal itemId={found.id} itemType={type} onClose={onClose} />
}

export function AgentOpener(props: OpenerProps) {
  return <MarketplaceItemOpener {...props} type="agent" />
}

export function PlaybookOpener(props: OpenerProps) {
  return <MarketplaceItemOpener {...props} type="recipe" />
}

export function PackageOpener({ item, onClose }: OpenerProps) {
  const found = useLinkedItem([DEEP_LINK_KEY, 'package', item], () => loadPackage(item))
  if (found === null) return null
  return <PackageDetailModal pkg={found} onClose={onClose} />
}

/** The Applications tab's add, with its words; a failure is logged and told, never thrown. */
async function addWithToast(app: ComposioApp): Promise<void> {
  try {
    const already = await addApplication(app)
    toast(already ? 'Already in Workspace' : 'Added to Workspace', {
      description: already
        ? `${app.display_name} is already in your workspace.`
        : `${app.display_name} has been added. Go to Tools > Applications to connect it.`,
    })
  } catch (error) {
    console.error(LOG_PREFIX, 'adding an application failed', error)
    toast.error('Failed to add', { description: `${app.display_name} could not be added to your workspace.` })
  }
}

/**
 * The app's place in this workspace, read and refreshed through the Applications tab's own query
 * (one cache, so the tab behind the modal shows an add too), and its Add.
 */
function useWorkspaceApp(app: ComposioApp | null) {
  const queryClient = useQueryClient()
  const tools = useQuery({
    queryKey: WORKSPACE_TOOLS_KEY,
    queryFn: workspaceTools,
    enabled: app !== null,
    staleTime: WORKSPACE_TOOLS_STALE_MS,
    onError: (error: unknown) => console.error(LOG_PREFIX, 'the workspace tools could not be read', error),
  })
  const [adding, setAdding] = useState(false)
  const add = () => {
    if (app === null) return
    setAdding(true)
    void addWithToast(app)
      .then(() => queryClient.invalidateQueries({ queryKey: WORKSPACE_TOOLS_KEY }))
      .finally(() => setAdding(false))
  }
  return { adding, add, connected: app !== null && isConnectedApp(tools.data ?? [], app) }
}

export function ApplicationOpener({ item, onClose }: OpenerProps) {
  const found = useLinkedItem([DEEP_LINK_KEY, 'application', item], () => loadApplication(item))
  const here = useWorkspaceApp(found)
  if (found === null) return null
  return (
    <MarketplaceAppDetailsModal
      open
      onClose={onClose}
      app={found}
      onConnect={here.add}
      loading={here.adding}
      isConnected={here.connected}
    />
  )
}

/** The Plugins tab's enable, with its words; true once the plugin is enabled here. */
async function enableWithToast(workspaceId: string, plugin: FoundPlugin): Promise<boolean> {
  if (!workspaceId) {
    toast.error('No workspace', { description: 'No active workspace found. Please select a workspace.' })
    return false
  }
  try {
    await enablePlugin(workspaceId, plugin.id)
    toast('Capability Enabled', { description: `${plugin.name} has been enabled for your workspace.` })
    return true
  } catch (error) {
    if ((error as ApiHttpError | null)?.status === CONFLICT) {
      toast('Already Enabled', { description: `${plugin.name} is already enabled for your workspace.` })
      return true
    }
    console.error(LOG_PREFIX, 'enabling a plugin failed', error)
    toast.error('Error', { description: `${plugin.name} could not be enabled.` })
    return false
  }
}

export function PluginOpener({ item, workspaceId, onClose }: OpenerProps) {
  const found = useLinkedItem([DEEP_LINK_KEY, 'plugin', item], () => loadPlugin(item))
  const enabledHere = useQuery({
    queryKey: [DEEP_LINK_KEY, 'enabled-plugins', workspaceId],
    queryFn: () => enabledPluginIds(workspaceId),
    enabled: found !== null && Boolean(workspaceId),
    retry: false,
    onError: (error: unknown) => console.error(LOG_PREFIX, 'the enabled plugins could not be read', error),
  })
  const [enabling, setEnabling] = useState(false)
  const [enabledNow, setEnabledNow] = useState(false)
  if (found === null) return null
  const enable = () => {
    setEnabling(true)
    void enableWithToast(workspaceId, found)
      .then((enabled) => setEnabledNow((was) => was || enabled))
      .finally(() => setEnabling(false))
  }
  return (
    <MarketplacePluginDetailModal
      open
      pluginId={found.id}
      onClose={onClose}
      isEnabled={enabledNow || Boolean(enabledHere.data?.has(found.id))}
      isEnabling={enabling}
      onEnable={enable}
    />
  )
}

export function LlmOpener({ item, onClose }: OpenerProps) {
  const queryClient = useQueryClient()
  const found = useLinkedItem([LLM_CATALOG_KEY, DEEP_LINK_KEY, item], () => loadLlmRoute(item))
  if (found === null) return null
  return (
    <LLMModelDetailModal
      model={found}
      open
      onOpenChange={(open) => {
        if (!open) onClose()
      }}
      onInstall={() => queryClient.invalidateQueries({ queryKey: [LLM_CATALOG_KEY] })}
    />
  )
}
