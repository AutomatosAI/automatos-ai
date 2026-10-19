/**
 * The state behind Settings > Orchestrator (PRD-55): the legacy LLM performance
 * settings, the orchestrator config (workspace.settings.orchestrator + the Auto
 * agent), its saves, the connected channels and the heartbeat's Run Now. Moved out
 * of SystemLLMSettingsTab.tsx unchanged (PRD-263 PR-1E-a) so each part fits the
 * length rule.
 */

import { useEffect, useState } from 'react'
import { toast } from 'sonner'
import { apiClient } from '@/lib/api-client'
import { fetchPlatformDefaultRoute, getDefaultModelConfig } from '@/lib/llm-defaults'
import { SystemSetting, getSettingsForCategory } from '@/lib/api/system-settings'
import {
  ConnectedChannel,
  EMPTY_LLM_CONFIG,
  OrchestratorConfig,
  connectedChannelsFrom,
  fallbackOrchestratorConfig,
  heartbeatPayload,
} from './orchestrator-settings-config'

const ORCHESTRATOR_ROUTE = '/api/workspaces/current/orchestrator'

type SetOrchConfig = (config: OrchestratorConfig | null) => void

/** The orchestrator_llm system settings (performance only), self-loaded when no props pass them. */
export function useStandaloneLLMSettings(externalSettings?: SystemSetting[]) {
  const isStandalone = !externalSettings
  const [selfSettings, setSelfSettings] = useState<SystemSetting[]>([])
  const [selfLoading, setSelfLoading] = useState(isStandalone)
  const settings = externalSettings ?? selfSettings
  const [formData, setFormData] = useState<Record<string, string>>({})

  // Self-load settings when in standalone mode
  useEffect(() => {
    if (!isStandalone) return
    const loadSelfSettings = async () => {
      try {
        setSelfLoading(true)
        const data = await getSettingsForCategory('orchestrator_llm')
        setSelfSettings(data)
      } catch (err) {
        console.error('Failed to self-load orchestrator_llm settings:', err)
        toast.error('Failed to load LLM settings')
      } finally {
        setSelfLoading(false)
      }
    }
    loadSelfSettings()
  }, [isStandalone])

  // Initialize LLM form data from system settings
  useEffect(() => {
    const initialData: Record<string, string> = {}
    settings.forEach(setting => {
      initialData[setting.key] = setting.value !== null && setting.value !== undefined
        ? setting.value
        : (setting.default_value || '')
    })
    setFormData(initialData)
  }, [settings])

  const handleInputChange = (key: string, value: string) => {
    setFormData(prev => ({ ...prev, [key]: value }))
  }
  return { selfLoading, formData, handleInputChange }
}

/** The orchestrator config from the workspace API, and its field editors. */
export function useOrchestratorConfig() {
  const [orchConfig, setOrchConfig] = useState<OrchestratorConfig | null>(null)
  const [orchLoading, setOrchLoading] = useState(true)

  useEffect(() => {
    const loadOrchConfig = async () => {
      try {
        setOrchLoading(true)
        setOrchConfig(await apiClient.request<OrchestratorConfig>(ORCHESTRATOR_ROUTE))
      } catch (err) {
        console.error('Failed to load orchestrator config:', err)
        // Use defaults if endpoint not available
        setOrchConfig(fallbackOrchestratorConfig())
      } finally {
        setOrchLoading(false)
      }
    }
    loadOrchConfig()
  }, [])

  const handleOrchChange = (key: string, value: string | boolean) => {
    if (orchConfig) setOrchConfig({ ...orchConfig, [key]: value })
  }
  const handleLLMChange = (key: string, value: string | number | string[] | null) => {
    if (orchConfig) setOrchConfig({ ...orchConfig, llm: { ...(orchConfig.llm || EMPTY_LLM_CONFIG), [key]: value } })
  }
  const handleHeartbeatChange = (key: string, value: string | boolean | number) => {
    if (orchConfig) setOrchConfig({ ...orchConfig, heartbeat: { ...orchConfig.heartbeat, [key]: value } })
  }
  const handleHarnessChange = (key: string, value: string) => {
    if (orchConfig) setOrchConfig({ ...orchConfig, harness: { ...orchConfig.harness, [key]: value } })
  }
  return {
    orchConfig, setOrchConfig, orchLoading,
    handleOrchChange, handleLLMChange, handleHeartbeatChange, handleHarnessChange,
  }
}

/** Save LLM through the orchestrator endpoint (→ Auto agent model_config), and reset it to defaults. */
export function useLLMSave(orchConfig: OrchestratorConfig | null, setOrchConfig: SetOrchConfig) {
  const [selfSaving, setSelfSaving] = useState(false)

  const handleSaveLLM = async () => {
    if (!orchConfig?.llm) return
    // A blank model (the config could not be loaded, or none picked yet) is never saved over Auto's
    if (!orchConfig.llm.model_id) {
      toast.error('Pick a model before saving the LLM settings')
      return
    }
    try {
      setSelfSaving(true)
      await apiClient.request(ORCHESTRATOR_ROUTE, { method: 'PUT', body: JSON.stringify({ llm: orchConfig.llm }) })
      toast.success('LLM settings saved')
    } catch (err) {
      toast.error('Failed to save LLM settings')
    } finally {
      setSelfSaving(false)
    }
  }

  // Reset LLM to the platform default: saving its pair keeps this workspace's Auto following it
  const handleReset = async () => {
    try {
      setSelfSaving(true)
      const llm = { ...getDefaultModelConfig(await fetchPlatformDefaultRoute()), max_tokens: 4000, stop: null, timeout: null }
      await apiClient.request(ORCHESTRATOR_ROUTE, { method: 'PUT', body: JSON.stringify({ llm }) })
      // Reload orchestrator config to get fresh values
      setOrchConfig(await apiClient.request<OrchestratorConfig>(ORCHESTRATOR_ROUTE))
      toast.success('LLM settings reset to defaults')
    } catch (err) {
      toast.error('Failed to reset LLM settings')
    } finally {
      setSelfSaving(false)
    }
  }
  return { selfSaving, handleSaveLLM, handleReset }
}

/** Save the orchestrator settings: Soul, Heartbeat and HARNESS. */
export function useOrchestratorSave(orchConfig: OrchestratorConfig | null, connectedChannels: ConnectedChannel[]) {
  const [orchSaving, setOrchSaving] = useState(false)

  const handleSaveOrchestrator = async () => {
    if (!orchConfig) return
    try {
      setOrchSaving(true)
      // Parse channel key to extract platform + channel_id for heartbeat delivery; an llm
      // block with no model (the config could not be loaded) is left out, not saved over Auto's
      const { llm, ...rest } = orchConfig
      const payload = {
        ...rest,
        ...(llm?.model_id ? { llm } : {}),
        heartbeat: heartbeatPayload(orchConfig.heartbeat, connectedChannels),
      }
      await apiClient.request(ORCHESTRATOR_ROUTE, { method: 'PUT', body: JSON.stringify(payload) })
      toast.success('Orchestrator settings saved')
    } catch (err) {
      console.error('Failed to save orchestrator settings:', err)
      toast.error('Failed to save orchestrator settings')
    } finally {
      setOrchSaving(false)
    }
  }
  return { orchSaving, handleSaveOrchestrator }
}

/** The connected channels, for the notification and approval dropdowns. */
export function useConnectedChannels(): ConnectedChannel[] {
  const [connectedChannels, setConnectedChannels] = useState<ConnectedChannel[]>([])

  useEffect(() => {
    Promise.all([
      apiClient.request<any>('/api/workspaces/current/integrations').catch(() => ({})),
      apiClient.request<any>('/api/channels').catch(() => []),
    ]).then(([integrations, channels]) => {
      setConnectedChannels(connectedChannelsFrom(integrations, channels))
    })
  }, [])
  return connectedChannels
}

/** The orchestrator heartbeat's last result, and Run Now. */
export function useOrchestratorHeartbeatRun() {
  const [heartbeatRunning, setHeartbeatRunning] = useState(false)
  const [lastHeartbeatResult, setLastHeartbeatResult] = useState<any>(null)

  useEffect(() => {
    apiClient.request<any>('/api/heartbeat/orchestrator/history?limit=1')
      .then((data) => {
        if (data?.results?.[0]) setLastHeartbeatResult(data.results[0])
      })
      .catch(() => {})
  }, [])

  const runHeartbeatNow = async () => {
    setHeartbeatRunning(true)
    try {
      const result = await apiClient.request<any>('/api/heartbeat/orchestrator/run', {
        method: 'POST',
      })
      setLastHeartbeatResult(result)
      toast.success('Orchestrator heartbeat executed')
    } catch (err) {
      console.error('Failed to run orchestrator heartbeat:', err)
      toast.error('Failed to run orchestrator heartbeat')
    } finally {
      setHeartbeatRunning(false)
    }
  }
  return { heartbeatRunning, lastHeartbeatResult, runHeartbeatNow }
}
