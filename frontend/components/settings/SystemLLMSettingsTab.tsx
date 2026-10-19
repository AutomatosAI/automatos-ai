/**
 * Orchestrator Settings Tab Component (PRD-55)
 * =============================================
 *
 * Expanded from "System LLM Settings" to full Orchestrator Soul Designer.
 * 4 sections:
 * 1. LLM Configuration (existing, enhanced with thinking level)
 * 2. Soul & Personality (NEW)
 * 3. Heartbeat (NEW)
 * 4. HARNESS — the weekly review of the agents (PRD-121; its mode, PRD-263)
 *
 * Each section is its own component and the state lives in
 * use-orchestrator-settings.ts (split for the component length rule, PRD-263 PR-1E-a).
 */

import React from 'react'
import { Button } from '@/components/ui/button'
import { Save, Loader2 } from 'lucide-react'
import { SystemSetting } from '@/lib/api/system-settings'
import { HarnessSettingsSection } from './harness-settings-section'
import { OrchestratorHeartbeatSection } from './orchestrator-heartbeat-section'
import { OrchestratorLLMSection, useOrchestratorModelOptions } from './orchestrator-llm-section'
import { OrchestratorSoulSection } from './orchestrator-soul-section'
import {
  useConnectedChannels,
  useLLMSave,
  useOrchestratorConfig,
  useOrchestratorHeartbeatRun,
  useOrchestratorSave,
  useStandaloneLLMSettings,
} from './use-orchestrator-settings'

interface SystemLLMSettingsTabProps {
  settings?: SystemSetting[]
  onSave?: (updates: Record<string, string>) => void
  saving?: boolean
  onReset?: () => void
}

export default function SystemLLMSettingsTab({
  settings: externalSettings,
  saving: externalSaving,
}: SystemLLMSettingsTabProps) {
  // Legacy LLM form data (from system_settings table — performance settings only),
  // self-loaded when no props are passed (top-level mode)
  const { selfLoading, formData, handleInputChange } = useStandaloneLLMSettings(externalSettings)

  // Orchestrator config (from workspace.settings.orchestrator + Auto agent)
  const config = useOrchestratorConfig()
  const { orchConfig, orchLoading } = config

  // Load workspace-installed models (same catalog agents use)
  const modelOptions = useOrchestratorModelOptions(orchConfig?.llm?.provider || '')

  // Heartbeat: connected channels, run now, last result
  const connectedChannels = useConnectedChannels()
  const heartbeatRun = useOrchestratorHeartbeatRun()

  const llmSave = useLLMSave(orchConfig, config.setOrchConfig)
  const { orchSaving, handleSaveOrchestrator } = useOrchestratorSave(orchConfig, connectedChannels)
  const saving = externalSaving ?? llmSave.selfSaving

  if (selfLoading) {
    return (
      <div className="flex items-center justify-center p-8">
        <Loader2 className="h-8 w-8 animate-spin" />
        <span className="ml-2">Loading orchestrator settings...</span>
      </div>
    )
  }

  return (
    <div className="space-y-6">
      <OrchestratorLLMSection
        orchConfig={orchConfig}
        modelOptions={modelOptions}
        formData={formData}
        saving={saving}
        onLLMChange={config.handleLLMChange}
        onOrchChange={config.handleOrchChange}
        onInputChange={handleInputChange}
        onSave={llmSave.handleSaveLLM}
        onReset={llmSave.handleReset}
      />

      <OrchestratorSoulSection
        orchConfig={orchConfig}
        orchLoading={orchLoading}
        connectedChannels={connectedChannels}
        onOrchChange={config.handleOrchChange}
      />

      <OrchestratorHeartbeatSection
        orchConfig={orchConfig}
        connectedChannels={connectedChannels}
        onHeartbeatChange={config.handleHeartbeatChange}
        heartbeatRunning={heartbeatRun.heartbeatRunning}
        lastHeartbeatResult={heartbeatRun.lastHeartbeatResult}
        onRunNow={heartbeatRun.runHeartbeatNow}
      />

      <HarnessSettingsSection orchConfig={orchConfig} onHarnessChange={config.handleHarnessChange} />

      {/* Save Orchestrator Settings (Soul + Heartbeat + HARNESS) */}
      <div className="flex justify-end">
        <Button onClick={handleSaveOrchestrator} disabled={orchSaving || orchLoading}>
          {orchSaving ? (
            <Loader2 className="h-4 w-4 animate-spin mr-2" />
          ) : (
            <Save className="h-4 w-4 mr-2" />
          )}
          Save Orchestrator Settings
        </Button>
      </div>
    </div>
  )
}
