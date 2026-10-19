/**
 * LLM Tiers Settings Tab (PRD-136)
 * ================================
 *
 * Three tier cards. "Platform default for Auto" is system_settings orchestrator_llm
 * provider + model: the model every new workspace's Auto, and every new agent given
 * no model, starts on. Saving it goes through bulk-update, which moves the Auto
 * agents still on the platform default to it (a workspace that chose its own model
 * in Settings > Orchestrator keeps it). System and Embeddings follow.
 */

import React from 'react'
import { Bot, Cog, Database } from 'lucide-react'
import { SystemSetting } from '@/lib/api/system-settings'
import LLMTierCard from './LLMTierCard'

/** The orchestrator_llm keys the platform default card shows: its route. */
const AUTO_DEFAULT_KEYS = ['provider', 'model']

interface LLMModelsSettingsTabProps {
  orchestratorSettings: SystemSetting[]
  systemSettings: SystemSetting[]
  embeddingsSettings: SystemSetting[]
  onSaveOrchestrator: (updates: Record<string, string>) => Promise<void> | void
  onSaveSystem: (updates: Record<string, string>) => Promise<void> | void
  onSaveEmbeddings: (updates: Record<string, string>) => Promise<void> | void
  onResetSystem: () => Promise<void> | void
  onResetEmbeddings: () => Promise<void> | void
  saving: boolean
}

export default function LLMModelsSettingsTab({
  orchestratorSettings,
  systemSettings,
  embeddingsSettings,
  onSaveOrchestrator,
  onSaveSystem,
  onSaveEmbeddings,
  onResetSystem,
  onResetEmbeddings,
  saving,
}: LLMModelsSettingsTabProps) {
  const autoDefaultSettings = orchestratorSettings.filter((s) => AUTO_DEFAULT_KEYS.includes(s.key))
  // Reset saves the shipped default through the same path as Save, so the Auto agents move too.
  const resetAutoDefault = () =>
    onSaveOrchestrator(Object.fromEntries(autoDefaultSettings.map((s) => [s.key, s.default_value || ''])))

  return (
    <div className="space-y-6">
      <div className="rounded-lg bg-secondary/30 p-4 text-sm text-muted-foreground">
        <p>
          <strong className="text-foreground">Platform default for Auto</strong> is the model each workspace&apos;s Auto runs on until that workspace picks its own in the <strong className="text-foreground">Orchestrator</strong> tab, and the model a new agent gets when none is picked. Changing it moves every Auto still on the platform default.
          <strong className="text-foreground"> System</strong> powers background work — codegraph, RAG, knowledge extraction, planners, verifiers, memory.
          <strong className="text-foreground"> Embeddings</strong> is the vector model used wherever vectors are needed (RAG, memory, semantic search).
        </p>
      </div>

      <LLMTierCard
        category="orchestrator_llm"
        title="Platform default for Auto"
        description="Provider and model for every Auto that has not picked its own, and for new agents given no model. Workspaces that chose a model keep it."
        icon={Bot}
        settings={autoDefaultSettings}
        onSave={onSaveOrchestrator}
        onReset={resetAutoDefault}
        saving={saving}
      />

      <LLMTierCard
        category="system_llm"
        title="System LLM"
        description="One model for all background workers (codegraph, chatbot context, RAG, NL2SQL, planner, verifier, knowledge graph extraction, memory)."
        icon={Cog}
        settings={systemSettings}
        onSave={onSaveSystem}
        onReset={onResetSystem}
        saving={saving}
      />

      <LLMTierCard
        category="embeddings"
        title="Embeddings"
        description="Vector model used by RAG, memory, and semantic search. Dimensions are derived from the model."
        icon={Database}
        settings={embeddingsSettings}
        onSave={onSaveEmbeddings}
        onReset={onResetEmbeddings}
        saving={saving}
      />
    </div>
  )
}
