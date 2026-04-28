/**
 * System & Embeddings Settings Tab (PRD-136)
 * ==========================================
 *
 * Two tier cards (System / Embeddings). The third tier — Auto — is configured
 * in the top-level Orchestrator tab and writes to the Auto agent's
 * model_config (per-workspace), NOT to system_settings.
 */

import React from 'react'
import { Cog, Database } from 'lucide-react'
import { SystemSetting } from '@/lib/api/system-settings'
import LLMTierCard from './LLMTierCard'

interface LLMModelsSettingsTabProps {
  systemSettings: SystemSetting[]
  embeddingsSettings: SystemSetting[]
  onSaveSystem: (updates: Record<string, string>) => Promise<void> | void
  onSaveEmbeddings: (updates: Record<string, string>) => Promise<void> | void
  onResetSystem: () => Promise<void> | void
  onResetEmbeddings: () => Promise<void> | void
  saving: boolean
}

export default function LLMModelsSettingsTab({
  systemSettings,
  embeddingsSettings,
  onSaveSystem,
  onSaveEmbeddings,
  onResetSystem,
  onResetEmbeddings,
  saving,
}: LLMModelsSettingsTabProps) {
  return (
    <div className="space-y-6">
      <div className="rounded-lg bg-secondary/30 p-4 text-sm text-muted-foreground">
        <p>
          Two tiers handle every background LLM call.
          <strong className="text-foreground"> System</strong> powers background work — codegraph, RAG, knowledge extraction, planners, verifiers, memory.
          <strong className="text-foreground"> Embeddings</strong> is the vector model used wherever vectors are needed (RAG, memory, semantic search).
          {' '}Auto (the brain that talks to users) is configured in the <strong className="text-foreground">Orchestrator</strong> tab.
        </p>
      </div>

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
