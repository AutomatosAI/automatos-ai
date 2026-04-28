/**
 * LLM Models Settings Tab (PRD-136)
 * =================================
 *
 * Three tier cards (Auto / System / Embeddings), one canonical schema each.
 * Replaces: System LLMs (CodeGraph + Knowledge Graph), Memory Management,
 * and the embedding section that lived in General Settings.
 */

import React from 'react'
import { Brain, Cog, Database } from 'lucide-react'
import { SystemSetting } from '@/lib/api/system-settings'
import LLMTierCard from './LLMTierCard'

interface LLMModelsSettingsTabProps {
  orchestratorSettings: SystemSetting[]
  systemSettings: SystemSetting[]
  embeddingsSettings: SystemSetting[]
  onSaveOrchestrator: (updates: Record<string, string>) => Promise<void> | void
  onSaveSystem: (updates: Record<string, string>) => Promise<void> | void
  onSaveEmbeddings: (updates: Record<string, string>) => Promise<void> | void
  onResetOrchestrator: () => Promise<void> | void
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
  onResetOrchestrator,
  onResetSystem,
  onResetEmbeddings,
  saving,
}: LLMModelsSettingsTabProps) {
  return (
    <div className="space-y-6">
      <div className="rounded-lg bg-secondary/30 p-4 text-sm text-muted-foreground">
        <p>
          Three tiers handle every LLM call across the platform.
          <strong className="text-foreground"> Auto</strong> is the brain that talks to users.
          <strong className="text-foreground"> System</strong> powers background work — codegraph, RAG, knowledge extraction, planners, verifiers.
          <strong className="text-foreground"> Embeddings</strong> is used wherever vectors are needed (RAG, memory, semantic search).
        </p>
      </div>

      <LLMTierCard
        category="orchestrator_llm"
        title="Auto — The Brain"
        description="The LLM that holds conversations with users and orchestrates agents."
        icon={Brain}
        settings={orchestratorSettings}
        onSave={onSaveOrchestrator}
        onReset={onResetOrchestrator}
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
