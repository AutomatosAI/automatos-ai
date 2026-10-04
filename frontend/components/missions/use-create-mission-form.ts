'use client'

/**
 * The New mission form's state: its fields, the Business Plan template's extra fields, and the
 * workspace's orchestrator model the Max power mode names. Moved out of create-mission-modal.tsx.
 */

import { useEffect, useState } from 'react'
import { apiClient } from '@/lib/api-client'
import { DEFAULT_POWER_MODE, type PowerMode } from './create-mission-options'

export function useMissionFormFields(initialGoal?: string, initialDescription?: string) {
  const [selectedTemplate, setSelectedTemplate] = useState<string | null>(null)
  const [name, setName] = useState(initialGoal ?? '')
  const [description, setDescription] = useState(initialDescription ?? '')
  const [tags, setTags] = useState('')
  const [budgetPauseEnabled, setBudgetPauseEnabled] = useState(true)
  const [powerMode, setPowerMode] = useState<PowerMode>(DEFAULT_POWER_MODE)
  const [checkEachStep, setCheckEachStep] = useState(false)

  const reset = () => {
    setSelectedTemplate(null)
    setName('')
    setDescription('')
    setTags('')
    setBudgetPauseEnabled(true)
    setPowerMode(DEFAULT_POWER_MODE)
    setCheckEachStep(false)
  }

  return {
    selectedTemplate, setSelectedTemplate,
    name, setName,
    description, setDescription,
    tags, setTags,
    budgetPauseEnabled, setBudgetPauseEnabled,
    powerMode, setPowerMode,
    checkEachStep, setCheckEachStep,
    reset,
  }
}

export type MissionFormFields = ReturnType<typeof useMissionFormFields>

// Business Plan template extra fields
export function useBusinessPlanFields() {
  const [businessName, setBusinessName] = useState('')
  const [businessType, setBusinessType] = useState('')
  const [industry, setIndustry] = useState('')
  const [targetMarket, setTargetMarket] = useState('')
  const [businessGoals, setBusinessGoals] = useState('')

  const reset = () => {
    setBusinessName('')
    setBusinessType('')
    setIndustry('')
    setTargetMarket('')
    setBusinessGoals('')
  }

  return {
    businessName, setBusinessName,
    businessType, setBusinessType,
    industry, setIndustry,
    targetMarket, setTargetMarket,
    businessGoals, setBusinessGoals,
    reset,
  }
}

export type BusinessPlanFields = ReturnType<typeof useBusinessPlanFields>

/** The workspace orchestrator's model id, read each time the form opens; null until it answers. */
export function useOrchestratorModel(open: boolean): string | null {
  const [orchestratorModel, setOrchestratorModel] = useState<string | null>(null)

  useEffect(() => {
    if (open) {
      apiClient.request<{ llm?: { model_id?: string } }>('/api/workspaces/current/orchestrator')
        .then((data) => {
          const modelId = data?.llm?.model_id
          if (modelId) setOrchestratorModel(modelId)
        })
        .catch(() => {})
    }
  }, [open])

  return orchestratorModel
}
