/**
 * Mission Control Zustand Store — PRD-82A
 *
 * Manages UI state for mission mode, plan review, and human review.
 * Data fetching is in hooks/use-missions-api.ts (React Query).
 */

import { create } from 'zustand'

interface MissionStore {
  // ── Chat integration ──
  isMissionMode: boolean
  setMissionMode: (on: boolean) => void
  activePlanningMissionId: string | null
  setActivePlanningMissionId: (id: string | null) => void

  // ── Plan review ──
  selectedTaskId: string | null
  setSelectedTaskId: (id: string | null) => void
  planModifications: PlanModifications
  setPlanModification: (taskId: string, field: string, value: unknown) => void
  clearPlanModifications: () => void

  // ── Human review ──
  taskFeedback: Record<string, string>
  setTaskFeedback: (taskId: string, feedback: string) => void
  removeTaskFeedback: (taskId: string) => void
  clearTaskFeedback: () => void
}

interface PlanModifications {
  task_overrides: Record<string, Record<string, unknown>>
  agent_overrides: Record<string, number>
  notes: string
}

const EMPTY_MODIFICATIONS: PlanModifications = {
  task_overrides: {},
  agent_overrides: {},
  notes: '',
}

export const useMissionStore = create<MissionStore>((set) => ({
  // ── Chat integration ──
  isMissionMode: false,
  setMissionMode: (on) => set(on ? { isMissionMode: true } : {
    isMissionMode: false,
    activePlanningMissionId: null,
  }),
  activePlanningMissionId: null,
  setActivePlanningMissionId: (id) => set({ activePlanningMissionId: id }),

  // ── Plan review ──
  selectedTaskId: null,
  setSelectedTaskId: (id) => set({ selectedTaskId: id }),
  planModifications: { ...EMPTY_MODIFICATIONS },
  setPlanModification: (taskId, field, value) =>
    set((state) => {
      if (field === 'assigned_agent_id') {
        return {
          planModifications: {
            ...state.planModifications,
            agent_overrides: {
              ...state.planModifications.agent_overrides,
              [taskId]: value as number,
            },
          },
        }
      }
      // Other fields go into task_overrides
      const existing = state.planModifications.task_overrides[taskId] ?? {}
      return {
        planModifications: {
          ...state.planModifications,
          task_overrides: {
            ...state.planModifications.task_overrides,
            [taskId]: { ...existing, [field]: value },
          },
        },
      }
    }),
  clearPlanModifications: () => set({ planModifications: { ...EMPTY_MODIFICATIONS } }),

  // ── Human review ──
  taskFeedback: {},
  setTaskFeedback: (taskId, feedback) =>
    set((state) => ({
      taskFeedback: { ...state.taskFeedback, [taskId]: feedback },
    })),
  removeTaskFeedback: (taskId) =>
    set((state) => {
      const { [taskId]: _, ...rest } = state.taskFeedback
      return { taskFeedback: rest }
    }),
  clearTaskFeedback: () => set({ taskFeedback: {} }),
}))
