/**
 * Mission Control Zustand Store — PRD-82A
 *
 * Manages UI state for mission mode, plan review, and human review.
 * Data fetching is in hooks/use-missions-api.ts (React Query).
 */

import { create } from 'zustand'

interface MissionStore {
  // ── Chat integration ──
  isPlanMode: boolean
  setPlanMode: (on: boolean) => void
  isMissionMode: boolean
  setMissionMode: (on: boolean) => void
  activePlanningMissionId: string | null
  setActivePlanningMissionId: (id: string | null) => void

  // ── Plan review ──
  selectedTaskId: string | null
  setSelectedTaskId: (id: string | null) => void

  // ── Human review ──
  taskFeedback: Record<string, string>
  setTaskFeedback: (taskId: string, feedback: string) => void
  removeTaskFeedback: (taskId: string) => void
  clearTaskFeedback: () => void
}

export const useMissionStore = create<MissionStore>((set) => ({
  // ── Chat integration ──
  isPlanMode: false,
  setPlanMode: (on) => set((state) => ({
    isPlanMode: on,
    // Plan and Mission modes are mutually exclusive
    ...(on && state.isMissionMode ? { isMissionMode: false } : {}),
  })),
  isMissionMode: false,
  setMissionMode: (on) => set((state) => ({
    isMissionMode: on,
    ...(on && state.isPlanMode ? { isPlanMode: false } : {}),
  })),
  activePlanningMissionId: null,
  setActivePlanningMissionId: (id) => set({ activePlanningMissionId: id }),

  // ── Plan review ──
  selectedTaskId: null,
  setSelectedTaskId: (id) => set({ selectedTaskId: id }),

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
