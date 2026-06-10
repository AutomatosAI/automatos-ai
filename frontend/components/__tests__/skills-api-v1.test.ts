import { describe, it, expect, vi, afterEach } from 'vitest'
import { readFileSync } from 'fs'
import path from 'path'
import { apiClient } from '@/lib/api-client'

// PRD-154 S10 — agent-skill assignment in api-client hit the dead pre-v1
// /api/skills/agents/* router (404). Migrated to the live PRD-22 endpoints
// /api/v1/skills/agents/{id}/skills (skills.py:770/838/910), the same surface
// the Agents page already uses via use-skills-api.

afterEach(() => vi.restoreAllMocks())

describe('S10 agent-skill assignment → /api/v1/skills', () => {
  it('addSkillToAgent POSTs to the v1 path with a [skillId] body', async () => {
    const spy = vi.spyOn(apiClient, 'request').mockResolvedValue({} as any)
    await apiClient.addSkillToAgent('5', '12')
    expect(spy).toHaveBeenCalledWith(
      '/api/v1/skills/agents/5/skills',
      expect.objectContaining({ method: 'POST', body: JSON.stringify([12]) }),
    )
  })

  it('removeSkillFromAgent DELETEs the v1 path with a skill_ids query', async () => {
    const spy = vi.spyOn(apiClient, 'request').mockResolvedValue({} as any)
    await apiClient.removeSkillFromAgent('5', '12')
    expect(spy).toHaveBeenCalledWith(
      '/api/v1/skills/agents/5/skills?skill_ids=12',
      expect.objectContaining({ method: 'DELETE' }),
    )
  })

  it('getAgentSkillsFromSkillsAPI GETs the v1 path', async () => {
    const spy = vi.spyOn(apiClient, 'request').mockResolvedValue([] as any)
    await apiClient.getAgentSkillsFromSkillsAPI('5')
    expect(spy).toHaveBeenCalledWith('/api/v1/skills/agents/5/skills')
  })

  it('drops the dead pre-v1 /api/skills/agents path', () => {
    const src = readFileSync(path.resolve(__dirname, '..', '..', 'lib', 'api-client.ts'), 'utf8')
    expect(src).not.toContain('/api/skills/agents')
  })
})
