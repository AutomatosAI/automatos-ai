'use client'

import { useState, useEffect, useCallback } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Input } from '@/components/ui/input'
import { Search, Download, Trash2, Shield, CheckCircle2, Loader2, Zap } from 'lucide-react'
import { ViewToggle } from '@/components/shared/view-toggle'
import { useViewMode } from '@/hooks/use-view-mode'
import { apiClient } from '@/lib/api-client'
import { toast } from 'sonner'

interface Skill {
  id: number
  name: string
  description: string | null
  category: string | null
  skill_version: string | null
  tags: string[] | null
  estimated_tokens: number
  skill_source: string | null
  is_enabled: boolean
}

interface EnabledSkill {
  skill_id: number
  name: string
  description: string | null
  category: string | null
  skill_version: string | null
  tags: string[] | null
  estimated_tokens: number
  skill_source: string | null
  enabled_at: string | null
}

export function MarketplaceSkillsTab({ searchQuery, workspaceId }: { searchQuery: string; workspaceId: string }) {
  const [viewMode, setViewMode] = useViewMode('mp-skills')
  const [available, setAvailable] = useState<Skill[]>([])
  const [enabled, setEnabled] = useState<EnabledSkill[]>([])
  const [loading, setLoading] = useState(false)
  const [enabling, setEnabling] = useState<number | null>(null)
  const [disabling, setDisabling] = useState<number | null>(null)
  const [localSearch, setLocalSearch] = useState(searchQuery || '')

  const fetchEnabled = useCallback(async () => {
    if (!workspaceId) return
    try {
      const data: any = await apiClient.get(`/api/workspaces/${workspaceId}/skills`)
      setEnabled(data.items || [])
    } catch {
      // Workspace skills endpoint may not exist yet on old deployments
    }
  }, [workspaceId])

  const fetchAvailable = useCallback(async () => {
    if (!workspaceId) return
    setLoading(true)
    try {
      const params = new URLSearchParams()
      if (localSearch.trim()) params.set('q', localSearch.trim())
      const qs = params.toString()
      const data: any = await apiClient.get(
        `/api/workspaces/${workspaceId}/skills/available${qs ? `?${qs}` : ''}`
      )
      setAvailable(data.items || [])
    } catch {
      // Endpoint may not exist yet
    } finally {
      setLoading(false)
    }
  }, [workspaceId, localSearch])

  useEffect(() => {
    fetchEnabled()
    fetchAvailable()
  }, [fetchEnabled, fetchAvailable])

  const enableSkill = async (skillId: number) => {
    setEnabling(skillId)
    try {
      await apiClient.post(`/api/workspaces/${workspaceId}/skills`, {
        skill_id: skillId,
      })
      toast.success('Skill enabled for workspace')
      await fetchEnabled()
      await fetchAvailable()
    } catch (error: any) {
      toast.error('Failed to enable skill', {
        description: error?.message || 'Unknown error',
      })
    } finally {
      setEnabling(null)
    }
  }

  const disableSkill = async (skillId: number, skillName: string) => {
    if (!confirm(`Disable ${skillName}? This will also unassign it from agents in this workspace.`)) return
    setDisabling(skillId)
    try {
      await apiClient.delete(`/api/workspaces/${workspaceId}/skills/${skillId}`)
      toast.success('Skill disabled')
      await fetchEnabled()
      await fetchAvailable()
    } catch (error: any) {
      toast.error('Failed to disable skill', {
        description: error?.message || 'Unknown error',
      })
    } finally {
      setDisabling(null)
    }
  }

  const enabledIds = new Set(enabled.map(s => s.skill_id))

  // Skill card component to avoid duplication
  const SkillCard = ({ skill, isEnabled, compact }: { skill: Skill | EnabledSkill; isEnabled: boolean; compact: boolean }) => {
    const id = 'skill_id' in skill ? skill.skill_id : skill.id
    const name = skill.name
    const tokens = skill.estimated_tokens

    if (compact) {
      return (
        <Card className="glass-card hover:border-primary/20 transition-all">
          <CardContent className="p-3">
            <div className="flex items-center gap-3">
              <div className="w-9 h-9 rounded-lg bg-primary/20 flex items-center justify-center shrink-0">
                <Zap className="w-5 h-5 text-primary" />
              </div>
              <div className="flex-1 min-w-0">
                <div className="flex items-center gap-2">
                  <span className="font-semibold text-sm truncate">{name}</span>
                  {skill.category && <Badge variant="outline" className="text-[10px] shrink-0">{skill.category}</Badge>}
                </div>
                <p className="text-xs text-muted-foreground truncate mt-0.5">{skill.description}</p>
                {tokens > 0 && <span className="text-[10px] text-muted-foreground">~{tokens} tokens</span>}
              </div>
              {isEnabled ? (
                <Button
                  size="sm"
                  variant="ghost"
                  className="h-8 w-8 p-0 shrink-0"
                  onClick={() => disableSkill(id, name)}
                  disabled={disabling === id}
                >
                  {disabling === id ? <Loader2 className="h-4 w-4 animate-spin" /> : <Trash2 className="h-4 w-4" />}
                </Button>
              ) : (
                <Button
                  size="sm"
                  variant="ghost"
                  className="h-8 w-8 p-0 shrink-0"
                  onClick={() => enableSkill(id)}
                  disabled={enabling === id}
                >
                  {enabling === id ? <Loader2 className="h-4 w-4 animate-spin" /> : <Download className="h-4 w-4" />}
                </Button>
              )}
            </div>
          </CardContent>
        </Card>
      )
    }

    return (
      <Card className="border-border/40 bg-card/50">
        <CardHeader className="pb-2">
          <div className="flex items-center justify-between">
            <CardTitle className="text-sm">{name}</CardTitle>
            {skill.category && <Badge variant="outline">{skill.category}</Badge>}
          </div>
          <CardDescription className="text-xs">{skill.description}</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              {skill.skill_version && <span className="text-xs text-muted-foreground">v{skill.skill_version}</span>}
              {tokens > 0 && <span className="text-xs text-muted-foreground">~{tokens} tokens</span>}
              {skill.tags?.map(tag => (
                <Badge key={tag} variant="secondary" className="text-[10px]">{tag}</Badge>
              ))}
            </div>
            {isEnabled ? (
              <Button
                size="sm"
                variant="ghost"
                onClick={() => disableSkill(id, name)}
                disabled={disabling === id}
              >
                {disabling === id ? <Loader2 className="h-3 w-3 animate-spin" /> : <Trash2 className="h-3 w-3 mr-1" />}
                Disable
              </Button>
            ) : (
              <Button
                size="sm"
                variant="outline"
                onClick={() => enableSkill(id)}
                disabled={enabling === id}
              >
                {enabling === id ? <Loader2 className="h-3 w-3 animate-spin mr-1" /> : <Download className="h-3 w-3 mr-1" />}
                Enable
              </Button>
            )}
          </div>
        </CardContent>
      </Card>
    )
  }

  return (
    <div className="space-y-6">
      {/* Search */}
      <div className="flex gap-2">
        <Input
          placeholder="Search marketplace skills..."
          value={localSearch}
          onChange={e => setLocalSearch(e.target.value)}
          onKeyDown={e => e.key === 'Enter' && fetchAvailable()}
        />
        <Button onClick={fetchAvailable} disabled={loading}>
          {loading ? <Loader2 className="h-4 w-4 animate-spin" /> : <Search className="h-4 w-4" />}
        </Button>
        <ViewToggle value={viewMode} onChange={setViewMode} />
      </div>

      {/* Available Skills */}
      {available.length > 0 && (
        <div>
          <h3 className="text-lg font-semibold mb-3">
            Available Skills
            <span className="text-sm text-muted-foreground font-normal ml-2">
              {available.filter(s => !s.is_enabled).length} available
            </span>
          </h3>
          <div className={viewMode === 'list'
            ? "grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3"
            : "grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4"
          }>
            {available.filter(s => !s.is_enabled).map(skill => (
              <SkillCard key={skill.id} skill={skill} isEnabled={false} compact={viewMode === 'list'} />
            ))}
          </div>
        </div>
      )}

      {available.length === 0 && !loading && (
        <div className="text-center py-12 text-muted-foreground">
          <Zap className="h-12 w-12 mx-auto mb-3 opacity-30" />
          <p className="text-sm">No marketplace skills available yet.</p>
          <p className="text-xs mt-1">Import skills via Capabilities &gt; Import from GitHub</p>
        </div>
      )}

      {/* Enabled Skills */}
      {enabled.length > 0 && (
        <div>
          <h3 className="text-lg font-semibold mb-3">
            Enabled Skills
            <Badge variant="secondary" className="ml-2 text-xs">{enabled.length}</Badge>
          </h3>
          <div className={viewMode === 'list'
            ? "grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3"
            : "grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4"
          }>
            {enabled.map(skill => (
              <SkillCard key={skill.skill_id} skill={skill} isEnabled={true} compact={viewMode === 'list'} />
            ))}
          </div>
        </div>
      )}
    </div>
  )
}
