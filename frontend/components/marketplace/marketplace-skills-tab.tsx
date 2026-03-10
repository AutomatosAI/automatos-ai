'use client'

import { useState, useEffect, useCallback, useMemo } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import {
  Zap,
  CheckCircle,
  Loader2,
  Download,
  Trash2,
  Search,
} from 'lucide-react'
import { Card, CardContent, CardHeader } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { StatusBadge, PremiumIcon } from '@/components/shared'
import { ViewToggle } from '@/components/shared/view-toggle'
import { useViewMode } from '@/hooks/use-view-mode'
import { useSystemIcons } from '@/hooks/use-system-config-api'
import { apiClient } from '@/lib/api-client'
import { toast } from 'sonner'

// ===================================================================
// Types
// ===================================================================

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

interface MarketplaceSkillsTabProps {
  searchQuery: string
  workspaceId: string
}

// ===================================================================
// Component
// ===================================================================

export function MarketplaceSkillsTab({ searchQuery, workspaceId }: MarketplaceSkillsTabProps) {
  const [viewMode, setViewMode] = useViewMode('mp-skills')
  const [available, setAvailable] = useState<Skill[]>([])
  const [enabled, setEnabled] = useState<EnabledSkill[]>([])
  const [loading, setLoading] = useState(true)
  const [enabling, setEnabling] = useState<number | null>(null)
  const [disabling, setDisabling] = useState<number | null>(null)
  const [localSearch, setLocalSearch] = useState('')

  const { data: iconMappings = {} } = useSystemIcons()

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
      const data: any = await apiClient.get(
        `/api/workspaces/${workspaceId}/skills/available`
      )
      setAvailable(data.items || [])
    } catch {
      // Endpoint may not exist yet
    } finally {
      setLoading(false)
    }
  }, [workspaceId])

  useEffect(() => {
    fetchEnabled()
    fetchAvailable()
  }, [fetchEnabled, fetchAvailable])

  const enableSkill = async (skillId: number, skillName: string) => {
    setEnabling(skillId)
    try {
      await apiClient.post(`/api/workspaces/${workspaceId}/skills`, {
        skill_id: skillId,
      })
      toast.success('Skill Enabled', {
        description: `${skillName} has been enabled for your workspace.`,
      })
      await fetchEnabled()
      await fetchAvailable()
    } catch (error: any) {
      const msg = error?.message || 'Failed to enable skill'
      if (msg.includes('already enabled') || msg.includes('409')) {
        toast.info('Already Enabled', {
          description: `${skillName} is already enabled for your workspace.`,
        })
      } else {
        toast.error('Failed to enable skill', { description: msg })
      }
    } finally {
      setEnabling(null)
    }
  }

  const disableSkill = async (skillId: number, skillName: string) => {
    if (!confirm(`Disable ${skillName}? This will also unassign it from agents in this workspace.`)) return
    setDisabling(skillId)
    try {
      await apiClient.delete(`/api/workspaces/${workspaceId}/skills/${skillId}`)
      toast.success('Skill Disabled', {
        description: `${skillName} has been disabled for your workspace.`,
      })
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

  // Client-side search filter (matches plugin tab pattern)
  const filteredAvailable = useMemo(() => {
    const query = (searchQuery || localSearch).toLowerCase().trim()
    if (!query) return available.filter(s => !s.is_enabled)

    return available
      .filter(s => !s.is_enabled)
      .filter(
        s =>
          s.name.toLowerCase().includes(query) ||
          s.description?.toLowerCase().includes(query) ||
          s.category?.toLowerCase().includes(query) ||
          (s.tags || []).some(t => t.toLowerCase().includes(query))
      )
  }, [available, searchQuery, localSearch])

  const enabledIds = new Set(enabled.map(s => s.skill_id))

  // Loading skeleton (matches plugin tab)
  if (loading) {
    return (
      <div className="space-y-6">
        {viewMode === 'list' ? (
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
            {[...Array(6)].map((_, i) => (
              <div key={i} className="h-16 glass-card animate-pulse rounded-xl" />
            ))}
          </div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
            {[...Array(8)].map((_, i) => (
              <Card key={i} className="glass-card animate-pulse">
                <CardHeader className="pb-2">
                  <div className="flex items-center gap-3">
                    <div className="w-10 h-10 bg-secondary/50 rounded-lg" />
                    <div className="space-y-2">
                      <div className="h-4 w-24 bg-secondary/50 rounded" />
                      <div className="h-3 w-32 bg-secondary/50 rounded" />
                    </div>
                  </div>
                </CardHeader>
                <CardContent>
                  <div className="h-8 w-full bg-secondary/50 rounded mt-2" />
                  <div className="h-6 w-20 bg-secondary/50 rounded mt-3" />
                </CardContent>
              </Card>
            ))}
          </div>
        )}
      </div>
    )
  }

  return (
    <div className="space-y-6">
      {/* Header Stats */}
      <div className="flex items-center justify-between">
        <div>
          <h3 className="text-xl font-semibold">Skills Library</h3>
          <p className="text-sm text-muted-foreground">
            Inject specialised methodology into your agents with curated skills
          </p>
        </div>
        <div className="flex items-center gap-3">
          <StatusBadge size="sm" status="success" dot>
            {enabled.length} Enabled
          </StatusBadge>
          <StatusBadge size="sm" status="info">
            {available.length} Available
          </StatusBadge>
        </div>
      </div>

      {/* Search + View Toggle */}
      <div className="flex items-center justify-between gap-4">
        <div className="flex gap-2 flex-1">
          <Input
            placeholder="Search skills..."
            value={localSearch}
            onChange={e => setLocalSearch(e.target.value)}
            className="bg-secondary/50 border-secondary"
          />
        </div>
        <ViewToggle value={viewMode} onChange={setViewMode} />
      </div>

      {/* Available Skills */}
      {filteredAvailable.length === 0 && !loading ? (
        <div className="text-center py-12">
          <div className="w-16 h-16 rounded-lg bg-secondary/30 flex items-center justify-center mx-auto mb-4">
            {iconMappings['global_skill'] ? (
              <PremiumIcon name={iconMappings['global_skill']} size={32} />
            ) : (
              <Zap className="w-8 h-8 text-muted-foreground" />
            )}
          </div>
          <h3 className="text-lg font-semibold mb-2">No skills found</h3>
          <p className="text-muted-foreground mb-4">
            {(searchQuery || localSearch)
              ? `No skills match "${searchQuery || localSearch}"`
              : 'No marketplace skills available yet. Import skills via Plugins > Import from GitHub.'}
          </p>
        </div>
      ) : viewMode === 'list' ? (
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
          {filteredAvailable.map(skill => (
            <SkillListCard
              key={skill.id}
              skill={skill}
              isEnabled={false}
              isEnabling={enabling === skill.id}
              onEnable={() => enableSkill(skill.id, skill.name)}
              iconName={(skill.category && iconMappings[skill.category]) || iconMappings['global_skill'] || null}
            />
          ))}
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
          <AnimatePresence>
            {filteredAvailable.map((skill, index) => (
              <SkillGridCard
                key={skill.id}
                skill={skill}
                index={index}
                isEnabled={false}
                isEnabling={enabling === skill.id}
                onEnable={() => enableSkill(skill.id, skill.name)}
                iconName={(skill.category && iconMappings[skill.category]) || iconMappings['global_skill'] || null}
              />
            ))}
          </AnimatePresence>
        </div>
      )}

      {/* Enabled Skills */}
      {enabled.length > 0 && (
        <div className="space-y-3">
          <div className="flex items-center gap-2">
            <CheckCircle className="w-4 h-4 text-primary" />
            <h4 className="text-sm font-semibold text-primary uppercase tracking-wider">
              Enabled for Workspace
            </h4>
            <StatusBadge size="sm" status="success">{enabled.length}</StatusBadge>
          </div>
          {viewMode === 'list' ? (
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
              {enabled.map(skill => (
                <SkillListCard
                  key={skill.skill_id}
                  skill={skill}
                  isEnabled
                  isDisabling={disabling === skill.skill_id}
                  onDisable={() => disableSkill(skill.skill_id, skill.name)}
                  iconName={(skill.category && iconMappings[skill.category]) || iconMappings['global_skill'] || null}
                />
              ))}
            </div>
          ) : (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
              <AnimatePresence>
                {enabled.map((skill, index) => (
                  <SkillGridCard
                    key={skill.skill_id}
                    skill={skill}
                    index={index}
                    isEnabled
                    isDisabling={disabling === skill.skill_id}
                    onDisable={() => disableSkill(skill.skill_id, skill.name)}
                    iconName={(skill.category && iconMappings[skill.category]) || iconMappings['global_skill'] || null}
                  />
                ))}
              </AnimatePresence>
            </div>
          )}
        </div>
      )}
    </div>
  )
}

// ===================================================================
// Skill List Card (compact view — matches plugin list card)
// ===================================================================

interface SkillListCardProps {
  skill: Skill | EnabledSkill
  isEnabled: boolean
  isEnabling?: boolean
  isDisabling?: boolean
  onEnable?: () => void
  onDisable?: () => void
  iconName?: string | null
}

function SkillListCard({ skill, isEnabled, isEnabling, isDisabling, onEnable, onDisable, iconName }: SkillListCardProps) {
  const name = skill.name
  const tokens = skill.estimated_tokens

  return (
    <Card className="glass-card hover:border-primary/20 transition-all">
      <CardContent className="p-3">
        <div className="flex items-center gap-3">
          {iconName ? (
            <PremiumIcon name={iconName} size={36} className="text-primary shrink-0" />
          ) : (
            <Zap className="w-9 h-9 text-primary shrink-0" />
          )}
          <div className="flex-1 min-w-0">
            <div className="flex items-center gap-2">
              <span className="font-semibold text-sm truncate">{name}</span>
              {skill.category && (
                <StatusBadge size="sm" status="neutral" className="shrink-0">
                  {skill.category}
                </StatusBadge>
              )}
            </div>
            <div className="flex items-center gap-2 text-xs text-muted-foreground mt-0.5">
              {tokens > 0 && <span>~{tokens} tokens</span>}
              {skill.skill_source && (
                <>
                  {tokens > 0 && <span>&middot;</span>}
                  <span>{skill.skill_source}</span>
                </>
              )}
            </div>
          </div>
          {isEnabled ? (
            <Button
              size="sm"
              variant="ghost"
              className="h-8 w-8 p-0 shrink-0 text-destructive hover:text-destructive"
              onClick={onDisable}
              disabled={isDisabling}
            >
              {isDisabling ? <Loader2 className="h-4 w-4 animate-spin" /> : <Trash2 className="h-4 w-4" />}
            </Button>
          ) : (
            <Button
              variant="ghost"
              size="sm"
              className="h-8 w-8 p-0 shrink-0"
              onClick={onEnable}
              disabled={isEnabling}
            >
              {isEnabling ? <Loader2 className="w-4 h-4 animate-spin" /> : <Download className="w-4 h-4" />}
            </Button>
          )}
        </div>
      </CardContent>
    </Card>
  )
}

// ===================================================================
// Skill Grid Card (expanded view — matches plugin grid card)
// ===================================================================

interface SkillGridCardProps {
  skill: Skill | EnabledSkill
  index: number
  isEnabled: boolean
  isEnabling?: boolean
  isDisabling?: boolean
  onEnable?: () => void
  onDisable?: () => void
  iconName?: string | null
}

function SkillGridCard({ skill, index, isEnabled, isEnabling, isDisabling, onEnable, onDisable, iconName }: SkillGridCardProps) {
  const name = skill.name
  const tokens = skill.estimated_tokens
  const version = skill.skill_version

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: 20 }}
      transition={{ delay: Math.min(index * 0.05, 0.5) }}
    >
      <Card className="glass-card card-glow hover:border-primary/20 transition-all duration-300">
        <CardHeader className="pb-3">
          <div className="flex items-start justify-between gap-3">
            <div className="flex items-center gap-3 flex-1 min-w-0">
              {iconName ? (
                <PremiumIcon name={iconName} size={40} className="text-primary shrink-0" />
              ) : (
                <Zap className="w-10 h-10 text-primary shrink-0" />
              )}
              <div className="flex-1 min-w-0">
                <h3 className="font-semibold text-foreground line-clamp-1">{name}</h3>
                <p className="text-xs text-muted-foreground">
                  {version && <>v{version}</>}
                  {skill.skill_source && (
                    <>{version && <> &middot; </>}{skill.skill_source}</>
                  )}
                </p>
              </div>
            </div>
            {isEnabled && (
              <StatusBadge size="sm" status="success" className="flex-shrink-0">
                <CheckCircle className="w-2.5 h-2.5 mr-0.5" />
                Enabled
              </StatusBadge>
            )}
          </div>
        </CardHeader>

        <CardContent className="space-y-3">
          <p className="text-sm text-muted-foreground line-clamp-2">
            {skill.description || 'No description available'}
          </p>

          {/* Category + Tags */}
          <div className="flex flex-wrap gap-1.5">
            {skill.category && (
              <StatusBadge size="sm" status="neutral">
                {skill.category}
              </StatusBadge>
            )}
            {(skill.tags || []).slice(0, 3).map(tag => (
              <StatusBadge key={tag} size="sm" status="neutral">
                {tag}
              </StatusBadge>
            ))}
          </div>

          {/* Stats + Action Row */}
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3 text-xs text-muted-foreground">
              {tokens > 0 && (
                <span>~{tokens} tokens</span>
              )}
            </div>
            {isEnabled ? (
              <Button
                size="sm"
                variant="ghost"
                className="text-destructive hover:text-destructive"
                onClick={onDisable}
                disabled={isDisabling}
              >
                {isDisabling ? (
                  <Loader2 className="h-3 w-3 animate-spin mr-1" />
                ) : (
                  <Trash2 className="h-3 w-3 mr-1" />
                )}
                Disable
              </Button>
            ) : (
              <Button
                size="sm"
                variant="outline"
                onClick={onEnable}
                disabled={isEnabling}
              >
                {isEnabling ? (
                  <Loader2 className="h-3 w-3 animate-spin mr-1" />
                ) : (
                  <Download className="h-3 w-3 mr-1" />
                )}
                Enable
              </Button>
            )}
          </div>
        </CardContent>
      </Card>
    </motion.div>
  )
}
