'use client'

/**
 * One Marketplace page for everyone (PRD-270 §5.7) — Capabilities › Skills for a signed-out
 * visitor: the member tab's own SkillGridCard and SkillListCard, header and view toggle, over the
 * public read of the global skills, by name.
 *
 * Read-only: nothing is "Added", there is no Enabled count or import, Add to Workspace goes to
 * sign-in carrying the skill's link, and Details opens the public detail. The workspace's own skills
 * are never read.
 */
import { useMemo, useState } from 'react'
import { AnimatePresence } from 'framer-motion'

import { StatusBadge } from '@/components/shared'
import { ViewToggle } from '@/components/shared/view-toggle'
import { Input } from '@/components/ui/input'
import { useViewMode } from '@/hooks/use-view-mode'

import { useMarketplaceAccess } from '../access/marketplace-access'
import { skillOf } from '../access/public-capability-shapes'
import { SkillGridCard, SkillListCard } from '../marketplace-skills-tab'
import type { Skill } from '../marketplace-skills-tab'
import { NO_ITEMS, NO_ITEM_MATCHES } from './copy'
import { PublicItemDetail } from './public-item-detail'
import { PublicReadState, usePublicEntries } from './public-tab-parts'

const SKILLS_TAB = 'skills'

/** A skill as its card draws it, with the key its link names it by. */
interface KeyedSkill {
  readonly key: string
  readonly skill: Skill
}

function matches({ skill }: KeyedSkill, query: string): boolean {
  const text = [skill.name, skill.description, skill.category, ...(skill.tags ?? [])]
  return !query || text.some((part) => part?.toLowerCase().includes(query))
}

function SkillsHeader({ available }: { available: number }) {
  return (
    <div className="flex items-center justify-between">
      <div>
        <h3 className="text-xl font-semibold">Skills Library</h3>
        <p className="text-sm text-muted-foreground">Inject specialised methodology into your agents with curated skills</p>
      </div>
      <StatusBadge size="sm" status="info">
        {available} Available
      </StatusBadge>
    </div>
  )
}

export function PublicSkillsTab({ searchQuery }: { searchQuery: string }) {
  const [viewMode, setViewMode] = useViewMode('mp-skills')
  const { requireSignIn } = useMarketplaceAccess()
  const { read, entries } = usePublicEntries('skills')
  const skills = useMemo(() => entries.map((entry, index) => ({ key: entry.item.key, skill: skillOf(entry, index) })), [entries])
  const [localSearch, setLocalSearch] = useState('')
  const [openKey, setOpenKey] = useState<string | null>(null)
  const query = (searchQuery || localSearch).trim().toLowerCase()
  const shown = useMemo(() => skills.filter((keyed) => matches(keyed, query)), [skills, query])

  if (read.isLoading || read.isError) return <PublicReadState read={read} />
  const add = (key: string) => () => requireSignIn(SKILLS_TAB, key)
  return (
    <div className="space-y-6" data-testid="public-skills-tab">
      <SkillsHeader available={skills.length} />
      <div className="flex items-center justify-between gap-4">
        <Input
          placeholder="Search skills..."
          value={localSearch}
          onChange={(e) => setLocalSearch(e.target.value)}
          className="bg-secondary/50 border-secondary"
        />
        <ViewToggle value={viewMode} onChange={setViewMode} />
      </div>
      {shown.length === 0 && <p className="text-sm text-muted-foreground">{skills.length ? NO_ITEM_MATCHES : NO_ITEMS}</p>}
      {viewMode === 'list' ? (
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
          {shown.map(({ key, skill }) => (
            <SkillListCard key={key} skill={skill} isEnabled={false} onEnable={add(key)} />
          ))}
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
          <AnimatePresence>
            {shown.map(({ key, skill }, index) => (
              <SkillGridCard key={key} skill={skill} index={index} isEnabled={false}
                onEnable={add(key)} onDetails={() => setOpenKey(key)} />
            ))}
          </AnimatePresence>
        </div>
      )}
      {openKey && <PublicItemDetail type="skills" tab={SKILLS_TAB} itemKey={openKey} onClose={() => setOpenKey(null)} />}
    </div>
  )
}
