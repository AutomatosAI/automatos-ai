'use client'

/**
 * The Playbooks tab's cards and type filter (marketplace-playbooks-tab.tsx), drawn the same for a
 * member and for a signed-out visitor (PRD-270 §5.7, one page for everyone).
 *
 * Signed out, the public read has no steps and no install counts, so a card shows none of them
 * rather than a made-up 0 (useMarketplaceAccess); its Add goes to sign-in (the tab's `onInstall`).
 */
import { Bot, ChefHat, Download, Eye, FileText, Loader2, MoreVertical } from 'lucide-react'

import { PremiumIcon } from '@/components/shared'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardHeader } from '@/components/ui/card'
import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuTrigger } from '@/components/ui/dropdown-menu'
import { Separator } from '@/components/ui/separator'

import { useMarketplaceAccess } from './access/marketplace-access'

/** A Marketplace Playbook as `/api/marketplace/items?type=recipe` answers it (or the public read, mapped). */
export interface PlaybookCardRecipe {
  id: number
  name: string
  description?: string | null
  creator_name?: string | null
  marketplace_category?: string | null
  is_approved?: boolean
  install_count?: number
  steps?: { agent_id?: unknown }[] | null
}

interface PlaybookCardProps {
  recipe: PlaybookCardRecipe
  iconMappings: Record<string, string>
  isAdmin: boolean
  installing: boolean
  onView: (recipe: PlaybookCardRecipe) => void
  onInstall: (e: React.MouseEvent, recipeId: number) => void
}

const THOUSAND = 1000
const FILTER_ACTIVE = 'bg-secondary border-primary/50 text-foreground font-semibold'
const FILTER_IDLE = 'border-secondary text-muted-foreground hover:bg-secondary'
const PLAYBOOK_TYPES = [
  { value: 'all', label: 'All Playbooks' },
  { value: 'simple', label: 'Simple' },
  { value: 'complex', label: 'Complex' },
]

const stepCount = (recipe: PlaybookCardRecipe) => recipe.steps?.length || 0
const agentCount = (recipe: PlaybookCardRecipe) => (recipe.steps ? new Set(recipe.steps.map((s) => s.agent_id)).size : 0)

function installsLabel(count: number | undefined): string {
  return count !== undefined && count >= THOUSAND ? `${(count / THOUSAND).toFixed(1)}k installs` : `${count || 0} installs`
}

function PlaybookIcon({ recipe, iconMappings, size }: { recipe: PlaybookCardRecipe; iconMappings: Record<string, string>; size: 36 | 40 }) {
  const premiumIconName = (recipe.marketplace_category && iconMappings[recipe.marketplace_category]) || iconMappings['global_recipe'] || null
  if (premiumIconName) return <PremiumIcon name={premiumIconName} size={size} className="text-primary shrink-0" />
  return <ChefHat className={`${size === 36 ? 'w-8 h-8' : 'w-10 h-10'} text-primary shrink-0`} />
}

function PendingBadge({ small }: { small?: boolean }) {
  return (
    <Badge variant="outline" className={`${small ? 'text-[10px] shrink-0' : 'text-xs'} border-warning/30 text-warning`}>
      Pending
    </Badge>
  )
}

/** All Playbooks, Simple, Complex. */
export function PlaybookTypeFilter({ selected, onSelect }: { selected: string; onSelect: (value: string) => void }) {
  return (
    <div className="flex gap-2 overflow-x-auto pb-1 scrollbar-thin scrollbar-thumb-secondary scrollbar-track-transparent flex-1">
      {PLAYBOOK_TYPES.map((type) => (
        <Button
          key={type.value}
          variant={selected === type.value ? 'default' : 'outline'}
          size="sm"
          onClick={() => onSelect(type.value)}
          className={`whitespace-nowrap flex-shrink-0 ${selected === type.value ? FILTER_ACTIVE : FILTER_IDLE}`}
        >
          {type.label}
        </Button>
      ))}
    </div>
  )
}

export function PlaybookListRow({ recipe, iconMappings, isAdmin, installing, onView, onInstall }: PlaybookCardProps) {
  const { signedIn } = useMarketplaceAccess()
  return (
    <Card className="glass-card card-glow hover:border-primary/20 transition-all cursor-pointer" onClick={() => onView(recipe)}>
      <CardContent className="p-3">
        <div className="flex items-center gap-3">
          <PlaybookIcon recipe={recipe} iconMappings={iconMappings} size={36} />
          <div className="flex-1 min-w-0">
            <div className="flex items-center gap-2">
              <span className="font-semibold text-sm truncate">{recipe.name}</span>
              {isAdmin && !recipe.is_approved && <PendingBadge small />}
            </div>
            {signedIn && (
              <div className="flex items-center gap-2 text-xs text-muted-foreground mt-0.5">
                <span>{stepCount(recipe)} Steps</span>
                <span>&middot;</span>
                <span>{agentCount(recipe)} Agents</span>
                <span>&middot;</span>
                <span>{recipe.install_count || 0} installs</span>
              </div>
            )}
          </div>
          <Button
            variant="ghost"
            size="sm"
            className="h-8 w-8 p-0 shrink-0"
            onClick={(e) => { e.stopPropagation(); onInstall(e, recipe.id) }}
            disabled={installing}
          >
            {installing ? <Loader2 className="w-4 h-4 animate-spin" /> : <Download className="w-4 h-4" />}
          </Button>
        </div>
      </CardContent>
    </Card>
  )
}

function PlaybookAdminMenu({ recipe, onView }: { recipe: PlaybookCardRecipe; onView: (recipe: PlaybookCardRecipe) => void }) {
  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <Button variant="ghost" size="sm" className="h-8 w-8 p-0" onClick={(e) => e.stopPropagation()}>
          <MoreVertical className="h-4 w-4" />
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="end">
        <DropdownMenuItem onClick={(e) => { e.stopPropagation(); onView(recipe) }}>
          <Eye className="w-4 h-4 mr-2" />
          View Details
        </DropdownMenuItem>
      </DropdownMenuContent>
    </DropdownMenu>
  )
}

/** Steps and agents, then installs: signed in only (the public read has neither). */
function PlaybookStats({ recipe }: { recipe: PlaybookCardRecipe }) {
  const { signedIn } = useMarketplaceAccess()
  if (!signedIn) return null
  return (
    <>
      <div className="flex items-center gap-3 text-xs text-muted-foreground">
        <div className="flex items-center gap-1">
          <FileText className="w-3.5 h-3.5" />
          <span>{stepCount(recipe)} Steps</span>
        </div>
        <div className="flex items-center gap-1">
          <Bot className="w-3.5 h-3.5" />
          <span>{agentCount(recipe)} Agents</span>
        </div>
      </div>
      <div className="text-xs text-muted-foreground pb-2">{installsLabel(recipe.install_count)}</div>
    </>
  )
}

export function PlaybookGridCard({ recipe, iconMappings, isAdmin, installing, onView, onInstall }: PlaybookCardProps) {
  return (
    <Card className="glass-card card-glow hover:border-primary/20 transition-all duration-300 cursor-pointer" onClick={() => onView(recipe)}>
      <CardHeader className="pb-3">
        <div className="flex items-start justify-between gap-3">
          <div className="flex items-center gap-3 flex-1 min-w-0">
            <PlaybookIcon recipe={recipe} iconMappings={iconMappings} size={40} />
            <div className="flex-1 min-w-0">
              <div className="flex items-center gap-2">
                <h3 className="font-semibold text-foreground line-clamp-1">{recipe.name}</h3>
                {isAdmin && !recipe.is_approved && <PendingBadge />}
              </div>
              <p className="text-xs text-muted-foreground">by {recipe.creator_name || 'Unknown'}</p>
            </div>
          </div>
          {isAdmin && <PlaybookAdminMenu recipe={recipe} onView={onView} />}
        </div>
      </CardHeader>
      <CardContent className="space-y-3">
        <p className="text-sm text-muted-foreground line-clamp-2">{recipe.description || 'No description available'}</p>
        {recipe.marketplace_category && (
          <div className="flex flex-col gap-2">
            <Badge variant="outline" className="text-xs border-border text-muted-foreground w-fit">
              {recipe.marketplace_category}
            </Badge>
          </div>
        )}
        <PlaybookStats recipe={recipe} />
      </CardContent>
      <Separator />
      <div className="flex items-center justify-between px-6 py-3">
        <Button variant="ghost" size="sm" onClick={(e) => { e.stopPropagation(); onView(recipe) }}
          className="text-muted-foreground hover:text-foreground p-0 h-auto">
          Details
        </Button>
        <Button size="sm" variant="outline" disabled={installing}
          onClick={(e) => { e.stopPropagation(); onInstall(e, recipe.id) }}>
          {installing ? 'Adding...' : 'Add to Workspace'}
        </Button>
      </div>
    </Card>
  )
}
