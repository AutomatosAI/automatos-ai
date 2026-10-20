'use client'

import { useMemo } from 'react'
import { useRouter } from 'next/navigation'
import { LayoutDashboard, ListTodo, Bot, Brain, Share2, Palette } from 'lucide-react'
import {
  CommandDialog,
  CommandInput,
  CommandList,
  CommandGroup,
  CommandItem,
  CommandEmpty,
  CommandSeparator,
} from '@/components/ui/command'
import { THEME_SEARCH_WORDS, useGlobalSearch, type SearchResult, type ThemeSearchResult } from '@/hooks/use-global-search'
import { useActiveTheme } from '@/hooks/use-studio-theme'
import { useWorkspaceOptional } from '@/components/workspace-provider'
import { useUiStyleOptional } from '@/contexts/ui-style-context'
import { ThemeChoiceSync, styleChoices, useChooseStyle, useWantInstalledThemes } from '@/lib/theme-host/choice-sync'

const CATEGORY_META = {
  pages: { heading: 'Pages', icon: LayoutDashboard },
  tasks: { heading: 'Tasks', icon: ListTodo },
  agents: { heading: 'Agents', icon: Bot },
  memories: { heading: 'Memories', icon: Brain },
  socials: { heading: 'Socials', icon: Share2 },
} as const

/** cmdk filters by value too: a result found by its keywords (a post's brief) must match them. */
function itemValue(result: SearchResult): string {
  return [`${result.category}-${result.label}`, result.keywords].filter(Boolean).join(' ')
}

function ResultGroup({
  results,
  category,
  onSelect,
}: {
  results: SearchResult[]
  category: SearchResult['category']
  onSelect: (result: SearchResult) => void
}) {
  if (results.length === 0) return null
  const meta = CATEGORY_META[category]
  const Icon = meta.icon

  return (
    <CommandGroup heading={meta.heading}>
      {results.map((result) => (
        <CommandItem
          key={result.id}
          value={itemValue(result)}
          onSelect={() => onSelect(result)}
        >
          <Icon className="mr-2 h-4 w-4 shrink-0 opacity-70" />
          <div className="flex flex-col gap-0.5 overflow-hidden">
            <span className="truncate">{result.label}</span>
            {result.description && (
              <span className="truncate text-xs text-muted-foreground">
                {result.description}
              </span>
            )}
          </div>
        </CommandItem>
      ))}
    </CommandGroup>
  )
}

const THEMES_HEADING = 'Themes'

/** cmdk filters by value too: a style is kept for its words ('theme', 'style') as for its name. */
function themeItemValue(result: ThemeSearchResult): string {
  return [`themes-${result.label}`, ...THEME_SEARCH_WORDS, result.description].filter(Boolean).join(' ')
}

/** PRD-261 A-02c: the styles ⌘K switches to; choosing one is the theme menu's pick. */
function ThemeGroup({ results, onSelect }: { results: ThemeSearchResult[]; onSelect: (result: ThemeSearchResult) => void }) {
  if (results.length === 0) return null
  return (
    <CommandGroup heading={THEMES_HEADING}>
      {results.map((result) => (
        <CommandItem key={result.id} value={themeItemValue(result)} onSelect={() => onSelect(result)}>
          <Palette className="mr-2 h-4 w-4 shrink-0 opacity-70" />
          <div className="flex flex-col gap-0.5 overflow-hidden">
            <span className="truncate">{result.label}</span>
            {result.description && <span className="truncate text-xs text-muted-foreground">{result.description}</span>}
          </div>
        </CommandItem>
      ))}
    </CommandGroup>
  )
}

/** Classic, Studio and the themes (D1-c applied), or none outside the style provider. */
function useStyleChoices() {
  const ui = useUiStyleOptional()
  const installed = ui?.installedThemes
  const style = ui?.style ?? null
  return useMemo(() => (installed === undefined ? undefined : styleChoices(installed, style)), [installed, style])
}

/**
 * The ⌘K dialog. It also mounts ThemeChoiceSync (PRD-261 A-02c): this is the one component that is
 * always mounted inside the workspace and query providers, where the theme choice is checked.
 */
export function GlobalSearch() {
  const router = useRouter()
  // PRD-251 D1: the Socials page and posts are searched only where Socials is offered and on.
  const socials = useWorkspaceOptional()?.workspace?.socials
  // PRD-260 M90, PRD-261 A-02b: under a theme, its own pages (its registry entry's nav) are found too.
  const themeNav = useActiveTheme()?.nav
  const chooseStyle = useChooseStyle()
  const {
    open,
    query,
    setQuery,
    loading,
    error,
    pages,
    themes,
    tasks,
    agents,
    memories,
    socials: posts,
    handleOpenChange,
  } = useGlobalSearch(socials, themeNav, useStyleChoices())
  // The workspace's installs (its themes in the list) are fetched only once ⌘K first opens.
  useWantInstalledThemes(open)

  const handleSelect = (result: SearchResult) => {
    handleOpenChange(false)
    router.push(result.path)
  }

  const handleThemeSelect = (result: ThemeSearchResult) => {
    handleOpenChange(false)
    chooseStyle(result.style)
  }

  const hasApiResults = tasks.length > 0 || agents.length > 0 || memories.length > 0 || posts.length > 0
  const hasAnyResults = pages.length > 0 || themes.length > 0 || hasApiResults

  return (
    <>
      <ThemeChoiceSync />
      <CommandDialog open={open} onOpenChange={handleOpenChange}>
        <CommandInput
          placeholder="Search pages, tasks, agents, memories..."
          value={query}
          onValueChange={setQuery}
        />
        <CommandList>
          {loading && <CommandEmpty>Searching...</CommandEmpty>}
          {!loading && error && (
            <div className="px-3 py-3 text-sm text-destructive" role="alert">
              {error}
            </div>
          )}
          {!loading && !error && !hasAnyResults && (
            <CommandEmpty>No results found.</CommandEmpty>
          )}

          <ResultGroup results={pages} category="pages" onSelect={handleSelect} />
          <ThemeGroup results={themes} onSelect={handleThemeSelect} />

          {(pages.length > 0 || themes.length > 0) && hasApiResults && <CommandSeparator />}

          <ResultGroup results={tasks} category="tasks" onSelect={handleSelect} />
          <ResultGroup results={agents} category="agents" onSelect={handleSelect} />
          <ResultGroup results={memories} category="memories" onSelect={handleSelect} />
          <ResultGroup results={posts} category="socials" onSelect={handleSelect} />
        </CommandList>
      </CommandDialog>
    </>
  )
}
