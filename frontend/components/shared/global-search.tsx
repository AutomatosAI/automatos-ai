'use client'

import { useRouter } from 'next/navigation'
import { LayoutDashboard, ListTodo, Bot, Brain, Share2 } from 'lucide-react'
import {
  CommandDialog,
  CommandInput,
  CommandList,
  CommandGroup,
  CommandItem,
  CommandEmpty,
  CommandSeparator,
} from '@/components/ui/command'
import { useGlobalSearch, type SearchResult } from '@/hooks/use-global-search'
import { useWorkspaceOptional } from '@/components/workspace-provider'

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

export function GlobalSearch() {
  const router = useRouter()
  // PRD-251 D1: the Socials page and posts are searched only where Socials is offered and on.
  const socials = useWorkspaceOptional()?.workspace?.socials
  const {
    open,
    query,
    setQuery,
    loading,
    error,
    pages,
    tasks,
    agents,
    memories,
    socials: posts,
    handleOpenChange,
  } = useGlobalSearch(socials)

  const handleSelect = (result: SearchResult) => {
    handleOpenChange(false)
    router.push(result.path)
  }

  const hasApiResults = tasks.length > 0 || agents.length > 0 || memories.length > 0 || posts.length > 0
  const hasAnyResults = pages.length > 0 || hasApiResults

  return (
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

        {pages.length > 0 && hasApiResults && <CommandSeparator />}

        <ResultGroup results={tasks} category="tasks" onSelect={handleSelect} />
        <ResultGroup results={agents} category="agents" onSelect={handleSelect} />
        <ResultGroup results={memories} category="memories" onSelect={handleSelect} />
        <ResultGroup results={posts} category="socials" onSelect={handleSelect} />
      </CommandList>
    </CommandDialog>
  )
}
