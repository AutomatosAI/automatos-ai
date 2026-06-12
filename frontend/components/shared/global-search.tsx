'use client'

import { useRouter } from 'next/navigation'
import { LayoutDashboard, ListTodo, Bot, Brain } from 'lucide-react'
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

const CATEGORY_META = {
  pages: { heading: 'Pages', icon: LayoutDashboard },
  tasks: { heading: 'Tasks', icon: ListTodo },
  agents: { heading: 'Agents', icon: Bot },
  memories: { heading: 'Memories', icon: Brain },
} as const

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
          value={`${result.category}-${result.label}`}
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
    handleOpenChange,
  } = useGlobalSearch()

  const handleSelect = (result: SearchResult) => {
    handleOpenChange(false)
    router.push(result.path)
  }

  const hasApiResults = tasks.length > 0 || agents.length > 0 || memories.length > 0
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
      </CommandList>
    </CommandDialog>
  )
}
