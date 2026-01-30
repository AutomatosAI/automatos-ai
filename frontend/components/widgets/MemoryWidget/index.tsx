'use client'

/**
 * MemoryWidget Component for PRD-38.2 Extended Widgets
 *
 * Displays AI memory/context that was injected or stored during conversation
 */

import { useState, useMemo, useCallback } from 'react'
import { Brain, Plus, Search } from 'lucide-react'
import { Tabs, TabsList, TabsTrigger, TabsContent } from '@/components/ui/tabs'
import { Button } from '@/components/ui/button'
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogFooter,
} from '@/components/ui/dialog'
import { Input } from '@/components/ui/input'
import { Textarea } from '@/components/ui/textarea'
import { Label } from '@/components/ui/label'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'
import { WidgetBase } from '../WidgetBase'
import { registerWidget } from '../registry'
import { MemoryList } from './MemoryList'
import { MemorySearch } from './MemorySearch'
import type {
  WidgetBaseProps,
  MemoryWidgetData,
  Memory,
  MemoryType,
  WidgetDefinition,
} from '../types'
import { toast } from 'sonner'

type TabValue = 'injected' | 'stored' | 'all'

export function MemoryWidget({
  id,
  title,
  data,
  metadata,
  isActive,
  isLoading,
  error,
  onClose,
  onMaximize,
  onRefresh,
}: WidgetBaseProps<MemoryWidgetData>) {
  const [activeTab, setActiveTab] = useState<TabValue>(data.mode || 'all')
  const [searchQuery, setSearchQuery] = useState('')
  const [showAddDialog, setShowAddDialog] = useState(false)
  const [newMemory, setNewMemory] = useState({
    type: 'fact' as MemoryType,
    content: '',
  })

  // Combine all memories based on tab
  const displayMemories = useMemo(() => {
    let memories: Memory[] = []

    switch (activeTab) {
      case 'injected':
        memories = data.injectedMemories || []
        break
      case 'stored':
        memories = data.storedMemories || []
        break
      case 'all':
        memories = [
          ...(data.injectedMemories || []),
          ...(data.storedMemories || []),
        ]
        break
    }

    // Apply search filter
    if (searchQuery) {
      const query = searchQuery.toLowerCase()
      memories = memories.filter(
        (m) =>
          m.content.toLowerCase().includes(query) ||
          m.type.toLowerCase().includes(query)
      )
    }

    // Sort by relevance (if available) or timestamp
    return memories.sort((a, b) => {
      if (a.relevance !== undefined && b.relevance !== undefined) {
        return b.relevance - a.relevance
      }
      return (
        new Date(b.source.timestamp).getTime() -
        new Date(a.source.timestamp).getTime()
      )
    })
  }, [data, activeTab, searchQuery])

  // Handle search
  const handleSearch = useCallback((query: string) => {
    setSearchQuery(query)
  }, [])

  // Handle delete memory
  const handleDelete = useCallback((memoryId: string) => {
    toast.info('Delete functionality requires backend integration')
  }, [])

  // Handle add memory
  const handleAddMemory = useCallback(() => {
    if (!newMemory.content.trim()) return

    toast.success('Memory added (simulated)')
    setShowAddDialog(false)
    setNewMemory({ type: 'fact', content: '' })
  }, [newMemory])

  // Count memories for badges
  const injectedCount = data.injectedMemories?.length || 0
  const storedCount = data.storedMemories?.length || 0
  const totalCount = injectedCount + storedCount

  return (
    <WidgetBase
      title={title || 'Memory'}
      icon={<Brain className="h-4 w-4" />}
      metadata={metadata}
      isActive={isActive}
      isLoading={isLoading}
      error={error}
      onClose={onClose}
      onMaximize={onMaximize}
      onRefresh={onRefresh}
      customActions={[
        {
          label: 'Add Memory',
          icon: <Plus className="h-4 w-4 mr-2" />,
          onClick: () => setShowAddDialog(true),
        },
      ]}
    >
      <div className="flex flex-col h-full">
        {/* Search bar */}
        <div className="px-3 py-2 border-b border-border/30">
          <MemorySearch onSearch={handleSearch} />
        </div>

        {/* Tabs */}
        <Tabs
          value={activeTab}
          onValueChange={(v) => setActiveTab(v as TabValue)}
          className="flex-1 flex flex-col"
        >
          <div className="px-3 pt-2">
            <TabsList className="w-full">
              <TabsTrigger value="injected" className="flex-1 text-xs">
                Injected ({injectedCount})
              </TabsTrigger>
              <TabsTrigger value="stored" className="flex-1 text-xs">
                Stored ({storedCount})
              </TabsTrigger>
              <TabsTrigger value="all" className="flex-1 text-xs">
                All ({totalCount})
              </TabsTrigger>
            </TabsList>
          </div>

          <TabsContent value="injected" className="flex-1 m-0">
            <MemoryList
              memories={displayMemories}
              onDelete={handleDelete}
              emptyMessage="No memories were injected into this conversation"
              showRelevance
            />
          </TabsContent>

          <TabsContent value="stored" className="flex-1 m-0">
            <MemoryList
              memories={displayMemories}
              onDelete={handleDelete}
              emptyMessage="No memories have been stored from this conversation"
              showRelevance={false}
            />
          </TabsContent>

          <TabsContent value="all" className="flex-1 m-0">
            <MemoryList
              memories={displayMemories}
              onDelete={handleDelete}
              emptyMessage="No memories available"
              showRelevance
            />
          </TabsContent>
        </Tabs>

        {/* Footer with stats */}
        <div className="px-3 py-1.5 border-t border-border/30 bg-muted/20 text-xs text-muted-foreground">
          Showing {displayMemories.length} of {totalCount} memories
          {searchQuery && ' (filtered)'}
        </div>
      </div>

      {/* Add memory dialog */}
      <Dialog open={showAddDialog} onOpenChange={setShowAddDialog}>
        <DialogContent className="sm:max-w-[425px]">
          <DialogHeader>
            <DialogTitle>Add Memory</DialogTitle>
          </DialogHeader>
          <div className="space-y-4 py-4">
            <div className="space-y-2">
              <Label htmlFor="memory-type">Type</Label>
              <Select
                value={newMemory.type}
                onValueChange={(v) =>
                  setNewMemory((prev) => ({ ...prev, type: v as MemoryType }))
                }
              >
                <SelectTrigger id="memory-type">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="fact">Fact</SelectItem>
                  <SelectItem value="preference">Preference</SelectItem>
                  <SelectItem value="context">Context</SelectItem>
                  <SelectItem value="instruction">Instruction</SelectItem>
                </SelectContent>
              </Select>
            </div>
            <div className="space-y-2">
              <Label htmlFor="memory-content">Content</Label>
              <Textarea
                id="memory-content"
                placeholder="What should I remember?"
                value={newMemory.content}
                onChange={(e) =>
                  setNewMemory((prev) => ({ ...prev, content: e.target.value }))
                }
                rows={4}
              />
            </div>
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={() => setShowAddDialog(false)}>
              Cancel
            </Button>
            <Button onClick={handleAddMemory} disabled={!newMemory.content.trim()}>
              Add Memory
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </WidgetBase>
  )
}

/**
 * Widget definition for registry
 */
export const MemoryWidgetDef: WidgetDefinition<MemoryWidgetData> = {
  type: 'memory',
  displayName: 'Memory',
  description: 'View and manage AI memory and context',
  icon: Brain,
  component: MemoryWidget,
  defaultSize: { width: 5, height: 5 },
  minSize: { width: 4, height: 3 },
  capabilities: ['refreshable', 'fullscreen'],
}

// Register the widget
registerWidget(MemoryWidgetDef)
