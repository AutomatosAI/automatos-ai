
'use client'

import { useState, useMemo } from 'react'
import { Search, Wrench, CheckCircle, AlertCircle, Filter } from 'lucide-react'
import { Input } from '@/components/ui/input'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Checkbox } from '@/components/ui/checkbox'
import { ToolLogo } from '@/components/ui/tool-logo'
import { Card, CardContent } from '@/components/ui/card'
import { useTools } from '@/hooks/use-tools-api'

interface AgentToolAssignmentProps {
    assignedToolIds: number[]
    onToggleTool: (toolId: number) => void
    disabled?: boolean
}

export function AgentToolAssignment({
    assignedToolIds,
    onToggleTool,
    disabled = false
}: AgentToolAssignmentProps) {
    const [searchQuery, setSearchQuery] = useState('')
    const [filter, setFilter] = useState<'all' | 'assigned' | 'unassigned'>('all')

    // Fetch only active (installed) tools to assign
    const { data: toolsData, isLoading } = useTools({
        status: 'active',
        limit: 1000
    })

    const tools = (toolsData as any)?.data || []

    // Filter tools
    const filteredTools = useMemo(() => {
        let result = tools

        // Search filter
        if (searchQuery.trim()) {
            const query = searchQuery.toLowerCase()
            result = result.filter((tool: any) =>
                tool.name.toLowerCase().includes(query) ||
                tool.description?.toLowerCase().includes(query) ||
                tool.provider?.toLowerCase().includes(query)
            )
        }

        // Status filter
        if (filter === 'assigned') {
            result = result.filter((tool: any) => assignedToolIds.includes(tool.id))
        } else if (filter === 'unassigned') {
            result = result.filter((tool: any) => !assignedToolIds.includes(tool.id))
        }

        return result
    }, [tools, searchQuery, filter, assignedToolIds])

    if (isLoading) {
        return (
            <div className="py-8 text-center text-muted-foreground">
                Loading tools...
            </div>
        )
    }

    if (tools.length === 0) {
        return (
            <div className="text-center py-8 bg-secondary/20 rounded-lg border border-dashed border-border">
                <Wrench className="w-10 h-10 text-muted-foreground mx-auto mb-3" />
                <h3 className="text-sm font-medium">No tools installed</h3>
                <p className="text-xs text-muted-foreground mt-1 max-w-xs mx-auto">
                    Go to the Tools Dashboard to install apps and tools first.
                </p>
            </div>
        )
    }

    return (
        <div className="space-y-4">
            {/* Search and Filters */}
            <div className="flex gap-2">
                <div className="relative flex-1">
                    <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" />
                    <Input
                        placeholder="Search installed tools..."
                        value={searchQuery}
                        onChange={(e) => setSearchQuery(e.target.value)}
                        className="pl-9 h-9"
                    />
                </div>
                <div className="flex bg-secondary/50 rounded-md p-1">
                    <Button
                        variant={filter === 'all' ? 'secondary' : 'ghost'}
                        size="sm"
                        onClick={() => setFilter('all')}
                        className="h-7 text-xs px-3"
                    >
                        All
                    </Button>
                    <Button
                        variant={filter === 'assigned' ? 'secondary' : 'ghost'}
                        size="sm"
                        onClick={() => setFilter('assigned')}
                        className="h-7 text-xs px-3"
                    >
                        Assigned
                    </Button>
                    <Button
                        variant={filter === 'unassigned' ? 'secondary' : 'ghost'}
                        size="sm"
                        onClick={() => setFilter('unassigned')}
                        className="h-7 text-xs px-3"
                    >
                        Unassigned
                    </Button>
                </div>
            </div>

            {/* Tools Grid */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-2 gap-3 max-h-[400px] overflow-y-auto pr-1">
                {filteredTools.map((tool: any) => {
                    const isAssigned = assignedToolIds.includes(tool.id)
                    return (
                        <div
                            key={tool.id}
                            className={`
                group relative flex items-start space-x-3 p-3 rounded-lg border transition-all duration-200
                ${isAssigned
                                    ? 'bg-primary/5 border-primary/30'
                                    : 'bg-background/50 border-border/50 hover:border-primary/20'}
              `}
                        >
                            <Checkbox
                                id={`tool-${tool.id}`}
                                checked={isAssigned}
                                onCheckedChange={() => !disabled && onToggleTool(tool.id)}
                                disabled={disabled}
                                className="mt-1"
                            />
                            <div className="flex-1 min-w-0">
                                <label
                                    htmlFor={`tool-${tool.id}`}
                                    className={`cursor-pointer ${disabled ? 'cursor-not-allowed' : ''}`}
                                >
                                    <div className="flex items-center justify-between mb-1">
                                        <div className="flex items-center gap-2">
                                            <ToolLogo
                                                name={tool.name}
                                                logo={tool.icon}
                                                size={16}
                                                showBackground={false}
                                            />
                                            <span className="font-medium text-sm truncate">{tool.name}</span>
                                        </div>
                                        <Badge variant="outline" className="text-[10px] h-4 px-1">
                                            {tool.provider}
                                        </Badge>
                                    </div>
                                    <p className="text-xs text-muted-foreground line-clamp-2 leading-relaxed">
                                        {tool.description || 'No description available'}
                                    </p>
                                </label>
                            </div>

                            {isAssigned && (
                                <div className="absolute top-2 right-2 text-primary opacity-0 group-hover:opacity-100 transition-opacity">
                                    <CheckCircle className="w-3 h-3" />
                                </div>
                            )}
                        </div>
                    )
                })}

                {filteredTools.length === 0 && (
                    <div className="col-span-full py-8 text-center text-muted-foreground bg-secondary/10 rounded border border-dashed text-sm">
                        No tools match your search
                    </div>
                )}
            </div>

            <div className="text-xs text-muted-foreground flex justify-between px-1">
                <span>{filteredTools.length} tools shown</span>
                <span>{assignedToolIds.length} assigned total</span>
            </div>
        </div>
    )
}
