'use client'

import { useState, useMemo } from 'react'
import { Search, Bot, Loader2, Download, Zap, Wrench } from 'lucide-react'
import { Input } from '@/components/ui/input'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Card, CardContent, CardHeader } from '@/components/ui/card'
import { useToast } from '@/hooks/use-toast'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription } from '@/components/ui/dialog'
import { ToolLogo } from '@/components/ui/tool-logo'

// Agent categories matching US-006
const AGENT_CATEGORIES = [
  { id: 'all', name: 'All Categories' },
  { id: 'Personal Assistant', name: 'Personal Assistant' },
  { id: 'Customer Support', name: 'Customer Support' },
  { id: 'DevOps', name: 'DevOps' },
  { id: 'Social Media', name: 'Social Media' },
  { id: 'Accounting', name: 'Accounting' },
  { id: 'E-commerce', name: 'E-commerce' },
  { id: 'Content Creation', name: 'Content Creation' },
  { id: 'HR', name: 'HR' },
  { id: 'Data Analysis', name: 'Data Analysis' },
  { id: 'Custom', name: 'Custom' },
]

interface MarketplaceAgent {
  id: number
  name: string
  description: string
  creator_name: string
  category: string
  install_count: number
  icon?: string
  metadata: {
    agent_type?: string
    model_id?: string
    skills?: string[]
    tools?: number[]
    tool_names?: string[]
    tool_icons?: string[]
  }
}

export function MarketplaceAgentsTab() {
  const { toast } = useToast()
  const queryClient = useQueryClient()
  const [searchQuery, setSearchQuery] = useState('')
  const [selectedCategory, setSelectedCategory] = useState('all')
  const [selectedAgent, setSelectedAgent] = useState<MarketplaceAgent | null>(null)
  const [isDetailModalOpen, setIsDetailModalOpen] = useState(false)

  // Fetch marketplace agents
  const { data: agents = [], isLoading } = useQuery({
    queryKey: ['marketplaceAgents', selectedCategory, searchQuery],
    queryFn: async () => {
      const params = new URLSearchParams({
        type: 'agent',
        ...(selectedCategory !== 'all' && { category: selectedCategory }),
        ...(searchQuery && { search: searchQuery }),
      })
      const response = await fetch(`/api/marketplace/items?${params}`)
      if (!response.ok) throw new Error('Failed to fetch agents')
      return response.json()
    },
  })

  // Install agent mutation
  const installMutation = useMutation({
    mutationFn: async (agentId: number) => {
      const response = await fetch(`/api/marketplace/items/${agentId}/install`, {
        method: 'POST',
      })
      if (!response.ok) {
        const error = await response.json()
        throw new Error(error.detail || 'Failed to install agent')
      }
      return response.json()
    },
    onSuccess: (data, agentId) => {
      const agent = agents.find((a: MarketplaceAgent) => a.id === agentId)
      toast({
        title: 'Agent Installed',
        description: `${agent?.name} installed successfully!`,
      })
      queryClient.invalidateQueries({ queryKey: ['agents'] })
      setIsDetailModalOpen(false)
    },
    onError: (error: any) => {
      toast({
        title: 'Installation Failed',
        description: error.message || 'Failed to install agent',
        variant: 'destructive',
      })
    },
  })

  const handleAgentClick = (agent: MarketplaceAgent) => {
    setSelectedAgent(agent)
    setIsDetailModalOpen(true)
  }

  const handleInstall = () => {
    if (selectedAgent) {
      installMutation.mutate(selectedAgent.id)
    }
  }

  const formatInstallCount = (count: number) => {
    if (count >= 1000000) return `${(count / 1000000).toFixed(1)}M`
    if (count >= 1000) return `${(count / 1000).toFixed(1)}k`
    return count.toString()
  }

  return (
    <div className="space-y-6">
      {/* Search */}
      <div className="relative">
        <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 h-4 w-4" />
        <Input
          type="text"
          placeholder="Search agents..."
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
          className="pl-10 bg-[#1a1a1a] border-gray-800 text-white placeholder:text-gray-500"
        />
      </div>

      {/* Category Filter Buttons */}
      <div className="flex flex-wrap gap-2">
        {AGENT_CATEGORIES.map((category) => (
          <Button
            key={category.id}
            variant={selectedCategory === category.id ? 'default' : 'outline'}
            size="sm"
            onClick={() => setSelectedCategory(category.id)}
            className={
              selectedCategory === category.id
                ? 'bg-orange-500 hover:bg-orange-600 text-white'
                : 'border-gray-700 text-gray-300 hover:bg-gray-800'
            }
          >
            {category.name}
          </Button>
        ))}
      </div>

      {/* Agents Grid */}
      {isLoading ? (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {[...Array(6)].map((_, i) => (
            <div key={i} className="h-64 bg-gray-800 animate-pulse rounded-lg" />
          ))}
        </div>
      ) : agents.length === 0 ? (
        <div className="text-center py-12 text-gray-400">
          <Bot className="w-12 h-12 mx-auto mb-4 text-gray-600" />
          <p>No agents found. Try adjusting your search or filters.</p>
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {agents.map((agent: MarketplaceAgent) => (
            <Card
              key={agent.id}
              className="bg-[#1a1a1a] border-gray-800 hover:border-orange-500/50 transition-all duration-200 cursor-pointer"
              onClick={() => handleAgentClick(agent)}
            >
              <CardHeader className="pb-3">
                <div className="flex items-start justify-between gap-3">
                  <div className="flex items-center gap-3 flex-1 min-w-0">
                    <div className="w-12 h-12 rounded-lg bg-orange-500/10 border border-orange-500/30 flex items-center justify-center shrink-0">
                      <Bot className="w-6 h-6 text-orange-400" />
                    </div>
                    <div className="flex-1 min-w-0">
                      <h3 className="font-semibold text-white line-clamp-1">
                        {agent.name}
                      </h3>
                      <p className="text-xs text-gray-500">
                        by {agent.creator_name}
                      </p>
                    </div>
                  </div>
                </div>
              </CardHeader>

              <CardContent className="space-y-3">
                <p className="text-sm text-gray-400 line-clamp-2">
                  {agent.description}
                </p>

                {/* Category badge */}
                <div className="flex items-center gap-2">
                  <Badge variant="outline" className="text-xs border-gray-700 text-gray-400">
                    {agent.category}
                  </Badge>
                  {agent.metadata.model_id && (
                    <Badge className="text-xs bg-purple-500/20 text-purple-300 border-purple-500/30">
                      <Zap className="w-3 h-3 mr-1" />
                      {agent.metadata.model_id.split('/').pop()?.substring(0, 15)}
                    </Badge>
                  )}
                </div>

                {/* Tools icons (max 4) */}
                {agent.metadata.tool_names && agent.metadata.tool_names.length > 0 && (
                  <div className="flex items-center gap-2">
                    <Wrench className="w-4 h-4 text-gray-500" />
                    <div className="flex items-center gap-1">
                      {agent.metadata.tool_names.slice(0, 4).map((toolName, idx) => (
                        <div key={idx} className="w-6 h-6 rounded bg-white/5 flex items-center justify-center">
                          <ToolLogo
                            name={toolName}
                            logo={agent.metadata.tool_icons?.[idx]}
                            size={16}
                          />
                        </div>
                      ))}
                      {agent.metadata.tool_names.length > 4 && (
                        <span className="text-xs text-gray-500 ml-1">
                          +{agent.metadata.tool_names.length - 4}
                        </span>
                      )}
                    </div>
                  </div>
                )}

                {/* Install count */}
                <div className="flex items-center justify-between pt-2 border-t border-gray-800">
                  <span className="text-xs text-gray-500">
                    {formatInstallCount(agent.install_count)} installs
                  </span>
                  <Button
                    size="sm"
                    className="bg-orange-500 hover:bg-orange-600 text-white"
                    onClick={(e) => {
                      e.stopPropagation()
                      handleAgentClick(agent)
                    }}
                  >
                    <Download className="w-3 h-3 mr-1" />
                    Install
                  </Button>
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
      )}

      {/* Detail Modal */}
      <Dialog open={isDetailModalOpen} onOpenChange={setIsDetailModalOpen}>
        <DialogContent className="bg-[#1a1a1a] border-gray-800 text-white max-w-2xl">
          {selectedAgent && (
            <>
              <DialogHeader>
                <div className="flex items-center gap-3 mb-2">
                  <div className="w-14 h-14 rounded-lg bg-orange-500/10 border border-orange-500/30 flex items-center justify-center">
                    <Bot className="w-8 h-8 text-orange-400" />
                  </div>
                  <div>
                    <DialogTitle className="text-xl text-white">{selectedAgent.name}</DialogTitle>
                    <DialogDescription className="text-gray-400">
                      by {selectedAgent.creator_name} • {formatInstallCount(selectedAgent.install_count)} installs
                    </DialogDescription>
                  </div>
                </div>
              </DialogHeader>

              <div className="space-y-4 mt-4">
                {/* Description */}
                <div>
                  <h4 className="text-sm font-medium text-gray-300 mb-2">Description</h4>
                  <p className="text-sm text-gray-400">{selectedAgent.description}</p>
                </div>

                {/* Category and Model */}
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <h4 className="text-sm font-medium text-gray-300 mb-2">Category</h4>
                    <Badge variant="outline" className="border-gray-700 text-gray-300">
                      {selectedAgent.category}
                    </Badge>
                  </div>
                  {selectedAgent.metadata.model_id && (
                    <div>
                      <h4 className="text-sm font-medium text-gray-300 mb-2">LLM Model</h4>
                      <Badge className="bg-purple-500/20 text-purple-300 border-purple-500/30">
                        <Zap className="w-3 h-3 mr-1" />
                        {selectedAgent.metadata.model_id}
                      </Badge>
                    </div>
                  )}
                </div>

                {/* Assigned Tools */}
                {selectedAgent.metadata.tool_names && selectedAgent.metadata.tool_names.length > 0 && (
                  <div>
                    <h4 className="text-sm font-medium text-gray-300 mb-2">Assigned Tools</h4>
                    <div className="flex flex-wrap gap-2">
                      {selectedAgent.metadata.tool_names.map((toolName, idx) => (
                        <div
                          key={idx}
                          className="flex items-center gap-2 px-3 py-2 rounded-lg bg-gray-800 border border-gray-700"
                        >
                          <ToolLogo
                            name={toolName}
                            logo={selectedAgent.metadata.tool_icons?.[idx]}
                            size={20}
                          />
                          <span className="text-sm text-gray-300">{toolName}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {/* Assigned Skills */}
                {selectedAgent.metadata.skills && selectedAgent.metadata.skills.length > 0 && (
                  <div>
                    <h4 className="text-sm font-medium text-gray-300 mb-2">Assigned Skills</h4>
                    <div className="flex flex-wrap gap-2">
                      {selectedAgent.metadata.skills.map((skill, idx) => (
                        <Badge key={idx} variant="outline" className="border-gray-700 text-gray-300">
                          {skill}
                        </Badge>
                      ))}
                    </div>
                  </div>
                )}

                {/* Install Button */}
                <div className="flex justify-end gap-3 pt-4 border-t border-gray-800">
                  <Button
                    variant="outline"
                    onClick={() => setIsDetailModalOpen(false)}
                    className="border-gray-700 text-gray-300 hover:bg-gray-800"
                  >
                    Cancel
                  </Button>
                  <Button
                    onClick={handleInstall}
                    disabled={installMutation.isPending}
                    className="bg-orange-500 hover:bg-orange-600 text-white"
                  >
                    {installMutation.isPending ? (
                      <>
                        <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                        Installing...
                      </>
                    ) : (
                      <>
                        <Download className="w-4 h-4 mr-2" />
                        Install Agent
                      </>
                    )}
                  </Button>
                </div>
              </div>
            </>
          )}
        </DialogContent>
      </Dialog>
    </div>
  )
}
