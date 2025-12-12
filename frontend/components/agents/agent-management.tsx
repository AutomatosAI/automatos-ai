'use client'

import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import { useInView } from 'react-intersection-observer'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Badge } from '@/components/ui/badge'
import { Card, CardContent } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import {
  Plus,
  Bot,
  Settings,
  BarChart,
  Users,
  Zap,
  Brain,
  Search,
  Filter,
  RefreshCw
} from 'lucide-react'

// Import all tab components
import { AgentRoster } from './agent-roster'
import { AgentSkills } from './agent-skills'
import { AgentConfiguration } from './agent-configuration'
import { AgentCoordination } from './agent-coordination'
import { AgentPerformance } from './agent-performance'
import { CreateAgentModal } from './create-agent-modal'
import { AgentDetailsModal } from './agent-details-modal'

// API hooks for real data
import { useAgents, useAgentStats, useAgentTypes } from '@/hooks/use-agent-api'
import { apiClient } from '@/lib/api-client'

export function AgentManagement() {
  const [activeTab, setActiveTab] = useState('roster')
  const [showCreateModal, setShowCreateModal] = useState(false)
  const [selectedAgentId, setSelectedAgentId] = useState<string | null>(null)
  const [viewDetailsAgentId, setViewDetailsAgentId] = useState<string | null>(null)
  const [mounted, setMounted] = useState(false)

  // Fetch real data from APIs
  const { data: agents = [], isLoading: agentsLoading, refetch: refetchAgents, error: agentsError } = useAgents()
  const { data: agentStats, isLoading: statsLoading } = useAgentStats()
  const { data: agentTypes = [] } = useAgentTypes()

  // Debug logging
  console.log('Agents API Response:', { agents, agentsLoading, agentsError })
  console.log('Agent count:', (agents as any[])?.length)

  // Debug log active tab changes
  useEffect(() => {
    console.log('Active tab changed:', activeTab)
    if (activeTab === 'configuration' && !selectedAgentId && (agents as any[])?.length > 0) {
      // Auto-select first agent when entering configuration tab with no agent selected
      setSelectedAgentId((agents as any[])[0].id.toString())
    }
  }, [activeTab, agents, selectedAgentId])

  useEffect(() => {
    setMounted(true)
    // Make sure viewDetailsAgentId starts as null
    setViewDetailsAgentId(null)
    // Set page context for API client to use real APIs
    apiClient.setCurrentPage('agents')
  }, [])
  const [searchTerm, setSearchTerm] = useState('')
  const [statusFilter, setStatusFilter] = useState('all')
  const [ref, inView] = useInView({
    triggerOnce: true,
    threshold: 0.1,
  })

  // Calculate real statistics from actual data
  const stats = [
    {
      label: 'Total Agents',
      value: (agentStats as any)?.total_agents || (agents as any[])?.length || '0',
      change: (agentStats as any)?.total_agents ? `${(agentStats as any).total_agents} agents` : '0 agents',
      icon: Bot,
      color: 'text-orange-400'
    },
    {
      label: 'Active Agents',
      value: (agentStats as any)?.active_agents || (agents as any[])?.filter((a: any) => a.status === 'active')?.length || '0',
      change: (agentStats as any)?.active_agents && (agentStats as any)?.total_agents ? `${Math.round(((agentStats as any).active_agents / (agentStats as any).total_agents) * 100)}% online` : '0% online',
      icon: Zap,
      color: 'text-green-400'
    },
    {
      label: 'Agent Types',
      value: (agentStats as any)?.agents_by_type ? Object.keys((agentStats as any).agents_by_type).length : (agentTypes as any[])?.length || '0',
      change: (agentStats as any)?.agents_by_type ? `${Object.keys((agentStats as any).agents_by_type).length} types` : '0 types',
      icon: Settings,
      color: 'text-blue-400'
    },
    {
      label: 'Avg Performance',
      value: `${(agentStats as any)?.average_performance?.toFixed(1) || '0.0'}%`,
      change: (agentStats as any)?.average_performance ? ((agentStats as any).average_performance > 90 ? '↑ Excellent performance' : '↓ Needs optimization') : 'No data',
      icon: BarChart,
      color: 'text-purple-400'
    }
  ]

  // Handle refresh
  const handleRefresh = async () => {
    await refetchAgents()
  }

  // Handle view details
  const handleViewDetails = (agentId: string | null) => {
    if (agentId) {
      setViewDetailsAgentId(agentId)
    }
  }

  return (
    <div ref={ref} className="space-y-6">
      {/* Header */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={inView ? { opacity: 1, y: 0 } : {}}
        transition={{ duration: 0.5 }}
        className="flex justify-between items-start"
      >
        <div>
          <h1 className="text-3xl font-bold mb-2">
            Agent <span className="gradient-text">Management</span>
          </h1>
          <p className="text-muted-foreground mt-1">
            Manage your AI agents, skills, and coordination strategies
          </p>
        </div>

        <div className="flex items-center gap-3">
          <Badge variant="outline" className="text-brand-primary border-brand-primary/30">
            <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse mr-2" />
            {agentsLoading ? 'Loading...' : `${(agents as any[])?.length || 0} Agents`}
          </Badge>

          <Button
            onClick={handleRefresh}
            variant="outline"
            size="sm"
            disabled={agentsLoading}
          >
            <RefreshCw className={`w-4 h-4 mr-2 ${agentsLoading ? 'animate-spin' : ''}`} />
            Refresh
          </Button>

          <Button
            onClick={() => setShowCreateModal(true)}
            className="bg-brand-primary hover:bg-brand-primary/90"
          >
            <Plus className="w-4 h-4 mr-2" />
            Create Agent
          </Button>
        </div>
      </motion.div>

      {/* Statistics Cards */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={inView ? { opacity: 1, y: 0 } : {}}
        transition={{ duration: 0.6, delay: 0.1 }}
        className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6"
      >
        {stats.map((stat, index) => (
          <Card key={stat.label} className="glass-card card-glow hover:border-primary/20 transition-all duration-300">
            <CardContent className="p-4">
              <div className="flex items-center justify-between gap-3">
                <div className="flex items-center gap-3 min-w-0">
                  <div className="w-10 h-10 rounded-2xl bg-black/20 border border-orange-500/10 flex items-center justify-center shrink-0">
                    <stat.icon
                      className={`w-5 h-5 ${
                        index === 0 ? 'text-orange-400' :
                        index === 1 ? 'text-green-400' :
                        index === 2 ? 'text-blue-400' :
                        index === 3 ? 'text-purple-400' :
                        'text-white'
                      }`}
                    />
                  </div>
                  <div className="min-w-0">
                    <div className="text-2xl font-bold leading-none">
                      {statsLoading ? '…' : stat.value}
                    </div>
                    <div className="text-sm text-muted-foreground truncate">{stat.label}</div>
                  </div>
                </div>
                <div className={`shrink-0 text-xs ${stat.color}`}>
                  {stat.change}
                </div>
              </div>
            </CardContent>
          </Card>
        ))}
      </motion.div>

      {/* Search and Filters */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={inView ? { opacity: 1, y: 0 } : {}}
        transition={{ duration: 0.6, delay: 0.2 }}
        className="flex flex-col sm:flex-row gap-4"
      >
        <div className="relative flex-1">
          <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-muted-foreground w-4 h-4" />
          <Input
            placeholder="Search agents by name, type, or skills..."
            value={searchTerm}
            onChange={(e: any) => setSearchTerm(e.target.value)}
            className="pl-10"
          />
        </div>

        <div className="flex gap-2">
          <Button
            variant={statusFilter === 'all' ? 'default' : 'outline'}
            size="sm"
            onClick={() => setStatusFilter('all')}
          >
            All
          </Button>
          <Button
            variant={statusFilter === 'active' ? 'default' : 'outline'}
            size="sm"
            onClick={() => setStatusFilter('active')}
          >
            Active
          </Button>
          <Button
            variant={statusFilter === 'idle' ? 'default' : 'outline'}
            size="sm"
            onClick={() => setStatusFilter('idle')}
          >
            Idle
          </Button>
        </div>
      </motion.div>

      {/* Main Tabs */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={inView ? { opacity: 1, y: 0 } : {}}
        transition={{ duration: 0.6, delay: 0.3 }}
      >
        <Tabs value={activeTab} onValueChange={setActiveTab} className="space-y-6">
          <TabsList className="grid w-full grid-cols-5 lg:w-auto lg:grid-cols-5">
            <TabsTrigger value="roster" className="flex items-center gap-2">
              <Users className="w-4 h-4" />
              <span className="hidden sm:inline">Agent Roster</span>
            </TabsTrigger>
            <TabsTrigger value="skills" className="flex items-center gap-2">
              <Brain className="w-4 h-4" />
              <span className="hidden sm:inline">Skills</span>
            </TabsTrigger>
            <TabsTrigger value="configuration" className="flex items-center gap-2">
              <Settings className="w-4 h-4" />
              <span className="hidden sm:inline">Configuration</span>
            </TabsTrigger>
            <TabsTrigger value="coordination" className="flex items-center gap-2">
              <Users className="w-4 h-4" />
              <span className="hidden sm:inline">Coordination</span>
            </TabsTrigger>
            <TabsTrigger value="performance" className="flex items-center gap-2">
              <BarChart className="w-4 h-4" />
              <span className="hidden sm:inline">Performance</span>
            </TabsTrigger>
          </TabsList>

          {/* Agent Roster Tab */}
          <TabsContent value="roster" className="space-y-6">
            <AgentRoster
              agents={agents as any[]}
              loading={agentsLoading}
              searchTerm={searchTerm}
              statusFilter={statusFilter}
              onAgentSelect={setSelectedAgentId}
              onViewDetails={handleViewDetails}
              selectedAgentId={selectedAgentId}
              onRefresh={() => refetchAgents()}
              setSearchTerm={setSearchTerm}
            />
          </TabsContent>

          {/* Skills Tab */}
          <TabsContent value="skills" className="space-y-6">
            <AgentSkills
              agents={agents as any[]}
              selectedAgentId={selectedAgentId}
              onAgentSelect={setSelectedAgentId}
            />
          </TabsContent>

          {/* Configuration Tab */}
          <TabsContent value="configuration" className="space-y-6">
            <AgentConfiguration
              agents={agents as any[]}
              selectedAgentId={selectedAgentId}
              onAgentSelect={setSelectedAgentId}
            />
          </TabsContent>

          {/* Coordination Tab */}
          <TabsContent value="coordination" className="space-y-6">
            <AgentCoordination
              agents={agents as any[]}
              selectedAgentId={selectedAgentId}
            />
          </TabsContent>

          {/* Performance Tab */}
          <TabsContent value="performance" className="space-y-6">
            <AgentPerformance
              agents={agents as any[]}
              agentStats={agentStats}
              selectedAgentId={selectedAgentId}
              onAgentSelect={setSelectedAgentId}
            />
          </TabsContent>
        </Tabs>
      </motion.div>


      {/* Agent Details Modal - ONLY opens when View Details is clicked */}
      {mounted && viewDetailsAgentId && (
        <AgentDetailsModal
          agentId={Number(viewDetailsAgentId)}
          open={!!viewDetailsAgentId}
          onClose={() => setViewDetailsAgentId(null)}
        />
      )}

      {/* Create Agent Modal */}
      <CreateAgentModal
        open={showCreateModal}
        onClose={() => setShowCreateModal(false)}
        onSuccess={() => {
          setShowCreateModal(false)
          handleRefresh()
        }}
      />
    </div>
  )
}
