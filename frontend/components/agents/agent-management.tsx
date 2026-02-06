'use client'

import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import { useInView } from 'react-intersection-observer'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import {
  Plus,
  Bot,
  Settings,
  BarChart,
  Users,
  Zap,
  Brain,
  RefreshCw
} from 'lucide-react'

// Shared components
import { PageHeader } from '@/components/shared/page-header'
import { StatsBar, type StatItem } from '@/components/shared/stats-bar'
import { SearchInput } from '@/components/shared/search-input'
import { FilterTabs, TabsContent } from '@/components/shared/filter-tabs'

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
      setSelectedAgentId((agents as any[])[0].id.toString())
    }
  }, [activeTab, agents, selectedAgentId])

  useEffect(() => {
    setMounted(true)
    setViewDetailsAgentId(null)
    apiClient.setCurrentPage('agents')
  }, [])
  const [searchTerm, setSearchTerm] = useState('')
  const [statusFilter, setStatusFilter] = useState('all')
  const [ref, inView] = useInView({
    triggerOnce: true,
    threshold: 0.1,
  })

  // Calculate real statistics from actual data
  const stats: StatItem[] = [
    {
      label: 'Total Agents',
      value: String((agentStats as any)?.total_agents || (agents as any[])?.length || '0'),
      change: (agentStats as any)?.total_agents ? `${(agentStats as any).total_agents} agents` : '0 agents',
      icon: Bot,
      iconColor: 'text-primary',
    },
    {
      label: 'Active Agents',
      value: String((agentStats as any)?.active_agents || (agents as any[])?.filter((a: any) => a.status === 'active')?.length || '0'),
      change: (agentStats as any)?.active_agents && (agentStats as any)?.total_agents ? `${Math.round(((agentStats as any).active_agents / (agentStats as any).total_agents) * 100)}% online` : '0% online',
      icon: Zap,
      iconColor: 'text-[hsl(var(--success))]',
    },
    {
      label: 'Categories',
      value: '10',
      change: '10 categories',
      icon: Settings,
      iconColor: 'text-[hsl(var(--info))]',
    },
    {
      label: 'Avg Performance',
      value: `${(agentStats as any)?.average_performance?.toFixed(1) || '0.0'}%`,
      change: (agentStats as any)?.average_performance ? ((agentStats as any).average_performance > 90 ? '↑ Excellent performance' : '↓ Needs optimization') : 'No data',
      icon: BarChart,
      iconColor: 'text-[hsl(var(--agent))]',
    }
  ]

  const handleRefresh = async () => {
    await refetchAgents()
  }

  const handleViewDetails = (agentId: string | null) => {
    if (agentId) {
      setViewDetailsAgentId(agentId)
    }
  }

  const tabDefs = [
    { value: 'roster', label: 'Agent Roster', icon: Users },
    { value: 'skills', label: 'Skills', icon: Brain },
    { value: 'configuration', label: 'Configuration', icon: Settings },
    { value: 'coordination', label: 'Coordination', icon: Users },
    { value: 'performance', label: 'Performance', icon: BarChart },
  ]

  return (
    <div ref={ref} className="space-y-6">
      {/* Header */}
      <PageHeader
        title="Agent"
        titleAccent="Management"
        subtitle="Manage your AI agents, skills, and coordination strategies"
        actions={
          <>
            {!!agentsError && (
              <Badge variant="outline" className="border-[hsl(var(--destructive))]/30 bg-[hsl(var(--destructive))]/10 text-[hsl(var(--destructive))]">
                Agents API error (HTTP {(agentsError as any)?.status || '500'})
              </Badge>
            )}
            <Button
              onClick={handleRefresh}
              variant="outline"
              size="sm"
              disabled={agentsLoading && !agentsError}
            >
              <RefreshCw className={`w-4 h-4 mr-2 ${agentsLoading ? 'animate-spin' : ''}`} />
              Refresh
            </Button>

            <Button
              variant="outline"
              onClick={() => setShowCreateModal(true)}
            >
              <Plus className="w-4 h-4 mr-2" />
              Create Agent
            </Button>
          </>
        }
      />

      {!!agentsError && (
        <div className="rounded-2xl border border-[hsl(var(--destructive))]/20 bg-[hsl(var(--destructive))]/5 p-4">
          <div className="text-sm font-semibold text-[hsl(var(--destructive))]">
            Agents failed to load (backend error)
          </div>
          <div className="mt-1 text-sm text-muted-foreground">
            The backend returned an error for the Agents endpoints. Check the backend logs for <code className="rounded bg-secondary/40 px-1.5 py-0.5 text-xs">/api/agents</code> and <code className="rounded bg-secondary/40 px-1.5 py-0.5 text-xs">/api/v1/skills</code>.
          </div>
          <div className="mt-3 flex gap-2">
            <Button variant="outline" size="sm" onClick={handleRefresh}>
              Retry
            </Button>
          </div>
        </div>
      )}

      {/* Statistics Cards */}
      <StatsBar stats={stats} loading={statsLoading} />

      {/* Search and Filters */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={inView ? { opacity: 1, y: 0 } : {}}
        transition={{ duration: 0.6, delay: 0.2 }}
        className="flex flex-col sm:flex-row gap-4"
      >
        <SearchInput
          value={searchTerm}
          onChange={setSearchTerm}
          placeholder="Search agents by name, type, or skills..."
          className="flex-1"
        />

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
        <FilterTabs
          tabs={tabDefs}
          value={activeTab}
          onValueChange={setActiveTab}
        >
          <TabsContent value="roster" className="space-y-6">
            <AgentRoster
              agents={agents as any[]}
              loading={agentsLoading && !agentsError}
              searchTerm={searchTerm}
              statusFilter={statusFilter}
              onAgentSelect={setSelectedAgentId}
              onViewDetails={handleViewDetails}
              selectedAgentId={selectedAgentId}
              onRefresh={() => refetchAgents()}
              setSearchTerm={setSearchTerm}
            />
          </TabsContent>

          <TabsContent value="skills" className="space-y-6">
            <AgentSkills
              agents={agents as any[]}
              selectedAgentId={selectedAgentId}
              onAgentSelect={setSelectedAgentId}
            />
          </TabsContent>

          <TabsContent value="configuration" className="space-y-6">
            <AgentConfiguration
              agents={agents as any[]}
              selectedAgentId={selectedAgentId}
              onAgentSelect={setSelectedAgentId}
            />
          </TabsContent>

          <TabsContent value="coordination" className="space-y-6">
            <AgentCoordination
              agents={agents as any[]}
              selectedAgentId={selectedAgentId}
            />
          </TabsContent>

          <TabsContent value="performance" className="space-y-6">
            <AgentPerformance
              agents={agents as any[]}
              agentStats={agentStats}
              selectedAgentId={selectedAgentId}
              onAgentSelect={setSelectedAgentId}
            />
          </TabsContent>
        </FilterTabs>
      </motion.div>

      {/* Agent Details Modal */}
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
