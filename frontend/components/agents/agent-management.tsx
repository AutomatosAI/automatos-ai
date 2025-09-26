'use client'

import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import { useInView } from 'react-intersection-observer'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { AgentRoster } from './agent-roster'
import { AgentsTable2 } from '@/components/agents2/AgentsTable2'
import { AgentDetailsPanel } from '@/components/agents/agent-details-panel'
import { AgentRunsPanel } from '@/components/agents/agent-runs-panel'
import { AgentConfiguration } from './agent-configuration'
import { AgentPerformance } from './agent-performance'
import { AgentSkills } from './agent-skills'
import { CreateAgentModal } from './create-agent-modal'
import { Badge } from '@/components/ui/badge'
import { Card, CardContent } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Plus, Bot, Settings, BarChart, Users, Zap, Brain } from 'lucide-react'
import { useAgents, useAgentStats } from '@/hooks/use-agent-api'

export function AgentManagement() {
  // Fetch agents and stats data
  const { data: agents = [], isLoading: agentsLoading } = useAgents()
  const { data: agentStats, isLoading: statsLoading } = useAgentStats()
  const [showCreateModal, setShowCreateModal] = useState(false)
  const [selectedAgentId, setSelectedAgentId] = useState<string | null>(null)
  const [ref, inView] = useInView({
    triggerOnce: true,
    threshold: 0.1,
  })

  // Fetch agents and stats data
  const { data: agents = [], isLoading: agentsLoading } = useAgents()
  const { data: agentStats, isLoading: statsLoading } = useAgentStats()

  const [stats, setStats] = useState([
    {
      label: 'Total Agents',
      value: 'Loading...',
      change: 'Loading...',
      icon: Bot,
      color: 'text-orange-400'
    },
    {
      label: 'Active Agents',
      value: 'Loading...',
      change: 'Loading...',
      icon: Zap,
      color: 'text-green-400'
    },
    {
      label: 'Agent Types',
      value: 'Loading...',
      change: 'Loading...',
      icon: Settings,
      color: 'text-blue-400'
    },
    {
      label: 'Avg Performance',
      value: 'Loading...',
      change: 'Loading...',
      icon: BarChart,
      color: 'text-purple-400'
    }
  ])

  // Fetch real agent data
  useEffect(() => {
    const fetchAgentStats = async () => {
      try {
        console.log('Fetching agent stats...')
        const response = await fetch('/api/agents/')
        const agents = await response.json()
        console.log('Agent data received:', agents.length, 'agents')
        
        if (Array.isArray(agents)) {
          const totalAgents = agents.length
          const activeAgents = agents.filter(agent => agent.status === 'active').length
          const agentTypes = new Set(agents.map(agent => agent.agent_type)).size
          const utilization = totalAgents > 0 ? Math.round((activeAgents / totalAgents) * 100) : 0
          
          setStats([
            {
              label: 'Total Agents',
              value: totalAgents.toString(),
              change: '+3 this week',
              icon: Bot,
              color: 'text-orange-400'
            },
            {
              label: 'Active Agents',
              value: activeAgents.toString(),
              change: `${utilization}% utilization`,
              icon: Zap,
              color: 'text-green-400'
            },
            {
              label: 'Agent Types',
              value: agentTypes.toString(),
              change: 'Multiple types',
              icon: Settings,
              color: 'text-blue-400'
            },
            {
              label: 'Avg Performance',
              value: '95.0%',
              change: '+2.1% this month',
              icon: BarChart,
              color: 'text-purple-400'
            }
          ])
        }
      } catch (error) {
        console.error('Error fetching agent stats:', error)
      }
    }

    fetchAgentStats()
  }, [])

  return (
    <div className="space-y-8">
      {/* Header */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
        className="flex justify-between items-center"
      >
        <div>
          <h1 className="text-4xl font-bold mb-2">Agent Management</h1>
          <p className="text-muted-foreground">
            Manage your AI agents, skills, and coordination settings
          </p>
        </div>
        <Button 
          onClick={() => setShowCreateModal(true)}
          className="bg-primary hover:bg-primary/90"
        >
          <Plus className="w-4 h-4 mr-2" />
          Create Agent
        </Button>
      </motion.div>

      {/* Stats Cards */}
      <motion.div
        ref={ref}
        initial={{ opacity: 0, y: 20 }}
        animate={inView ? { opacity: 1, y: 0 } : {}}
        transition={{ duration: 0.5, delay: 0.2 }}
        className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6"
      >
        {stats.map((stat, index) => (
          <motion.div
            key={stat.label}
            initial={{ opacity: 0, y: 20 }}
            animate={inView ? { opacity: 1, y: 0 } : {}}
            transition={{ duration: 0.5, delay: 0.3 + index * 0.1 }}
          >
            <Card className="glass-card card-glow hover:border-primary/20 transition-all duration-300">
              <CardContent className="p-6">
                <div className="flex items-center space-x-4">
                  <div className={`w-12 h-12 rounded-lg bg-secondary/50 flex items-center justify-center`}>
                    <stat.icon className={`w-6 h-6 ${stat.color}`} />
                  </div>
                  <div className="flex-1">
                    <div className="flex items-center justify-between">
                      <h3 className="text-2xl font-bold">{stat.value}</h3>
                      <Badge variant="secondary" className="text-xs">
                        {stat.change}
                      </Badge>
                    </div>
                    <p className="text-muted-foreground text-sm mt-1">{stat.label}</p>
                  </div>
                </div>
              </CardContent>
            </Card>
          </motion.div>
        ))}
      </motion.div>

      {/* Agent Management Tabs */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={inView ? { opacity: 1, y: 0 } : {}}
        transition={{ duration: 0.5, delay: 0.4 }}
      >
        <Tabs defaultValue="multi-agent" className="space-y-6">
          <TabsList className="grid grid-cols-6 lg:grid-cols-6 glass-card">
            <TabsTrigger value="multi-agent" className="flex items-center space-x-2">
              <Brain className="w-4 h-4" />
              <span>Multi-Agent</span>
            </TabsTrigger>
            <TabsTrigger value="roster" className="flex items-center space-x-2">
              <Users className="w-4 h-4" />
              <span>Roster</span>
            </TabsTrigger>
            <TabsTrigger value="skills" className="flex items-center space-x-2">
              <Zap className="w-4 h-4" />
              <span>Skills</span>
            </TabsTrigger>
            <TabsTrigger value="config" className="flex items-center space-x-2">
              <Settings className="w-4 h-4" />
              <span>Config</span>
            </TabsTrigger>
            <TabsTrigger value="coordination" className="flex items-center space-x-2">
              <Brain className="w-4 h-4" />
              <span>Coordination</span>
            </TabsTrigger>
            <TabsTrigger value="analytics" className="flex items-center space-x-2">
              <BarChart className="w-4 h-4" />
              <span>Analytics</span>
            </TabsTrigger>
          </TabsList>

          <TabsContent value="multi-agent" className="space-y-6">
            <AgentPerformance 
              agents={agents}
              agentStats={agentStats}
              selectedAgentId={selectedAgentId}
              onAgentSelect={setSelectedAgentId}
              showAnalytics={true}
            />
          </TabsContent>

          <TabsContent value="roster" className="space-y-6">
            <AgentRoster />
          </TabsContent>

          <TabsContent value="skills" className="space-y-6">
            <AgentSkills />
          </TabsContent>

          <TabsContent value="config" className="space-y-6">
            <AgentConfiguration />
          </TabsContent>

          <TabsContent value="coordination" className="space-y-6">
            <div className="text-center py-12">
              <Brain className="w-16 h-16 mx-auto mb-4 text-muted-foreground" />
              <h3 className="text-lg font-semibold mb-2">Agent Coordination</h3>
              <p className="text-muted-foreground">
                Multi-agent coordination features coming soon
              </p>
            </div>
          </TabsContent>

          <TabsContent value="analytics" className="space-y-6">
            <div className="text-center py-12">
              <BarChart className="w-16 h-16 mx-auto mb-4 text-muted-foreground" />
              <h3 className="text-lg font-semibold mb-2">Agent Analytics</h3>
              <p className="text-muted-foreground">
                Detailed agent analytics coming soon
              </p>
            </div>
          </TabsContent>
        </Tabs>
      </motion.div>

      {/* Create Agent Modal */}
      <CreateAgentModal 
        isOpen={showCreateModal}
        onClose={() => setShowCreateModal(false)}
      />

      {/* Agent Detail Panel */}
      {selectedAgentId && (
        <AgentDetailsPanel 
          agentId={selectedAgentId}
          onClose={() => setSelectedAgentId(null)}
        />
      )}
    </div>
  )
}
