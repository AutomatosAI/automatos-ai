
'use client'

import { useState } from 'react'
import { motion } from 'framer-motion'
import { useInView } from 'react-intersection-observer'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { AgentRoster } from './agent-roster'
import { AgentConfiguration } from './agent-configuration'
import { AgentPerformance } from './agent-performance'
import { AgentSkills } from './agent-skills'
import { AgentCoordination } from './agent-coordination'
import { MultiAgentDashboard } from './multi-agent-dashboard'
import { CreateAgentModal } from './create-agent-modal'
import { Button } from '@/components/ui/button'
import { Plus, Bot, Settings, BarChart, Users, Zap, Brain } from 'lucide-react'

export function AgentManagement() {
  const [showCreateModal, setShowCreateModal] = useState(false)
  const [ref, inView] = useInView({
    triggerOnce: true,
    threshold: 0.1,
  })

  const stats = [
    {
      label: 'Total Agents',
      value: '24',
      change: '+3 this week',
      icon: Bot,
      color: 'text-orange-400'
    },
    {
      label: 'Active Agents',
      value: '18',
      change: '75% utilization',
      icon: Zap,
      color: 'text-green-400'
    },
    {
      label: 'Agent Types',
      value: '8',
      change: '4 custom types',
      icon: Settings,
      color: 'text-blue-400'
    },
    {
      label: 'Avg Performance',
      value: '94.2%',
      change: '+2.1% this month',
      icon: BarChart,
      color: 'text-purple-400'
    }
  ]

  return (
    <div className="space-y-8">
      {/* Header */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.8 }}
        className="flex items-center justify-between"
      >
        <div>
          <h1 className="text-3xl font-bold mb-2">
            Agent <span className="gradient-text">Management</span>
          </h1>
          <p className="text-muted-foreground text-lg">
            Manage your AI agents, skills, and coordination settings
          </p>
        </div>
        
        <Button
          onClick={() => setShowCreateModal(true)}
          className="gradient-accent hover:opacity-90 transition-opacity"
        >
          <Plus className="w-4 h-4 mr-2" />
          Create Agent
        </Button>
      </motion.div>

      {/* Stats Overview */}
      <motion.div
        ref={ref}
        className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6"
        initial={{ opacity: 0, y: 20 }}
        animate={inView ? { opacity: 1, y: 0 } : {}}
        transition={{ duration: 0.8, delay: 0.2 }}
      >
        {stats.map((stat, index) => (
          <motion.div
            key={stat.label}
            className="glass-card p-6 card-glow hover:border-primary/20 transition-all duration-300"
            initial={{ opacity: 0, y: 20 }}
            animate={inView ? { opacity: 1, y: 0 } : {}}
            transition={{ duration: 0.8, delay: index * 0.1 }}
          >
            <div className="flex items-center justify-between mb-4">
              <div className="w-10 h-10 rounded-lg bg-secondary/50 flex items-center justify-center">
                <stat.icon className={`w-5 h-5 ${stat.color}`} />
              </div>
            </div>
            <div className="space-y-1">
              <h3 className="text-2xl font-bold">{stat.value}</h3>
              <p className="text-muted-foreground text-sm">{stat.label}</p>
              <p className="text-xs text-green-400">{stat.change}</p>
            </div>
          </motion.div>
        ))}
      </motion.div>

      {/* Main Content Tabs */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={inView ? { opacity: 1, y: 0 } : {}}
        transition={{ duration: 0.8, delay: 0.4 }}
      >
        <Tabs defaultValue="dashboard" className="space-y-6">
          <TabsList className="grid w-full grid-cols-6 lg:w-auto lg:inline-grid bg-secondary/50">
            <TabsTrigger value="dashboard" className="flex items-center space-x-2">
              <Brain className="w-4 h-4" />
              <span className="hidden sm:inline">Multi-Agent</span>
            </TabsTrigger>
            <TabsTrigger value="roster" className="flex items-center space-x-2">
              <Users className="w-4 h-4" />
              <span className="hidden sm:inline">Roster</span>
            </TabsTrigger>
            <TabsTrigger value="skills" className="flex items-center space-x-2">
              <Zap className="w-4 h-4" />
              <span className="hidden sm:inline">Skills</span>
            </TabsTrigger>
            <TabsTrigger value="configuration" className="flex items-center space-x-2">
              <Settings className="w-4 h-4" />
              <span className="hidden sm:inline">Config</span>
            </TabsTrigger>
            <TabsTrigger value="coordination" className="flex items-center space-x-2">
              <Bot className="w-4 h-4" />
              <span className="hidden sm:inline">Coordination</span>
            </TabsTrigger>
            <TabsTrigger value="performance" className="flex items-center space-x-2">
              <BarChart className="w-4 h-4" />
              <span className="hidden sm:inline">Analytics</span>
            </TabsTrigger>
          </TabsList>

          <TabsContent value="dashboard" className="space-y-6">
            <MultiAgentDashboard />
          </TabsContent>

          <TabsContent value="roster" className="space-y-6">
            <AgentRoster />
          </TabsContent>

          <TabsContent value="skills" className="space-y-6">
            <AgentSkills />
          </TabsContent>

          <TabsContent value="configuration" className="space-y-6">
            <AgentConfiguration />
          </TabsContent>

          <TabsContent value="coordination" className="space-y-6">
            <AgentCoordination />
          </TabsContent>

          <TabsContent value="performance" className="space-y-6">
            <AgentPerformance />
          </TabsContent>
        </Tabs>
      </motion.div>

      {/* Create Agent Modal */}
      <CreateAgentModal 
        open={showCreateModal} 
        onClose={() => setShowCreateModal(false)} 
      />
    </div>
  )
}
