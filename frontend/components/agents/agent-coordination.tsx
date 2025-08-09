
'use client'

import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import { 
  Network, 
  Users, 
  MessageSquare, 
  GitBranch, 
  Settings,
  Plus,
  Edit,
  Trash2,
  Play,
  Pause,
  Loader2,
  Activity
} from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Switch } from '@/components/ui/switch'
import { Label } from '@/components/ui/label'
import { useAgentCoordination, useBehaviorMonitoring, useMultiAgentStatistics } from '@/hooks/use-multi-agent'
import { toast } from 'sonner'

const coordinationPatterns = [
  {
    id: 'pipeline-1',
    name: 'Code Review Pipeline',
    description: 'Sequential workflow for code analysis, security check, and quality assurance',
    type: 'sequential',
    status: 'active',
    agents: ['CodeArchitect', 'SecurityGuard', 'TestMaster'],
    steps: [
      { agent: 'CodeArchitect', task: 'Analyze code structure and patterns', order: 1 },
      { agent: 'SecurityGuard', task: 'Perform security vulnerability scan', order: 2 },
      { agent: 'TestMaster', task: 'Generate and execute test cases', order: 3 }
    ],
    performance: 94.5,
    completedTasks: 127
  },
  {
    id: 'parallel-1',
    name: 'Bug Investigation Team',
    description: 'Parallel analysis of bug reports from multiple perspectives',
    type: 'parallel',
    status: 'active',
    agents: ['BugHunter', 'PerformanceOptimizer', 'SecurityGuard'],
    performance: 91.2,
    completedTasks: 89
  },
  {
    id: 'hybrid-1',
    name: 'Full Stack Analysis',
    description: 'Hybrid workflow combining parallel and sequential coordination',
    type: 'hybrid',
    status: 'paused',
    agents: ['CodeArchitect', 'BugHunter', 'PerformanceOptimizer', 'DocuMentor'],
    performance: 87.8,
    completedTasks: 45
  }
]

const communicationChannels = [
  {
    id: 'channel-1',
    name: 'Code Review Channel',
    type: 'task-specific',
    agents: ['CodeArchitect', 'SecurityGuard', 'TestMaster'],
    messages: 1247,
    active: true
  },
  {
    id: 'channel-2',
    name: 'Performance Optimization',
    type: 'collaborative',
    agents: ['PerformanceOptimizer', 'BugHunter'],
    messages: 892,
    active: true
  },
  {
    id: 'channel-3',
    name: 'Global Coordination',
    type: 'broadcast',
    agents: ['All Agents'],
    messages: 456,
    active: true
  }
]

const typeStyles: Record<string, string> = {
  sequential: 'bg-blue-500/10 text-blue-400 border-blue-500/20',
  parallel: 'bg-green-500/10 text-green-400 border-green-500/20',
  hybrid: 'bg-purple-500/10 text-purple-400 border-purple-500/20'
}

const statusStyles: Record<string, string> = {
  active: 'bg-green-500/10 text-green-400 border-green-500/20',
  paused: 'bg-yellow-500/10 text-yellow-400 border-yellow-500/20',
  inactive: 'bg-red-500/10 text-red-400 border-red-500/20'
}

export function AgentCoordination() {
  const [selectedPattern, setSelectedPattern] = useState<string | null>(null)
  const [activeAgents, setActiveAgents] = useState([
    { id: 'agent-001', name: 'CodeArchitect', type: 'development', status: 'active' as const },
    { id: 'agent-002', name: 'BugHunter', type: 'debugging', status: 'active' as const },
    { id: 'agent-003', name: 'SecurityGuard', type: 'security', status: 'active' as const },
    { id: 'agent-004', name: 'TestMaster', type: 'testing', status: 'active' as const },
  ])

  // Use our new multi-agent hooks
  const { coordinateAgents, rebalanceAgents, statistics: coordStats, isLoading: coordLoading } = useAgentCoordination()
  const { monitorBehavior, connectRealtimeMonitoring, isConnected, realtimeData } = useBehaviorMonitoring()
  const { statistics: allStats, isLoading: statsLoading } = useMultiAgentStatistics()

  // Connect to real-time monitoring on component mount
  useEffect(() => {
    const ws = connectRealtimeMonitoring()
    return () => {
      if (ws) ws.close()
    }
  }, [connectRealtimeMonitoring])

  // Handle coordination pattern execution
  const handleExecutePattern = async (patternId: string, strategy: string) => {
    try {
      const response = await coordinateAgents(activeAgents, strategy as any)
      if (response.data) {
        toast.success(`Coordination pattern "${patternId}" executed successfully!`)
        // Update pattern with real results
        console.log('Coordination result:', response.data)
      }
    } catch (error) {
      toast.error('Failed to execute coordination pattern')
      console.error(error)
    }
  }

  // Handle agent rebalancing
  const handleRebalanceAgents = async () => {
    try {
      const response = await rebalanceAgents(activeAgents)
      if (response.data) {
        toast.success('Agent workloads rebalanced successfully!')
        console.log('Rebalancing result:', response.data)
      }
    } catch (error) {
      toast.error('Failed to rebalance agents')
      console.error(error)
    }
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <div className="flex items-center space-x-3 mb-2">
            <h2 className="text-2xl font-bold">Agent Coordination</h2>
            <div className="flex items-center space-x-2">
              <div className={`w-2 h-2 rounded-full ${isConnected ? 'bg-green-400' : 'bg-red-400'}`} />
              <span className="text-xs text-muted-foreground">
                {isConnected ? 'Live Monitoring' : 'Disconnected'}
              </span>
              {(coordLoading || statsLoading) && (
                <Loader2 className="w-4 h-4 animate-spin text-muted-foreground" />
              )}
            </div>
          </div>
          <p className="text-muted-foreground">
            Manage agent collaboration patterns and communication channels
          </p>
          {allStats?.coordination && (
            <div className="flex items-center space-x-4 mt-2 text-sm text-muted-foreground">
              <span>Active Patterns: {coordStats?.active_patterns || 3}</span>
              <span>Efficiency: {allStats.coordination.efficiency || '94.2%'}</span>
              <span>Balance Score: {allStats.coordination.balance_score || '0.86'}</span>
            </div>
          )}
        </div>
        <div className="flex items-center space-x-3">
          <Button 
            variant="outline" 
            onClick={handleRebalanceAgents}
            disabled={coordLoading}
            className="hover:border-primary/50"
          >
            {coordLoading ? (
              <Loader2 className="w-4 h-4 mr-2 animate-spin" />
            ) : (
              <Activity className="w-4 h-4 mr-2" />
            )}
            Rebalance
          </Button>
          <Button className="gradient-accent hover:opacity-90">
            <Plus className="w-4 h-4 mr-2" />
            Create Pattern
          </Button>
        </div>
      </div>

      {/* Coordination Patterns */}
      <div className="space-y-4">
        <h3 className="text-lg font-semibold flex items-center">
          <Network className="w-5 h-5 mr-2" />
          Coordination Patterns
        </h3>
        
        <div className="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-3 gap-6">
          {coordinationPatterns.map((pattern, index) => (
            <motion.div
              key={pattern.id}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, delay: index * 0.1 }}
            >
              <Card className="glass-card card-glow hover:border-primary/20 transition-all duration-300 cursor-pointer">
                <CardHeader>
                  <div className="flex items-center justify-between">
                    <CardTitle className="text-lg">{pattern.name}</CardTitle>
                    <div className="flex items-center space-x-2">
                      <Badge className={typeStyles[pattern.type]}>
                        {pattern.type}
                      </Badge>
                      <Badge className={statusStyles[pattern.status]}>
                        {pattern.status}
                      </Badge>
                    </div>
                  </div>
                  <p className="text-sm text-muted-foreground">{pattern.description}</p>
                </CardHeader>
                
                <CardContent className="space-y-4">
                  {/* Performance Metrics */}
                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <p className="text-sm text-muted-foreground">Performance</p>
                      <p className="text-xl font-bold">{pattern.performance}%</p>
                    </div>
                    <div>
                      <p className="text-sm text-muted-foreground">Completed</p>
                      <p className="text-xl font-bold">{pattern.completedTasks}</p>
                    </div>
                  </div>

                  {/* Progress Bar */}
                  <div className="space-y-2">
                    <div className="flex justify-between text-xs text-muted-foreground">
                      <span>Efficiency</span>
                      <span>{pattern.performance}%</span>
                    </div>
                    <div className="w-full bg-secondary rounded-full h-2">
                      <div 
                        className="bg-gradient-to-r from-orange-500 to-red-500 h-2 rounded-full transition-all duration-300"
                        style={{ width: `${pattern.performance}%` }}
                      />
                    </div>
                  </div>

                  {/* Agents */}
                  <div>
                    <p className="text-sm text-muted-foreground mb-2">Agents ({pattern.agents.length})</p>
                    <div className="flex flex-wrap gap-1">
                      {pattern.agents.map(agent => (
                        <Badge key={agent} variant="outline" className="text-xs">
                          {agent}
                        </Badge>
                      ))}
                    </div>
                  </div>

                  {/* Actions */}
                  <div className="flex justify-between items-center pt-2 border-t border-border/30">
                    <div className="flex space-x-2">
                      <Button variant="ghost" size="icon" className="h-8 w-8">
                        <Edit className="w-4 h-4" />
                      </Button>
                      <Button variant="ghost" size="icon" className="h-8 w-8 text-red-400 hover:text-red-300">
                        <Trash2 className="w-4 h-4" />
                      </Button>
                    </div>
                    
                    <Button 
                      size="sm" 
                      variant={pattern.status === 'active' ? 'secondary' : 'default'}
                      className={pattern.status === 'active' ? '' : 'gradient-accent hover:opacity-90'}
                      onClick={() => handleExecutePattern(pattern.id, pattern.type)}
                      disabled={coordLoading}
                    >
                      {coordLoading ? (
                        <Loader2 className="w-3 h-3 mr-1 animate-spin" />
                      ) : pattern.status === 'active' ? (
                        <>
                          <Pause className="w-3 h-3 mr-1" />
                          Pause
                        </>
                      ) : (
                        <>
                          <Play className="w-3 h-3 mr-1" />
                          Execute
                        </>
                      )}
                    </Button>
                  </div>
                </CardContent>
              </Card>
            </motion.div>
          ))}
        </div>
      </div>

      {/* Communication Channels */}
      <div className="space-y-4">
        <h3 className="text-lg font-semibold flex items-center">
          <MessageSquare className="w-5 h-5 mr-2" />
          Communication Channels
        </h3>
        
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {communicationChannels.map((channel, index) => (
            <motion.div
              key={channel.id}
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.5, delay: index * 0.1 }}
            >
              <Card className="glass-card hover:border-primary/20 transition-all duration-300">
                <CardHeader>
                  <div className="flex items-center justify-between">
                    <CardTitle className="text-lg">{channel.name}</CardTitle>
                    <Switch 
                      checked={channel.active}
                      onCheckedChange={() => {}}
                    />
                  </div>
                  <Badge variant="outline" className="w-fit">
                    {channel.type}
                  </Badge>
                </CardHeader>
                
                <CardContent className="space-y-4">
                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <p className="text-sm text-muted-foreground">Messages</p>
                      <p className="text-xl font-bold">{channel.messages.toLocaleString()}</p>
                    </div>
                    <div>
                      <p className="text-sm text-muted-foreground">Agents</p>
                      <p className="text-xl font-bold">{Array.isArray(channel.agents) ? channel.agents.length : 'All'}</p>
                    </div>
                  </div>

                  <div>
                    <p className="text-sm text-muted-foreground mb-2">Connected Agents</p>
                    <div className="flex flex-wrap gap-1">
                      {Array.isArray(channel.agents) ? (
                        channel.agents.map(agent => (
                          <Badge key={agent} variant="secondary" className="text-xs">
                            {agent}
                          </Badge>
                        ))
                      ) : (
                        <Badge variant="secondary" className="text-xs">
                          {channel.agents}
                        </Badge>
                      )}
                    </div>
                  </div>
                </CardContent>
              </Card>
            </motion.div>
          ))}
        </div>
      </div>

      {/* Coordination Settings */}
      <Card className="glass-card">
        <CardHeader>
          <CardTitle className="flex items-center">
            <Settings className="w-5 h-5 mr-2" />
            Global Coordination Settings
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-6">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <div>
                  <Label htmlFor="auto-coordination">Auto Coordination</Label>
                  <p className="text-sm text-muted-foreground">
                    Automatically create coordination patterns based on task requirements
                  </p>
                </div>
                <Switch id="auto-coordination" defaultChecked />
              </div>
              
              <div className="flex items-center justify-between">
                <div>
                  <Label htmlFor="smart-routing">Smart Task Routing</Label>
                  <p className="text-sm text-muted-foreground">
                    Intelligently route tasks to most suitable agents
                  </p>
                </div>
                <Switch id="smart-routing" defaultChecked />
              </div>
            </div>
            
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <div>
                  <Label htmlFor="load-balancing">Load Balancing</Label>
                  <p className="text-sm text-muted-foreground">
                    Distribute tasks evenly across available agents
                  </p>
                </div>
                <Switch id="load-balancing" defaultChecked />
              </div>
              
              <div className="flex items-center justify-between">
                <div>
                  <Label htmlFor="failure-recovery">Failure Recovery</Label>
                  <p className="text-sm text-muted-foreground">
                    Automatically reassign tasks when agents fail
                  </p>
                </div>
                <Switch id="failure-recovery" defaultChecked />
              </div>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}
