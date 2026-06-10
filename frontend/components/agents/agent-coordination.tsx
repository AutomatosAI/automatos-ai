
'use client'

import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import { 
  Users, 
  Play, 
  Pause, 
  RefreshCw,
  Settings,
  GitBranch,
  Zap,
  Clock,
  AlertCircle,
  CheckCircle,
  MessageCircle,
  Brain,
  Network,
  Target,
  Activity,
  BarChart3
} from 'lucide-react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { 
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'
import { Textarea } from '@/components/ui/textarea'
import { Label } from '@/components/ui/label'
import { Input } from '@/components/ui/input'
import { Separator } from '@/components/ui/separator'
import { Progress } from '@/components/ui/progress'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Alert, AlertDescription } from '@/components/ui/alert'
import { Skeleton } from '@/components/ui/skeleton'
import { toast } from 'sonner'

// API hooks
import { 
  useCoordinationStatus, 
  useCoordinateAgents, 
  useCollaborativeReasoning 
} from '@/hooks/use-agent-api'

const coordinationStrategies = [
  {
    id: 'sequential',
    name: 'Sequential',
    description: 'Execute tasks one after another',
    icon: GitBranch,
    color: 'text-[hsl(var(--info))]',
    bgColor: 'bg-[hsl(var(--info))]/10'
  },
  {
    id: 'parallel',
    name: 'Parallel',
    description: 'Execute multiple tasks simultaneously',
    icon: Users,
    color: 'text-[hsl(var(--success))]',
    bgColor: 'bg-[hsl(var(--success))]/10'
  },
  {
    id: 'hierarchical',
    name: 'Hierarchical',
    description: 'Organized hierarchy with lead agents',
    icon: Network,
    color: 'text-[hsl(var(--agent))]',
    bgColor: 'bg-[hsl(var(--agent))]/10'
  },
  {
    id: 'mesh',
    name: 'Mesh Network',
    description: 'All agents communicate with each other',
    icon: Activity,
    color: 'text-primary',
    bgColor: 'bg-primary/10'
  },
  {
    id: 'adaptive',
    name: 'Adaptive',
    description: 'Strategy adapts based on task complexity',
    icon: Brain,
    color: 'text-[hsl(var(--destructive))]',
    bgColor: 'bg-[hsl(var(--destructive))]/10'
  }
]

const reasoningStrategies = [
  {
    id: 'majority_vote',
    name: 'Majority Vote',
    description: 'Decision based on majority consensus'
  },
  {
    id: 'weighted_consensus',
    name: 'Weighted Consensus',
    description: 'Weighted by agent expertise and performance'
  },
  {
    id: 'expert_override',
    name: 'Expert Override',
    description: 'Domain expert has final decision authority'
  },
  {
    id: 'iterative_refinement',
    name: 'Iterative Refinement',
    description: 'Multiple rounds of refinement and consensus'
  }
]

interface AgentCoordinationProps {
  agents: any[]
  selectedAgentId: string | null
}

export function AgentCoordination({ agents, selectedAgentId }: AgentCoordinationProps) {
  const [coordinationConfig, setCoordinationConfig] = useState({
    strategy: 'adaptive',
    selectedAgents: [] as string[],
    loadBalance: true,
    timeout: 300,
    priority: 'medium',
    context: ''
  })

  const [reasoningConfig, setReasoningConfig] = useState({
    strategy: 'majority_vote',
    selectedAgents: [] as string[],
    taskDescription: '',
    timeout: 180,
    context: {}
  })

  // Fetch coordination status
  const { data: coordinationStatus, isLoading: statusLoading, refetch: refetchStatus } = useCoordinationStatus()
  
  // Mutations
  const coordinateAgentsMutation = useCoordinateAgents()
  const collaborativeReasoningMutation = useCollaborativeReasoning()

  // Get active agents
  const activeAgents = agents.filter(agent => agent.status === 'active')

  // Handle coordination
  const handleCoordination = async () => {
    if (coordinationConfig.selectedAgents.length < 2) {
      toast.error('Please select at least 2 agents for coordination')
      return
    }

    const coordinationData = {
      task_id: Date.now(),
      user_id: 1, // TODO: Get from auth context
      agents: coordinationConfig.selectedAgents,
      strategy: coordinationConfig.strategy,
      load_balance: coordinationConfig.loadBalance,
      context: {
        priority: coordinationConfig.priority,
        timeout_seconds: coordinationConfig.timeout,
        description: coordinationConfig.context
      }
    }

    try {
      await coordinateAgentsMutation.mutateAsync(coordinationData)
      await refetchStatus()
    } catch (error) {
      // Error handled by mutation hook
    }
  }

  // Handle collaborative reasoning
  const handleCollaborativeReasoning = async () => {
    if (reasoningConfig.selectedAgents.length < 2) {
      toast.error('Please select at least 2 agents for collaborative reasoning')
      return
    }

    if (!reasoningConfig.taskDescription.trim()) {
      toast.error('Please provide a task description')
      return
    }

    const reasoningData = {
      task_id: Date.now(),
      user_id: 1, // TODO: Get from auth context
      agents: reasoningConfig.selectedAgents,
      strategy: reasoningConfig.strategy,
      context: {
        task_description: reasoningConfig.taskDescription,
        timeout_seconds: reasoningConfig.timeout,
        ...reasoningConfig.context
      }
    }

    try {
      await collaborativeReasoningMutation.mutateAsync(reasoningData)
    } catch (error) {
      // Error handled by mutation hook
    }
  }

  // Toggle agent selection
  const toggleAgentSelection = (agentId: string, configType: 'coordination' | 'reasoning') => {
    if (configType === 'coordination') {
      setCoordinationConfig(prev => ({
        ...prev,
        selectedAgents: prev.selectedAgents.includes(agentId)
          ? prev.selectedAgents.filter(id => id !== agentId)
          : [...prev.selectedAgents, agentId]
      }))
    } else {
      setReasoningConfig(prev => ({
        ...prev,
        selectedAgents: prev.selectedAgents.includes(agentId)
          ? prev.selectedAgents.filter(id => id !== agentId)
          : [...prev.selectedAgents, agentId]
      }))
    }
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold">Agent Coordination</h2>
          <p className="text-muted-foreground">
            Coordinate multiple agents and manage collaborative reasoning
          </p>
        </div>

        <div className="flex items-center gap-2">
          <Button
            variant="outline"
            size="sm"
            onClick={() => refetchStatus()}
            disabled={statusLoading}
          >
            <RefreshCw className={`w-4 h-4 mr-2 ${statusLoading ? 'animate-spin' : ''}`} />
            Refresh Status
          </Button>
        </div>
      </div>

      {/* Current Status */}
      <Card className="glass-card">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Activity className="w-5 h-5" />
            Coordination Status
          </CardTitle>
        </CardHeader>
        <CardContent>
          {statusLoading ? (
            <div className="space-y-4">
              <Skeleton className="h-4 w-48" />
              <Skeleton className="h-4 w-64" />
              <Skeleton className="h-4 w-32" />
            </div>
          ) : (
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div className="text-center">
                <div className="text-2xl font-bold text-[hsl(var(--success))]">
                  {coordinationStatus?.active_coordinations || 0}
                </div>
                <div className="text-sm text-muted-foreground">Active Coordinations</div>
              </div>
              <div className="text-center">
                <div className="text-2xl font-bold text-[hsl(var(--info))]">
                  {coordinationStatus?.total_agents_coordinated || 0}
                </div>
                <div className="text-sm text-muted-foreground">Agents Coordinated</div>
              </div>
              <div className="text-center">
                <div className="text-2xl font-bold text-[hsl(var(--agent))]">
                  {coordinationStatus?.success_rate || 0}%
                </div>
                <div className="text-sm text-muted-foreground">Success Rate</div>
              </div>
            </div>
          )}
        </CardContent>
      </Card>

      {/* Main Tabs */}
      <Tabs defaultValue="coordination" className="space-y-6">
        <TabsList className="grid w-full grid-cols-2">
          <TabsTrigger value="coordination">Agent Coordination</TabsTrigger>
          <TabsTrigger value="reasoning">Collaborative Reasoning</TabsTrigger>
        </TabsList>

        {/* Agent Coordination Tab */}
        <TabsContent value="coordination" className="space-y-6">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Configuration */}
            <Card className="glass-card">
              <CardHeader>
                <CardTitle>Coordination Setup</CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="space-y-2">
                  <Label>Coordination Strategy</Label>
                  <Select
                    value={coordinationConfig.strategy}
                    onValueChange={(value) => setCoordinationConfig(prev => ({ ...prev, strategy: value }))}
                  >
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      {coordinationStrategies.map(strategy => (
                        <SelectItem key={strategy.id} value={strategy.id}>
                          {strategy.name}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                  <p className="text-xs text-muted-foreground">
                    {coordinationStrategies.find(s => s.id === coordinationConfig.strategy)?.description}
                  </p>
                </div>

                <div className="space-y-2">
                  <Label>Priority Level</Label>
                  <Select
                    value={coordinationConfig.priority}
                    onValueChange={(value) => setCoordinationConfig(prev => ({ ...prev, priority: value }))}
                  >
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="low">Low Priority</SelectItem>
                      <SelectItem value="medium">Medium Priority</SelectItem>
                      <SelectItem value="high">High Priority</SelectItem>
                      <SelectItem value="critical">Critical Priority</SelectItem>
                    </SelectContent>
                  </Select>
                </div>

                <div className="space-y-2">
                  <Label>Timeout (seconds)</Label>
                  <Input
                    type="number"
                    value={coordinationConfig.timeout}
                    onChange={(e) => setCoordinationConfig(prev => ({ 
                      ...prev, 
                      timeout: parseInt(e.target.value) || 300 
                    }))}
                    min="10"
                    max="3600"
                  />
                </div>

                <div className="space-y-2">
                  <Label>Context Description</Label>
                  <Textarea
                    value={coordinationConfig.context}
                    onChange={(e) => setCoordinationConfig(prev => ({ ...prev, context: e.target.value }))}
                    placeholder="Describe the coordination context and objectives..."
                    rows={3}
                  />
                </div>

                <Button 
                  onClick={handleCoordination}
                  disabled={coordinationConfig.selectedAgents.length < 2 || coordinateAgentsMutation.isPending}
                  className="w-full"
                >
                  <Users className="w-4 h-4 mr-2" />
                  {coordinateAgentsMutation.isPending ? 'Starting Coordination...' : 'Start Coordination'}
                </Button>
              </CardContent>
            </Card>

            {/* Agent Selection */}
            <Card className="glass-card">
              <CardHeader>
                <CardTitle>Select Agents ({coordinationConfig.selectedAgents.length} selected)</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-2 max-h-80 overflow-y-auto">
                  {activeAgents.map(agent => (
                    <div
                      key={agent.id}
                      className={`p-3 rounded-lg border cursor-pointer transition-all ${
                        coordinationConfig.selectedAgents.includes(agent.id)
                          ? 'border-primary bg-primary/5'
                          : 'border-border hover:border-primary/50'
                      }`}
                      onClick={() => toggleAgentSelection(agent.id, 'coordination')}
                    >
                      <div className="flex items-center gap-3">
                        <div className="w-8 h-8 rounded-full bg-gradient-to-br from-primary to-[hsl(var(--destructive))] flex items-center justify-center text-white text-sm">
                          🤖
                        </div>
                        <div className="flex-1">
                          <div className="font-medium">{agent.name}</div>
                          <div className="text-xs text-muted-foreground capitalize">
                            {agent.agent_type?.replace('_', ' ')}
                          </div>
                        </div>
                        {coordinationConfig.selectedAgents.includes(agent.id) && (
                          <CheckCircle className="w-5 h-5 text-primary" />
                        )}
                      </div>
                    </div>
                  ))}
                </div>

                {activeAgents.length === 0 && (
                  <Alert>
                    <AlertCircle className="h-4 w-4" />
                    <AlertDescription>
                      No active agents available for coordination. Please ensure agents are running.
                    </AlertDescription>
                  </Alert>
                )}
              </CardContent>
            </Card>
          </div>

          {/* Strategy Overview */}
          <Card className="glass-card">
            <CardHeader>
              <CardTitle>Coordination Strategies</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-1 md:grid-cols-3 lg:grid-cols-5 gap-4">
                {coordinationStrategies.map(strategy => {
                  const Icon = strategy.icon
                  const isSelected = coordinationConfig.strategy === strategy.id
                  
                  return (
                    <div
                      key={strategy.id}
                      className={`p-4 rounded-lg border cursor-pointer transition-all ${
                        isSelected
                          ? 'border-primary bg-primary/5'
                          : 'border-border hover:border-primary/50'
                      }`}
                      onClick={() => setCoordinationConfig(prev => ({ ...prev, strategy: strategy.id }))}
                    >
                      <div className={`w-10 h-10 rounded-lg ${strategy.bgColor} flex items-center justify-center mb-3`}>
                        <Icon className={`w-5 h-5 ${strategy.color}`} />
                      </div>
                      <div className="font-medium text-sm mb-1">{strategy.name}</div>
                      <div className="text-xs text-muted-foreground">{strategy.description}</div>
                    </div>
                  )
                })}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        {/* Collaborative Reasoning Tab */}
        <TabsContent value="reasoning" className="space-y-6">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Reasoning Setup */}
            <Card className="glass-card">
              <CardHeader>
                <CardTitle>Collaborative Reasoning Setup</CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="space-y-2">
                  <Label>Reasoning Strategy</Label>
                  <Select
                    value={reasoningConfig.strategy}
                    onValueChange={(value) => setReasoningConfig(prev => ({ ...prev, strategy: value }))}
                  >
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      {reasoningStrategies.map(strategy => (
                        <SelectItem key={strategy.id} value={strategy.id}>
                          {strategy.name}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                  <p className="text-xs text-muted-foreground">
                    {reasoningStrategies.find(s => s.id === reasoningConfig.strategy)?.description}
                  </p>
                </div>

                <div className="space-y-2">
                  <Label>Task Description *</Label>
                  <Textarea
                    value={reasoningConfig.taskDescription}
                    onChange={(e) => setReasoningConfig(prev => ({ ...prev, taskDescription: e.target.value }))}
                    placeholder="Describe the problem or question for collaborative reasoning..."
                    rows={4}
                  />
                </div>

                <div className="space-y-2">
                  <Label>Timeout (seconds)</Label>
                  <Input
                    type="number"
                    value={reasoningConfig.timeout}
                    onChange={(e) => setReasoningConfig(prev => ({ 
                      ...prev, 
                      timeout: parseInt(e.target.value) || 180 
                    }))}
                    min="10"
                    max="1800"
                  />
                </div>

                <Button 
                  onClick={handleCollaborativeReasoning}
                  disabled={
                    reasoningConfig.selectedAgents.length < 2 || 
                    !reasoningConfig.taskDescription.trim() ||
                    collaborativeReasoningMutation.isPending
                  }
                  className="w-full"
                >
                  <Brain className="w-4 h-4 mr-2" />
                  {collaborativeReasoningMutation.isPending ? 'Processing...' : 'Start Collaborative Reasoning'}
                </Button>
              </CardContent>
            </Card>

            {/* Agent Selection for Reasoning */}
            <Card className="glass-card">
              <CardHeader>
                <CardTitle>Select Agents ({reasoningConfig.selectedAgents.length} selected)</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-2 max-h-80 overflow-y-auto">
                  {activeAgents.map(agent => (
                    <div
                      key={agent.id}
                      className={`p-3 rounded-lg border cursor-pointer transition-all ${
                        reasoningConfig.selectedAgents.includes(agent.id)
                          ? 'border-primary bg-primary/5'
                          : 'border-border hover:border-primary/50'
                      }`}
                      onClick={() => toggleAgentSelection(agent.id, 'reasoning')}
                    >
                      <div className="flex items-center gap-3">
                        <div className="w-8 h-8 rounded-full bg-gradient-to-br from-primary to-[hsl(var(--destructive))] flex items-center justify-center text-white text-sm">
                          🤖
                        </div>
                        <div className="flex-1">
                          <div className="font-medium">{agent.name}</div>
                          <div className="text-xs text-muted-foreground capitalize">
                            {agent.agent_type?.replace('_', ' ')} • {agent.skills?.length || 0} skills
                          </div>
                        </div>
                        {reasoningConfig.selectedAgents.includes(agent.id) && (
                          <CheckCircle className="w-5 h-5 text-primary" />
                        )}
                      </div>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>
      </Tabs>
    </div>
  )
}

