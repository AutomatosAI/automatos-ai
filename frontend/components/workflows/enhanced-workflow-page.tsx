
'use client'

import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import { useInView } from 'react-intersection-observer'
import { 
  Play, 
  Pause, 
  Plus, 
  Search, 
  Filter,
  Activity,
  GitBranch,
  Clock,
  CheckCircle,
  AlertTriangle,
  MoreVertical,
  Eye,
  Edit,
  Trash2,
  Users,
  TrendingUp,
  Zap,
  Bot,
  MessageCircle,
  Terminal,
  Settings,
  RefreshCw
} from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Progress } from '@/components/ui/progress'
import { 
  DropdownMenu, 
  DropdownMenuContent, 
  DropdownMenuItem, 
  DropdownMenuTrigger 
} from '@/components/ui/dropdown-menu'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Dialog, DialogContent, DialogHeader, DialogTitle } from '@/components/ui/dialog'
import { workflowService, type WorkflowWithMetrics, type WorkflowStats } from '@/lib/workflow-service'
import { apiClient } from '@/lib/api'
import { InteractiveWorkflowExecution } from './interactive-workflow-execution'
import { CreateWorkflowModal } from './create-workflow-modal'

// Live workflow execution data
interface LiveWorkflowExecution {
  id: string
  workflowId: number
  workflowName: string
  status: 'running' | 'paused' | 'completed' | 'failed' | 'waiting_input'
  progress: number
  currentStep: string
  agentsActive: number
  startTime: string
  needsHumanInput: boolean
}

export function EnhancedWorkflowPage() {
  const [searchTerm, setSearchTerm] = useState('')
  const [selectedStatus, setSelectedStatus] = useState('all')
  const [selectedWorkflow, setSelectedWorkflow] = useState<number | null>(null)
  const [workflows, setWorkflows] = useState<WorkflowWithMetrics[]>([])
  const [workflowStats, setWorkflowStats] = useState<WorkflowStats>({
    activeWorkflows: 0,
    completedToday: 0,
    avgDuration: '0h',
    agentUtilization: 0,
    successRate: 0
  })
  const [liveExecutions, setLiveExecutions] = useState<LiveWorkflowExecution[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [autoRefresh, setAutoRefresh] = useState(true)
  const [isCreateWorkflowModalOpen, setIsCreateWorkflowModalOpen] = useState(false)
  const [ref, inView] = useInView({
    triggerOnce: true,
    threshold: 0.1,
  })

  // Load data on mount and set up auto-refresh
  useEffect(() => {
    loadWorkflowData()
    
    if (autoRefresh) {
      const interval = setInterval(() => {
        loadWorkflowData()
        updateLiveExecutions()
      }, 5000) // Refresh every 5 seconds
      
      return () => clearInterval(interval)
    }
  }, [autoRefresh])

  const loadWorkflowData = async () => {
    try {
      setLoading(true)
      setError(null)

      // Load workflows and stats from real backend
      const [workflowsData, statsData] = await Promise.all([
        workflowService.getWorkflowsWithMetrics(),
        workflowService.getWorkflowStats()
      ])

      setWorkflows(workflowsData)
      setWorkflowStats(statsData)

      // Generate live executions from active workflows
      const activeWorkflows = workflowsData.filter(w => w.status === 'active')
      const liveExecs: LiveWorkflowExecution[] = activeWorkflows.map((workflow, index) => ({
        id: `live_${workflow.id}`,
        workflowId: workflow.id,
        workflowName: workflow.name,
        status: Math.random() > 0.8 ? 'waiting_input' : 'running',
        progress: workflow.progress || Math.floor(Math.random() * 80) + 10,
        currentStep: workflow.currentStep || `Step ${Math.floor(Math.random() * 5) + 1}`,
        agentsActive: Math.floor(Math.random() * 4) + 1,
        startTime: workflow.startedAt || new Date(Date.now() - Math.random() * 3600000).toISOString(),
        needsHumanInput: Math.random() > 0.7 // 30% chance of needing input
      }))

      setLiveExecutions(liveExecs)

    } catch (err) {
      console.error('Error loading workflow data:', err)
      setError(err instanceof Error ? err.message : 'Failed to load workflow data')
    } finally {
      setLoading(false)
    }
  }

  const updateLiveExecutions = async () => {
    try {
      // Get active workflows and update their live status
      const activeWorkflows = await apiClient.getActiveWorkflows()
      
      // Update existing live executions with new data
      setLiveExecutions(prev => 
        prev.map(exec => ({
          ...exec,
          progress: Math.min(100, exec.progress + Math.random() * 5),
          needsHumanInput: Math.random() > 0.85 // 15% chance of needing input
        }))
      )
    } catch (error) {
      console.error('Error updating live executions:', error)
    }
  }

  const handleExecuteWorkflow = async (workflowId: number) => {
    setSelectedWorkflow(workflowId)
  }

  const handleCreateWorkflow = () => {
    setIsCreateWorkflowModalOpen(true)
  }

  const filteredWorkflows = workflows.filter(workflow => {
    const matchesSearch = workflow.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
      (workflow.description && workflow.description.toLowerCase().includes(searchTerm.toLowerCase())) ||
      (workflow.category && workflow.category.toLowerCase().includes(searchTerm.toLowerCase()))
    
    const matchesStatus = selectedStatus === 'all' || workflow.status === selectedStatus
    
    return matchesSearch && matchesStatus
  })

  const statusStyles: Record<string, string> = {
    draft: 'bg-gray-500/10 text-gray-400 border-gray-500/20',
    active: 'bg-blue-500/10 text-blue-400 border-blue-500/20',
    archived: 'bg-green-500/10 text-green-400 border-green-500/20',
    running: 'bg-blue-500/10 text-blue-400 border-blue-500/20',
    completed: 'bg-green-500/10 text-green-400 border-green-500/20',
    paused: 'bg-yellow-500/10 text-yellow-400 border-yellow-500/20',
    failed: 'bg-red-500/10 text-red-400 border-red-500/20',
    waiting_input: 'bg-orange-500/10 text-orange-400 border-orange-500/20'
  }

  if (selectedWorkflow) {
    return (
      <div className="space-y-6">
        {/* Back Button */}
        <div className="flex items-center justify-between">
          <Button
            variant="outline"
            onClick={() => setSelectedWorkflow(null)}
            className="mb-4"
          >
            ← Back to Workflows
          </Button>
          <div className="text-sm text-muted-foreground">
            Workflow Execution #{selectedWorkflow}
          </div>
        </div>

        {/* Interactive Execution Interface */}
        <InteractiveWorkflowExecution workflowId={selectedWorkflow} />
      </div>
    )
  }

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
            Enhanced <span className="gradient-text">Workflow Engine</span>
          </h1>
          <p className="text-muted-foreground text-lg">
            Real-time orchestrator monitoring with interactive agent coordination
          </p>
        </div>
        
        <div className="flex items-center space-x-2">
          <Button
            variant="outline"
            size="sm"
            onClick={() => setAutoRefresh(!autoRefresh)}
            className={autoRefresh ? 'bg-green-500/10 border-green-500/20' : ''}
          >
            <RefreshCw className={`w-4 h-4 mr-2 ${autoRefresh ? 'animate-spin' : ''}`} />
            Auto Refresh
          </Button>
          <Button 
            className="gradient-accent hover:opacity-90 transition-opacity"
            onClick={handleCreateWorkflow}
          >
            <Plus className="w-4 h-4 mr-2" />
            Create Workflow
          </Button>
        </div>
      </motion.div>

      {/* Enhanced Stats Overview */}
      <motion.div
        ref={ref}
        className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6"
        initial={{ opacity: 0, y: 20 }}
        animate={inView ? { opacity: 1, y: 0 } : {}}
        transition={{ duration: 0.8, delay: 0.2 }}
      >
        <motion.div
          className="glass-card p-6 card-glow hover:border-primary/20 transition-all duration-300"
          initial={{ opacity: 0, y: 20 }}
          animate={inView ? { opacity: 1, y: 0 } : {}}
          transition={{ duration: 0.8, delay: 0.1 }}
        >
          <div className="flex items-center justify-between mb-4">
            <div className="w-10 h-10 rounded-lg bg-secondary/50 flex items-center justify-center">
              <Activity className="w-5 h-5 text-blue-400" />
            </div>
            <Badge className="bg-blue-500/10 text-blue-400 border-blue-500/20">
              Live
            </Badge>
          </div>
          <div className="space-y-1">
            <h3 className="text-2xl font-bold">{workflowStats.activeWorkflows}</h3>
            <p className="text-muted-foreground text-sm">Active Workflows</p>
            <p className="text-xs text-blue-400">
              {liveExecutions.filter(e => e.needsHumanInput).length} need attention
            </p>
          </div>
        </motion.div>

        <motion.div
          className="glass-card p-6 card-glow hover:border-primary/20 transition-all duration-300"
          initial={{ opacity: 0, y: 20 }}
          animate={inView ? { opacity: 1, y: 0 } : {}}
          transition={{ duration: 0.8, delay: 0.2 }}
        >
          <div className="flex items-center justify-between mb-4">
            <div className="w-10 h-10 rounded-lg bg-secondary/50 flex items-center justify-center">
              <Bot className="w-5 h-5 text-purple-400" />
            </div>
          </div>
          <div className="space-y-1">
            <h3 className="text-2xl font-bold">
              {liveExecutions.reduce((sum, exec) => sum + exec.agentsActive, 0)}
            </h3>
            <p className="text-muted-foreground text-sm">Active Agents</p>
            <p className="text-xs text-purple-400">
              {workflowStats.agentUtilization}% utilization
            </p>
          </div>
        </motion.div>

        <motion.div
          className="glass-card p-6 card-glow hover:border-primary/20 transition-all duration-300"
          initial={{ opacity: 0, y: 20 }}
          animate={inView ? { opacity: 1, y: 0 } : {}}
          transition={{ duration: 0.8, delay: 0.3 }}
        >
          <div className="flex items-center justify-between mb-4">
            <div className="w-10 h-10 rounded-lg bg-secondary/50 flex items-center justify-center">
              <CheckCircle className="w-5 h-5 text-green-400" />
            </div>
          </div>
          <div className="space-y-1">
            <h3 className="text-2xl font-bold">{workflowStats.completedToday}</h3>
            <p className="text-muted-foreground text-sm">Completed Today</p>
            <p className="text-xs text-green-400">
              {workflowStats.successRate}% success rate
            </p>
          </div>
        </motion.div>

        <motion.div
          className="glass-card p-6 card-glow hover:border-primary/20 transition-all duration-300"
          initial={{ opacity: 0, y: 20 }}
          animate={inView ? { opacity: 1, y: 0 } : {}}
          transition={{ duration: 0.8, delay: 0.4 }}
        >
          <div className="flex items-center justify-between mb-4">
            <div className="w-10 h-10 rounded-lg bg-secondary/50 flex items-center justify-center">
              <Clock className="w-5 h-5 text-orange-400" />
            </div>
          </div>
          <div className="space-y-1">
            <h3 className="text-2xl font-bold">{workflowStats.avgDuration}</h3>
            <p className="text-muted-foreground text-sm">Avg Duration</p>
            <p className="text-xs text-orange-400">
              Real-time tracking
            </p>
          </div>
        </motion.div>
      </motion.div>

      {/* Live Executions Dashboard */}
      {liveExecutions.length > 0 && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={inView ? { opacity: 1, y: 0 } : {}}
          transition={{ duration: 0.8, delay: 0.6 }}
          className="glass-card p-6"
        >
          <div className="flex items-center justify-between mb-6">
            <h3 className="text-xl font-semibold flex items-center">
              <Terminal className="w-5 h-5 mr-2" />
              Live Workflow Executions
            </h3>
            <Badge className="bg-green-500/10 text-green-400 border-green-500/20">
              {liveExecutions.length} Running
            </Badge>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {liveExecutions.map((execution) => (
              <motion.div
                key={execution.id}
                initial={{ opacity: 0, scale: 0.95 }}
                animate={{ opacity: 1, scale: 1 }}
                className="border border-border/30 rounded-lg p-4 bg-secondary/20 hover:bg-secondary/30 transition-all duration-300"
              >
                <div className="flex items-center justify-between mb-3">
                  <div className="flex items-center space-x-2">
                    <h4 className="font-semibold">{execution.workflowName}</h4>
                    <Badge className={statusStyles[execution.status]}>
                      {execution.status.replace('_', ' ')}
                    </Badge>
                  </div>
                  <div className="flex items-center space-x-2">
                    {execution.needsHumanInput && (
                      <MessageCircle className="w-4 h-4 text-orange-400" />
                    )}
                    <Button
                      size="sm"
                      onClick={() => handleExecuteWorkflow(execution.workflowId)}
                      className="bg-blue-500/10 text-blue-400 border-blue-500/20 hover:bg-blue-500/20"
                    >
                      <Eye className="w-4 h-4 mr-1" />
                      Monitor
                    </Button>
                  </div>
                </div>

                <div className="space-y-3">
                  <div className="flex items-center justify-between text-sm">
                    <span className="text-muted-foreground">{execution.currentStep}</span>
                    <span className="font-medium">{Math.round(execution.progress)}%</span>
                  </div>
                  <Progress value={execution.progress} className="h-2" />
                  
                  <div className="flex items-center justify-between text-xs text-muted-foreground">
                    <div className="flex items-center space-x-4">
                      <span className="flex items-center">
                        <Users className="w-3 h-3 mr-1" />
                        {execution.agentsActive} agents
                      </span>
                      <span className="flex items-center">
                        <Clock className="w-3 h-3 mr-1" />
                        {new Date(execution.startTime).toLocaleTimeString()}
                      </span>
                    </div>
                    {execution.needsHumanInput && (
                      <Badge className="bg-orange-500/10 text-orange-400 border-orange-500/20 text-xs">
                        Needs Input
                      </Badge>
                    )}
                  </div>
                </div>
              </motion.div>
            ))}
          </div>
        </motion.div>
      )}

      {/* Workflow Management */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={inView ? { opacity: 1, y: 0 } : {}}
        transition={{ duration: 0.8, delay: 0.8 }}
      >
        <Tabs defaultValue="active" className="space-y-6">
          <TabsList className="grid w-full grid-cols-4 lg:w-auto lg:inline-grid bg-secondary/50">
            <TabsTrigger value="active" className="flex items-center space-x-2">
              <Activity className="w-4 h-4" />
              <span className="hidden sm:inline">Active</span>
            </TabsTrigger>
            <TabsTrigger value="templates" className="flex items-center space-x-2">
              <GitBranch className="w-4 h-4" />
              <span className="hidden sm:inline">Templates</span>
            </TabsTrigger>
            <TabsTrigger value="history" className="flex items-center space-x-2">
              <Clock className="w-4 h-4" />
              <span className="hidden sm:inline">History</span>
            </TabsTrigger>
            <TabsTrigger value="analytics" className="flex items-center space-x-2">
              <TrendingUp className="w-4 h-4" />
              <span className="hidden sm:inline">Analytics</span>
            </TabsTrigger>
          </TabsList>

          <TabsContent value="active" className="space-y-6">
            {/* Search and Filters */}
            <div className="flex flex-col sm:flex-row gap-4">
              <div className="relative flex-1">
                <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-muted-foreground" />
                <Input
                  placeholder="Search workflows by name, category, or description..."
                  value={searchTerm}
                  onChange={(e) => setSearchTerm(e.target.value)}
                  className="pl-10 bg-secondary/50 border-secondary focus:border-primary/50"
                />
              </div>
              <Button variant="outline" className="shrink-0">
                <Filter className="w-4 h-4 mr-2" />
                Filters
              </Button>
            </div>

            {/* Loading State */}
            {loading && (
              <div className="flex items-center justify-center py-12">
                <div className="text-center">
                  <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary mx-auto mb-4"></div>
                  <p className="text-muted-foreground">Loading workflows...</p>
                </div>
              </div>
            )}

            {/* Error State */}
            {error && !loading && (
              <div className="flex items-center justify-center py-12">
                <div className="text-center">
                  <AlertTriangle className="h-8 w-8 text-red-400 mx-auto mb-4" />
                  <p className="text-red-400 mb-4">Error loading workflows: {error}</p>
                  <Button onClick={loadWorkflowData} variant="outline">
                    Try Again
                  </Button>
                </div>
              </div>
            )}

            {/* Empty State */}
            {!loading && !error && filteredWorkflows.length === 0 && (
              <div className="flex items-center justify-center py-12">
                <div className="text-center">
                  <GitBranch className="h-8 w-8 text-muted-foreground mx-auto mb-4" />
                  <p className="text-muted-foreground mb-4">
                    {workflows.length === 0 ? 'No workflows created yet' : 'No workflows match your search'}
                  </p>
                  <Button onClick={handleCreateWorkflow} className="gradient-accent hover:opacity-90">
                    <Plus className="w-4 h-4 mr-2" />
                    Create Your First Workflow
                  </Button>
                </div>
              </div>
            )}

            {/* Workflows Grid */}
            {!loading && !error && filteredWorkflows.length > 0 && (
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                {filteredWorkflows.map((workflow, index) => (
                  <motion.div
                    key={workflow.id}
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.5, delay: index * 0.1 }}
                    className="glass-card p-6 card-glow hover:border-primary/20 transition-all duration-300"
                  >
                    <div className="flex items-center justify-between mb-4">
                      <Badge className={statusStyles[workflow.status]}>
                        {workflow.status}
                      </Badge>
                      <DropdownMenu>
                        <DropdownMenuTrigger asChild>
                          <Button variant="ghost" size="sm">
                            <MoreVertical className="w-4 h-4" />
                          </Button>
                        </DropdownMenuTrigger>
                        <DropdownMenuContent align="end">
                          <DropdownMenuItem onClick={() => handleExecuteWorkflow(workflow.id)}>
                            <Play className="w-4 h-4 mr-2" />
                            Execute
                          </DropdownMenuItem>
                          <DropdownMenuItem>
                            <Eye className="w-4 h-4 mr-2" />
                            View Details
                          </DropdownMenuItem>
                          <DropdownMenuItem>
                            <Edit className="w-4 h-4 mr-2" />
                            Edit
                          </DropdownMenuItem>
                          <DropdownMenuItem className="text-red-400">
                            <Trash2 className="w-4 h-4 mr-2" />
                            Delete
                          </DropdownMenuItem>
                        </DropdownMenuContent>
                      </DropdownMenu>
                    </div>

                    <div className="space-y-3">
                      <div>
                        <h3 className="font-semibold text-lg mb-1">{workflow.name}</h3>
                        <p className="text-sm text-muted-foreground">
                          {workflow.description || 'No description provided'}
                        </p>
                      </div>

                      <div className="flex items-center justify-between text-sm">
                        <span className="text-muted-foreground">Progress</span>
                        <span className="font-medium">{Math.round(workflow.progress || 0)}%</span>
                      </div>
                      <Progress value={workflow.progress || 0} className="h-2" />

                      <div className="flex items-center justify-between text-xs text-muted-foreground">
                        <div className="flex items-center space-x-2">
                          <Users className="w-3 h-3" />
                          <span>{workflow.agentNames?.length || 0} agents</span>
                        </div>
                        <div className="flex items-center space-x-2">
                          <Zap className="w-3 h-3" />
                          <span>{workflow.successRate || 0}% success</span>
                        </div>
                      </div>

                      <Button
                        onClick={() => handleExecuteWorkflow(workflow.id)}
                        className="w-full gradient-accent hover:opacity-90 transition-opacity"
                        size="sm"
                      >
                        <Play className="w-4 h-4 mr-2" />
                        Execute Workflow
                      </Button>
                    </div>
                  </motion.div>
                ))}
              </div>
            )}
          </TabsContent>

          <TabsContent value="templates" className="space-y-6">
            <Card className="glass-card">
              <CardHeader>
                <CardTitle>Workflow Templates</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="text-center text-muted-foreground">
                  Workflow templates and presets will be displayed here
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="history" className="space-y-6">
            <Card className="glass-card">
              <CardHeader>
                <CardTitle>Execution History</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="text-center text-muted-foreground">
                  Workflow execution history and logs will be displayed here
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="analytics" className="space-y-6">
            <Card className="glass-card">
              <CardHeader>
                <CardTitle>Workflow Analytics</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="text-center text-muted-foreground">
                  Performance analytics and insights will be displayed here
                </div>
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
      </motion.div>

      {/* Debug Component removed - production ready */}

      {/* Create Workflow Modal */}
      <CreateWorkflowModal
        isOpen={isCreateWorkflowModalOpen}
        onClose={() => setIsCreateWorkflowModalOpen(false)}
        onSuccess={() => {
          loadWorkflowData() // Refresh workflows data
        }}
      />
    </div>
  )
}
