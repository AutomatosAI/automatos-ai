
'use client'

import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import { useInView } from 'react-intersection-observer'
import { useQueryClient } from '@tanstack/react-query'
import { 
  Play, 
  Pause, 
  Square, 
  Plus, 
  Search, 
  Filter,
  GitBranch,
  Clock,
  CheckCircle,
  AlertTriangle,
  MoreVertical,
  Eye,
  Edit,
  Trash2,
  Users,
  Activity,
  X,
  ChevronRight,
  ChevronLeft,
  Bot,
  Zap
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
import { Label } from '@/components/ui/label'
import { Textarea } from '@/components/ui/textarea'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Checkbox } from '@/components/ui/checkbox'
import { workflowService, type WorkflowWithMetrics, type WorkflowStats } from '@/lib/workflow-service'
import { apiClient } from '@/lib/api-client'
import { ActiveWorkflowsPanel } from './active-workflows-panel'
import { HistoryTab } from './history-tab'
import { MonitoringTab } from './monitoring-tab'
import { TemplatesTab } from './templates-tab'
import { LiveProgressTab } from './live-progress-tab'
import { ExecutionTheater } from './execution-theater'

// Real data will be loaded from backend
const initialWorkflowStats = [
  {
    label: 'Active Workflows',
    value: '0',
    change: 'Loading...',
    icon: GitBranch,
    color: 'text-blue-400'
  },
  {
    label: 'Completed Today',
    value: '0',
    change: 'Loading...',
    icon: CheckCircle,
    color: 'text-green-400'
  },
  {
    label: 'Avg Duration',
    value: '0h',
    change: 'Loading...',
    icon: Clock,
    color: 'text-orange-400'
  },
  {
    label: 'Agent Utilization',
    value: '0%',
    change: 'Loading...',
    icon: Activity,
    color: 'text-purple-400'
  }
]

const statusStyles: Record<string, string> = {
  draft: 'bg-gray-500/10 text-gray-400 border-gray-500/20',
  active: 'bg-blue-500/10 text-blue-400 border-blue-500/20',
  archived: 'bg-green-500/10 text-green-400 border-green-500/20',
  // Legacy status support for UI display
  running: 'bg-blue-500/10 text-blue-400 border-blue-500/20',
  completed: 'bg-green-500/10 text-green-400 border-green-500/20',
  paused: 'bg-yellow-500/10 text-yellow-400 border-yellow-500/20',
  failed: 'bg-red-500/10 text-red-400 border-red-500/20',
  queued: 'bg-gray-500/10 text-gray-400 border-gray-500/20'
}

const priorityStyles: Record<string, string> = {
  low: 'bg-gray-500/10 text-gray-400 border-gray-500/20',
  medium: 'bg-blue-500/10 text-blue-400 border-blue-500/20',
  high: 'bg-orange-500/10 text-orange-400 border-orange-500/20',
  critical: 'bg-red-500/10 text-red-400 border-red-500/20'
}

const statusIcons: Record<string, any> = {
  draft: Clock,
  active: Play,
  archived: CheckCircle,
  // Legacy status support
  running: Play,
  completed: CheckCircle,
  paused: Pause,
  failed: AlertTriangle,
  queued: Clock
}

const availableAgents = [
  {
    id: 'code-architect',
    name: 'CodeArchitect',
    description: 'Senior software architect for system design and code review',
    category: 'Development',
    skills: ['Architecture', 'Code Review', 'Best Practices']
  },
  {
    id: 'security-guard',
    name: 'SecurityGuard',
    description: 'Cybersecurity specialist for vulnerability assessment',
    category: 'Security',
    skills: ['Security Audit', 'Vulnerability Scanning', 'Compliance']
  },
  {
    id: 'bug-hunter',
    name: 'BugHunter',
    description: 'Expert debugger for identifying and resolving issues',
    category: 'Development',
    skills: ['Bug Detection', 'Root Cause Analysis', 'Testing']
  },
  {
    id: 'performance-optimizer',
    name: 'PerformanceOptimizer',
    description: 'Performance analysis and optimization specialist',
    category: 'Optimization',
    skills: ['Performance Analysis', 'Database Optimization', 'Caching']
  },
  {
    id: 'test-master',
    name: 'TestMaster',
    description: 'Automated testing and quality assurance expert',
    category: 'Testing',
    skills: ['Test Automation', 'Quality Assurance', 'Test Planning']
  },
  {
    id: 'data-analyst',
    name: 'DataAnalyst',
    description: 'Data analysis and insights generation specialist',
    category: 'Analytics',
    skills: ['Data Analysis', 'Reporting', 'Visualization']
  }
]

const workflowTemplates = [
  {
    id: 'code-review',
    name: 'Code Review Pipeline',
    description: 'Comprehensive code review with security and quality checks',
    agents: ['code-architect', 'security-guard', 'test-master'],
    steps: ['Code Analysis', 'Security Scan', 'Quality Check', 'Documentation Review']
  },
  {
    id: 'bug-investigation',
    name: 'Bug Investigation',
    description: 'Systematic bug analysis and resolution workflow',
    agents: ['bug-hunter', 'performance-optimizer'],
    steps: ['Issue Analysis', 'Root Cause Investigation', 'Solution Development', 'Testing', 'Documentation']
  },
  {
    id: 'security-audit',
    name: 'Security Audit',
    description: 'Complete security assessment and compliance check',
    agents: ['security-guard'],
    steps: ['Vulnerability Scan', 'Code Security Review', 'Infrastructure Audit', 'Compliance Check', 'Report Generation']
  }
]

export function WorkflowManagement() {
  const queryClient = useQueryClient()
  const [searchTerm, setSearchTerm] = useState('')
  const [selectedStatus, setSelectedStatus] = useState('all')
  const [showCreateModal, setShowCreateModal] = useState(false)
  const [currentStep, setCurrentStep] = useState(1)
  const [isCreating, setIsCreating] = useState(false)
  const [workflows, setWorkflows] = useState<WorkflowWithMetrics[]>([])
  const [workflowStats, setWorkflowStats] = useState(initialWorkflowStats)
  const [availableAgents, setAvailableAgents] = useState<any[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [ref, inView] = useInView({
    triggerOnce: true,
    threshold: 0.1,
  })
  
  // Execution Theater state
  const [showExecutionTheater, setShowExecutionTheater] = useState(false)
  const [selectedWorkflowId, setSelectedWorkflowId] = useState<number | null>(null)
  const [autoStartExecution, setAutoStartExecution] = useState(false)

  // Workflow creation form state
  const [workflowForm, setWorkflowForm] = useState({
    name: '',
    description: '',
    goal: '',
    context: {} as Record<string, any>,
    template: '',
    priority: 'medium',
    selectedAgents: [] as string[],
    customSteps: [] as string[],
    configuration: {
      maxRetries: 3,
      timeout: 30,
      parallelExecution: false
    }
  })
  const [showAdvanced, setShowAdvanced] = useState(false)
  const [contextJson, setContextJson] = useState('')

  // Load real data from backend
  useEffect(() => {
    loadWorkflowData()
  }, [])

  const loadWorkflowData = async () => {
    try {
      setLoading(true)
      setError(null)

      // Load workflows and stats in parallel
      const [workflowsData, statsData, agentsData] = await Promise.all([
        workflowService.getWorkflowsWithMetrics(),
        workflowService.getWorkflowStats(),
        apiClient.getAgents().catch(() => []) // Fallback to empty array if agents fail
      ])

      setWorkflows(workflowsData)
      
      // Update stats with real data
      setWorkflowStats([
        {
          label: 'Active Workflows',
          value: statsData.activeWorkflows.toString(),
          change: `${statsData.activeWorkflows > 0 ? '+' : ''}${statsData.activeWorkflows} total`,
          icon: GitBranch,
          color: 'text-blue-400'
        },
        {
          label: 'Completed Today',
          value: statsData.completedToday.toString(),
          change: `${statsData.successRate}% success rate`,
          icon: CheckCircle,
          color: 'text-green-400'
        },
        {
          label: 'Avg Duration',
          value: statsData.avgDuration,
          change: 'Based on history',
          icon: Clock,
          color: 'text-orange-400'
        },
        {
          label: 'Agent Utilization',
          value: `${statsData.agentUtilization}%`,
          change: 'CPU usage based',
          icon: Activity,
          color: 'text-purple-400'
        }
      ])

      // Transform agents for UI
      const transformedAgents = agentsData.map(agent => ({
        id: agent.id.toString(),
        name: agent.name || 'Unknown Agent',
        description: agent.description || 'AI Agent',
        category: agent.agent_type ? agent.agent_type.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase()) : 'General',
        skills: agent.skills?.map(skill => skill.name ? skill.name : 'Unknown Skill') || ['General']
      }))
      
      setAvailableAgents(transformedAgents)

    } catch (err) {
      console.error('Error loading workflow data:', err)
      setError(err instanceof Error ? err.message : 'Failed to load workflow data')
    } finally {
      setLoading(false)
    }
  }

  const handleCreateWorkflow = () => {
    setShowCreateModal(true)
    setCurrentStep(1)
    setShowAdvanced(false)
    setContextJson('')
    setWorkflowForm({
      name: '',
      description: '',
      goal: '',
      context: {},
      template: '',
      priority: 'medium',
      selectedAgents: [],
      customSteps: [],
      configuration: {
        maxRetries: 3,
        timeout: 30,
        parallelExecution: false
      }
    })
  }

  const handleSubmitWorkflow = async () => {
    setIsCreating(true)
    
    // Parse context JSON if provided
    let contextData = workflowForm.context
    if (contextJson.trim()) {
      try {
        contextData = JSON.parse(contextJson)
      } catch (e) {
        setError('Invalid JSON in context field')
        setIsCreating(false)
        return
      }
    }
    
    // Get agent IDs from selected agent names
    const selectedAgentIds = availableAgents
      .filter(agent => workflowForm.selectedAgents.includes(agent.id))
      .map(agent => parseInt(agent.id))

    // Prepare the workflow data in the expected backend API format
    const workflowData = {
      name: workflowForm.name,
      description: workflowForm.description,
      goal: workflowForm.goal || undefined, // Only include if provided
      context: Object.keys(contextData).length > 0 ? contextData : undefined, // Only include if not empty
      category: workflowForm.template ? workflowTemplates.find(t => t.id === workflowForm.template)?.name : 'automation',
      priority: workflowForm.priority,
      config: workflowForm.configuration,
      steps: workflowForm.customSteps,
      agents: workflowForm.selectedAgents,
      tags: workflowForm.template ? [workflowForm.template] : ['custom']
    }
    
    try {
      // Submit to backend API using the service
      const result = await workflowService.createWorkflow(workflowData)
      console.log('Workflow created successfully:', result)

      // Reload workflow data to show the new workflow
      await loadWorkflowData()
      
      // Invalidate React Query cache to refresh active workflows panel
      queryClient.invalidateQueries({ queryKey: ['workflows', 'active'] })
      queryClient.invalidateQueries({ queryKey: ['workflows'] })

      // Close modal and reset form
      setShowCreateModal(false)
      setShowAdvanced(false)
      setContextJson('')
      setWorkflowForm({
        name: '',
        description: '',
        goal: '',
        context: {},
        template: '',
        priority: 'medium',
        selectedAgents: [],
        customSteps: [],
        configuration: {
          maxRetries: 3,
          timeout: 30,
          parallelExecution: false
        }
      })

      // Automatically open Execution Theater for the new workflow with auto-start
      if (result && result.id) {
        setTimeout(() => {
          setSelectedWorkflowId(result.id)
          setAutoStartExecution(true)
          setShowExecutionTheater(true)
        }, 500) // Small delay to allow UI to update
      }
      
    } catch (error: any) {
      console.error('Full error creating workflow:', error)
      console.error('Payload sent:', workflowData)
      
      // Extract detailed error message from API response
      let errorMessage = 'Unknown error'
      if (error?.response?.data?.detail) {
        errorMessage = error.response.data.detail
      } else if (error?.message) {
        errorMessage = error.message
      } else if (typeof error === 'string') {
        errorMessage = error
      }
      
      setError(errorMessage)
    } finally {
      setIsCreating(false)
    }
  }

  const handleTemplateChange = (template: any) => {
    // Handle both old format (string ID) and new format (full template object)
    if (typeof template === 'string') {
      // Old format - hardcoded template ID
      const hardcodedTemplate = workflowTemplates.find(t => t.id === template)
      if (hardcodedTemplate) {
        setWorkflowForm({
          ...workflowForm,
          template: template,
          name: hardcodedTemplate.name,
          description: hardcodedTemplate.description,
          selectedAgents: hardcodedTemplate.agents,
          customSteps: hardcodedTemplate.steps
        })
      }
    } else {
      // New format - full template object from backend API
      // Add timestamp to avoid name conflicts
      const timestamp = new Date().toISOString().slice(11, 19).replace(/:/g, '')
      setWorkflowForm({
        ...workflowForm,
        template: template.template_id || template.id,
        name: `${template.name} ${timestamp}`,
        description: template.description,
        priority: template.priority || 'medium',
        selectedAgents: template.recommended_agents || [],
        customSteps: template.template_definition?.steps || []
      })
    }
  }

  const toggleAgent = (agentId: string) => {
    setWorkflowForm({
      ...workflowForm,
      selectedAgents: workflowForm.selectedAgents.includes(agentId)
        ? workflowForm.selectedAgents.filter(id => id !== agentId)
        : [...workflowForm.selectedAgents, agentId]
    })
  }

  const handleWorkflowClick = (workflowId: number) => {
    setSelectedWorkflowId(workflowId)
    setShowExecutionTheater(true)
  }

  const handleBackFromTheater = () => {
    setShowExecutionTheater(false)
    setSelectedWorkflowId(null)
    setAutoStartExecution(false) // Reset auto-start flag
    // Reload workflows to get fresh data
    loadWorkflowData()
  }

  const filteredWorkflows = workflows.filter(workflow => {
    const matchesSearch = (workflow.name && workflow.name.toLowerCase().includes(searchTerm.toLowerCase())) ||
      (workflow.description && workflow.description.toLowerCase().includes(searchTerm.toLowerCase())) ||
      (workflow.category && workflow.category.toLowerCase().includes(searchTerm.toLowerCase())) ||
      (workflow.tags && workflow.tags.some(tag => tag && tag.toLowerCase().includes(searchTerm.toLowerCase())))
    
    const matchesStatus = selectedStatus === 'all' || workflow.status === selectedStatus
    
    return matchesSearch && matchesStatus
  })

  // Show Execution Theater if workflow is selected
  if (showExecutionTheater && selectedWorkflowId) {
    return (
      <ExecutionTheater 
        workflowId={selectedWorkflowId}
        onBack={handleBackFromTheater}
        autoStart={autoStartExecution}
      />
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
            Workflow <span className="gradient-text">Management</span>
          </h1>
          <p className="text-muted-foreground text-lg">
            Create, monitor, and manage your multi-agent workflows
          </p>
        </div>
        
        <Button 
          className="bg-gray-800 border border-orange-400/50 hover:border-orange-400 hover:bg-gray-700 text-white transition-all duration-200"
          onClick={handleCreateWorkflow}
        >
          <Plus className="w-4 h-4 mr-2" />
          Create Workflow
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
        {workflowStats.map((stat, index) => (
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

      {/* Workflow Management */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={inView ? { opacity: 1, y: 0 } : {}}
        transition={{ duration: 0.8, delay: 0.4 }}
      >
        <Tabs defaultValue="active" className="space-y-6">
          <TabsList className="grid w-full grid-cols-5 lg:w-auto lg:inline-grid bg-secondary/50">
            <TabsTrigger value="active" className="flex items-center space-x-2">
              <Play className="w-4 h-4" />
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
            <TabsTrigger value="monitoring" className="flex items-center space-x-2">
              <Activity className="w-4 h-4" />
              <span className="hidden sm:inline">Monitoring</span>
            </TabsTrigger>
            <TabsTrigger value="live" className="flex items-center space-x-2">
              <Eye className="w-4 h-4" />
              <span className="hidden sm:inline">Live</span>
            </TabsTrigger>
          </TabsList>

          <TabsContent value="active" className="space-y-6">
            {/* Search and Filters */}
            <div className="flex flex-col sm:flex-row gap-4">
              <div className="relative flex-1">
                <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-muted-foreground" />
                <Input
                  placeholder="Search workflows by name, category, or tags..."
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

            {/* Active Workflows Panel - handles its own empty state */}
            <ActiveWorkflowsPanel onWorkflowClick={handleWorkflowClick} />
          </TabsContent>

          <TabsContent value="templates" className="space-y-6">
            <TemplatesTab 
              onUseTemplate={(templateId) => {
                setError(null) // Clear any previous errors
                handleTemplateChange(templateId)
                setShowCreateModal(true) // Open modal after populating template data
              }}
              onOpenCreateModal={() => {
                setError(null) // Clear any previous errors
                setShowCreateModal(true)
              }}
            />
          </TabsContent>

          <TabsContent value="history" className="space-y-6">
            <HistoryTab />
          </TabsContent>

          <TabsContent value="monitoring" className="space-y-6">
            <MonitoringTab />
          </TabsContent>

          <TabsContent value="live" className="space-y-6">
            <LiveProgressTab />
          </TabsContent>
        </Tabs>
      </motion.div>

      {/* Create Workflow Modal - Simplified for Intelligent Orchestration */}
      <Dialog open={showCreateModal} onOpenChange={setShowCreateModal}>
        <DialogContent className="glass-card max-w-2xl">
          <DialogHeader>
            <DialogTitle>Create New Workflow</DialogTitle>
            <p className="text-sm text-muted-foreground mt-2">
              Describe your task and the Orchestrator will intelligently select the best agents and execute it.
            </p>
          </DialogHeader>

          <div className="space-y-6">
            {/* Simplified Single-Step Form */}
            <motion.div
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              className="space-y-4"
            >
              <div>
                <Label htmlFor="name">Workflow Name <span className="text-red-400">*</span></Label>
                <Input
                  id="name"
                  value={workflowForm.name}
                  onChange={(e) => setWorkflowForm({...workflowForm, name: e.target.value})}
                  placeholder="e.g., Analyze API Documentation"
                  className="bg-secondary/50 border-secondary"
                />
              </div>

              <div>
                <Label htmlFor="description">Task Description <span className="text-red-400">*</span></Label>
                <Textarea
                  id="description"
                  value={workflowForm.description}
                  onChange={(e) => setWorkflowForm({...workflowForm, description: e.target.value})}
                  placeholder="Describe what you want the workflow to accomplish. Be as specific as possible - the Orchestrator uses this to intelligently select agents and break down the task."
                  className="bg-secondary/50 border-secondary min-h-[120px]"
                  rows={5}
                />
                <p className="text-xs text-muted-foreground mt-2">
                  💡 <strong>Tip:</strong> Include context, goals, and expected outcomes for better results
                </p>
              </div>

              <div>
                <Label htmlFor="priority">Priority</Label>
                <Select 
                  value={workflowForm.priority || "medium"} 
                  onValueChange={(value) => setWorkflowForm({...workflowForm, priority: value})}
                >
                  <SelectTrigger className="bg-secondary/50 border-secondary">
                    <SelectValue placeholder="Select priority" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="low">Low</SelectItem>
                    <SelectItem value="medium">Medium (Default)</SelectItem>
                    <SelectItem value="high">High</SelectItem>
                    <SelectItem value="urgent">Urgent</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              {/* Advanced Options Toggle */}
              <div className="flex items-center justify-between">
                <Button
                  type="button"
                  variant="ghost"
                  size="sm"
                  onClick={() => setShowAdvanced(!showAdvanced)}
                  className="text-blue-400 hover:text-blue-300"
                >
                  <ChevronRight className={`w-4 h-4 mr-2 transition-transform ${showAdvanced ? 'rotate-90' : ''}`} />
                  Advanced Options (Goal & Context)
                </Button>
              </div>

              {/* Advanced Options */}
              {showAdvanced && (
                <motion.div
                  initial={{ opacity: 0, height: 0 }}
                  animate={{ opacity: 1, height: 'auto' }}
                  exit={{ opacity: 0, height: 0 }}
                  className="space-y-4 p-4 bg-secondary/20 border border-secondary rounded-lg"
                >
                  <div>
                    <Label htmlFor="goal">Workflow Goal (Optional)</Label>
                    <Input
                      id="goal"
                      value={workflowForm.goal}
                      onChange={(e) => setWorkflowForm({...workflowForm, goal: e.target.value})}
                      placeholder="e.g., Review PR #123 for security vulnerabilities"
                      className="bg-secondary/50 border-secondary"
                    />
                    <p className="text-xs text-muted-foreground mt-1">
                      High-level objective - overrides description if provided
                    </p>
                  </div>

                  <div>
                    <Label htmlFor="context">Workflow Context (Optional JSON)</Label>
                    <Textarea
                      id="context"
                      value={contextJson}
                      onChange={(e) => setContextJson(e.target.value)}
                      placeholder={`{\n  "codegraph_project": "my-app",\n  "pr_number": 123,\n  "git_url": "https://github.com/..."\n}`}
                      className="bg-secondary/50 border-secondary font-mono text-sm min-h-[100px]"
                      rows={5}
                    />
                    <p className="text-xs text-muted-foreground mt-1">
                      Additional context for execution (JSON format). Useful for CodeGraph integration, PR reviews, etc.
                    </p>
                  </div>

                  <div className="text-xs text-yellow-400 flex items-start space-x-2">
                    <Zap className="w-4 h-4 mt-0.5 flex-shrink-0" />
                    <div>
                      <strong>Pro Tip:</strong> Use <code className="bg-black/30 px-1 rounded">codegraph_project</code> in context to give agents access to indexed code.
                    </div>
                  </div>
                </motion.div>
              )}

              <div className="p-4 bg-blue-500/10 border border-blue-500/30 rounded-lg">
                <div className="flex items-start space-x-3">
                  <Bot className="w-5 h-5 text-blue-400 mt-0.5 flex-shrink-0" />
                  <div className="flex-1 text-sm">
                    <p className="font-semibold text-blue-300 mb-1">Intelligent Orchestration</p>
                    <p className="text-muted-foreground">
                      The Orchestrator will automatically analyze your task, select the most capable agents, 
                      decompose the work into subtasks, and manage execution - no manual configuration needed.
                    </p>
                  </div>
                </div>
              </div>
            </motion.div>

            {/* Error Display */}
            {error && (
              <div className="p-3 bg-red-500/10 border border-red-500/50 rounded-lg">
                <p className="text-red-400 text-sm">{error}</p>
              </div>
            )}

            {/* Modal Actions */}
            <div className="flex items-center justify-end space-x-2 pt-4 border-t border-border/30">
              <Button
                variant="outline"
                onClick={() => {
                  setShowCreateModal(false)
                  setError(null)
                  setShowAdvanced(false)
                  setContextJson('')
                  setWorkflowForm({
                    name: '',
                    description: '',
                    goal: '',
                    context: {},
                    template: '',
                    priority: 'medium',
                    selectedAgents: [],
                    customSteps: [],
                    configuration: {
                      maxRetries: 3,
                      timeout: 30,
                      parallelExecution: false
                    }
                  })
                }}
                disabled={isCreating}
              >
                Cancel
              </Button>
              
              <Button
                onClick={handleSubmitWorkflow}
                disabled={isCreating || !workflowForm.name || !workflowForm.description}
                className="bg-gradient-to-r from-orange-500 to-pink-500 hover:from-orange-600 hover:to-pink-600 text-white"
              >
                {isCreating ? (
                  <>
                    <Bot className="w-4 h-4 mr-2 animate-spin" />
                    Creating Workflow...
                  </>
                ) : (
                  <>
                    <Zap className="w-4 h-4 mr-2" />
                    Create & Deploy
                  </>
                )}
              </Button>
            </div>
          </div>
        </DialogContent>
      </Dialog>

    </div>
  )
}
