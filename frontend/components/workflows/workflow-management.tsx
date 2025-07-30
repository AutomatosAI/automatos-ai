
'use client'

import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import { useInView } from 'react-intersection-observer'
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
  ChevronLeft
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
import { apiClient } from '@/lib/api'
import { ActiveWorkflowsPanel } from './active-workflows-panel'

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

  // Workflow creation form state
  const [workflowForm, setWorkflowForm] = useState({
    name: '',
    description: '',
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
    setWorkflowForm({
      name: '',
      description: '',
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
    try {
      // Get agent IDs from selected agent names
      const selectedAgentIds = availableAgents
        .filter(agent => workflowForm.selectedAgents.includes(agent.id))
        .map(agent => parseInt(agent.id))

      // Prepare the workflow data in the expected API format
      const workflowData = {
        name: workflowForm.name,
        description: workflowForm.description,
        workflow_definition: {
          template: workflowForm.template,
          priority: workflowForm.priority,
          agents: workflowForm.selectedAgents,
          steps: workflowForm.customSteps,
          configuration: workflowForm.configuration,
          category: workflowForm.template ? workflowTemplates.find(t => t.id === workflowForm.template)?.name : 'Custom',
          tags: workflowForm.template ? [workflowForm.template] : ['custom']
        },
        agent_ids: selectedAgentIds
      }

      // Submit to backend API using the service
      const result = await workflowService.createWorkflow(workflowData)
      console.log('Workflow created successfully:', result)

      // Reload workflow data to show the new workflow
      await loadWorkflowData()

      // Close modal and reset form
      setShowCreateModal(false)
      setWorkflowForm({
        name: '',
        description: '',
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

      // Show success message (you can add a toast notification here)
      alert('Workflow created successfully!')
      
    } catch (error) {
      console.error('Error creating workflow:', error)
      // Show error message (you can add a toast notification here)
      const errorMessage = error instanceof Error ? error.message : 'Unknown error occurred'
      alert(`Error creating workflow: ${errorMessage}`)
    } finally {
      setIsCreating(false)
    }
  }

  const handleTemplateChange = (templateId: string) => {
    const template = workflowTemplates.find(t => t.id === templateId)
    if (template) {
      setWorkflowForm({
        ...workflowForm,
        template: templateId,
        name: template.name,
        description: template.description,
        selectedAgents: template.agents,
        customSteps: template.steps
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

  const filteredWorkflows = workflows.filter(workflow => {
    const matchesSearch = (workflow.name && workflow.name.toLowerCase().includes(searchTerm.toLowerCase())) ||
      (workflow.description && workflow.description.toLowerCase().includes(searchTerm.toLowerCase())) ||
      (workflow.category && workflow.category.toLowerCase().includes(searchTerm.toLowerCase())) ||
      (workflow.tags && workflow.tags.some(tag => tag && tag.toLowerCase().includes(searchTerm.toLowerCase())))
    
    const matchesStatus = selectedStatus === 'all' || workflow.status === selectedStatus
    
    return matchesSearch && matchesStatus
  })

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
          className="gradient-accent hover:opacity-90 transition-opacity"
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
          <TabsList className="grid w-full grid-cols-4 lg:w-auto lg:inline-grid bg-secondary/50">
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

            {/* Active Workflows Panel */}
            <ActiveWorkflowsPanel />
          </TabsContent>

          <TabsContent value="templates" className="space-y-6">
            <Card className="glass-card">
              <CardHeader>
                <CardTitle>Workflow Templates</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="text-center text-muted-foreground">
                  Workflow templates will be displayed here
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="history" className="space-y-6">
            <Card className="glass-card">
              <CardHeader>
                <CardTitle>Workflow History</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="text-center text-muted-foreground">
                  Workflow execution history will be displayed here
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="monitoring" className="space-y-6">
            <Card className="glass-card">
              <CardHeader>
                <CardTitle>Real-time Monitoring</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="text-center text-muted-foreground">
                  Real-time workflow monitoring dashboard will be displayed here
                </div>
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
      </motion.div>

      {/* Create Workflow Modal */}
      <Dialog open={showCreateModal} onOpenChange={setShowCreateModal}>
        <DialogContent className="glass-card max-w-4xl max-h-[90vh] overflow-y-auto">
          <DialogHeader>
            <DialogTitle className="flex items-center justify-between">
              <span>Create New Workflow</span>
              <div className="flex items-center space-x-2 text-sm text-muted-foreground">
                <span>Step {currentStep} of 3</span>
                <div className="flex space-x-1">
                  {[1, 2, 3].map(step => (
                    <div
                      key={step}
                      className={`w-2 h-2 rounded-full ${
                        step <= currentStep ? 'bg-primary' : 'bg-secondary'
                      }`}
                    />
                  ))}
                </div>
              </div>
            </DialogTitle>
          </DialogHeader>

          <div className="space-y-6">
            {/* Step 1: Basic Information */}
            {currentStep === 1 && (
              <motion.div
                initial={{ opacity: 0, x: 20 }}
                animate={{ opacity: 1, x: 0 }}
                className="space-y-4"
              >
                <div className="space-y-4">
                  <div>
                    <Label htmlFor="template">Choose Template</Label>
                    <Select value={workflowForm.template || ""} onValueChange={handleTemplateChange}>
                      <SelectTrigger className="bg-secondary/50 border-secondary">
                        <SelectValue placeholder="Select a workflow template" />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="">Custom Workflow</SelectItem>
                        {workflowTemplates.map(template => (
                          <SelectItem key={template.id} value={template.id}>
                            {template.name}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </div>

                  <div>
                    <Label htmlFor="name">Workflow Name</Label>
                    <Input
                      id="name"
                      value={workflowForm.name}
                      onChange={(e) => setWorkflowForm({...workflowForm, name: e.target.value})}
                      placeholder="Enter workflow name"
                      className="bg-secondary/50 border-secondary"
                    />
                  </div>

                  <div>
                    <Label htmlFor="description">Description</Label>
                    <Textarea
                      id="description"
                      value={workflowForm.description}
                      onChange={(e) => setWorkflowForm({...workflowForm, description: e.target.value})}
                      placeholder="Describe what this workflow does"
                      className="bg-secondary/50 border-secondary min-h-[100px]"
                    />
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
                        <SelectItem value="medium">Medium</SelectItem>
                        <SelectItem value="high">High</SelectItem>
                        <SelectItem value="critical">Critical</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                </div>
              </motion.div>
            )}

            {/* Step 2: Agent Selection */}
            {currentStep === 2 && (
              <motion.div
                initial={{ opacity: 0, x: 20 }}
                animate={{ opacity: 1, x: 0 }}
                className="space-y-4"
              >
                <div>
                  <Label>Select Agents</Label>
                  <p className="text-sm text-muted-foreground mb-4">
                    Choose the agents that will participate in this workflow
                  </p>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    {availableAgents.map(agent => (
                      <div
                        key={agent.id}
                        className={`p-4 border rounded-lg cursor-pointer transition-all ${
                          workflowForm.selectedAgents.includes(agent.id)
                            ? 'border-primary bg-primary/5'
                            : 'border-border hover:border-primary/50'
                        }`}
                        onClick={() => toggleAgent(agent.id)}
                      >
                        <div className="flex items-start space-x-3">
                          <Checkbox
                            checked={workflowForm.selectedAgents.includes(agent.id)}
                            className="mt-1"
                          />
                          <div className="flex-1">
                            <h4 className="font-semibold">{agent.name}</h4>
                            <p className="text-sm text-muted-foreground mb-2">
                              {agent.description}
                            </p>
                            <div className="flex flex-wrap gap-1">
                              {agent.skills.map((skill: string) => (
                                <Badge key={skill} variant="outline" className="text-xs">
                                  {skill}
                                </Badge>
                              ))}
                            </div>
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              </motion.div>
            )}

            {/* Step 3: Configuration */}
            {currentStep === 3 && (
              <motion.div
                initial={{ opacity: 0, x: 20 }}
                animate={{ opacity: 1, x: 0 }}
                className="space-y-4"
              >
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <div className="space-y-4">
                    <h4 className="font-semibold">Execution Settings</h4>
                    
                    <div>
                      <Label htmlFor="maxRetries">Max Retries</Label>
                      <Input
                        id="maxRetries"
                        type="number"
                        value={workflowForm.configuration.maxRetries}
                        onChange={(e) => setWorkflowForm({
                          ...workflowForm,
                          configuration: {
                            ...workflowForm.configuration,
                            maxRetries: parseInt(e.target.value) || 0
                          }
                        })}
                        className="bg-secondary/50 border-secondary"
                      />
                    </div>

                    <div>
                      <Label htmlFor="timeout">Timeout (minutes)</Label>
                      <Input
                        id="timeout"
                        type="number"
                        value={workflowForm.configuration.timeout}
                        onChange={(e) => setWorkflowForm({
                          ...workflowForm,
                          configuration: {
                            ...workflowForm.configuration,
                            timeout: parseInt(e.target.value) || 0
                          }
                        })}
                        className="bg-secondary/50 border-secondary"
                      />
                    </div>

                    <div className="flex items-center space-x-2">
                      <Checkbox
                        id="parallel"
                        checked={workflowForm.configuration.parallelExecution}
                        onCheckedChange={(checked) => setWorkflowForm({
                          ...workflowForm,
                          configuration: {
                            ...workflowForm.configuration,
                            parallelExecution: Boolean(checked)
                          }
                        })}
                      />
                      <Label htmlFor="parallel">Enable parallel execution</Label>
                    </div>
                  </div>

                  <div className="space-y-4">
                    <h4 className="font-semibold">Workflow Summary</h4>
                    <div className="p-4 bg-secondary/30 rounded-lg space-y-2">
                      <div>
                        <span className="text-sm font-medium">Name:</span>
                        <span className="text-sm ml-2">{workflowForm.name || 'Untitled'}</span>
                      </div>
                      <div>
                        <span className="text-sm font-medium">Priority:</span>
                        <Badge className={`ml-2 ${priorityStyles[workflowForm.priority]}`}>
                          {workflowForm.priority}
                        </Badge>
                      </div>
                      <div>
                        <span className="text-sm font-medium">Agents:</span>
                        <span className="text-sm ml-2">{workflowForm.selectedAgents.length} selected</span>
                      </div>
                      <div className="space-y-1">
                        <span className="text-sm font-medium">Selected Agents:</span>
                        <div className="flex flex-wrap gap-1">
                          {workflowForm.selectedAgents.map(agentId => {
                            const agent = availableAgents.find(a => a.id === agentId)
                            return agent ? (
                              <Badge key={agentId} variant="outline" className="text-xs">
                                {agent.name}
                              </Badge>
                            ) : null
                          })}
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              </motion.div>
            )}

            {/* Modal Actions */}
            <div className="flex items-center justify-between pt-6 border-t border-border/30">
              <div className="flex space-x-2">
                {currentStep > 1 && (
                  <Button
                    variant="outline"
                    onClick={() => setCurrentStep(currentStep - 1)}
                  >
                    <ChevronLeft className="w-4 h-4 mr-2" />
                    Previous
                  </Button>
                )}
              </div>

              <div className="flex space-x-2">
                <Button
                  variant="outline"
                  onClick={() => setShowCreateModal(false)}
                >
                  Cancel
                </Button>
                
                {currentStep < 3 ? (
                  <Button
                    onClick={() => setCurrentStep(currentStep + 1)}
                    disabled={
                      (currentStep === 1 && !workflowForm.name) ||
                      (currentStep === 2 && workflowForm.selectedAgents.length === 0)
                    }
                  >
                    Next
                    <ChevronRight className="w-4 h-4 ml-2" />
                  </Button>
                ) : (
                  <Button
                    onClick={handleSubmitWorkflow}
                    disabled={isCreating}
                    className="gradient-accent hover:opacity-90"
                  >
                    {isCreating ? 'Creating...' : 'Create Workflow'}
                  </Button>
                )}
              </div>
            </div>
          </div>
        </DialogContent>
      </Dialog>
    </div>
  )
}
