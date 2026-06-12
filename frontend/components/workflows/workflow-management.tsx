
'use client'

import { useState, useEffect, useCallback } from 'react'
import { useSearchParams } from 'next/navigation'
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
import { PageHeader } from '@/components/shared/page-header'
import { StatsBar, type StatItem } from '@/components/shared/stats-bar'
import { SearchInput } from '@/components/shared/search-input'
import { FilterTabs, TabsContent as SharedTabsContent } from '@/components/shared/filter-tabs'
import { Dialog, DialogContent, DialogHeader, DialogTitle } from '@/components/ui/dialog'
import { Label } from '@/components/ui/label'
import { Textarea } from '@/components/ui/textarea'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Checkbox } from '@/components/ui/checkbox'
import { workflowService, type WorkflowWithMetrics, type WorkflowStats } from '@/lib/workflow-service'
import { apiClient } from '@/lib/api-client'
import { ViewToggle } from '@/components/shared/view-toggle'
import { useViewMode } from '@/hooks/use-view-mode'
import { ActiveWorkflowsPanel } from './active-workflows-panel'
import { toast } from 'sonner'
import { HistoryTab } from './history-tab'
// MonitoringTab removed — analytics consolidated into /analytics
import { PlaybooksTab } from './playbooks-tab'
import { ExecutionKitchen } from './execution-kitchen'

// Real data will be loaded from backend
const initialWorkflowStats = [
  {
    label: 'Cooking Now',
    value: '0',
    change: 'Loading...',
    icon: GitBranch,
    color: 'text-[hsl(var(--info))]'
  },
  {
    label: 'Completed Today',
    value: '0',
    change: 'Loading...',
    icon: CheckCircle,
    color: 'text-[hsl(var(--success))]'
  },
  {
    label: 'Avg Duration',
    value: '0h',
    change: 'Loading...',
    icon: Clock,
    color: 'text-primary'
  },
  {
    label: 'Agent Utilization',
    value: '0%',
    change: 'Loading...',
    icon: Activity,
    color: 'text-[hsl(var(--agent))]'
  }
]

const statusStyles: Record<string, string> = {
  draft: 'bg-muted/10 text-muted-foreground border-muted/20',
  active: 'bg-[hsl(var(--info))]/10 text-[hsl(var(--info))] border-[hsl(var(--info))]/20',
  archived: 'bg-[hsl(var(--success))]/10 text-[hsl(var(--success))] border-[hsl(var(--success))]/20',
  // Legacy status support for UI display
  running: 'bg-[hsl(var(--info))]/10 text-[hsl(var(--info))] border-[hsl(var(--info))]/20',
  completed: 'bg-[hsl(var(--success))]/10 text-[hsl(var(--success))] border-[hsl(var(--success))]/20',
  paused: 'bg-[hsl(var(--warning))]/10 text-[hsl(var(--warning))] border-[hsl(var(--warning))]/20',
  failed: 'bg-[hsl(var(--destructive))]/10 text-[hsl(var(--destructive))] border-[hsl(var(--destructive))]/20',
  queued: 'bg-muted/10 text-muted-foreground border-muted/20'
}

const priorityStyles: Record<string, string> = {
  low: 'bg-muted/10 text-muted-foreground border-muted/20',
  medium: 'bg-[hsl(var(--info))]/10 text-[hsl(var(--info))] border-[hsl(var(--info))]/20',
  high: 'bg-primary/10 text-primary border-primary/20',
  critical: 'bg-[hsl(var(--destructive))]/10 text-[hsl(var(--destructive))] border-[hsl(var(--destructive))]/20'
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

// Removed hardcoded templates - all templates now come from database via workflow_templates table
const workflowTemplates: any[] = []

export function WorkflowManagement() {
  const queryClient = useQueryClient()
  const [searchTerm, setSearchTerm] = useState('')
  const [selectedStatus, setSelectedStatus] = useState('all')
  const [activeWorkflowTab, setActiveWorkflowTab] = useState('templates')
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
  const [showExecutionKitchen, setShowExecutionKitchen] = useState(false)
  const [selectedWorkflowId, setSelectedWorkflowId] = useState<number | null>(null)
  const [autoStartExecution, setAutoStartExecution] = useState(false)

  // View mode
  const [viewMode, setViewMode] = useViewMode('wf-recipes')

  // Recipe creation modal (triggered from top "Create Recipe" button)
  const [recipeCreateOpen, setRecipeCreateOpen] = useState(false)
  const [recipeSearchTerm, setRecipeSearchTerm] = useState('')

  // Recipe direct execution state
  const [recipeExecInfo, setRecipeExecInfo] = useState<{
    recipeExecutionId: string
    recipeId: string
    recipeSteps: Array<{ step_id: string; order: number; prompt_template: string; agent_id: number }>
    recipeName: string
  } | null>(null)

  // Deep-link from Activity Command Centre: /workflows?openExecution=X&recipeId=Y
  const searchParams = useSearchParams()
  useEffect(() => {
    const openExecution = searchParams.get('openExecution')
    const recipeIdParam = searchParams.get('recipeId')
    if (openExecution && recipeIdParam) {
      setRecipeExecInfo({
        recipeExecutionId: openExecution,
        recipeId: recipeIdParam,
        recipeSteps: [],
        recipeName: '',
      })
      setSelectedWorkflowId(null)
      setAutoStartExecution(false)
      setShowExecutionKitchen(true)
    }
  }, [searchParams])

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
          label: 'Cooking Now',
          value: statsData.activeWorkflows.toString(),
          change: `${statsData.activeWorkflows > 0 ? '+' : ''}${statsData.activeWorkflows} total`,
          icon: GitBranch,
          color: 'text-[hsl(var(--info))]'
        },
        {
          label: 'Completed Today',
          value: statsData.completedToday.toString(),
          change: `${statsData.successRate}% success rate`,
          icon: CheckCircle,
          color: 'text-[hsl(var(--success))]'
        },
        {
          label: 'Avg Duration',
          value: statsData.avgDuration,
          change: 'Based on history',
          icon: Clock,
          color: 'text-primary'
        },
        {
          label: 'Agent Utilization',
          value: `${statsData.agentUtilization}%`,
          change: 'CPU usage based',
          icon: Activity,
          color: 'text-[hsl(var(--agent))]'
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

    // Normalize step objects: ensure integer agent_ids and proper structure
    const steps = Array.isArray(workflowForm.customSteps) && workflowForm.customSteps.length > 0
      ? workflowForm.customSteps.map((step: any, idx: number) => {
          if (typeof step === 'object' && step !== null) {
            return {
              ...step,
              agent_id: typeof step.agent_id === 'string' ? parseInt(step.agent_id, 10) : step.agent_id,
              order: step.order || idx + 1,
              step_id: step.step_id || `step_${idx + 1}`,
            }
          }
          return step // string steps pass through as-is
        })
      : workflowForm.customSteps

    // Prepare the workflow data in the expected backend API format
    const workflowData = {
      name: workflowForm.name,
      description: workflowForm.description,
      goal: workflowForm.goal || undefined, // Only include if provided
      context: Object.keys(contextData).length > 0 ? contextData : undefined, // Only include if not empty
      category: workflowForm.template ? workflowTemplates.find(t => t.id === workflowForm.template)?.name : 'automation',
      priority: workflowForm.priority,
      config: workflowForm.configuration,
      steps: steps,
      agents: workflowForm.selectedAgents,
      tags: workflowForm.template ? [workflowForm.template] : ['custom']
    }
    
    try {
      // Submit to backend API using the service
      const result = await workflowService.createWorkflow(workflowData)
      console.log('Workflow created successfully:', result)
      
      // Show success toast
      toast("✅ Workflow Created", { description: `"${workflowForm.name}" has been created successfully and is now active.` })

      // Reload workflow data to show the new workflow
      await loadWorkflowData()
      
      // Invalidate all workflow-related queries to ensure UI refreshes
      await Promise.all([
        queryClient.invalidateQueries({ queryKey: ['workflows'] }),
        queryClient.invalidateQueries({ queryKey: ['activeWorkflows'] }),
        queryClient.invalidateQueries({ queryKey: ['workflows', 'active'] }),
        queryClient.invalidateQueries({ queryKey: ['workflowStats'] })
      ])

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

      // Open Execution Theater for the new workflow WITHOUT auto-start
      // User can choose when to run it
      if (result && result.id) {
        setTimeout(() => {
          setSelectedWorkflowId(result.id)
          setAutoStartExecution(false) // Don't auto-start, let user decide
          setShowExecutionKitchen(true)
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
      
      // Show error toast
      toast.error("❌ Failed to Create Workflow", { description: errorMessage })
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
      // Fix: Read from template.steps (JSONB column), not template_definition.steps
      const recipeSteps = template.steps || template.template_definition?.steps || []
      const agentIdsFromSteps = recipeSteps
        .map((step: any) => step.agent_id?.toString())
        .filter(Boolean)
      setWorkflowForm({
        ...workflowForm,
        template: template.template_id || template.id,
        name: `${template.name} ${timestamp}`,
        description: template.description,
        priority: template.priority || 'medium',
        selectedAgents: agentIdsFromSteps.length > 0 ? agentIdsFromSteps : template.recommended_agents || [],
        customSteps: recipeSteps,
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
    setShowExecutionKitchen(true)
  }

  const handleBackFromKitchen = () => {
    setShowExecutionKitchen(false)
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
  if (showExecutionKitchen && (selectedWorkflowId || recipeExecInfo)) {
    return (
      <ExecutionKitchen
        workflowId={selectedWorkflowId || 0}
        onBack={handleBackFromKitchen}
        autoStart={autoStartExecution}
        executionType={recipeExecInfo ? 'recipe' : 'workflow'}
        recipeExecutionId={recipeExecInfo?.recipeExecutionId}
        recipeId={recipeExecInfo?.recipeId}
        recipeSteps={recipeExecInfo?.recipeSteps}
      />
    )
  }

  return (
    <div className="space-y-8">
      {/* Header */}
      <div data-tour="workflows-page-header">
      <PageHeader
        title="Workflow"
        titleAccent="Management"
        subtitle="Create, monitor, and manage your multi-agent workflows"
        actions={
          <Button
            variant="outline"
            data-tour="workflows-create-btn"
            onClick={() => setRecipeCreateOpen(true)}
          >
            <Plus className="w-4 h-4 mr-2" />
            Create Recipe
          </Button>
        }
      />
      </div>

      {/* Stats Overview */}
      <div ref={ref}>
        <StatsBar
          stats={workflowStats.map((stat) => ({
            label: stat.label,
            value: stat.value,
            change: stat.change,
            icon: stat.icon,
            iconColor: stat.color,
          }))}
          loading={loading}
        />
      </div>

      {/* Workflow Management */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={inView ? { opacity: 1, y: 0 } : {}}
        transition={{ duration: 0.8, delay: 0.4 }}
      >
        <Tabs defaultValue="templates" className="space-y-6">
          <div className="flex items-center gap-4">
            <TabsList className="bg-secondary/50 shrink-0">
              <TabsTrigger value="templates" data-tour="workflows-recipes-tab" className="flex items-center space-x-2">
                <GitBranch className="w-4 h-4" />
                <span className="hidden sm:inline">Playbooks</span>
              </TabsTrigger>
              <TabsTrigger value="active" data-tour="workflows-active-tab" className="flex items-center space-x-2">
                <Play className="w-4 h-4" />
                <span className="hidden sm:inline">Cooking</span>
              </TabsTrigger>
              {/* Monitoring tab removed — see /analytics */}
            </TabsList>
            <div className="relative flex-1">
              <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-muted-foreground w-4 h-4" />
              <Input
                placeholder="Search recipes..."
                value={recipeSearchTerm}
                onChange={(e) => setRecipeSearchTerm(e.target.value)}
                className="pl-10 bg-secondary/50 border-secondary"
              />
            </div>
            <ViewToggle value={viewMode} onChange={setViewMode} />
          </div>

          <SharedTabsContent value="active" className="space-y-6">
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
                  <AlertTriangle className="h-8 w-8 text-[hsl(var(--destructive))] mx-auto mb-4" />
                  <p className="text-[hsl(var(--destructive))] mb-4">Error loading workflows: {error}</p>
                  <Button onClick={loadWorkflowData} variant="outline">
                    Try Again
                  </Button>
                </div>
              </div>
            )}

            {/* Active Workflows Panel - handles its own empty state */}
            <ActiveWorkflowsPanel
              onWorkflowClick={handleWorkflowClick}
              onPlaybookRunClick={(run) => {
                if (!run.recipe_template_id) return
                setRecipeExecInfo({
                  recipeExecutionId: run.execution_id,
                  recipeId: run.recipe_template_id,
                  recipeSteps: (run.step_results || []).map((s: any) => ({
                    step_id: s.step_id,
                    order: s.order,
                    prompt_template: s.prompt_template || '',
                    agent_id: s.agent_id ?? 0,
                  })),
                  recipeName: run.recipe_name,
                })
                setSelectedWorkflowId(null)
                setAutoStartExecution(false)
                setShowExecutionKitchen(true)
              }}
            />
          </SharedTabsContent>

          <SharedTabsContent value="templates" className="space-y-6">
            <PlaybooksTab
              searchTerm={recipeSearchTerm}
              viewMode={viewMode}
              externalCreateOpen={recipeCreateOpen}
              onCreateModalClosed={() => setRecipeCreateOpen(false)}
              onUseRecipe={(recipeId) => {
                setError(null)
                handleTemplateChange(recipeId)
                setShowCreateModal(true)
              }}
              onExecuteRecipe={(workflowId: number, recipeExecution?: any) => {
                if (recipeExecution) {
                  setRecipeExecInfo(recipeExecution)
                  setSelectedWorkflowId(null)
                } else {
                  setSelectedWorkflowId(workflowId)
                  setRecipeExecInfo(null)
                }
                setAutoStartExecution(false)
                setShowExecutionKitchen(true)
              }}
            />
          </SharedTabsContent>

          {/* Monitoring tab removed — see /analytics */}
        </Tabs>
      </motion.div>

      {/* Cook Workflow Confirmation Modal */}
      <Dialog open={showCreateModal} onOpenChange={setShowCreateModal}>
        <DialogContent className="glass-card max-w-lg">
          <DialogHeader>
            <DialogTitle className="text-xl font-semibold">Cook Workflow?</DialogTitle>
          </DialogHeader>

          <div className="space-y-4 py-4">
            {/* Workflow Name - Read Only */}
            <div className="space-y-2">
              <Label className="text-sm text-muted-foreground">Playbook</Label>
              <div className="px-4 py-3 bg-secondary/50 rounded-lg border border-border/50">
                <p className="font-medium">{workflowForm.name || 'Untitled Workflow'}</p>
              </div>
            </div>

            {/* Description - Read Only */}
            <div className="space-y-2">
              <Label className="text-sm text-muted-foreground">Description</Label>
              <div className="px-4 py-3 bg-secondary/50 rounded-lg border border-border/50 min-h-[100px]">
                <p className="text-sm text-muted-foreground leading-relaxed">
                  {workflowForm.description || 'No description provided'}
                </p>
              </div>
            </div>
          </div>

          {/* Error Display */}
          {error && (
            <div className="p-3 bg-[hsl(var(--destructive))]/10 border border-[hsl(var(--destructive))]/50 rounded-lg">
              <p className="text-[hsl(var(--destructive))] text-sm">{error}</p>
            </div>
          )}

          {/* Actions */}
          <div className="flex gap-3 pt-4">
            <Button
              variant="outline"
              onClick={() => {
                setShowCreateModal(false)
                setError(null)
              }}
              disabled={isCreating}
              className="flex-1"
            >
              Cancel
            </Button>

            <Button
              onClick={handleSubmitWorkflow}
              disabled={isCreating || !workflowForm.name || !workflowForm.description}
              className="flex-1 bg-primary hover:bg-primary/90"
            >
              {isCreating ? (
                <>
                  <Bot className="w-4 h-4 mr-2 animate-spin" />
                  Cooking...
                </>
              ) : (
                <>
                  <Play className="w-4 h-4 mr-2" />
                  Cook
                </>
              )}
            </Button>
          </div>
        </DialogContent>
      </Dialog>

    </div>
  )
}
