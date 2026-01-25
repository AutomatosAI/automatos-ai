
'use client'

import { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import {
  X,
  Bot,
  Code,
  Shield,
  Zap,
  Database,
  FileText,
  BarChart,
  Settings
} from 'lucide-react'
import { toast } from 'react-hot-toast'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Textarea } from '@/components/ui/textarea'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Switch } from '@/components/ui/switch'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Slider } from '@/components/ui/slider'
import { useTools } from '@/hooks/use-tools-api'

// API hooks
import { useCreateAgent, useSkills } from '@/hooks/use-agent-api'
import { useModels, useUpdateAgentModelConfig } from '@/hooks/use-model-api'
import { ModelSelector } from './model-selector'
import { ToolLogo } from '@/components/ui/tool-logo'

interface CreateAgentModalProps {
  open: boolean
  onClose: () => void
  onSuccess: () => void
}

const agentTypes = [
  {
    type: 'code_architect',
    name: 'Code Architect',
    description: 'Specialized in code analysis, architecture design, and best practices',
    icon: Code,
    color: 'text-blue-400',
    skills: ['code_analysis', 'architecture_design', 'best_practices', 'refactoring']
  },
  {
    type: 'security_expert',
    name: 'Security Expert',
    description: 'Focused on security analysis, vulnerability detection, and compliance',
    icon: Shield,
    color: 'text-red-400',
    skills: ['vulnerability_scanning', 'threat_modeling', 'compliance_check', 'security_audit']
  },
  {
    type: 'performance_optimizer',
    name: 'Performance Optimizer',
    description: 'Optimizes system performance, identifies bottlenecks, and improves efficiency',
    icon: Zap,
    color: 'text-yellow-400',
    skills: ['performance_analysis', 'bottleneck_detection', 'optimization', 'profiling']
  },
  {
    type: 'data_analyst',
    name: 'Data Analyst',
    description: 'Processes data, generates insights, and creates analytical reports',
    icon: BarChart,
    color: 'text-purple-400',
    skills: ['data_processing', 'pattern_recognition', 'report_generation', 'visualization']
  },
  {
    type: 'infrastructure_manager',
    name: 'Infrastructure Manager',
    description: 'Manages deployment, scaling, and infrastructure operations',
    icon: Database,
    color: 'text-green-400',
    skills: ['deployment', 'scaling', 'monitoring', 'resource_management']
  },
  {
    type: 'custom',
    name: 'Custom Agent',
    description: 'Create a custom agent with specific skills and capabilities',
    icon: Settings,
    color: 'text-orange-400',
    skills: []
  }
]

// Skills will be loaded dynamically from the API

export function CreateAgentModal({ open, onClose, onSuccess }: CreateAgentModalProps) {
  const [step, setStep] = useState(1)
  const [agentData, setAgentData] = useState({
    name: '',
    type: '',
    description: '',
    tags: '',
    skills: [] as string[],
    tools: [] as number[],
    specializations: [] as string[],
    maxConcurrentTasks: 3,
    priority: 'medium',  // Backend expects: low, medium, high, critical
    autoStart: true
  })

  // PRD-15: Model configuration state
  const [modelConfig, setModelConfig] = useState({
    provider: 'openai',
    model_id: 'gpt-4',
    temperature: 0.7,
    max_tokens: 2000,
    top_p: 1.0,
    frequency_penalty: 0.0,
    presence_penalty: 0.0,
    fallback_model_id: null as string | null
  })

  // API hooks
  const createAgentMutation = useCreateAgent()
  const { data: availableSkillsData = [], isLoading: skillsLoading } = useSkills()
  const { data: toolsResponse, isLoading: toolsLoading } = useTools({ status: 'active', limit: 100 })
  const availableTools = toolsResponse?.data || []
  const { data: models = [], isLoading: modelsLoading } = useModels()
  const updateModelConfigMutation = useUpdateAgentModelConfig()

  const handleSkillToggle = (skill: string) => {
    setAgentData(prev => ({
      ...prev,
      skills: prev.skills.includes(skill)
        ? prev.skills.filter(s => s !== skill)
        : [...prev.skills, skill]
    }))
  }

  const handleToolToggle = (toolId: number) => {
    setAgentData(prev => ({
      ...prev,
      tools: prev.tools.includes(toolId)
        ? prev.tools.filter(t => t !== toolId)
        : [...prev.tools, toolId]
    }))
  }

  const handleModelConfigChange = (key: string, value: any) => {
    setModelConfig(prev => ({ ...prev, [key]: value }))
  }

  const handleCreate = async () => {
    console.log('🔥 CREATE AGENT CLICKED', { agentData, modelConfig })

    if (!agentData.name || !agentData.type) {
      console.error('❌ Validation failed:', { name: agentData.name, type: agentData.type })
      toast.error('Please provide agent name and type')
      return
    }

    console.log('✅ Validation passed, creating agent...')
    try {
      // Prepare agent payload matching backend API expectations
      const tags = agentData.tags
        .split(',')
        .map(tag => tag.trim())
        .filter(Boolean)

      const agentPayload = {
        name: agentData.name,
        agent_type: agentData.type,
        description: agentData.description || '',
        skill_ids: agentData.skills, // Backend expects skill_ids array
        tool_ids: agentData.tools, // Agent app assignments (Composio apps)
        priority_level: agentData.priority || 'medium', // Backend expects priority_level
        max_concurrent_tasks: agentData.maxConcurrentTasks || 3, // Backend expects snake_case
        auto_start: agentData.autoStart !== undefined ? agentData.autoStart : true, // Backend expects snake_case
        tags,
        configuration: {
          specializations: agentData.specializations || [],
          tags
        }
      }

      console.log('Creating agent with payload:', JSON.stringify(agentPayload, null, 2))
      alert('Creating agent: ' + agentData.name)

      // Create the agent
      const newAgent: any = await (createAgentMutation as any).mutateAsync(agentPayload)

      console.log('Agent created successfully:', newAgent)
      alert('Agent created! ID: ' + newAgent.id)

      // PRD-15: Update model configuration
      if (newAgent?.id) {
        try {
          console.log('Setting model config for agent:', newAgent.id)
          await (updateModelConfigMutation as any).mutateAsync({
            agentId: newAgent.id,
            modelConfig
          })
          console.log('Model config set successfully')
        } catch (error) {
          console.error('Failed to set model config:', error)
          // Don't alert, just log - non-critical
        }

      }

      toast.success(`Agent "${agentData.name}" created successfully!`)

      // Notify parent component and close modal
      onSuccess()
      onClose()

      // Reset form
      setAgentData({
        name: '',
        type: '',
        description: '',
        tags: '',
        tags: '',
        skills: [],
        tools: [],
        specializations: [],
        maxConcurrentTasks: 3,
        priority: 'medium',
        autoStart: true
      })
      setModelConfig({
        provider: 'openai',
        model_id: 'gpt-4',
        temperature: 0.7,
        max_tokens: 2000,
        top_p: 1.0,
        frequency_penalty: 0.0,
        presence_penalty: 0.0,
        fallback_model_id: null
      })
      setStep(1)
    } catch (error: any) {
      console.error('❌ CREATE AGENT ERROR:', error)
      const errorMsg = error?.response?.data?.detail || error?.message || JSON.stringify(error)
      console.error('Error details:', errorMsg)
      alert('FAILED TO CREATE: ' + errorMsg)
      toast.error('Failed: ' + errorMsg)
    }
  }

  const selectedType = agentTypes.find(type => type.type === agentData.type)

  return (
    <AnimatePresence>
      {open && (
        <>
          {/* Backdrop */}
          <motion.div
            className="fixed inset-0 bg-black/50 backdrop-blur-sm z-50"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={onClose}
          />

          {/* Modal */}
          <motion.div
            className="fixed inset-0 z-50 flex items-center justify-center p-4"
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.95 }}
            transition={{ duration: 0.2 }}
          >
            <Card className="glass-card w-full max-w-4xl max-h-[90vh] overflow-hidden">
              <CardHeader className="flex flex-row items-center justify-between">
                <CardTitle className="flex items-center space-x-2">
                  <Bot className="w-6 h-6" />
                  <span>Create New Agent</span>
                </CardTitle>
                <Button variant="ghost" size="icon" onClick={onClose}>
                  <X className="w-5 h-5" />
                </Button>
              </CardHeader>

              <CardContent className="pr-2">
                <Tabs value={`step-${step}`} className="space-y-6">
                  <TabsList className="grid w-full grid-cols-5 bg-secondary/50">
                    <TabsTrigger value="step-1" disabled={step < 1}>
                      1. Agent Type
                    </TabsTrigger>
                    <TabsTrigger value="step-2" disabled={step < 2}>
                      2. Config
                    </TabsTrigger>
                    <TabsTrigger value="step-3" disabled={step < 3}>
                      3. Model
                    </TabsTrigger>
                    <TabsTrigger value="step-4" disabled={step < 4}>
                      4. Tools
                    </TabsTrigger>
                    <TabsTrigger value="step-5" disabled={step < 5}>
                      5. Skills
                    </TabsTrigger>
                  </TabsList>

                  {/* Step 1: Agent Type Selection */}
                  <TabsContent value="step-1" className="space-y-6 max-h-[50vh] overflow-y-auto">
                    <div>
                      <h3 className="text-lg font-semibold mb-2">Choose Agent Type</h3>
                      <p className="text-muted-foreground mb-6">
                        Select the type of agent that best fits your needs
                      </p>
                    </div>

                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                      {agentTypes.map(type => (
                        <motion.div
                          key={type.type}
                          className={`p-4 rounded-lg border-2 cursor-pointer transition-all duration-200 ${agentData.type === type.type
                            ? 'border-primary bg-primary/5'
                            : 'border-border/50 hover:border-primary/30 hover:bg-secondary/20'
                            }`}
                          onClick={() => setAgentData(prev => ({ ...prev, type: type.type }))}
                          whileHover={{ scale: 1.02 }}
                          whileTap={{ scale: 0.98 }}
                        >
                          <div className="flex items-center space-x-3 mb-3">
                            <div className="w-10 h-10 rounded-lg bg-secondary/50 flex items-center justify-center">
                              <type.icon className={`w-5 h-5 ${type.color}`} />
                            </div>
                            <div>
                              <h4 className="font-semibold">{type.name}</h4>
                              <p className="text-xs text-muted-foreground">{type.type}</p>
                            </div>
                          </div>
                          <p className="text-sm text-muted-foreground mb-3">
                            {type.description}
                          </p>
                          {type.skills.length > 0 && (
                            <div className="flex flex-wrap gap-1">
                              {type.skills.slice(0, 3).map(skill => (
                                <Badge key={skill} variant="outline" className="text-xs">
                                  {skill.replace('_', ' ')}
                                </Badge>
                              ))}
                              {type.skills.length > 3 && (
                                <Badge variant="outline" className="text-xs">
                                  +{type.skills.length - 3} more
                                </Badge>
                              )}
                            </div>
                          )}
                        </motion.div>
                      ))}
                    </div>
                  </TabsContent>

                  {/* Step 2: Configuration */}
                  <TabsContent value="step-2" className="space-y-6 max-h-[50vh] overflow-y-auto">
                    <div>
                      <h3 className="text-lg font-semibold mb-2">Agent Configuration</h3>
                      <p className="text-muted-foreground mb-6">
                        Configure your agent's basic information and behavior
                      </p>
                    </div>

                    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                      <div className="space-y-4">
                        <div>
                          <Label htmlFor="agent-name">Agent Name</Label>
                          <Input
                            id="agent-name"
                            placeholder="Enter agent name..."
                            value={agentData.name}
                            onChange={(e) => setAgentData(prev => ({ ...prev, name: e.target.value }))}
                            className="bg-secondary/50"
                          />
                        </div>

                        <div>
                          <Label htmlFor="agent-description">Description</Label>
                          <Textarea
                            id="agent-description"
                            placeholder="Describe the agent's purpose and capabilities..."
                            value={agentData.description}
                            onChange={(e) => setAgentData(prev => ({ ...prev, description: e.target.value }))}
                            className="bg-secondary/50 min-h-[100px]"
                          />
                        </div>

                        <div>
                          <Label htmlFor="agent-tags">Tags (comma separated)</Label>
                          <Input
                            id="agent-tags"
                            placeholder="e.g. writing, pdf, research"
                            value={agentData.tags}
                            onChange={(e) => setAgentData(prev => ({ ...prev, tags: e.target.value }))}
                            className="bg-secondary/50"
                          />
                          <p className="text-xs text-muted-foreground mt-1">
                            Lightweight keywords used for semantic matching (e.g., writing, pdf, research).
                          </p>
                        </div>
                      </div>

                      <div className="space-y-4">
                        <div>
                          <Label htmlFor="priority">Priority Level</Label>
                          <Select
                            value={agentData.priority}
                            onValueChange={(value) => setAgentData(prev => ({ ...prev, priority: value }))}
                          >
                            <SelectTrigger className="bg-secondary/50">
                              <SelectValue />
                            </SelectTrigger>
                            <SelectContent>
                              <SelectItem value="low">Low</SelectItem>
                              <SelectItem value="medium">Medium</SelectItem>
                              <SelectItem value="high">High</SelectItem>
                              <SelectItem value="critical">Critical</SelectItem>
                            </SelectContent>
                          </Select>
                        </div>

                        <div>
                          <Label htmlFor="max-tasks">Max Concurrent Tasks</Label>
                          <Input
                            id="max-tasks"
                            type="number"
                            min="1"
                            max="10"
                            value={agentData.maxConcurrentTasks}
                            onChange={(e) => setAgentData(prev => ({
                              ...prev,
                              maxConcurrentTasks: parseInt(e.target.value) || 3
                            }))}
                            className="bg-secondary/50"
                          />
                        </div>

                        <div className="flex items-center justify-between">
                          <div>
                            <Label htmlFor="auto-start">Auto Start</Label>
                            <p className="text-sm text-muted-foreground">
                              Start the agent automatically after creation
                            </p>
                          </div>
                          <Switch
                            id="auto-start"
                            checked={agentData.autoStart}
                            onCheckedChange={(checked) => setAgentData(prev => ({ ...prev, autoStart: checked }))}
                          />
                        </div>
                      </div>
                    </div>

                    {selectedType && (
                      <Card className="bg-secondary/20">
                        <CardHeader>
                          <CardTitle className="flex items-center space-x-2 text-base">
                            <selectedType.icon className={`w-5 h-5 ${selectedType.color}`} />
                            <span>{selectedType.name} Preview</span>
                          </CardTitle>
                        </CardHeader>
                        <CardContent>
                          <p className="text-sm text-muted-foreground mb-3">
                            {selectedType.description}
                          </p>
                          {selectedType.skills.length > 0 && (
                            <div>
                              <p className="text-sm font-medium mb-2">Default Skills:</p>
                              <div className="flex flex-wrap gap-1">
                                {selectedType.skills.map(skill => (
                                  <Badge key={skill} variant="secondary" className="text-xs">
                                    {skill.replace('_', ' ')}
                                  </Badge>
                                ))}
                              </div>
                            </div>
                          )}
                        </CardContent>
                      </Card>
                    )}
                  </TabsContent>

                  {/* Step 3: Model Configuration (PRD-15) */}
                  <TabsContent value="step-3" className="space-y-6 max-h-[50vh] overflow-y-auto">
                    <div>
                      <h3 className="text-lg font-semibold mb-2">Model Configuration</h3>
                      <p className="text-muted-foreground mb-6">
                        Select and configure the LLM model for your agent
                      </p>
                    </div>

                    <ModelSelector
                      value={modelConfig.model_id}
                      onChange={(modelId) => {
                        const model = (models as any)?.find((m: any) => m.model_id === modelId)
                        handleModelConfigChange('model_id', modelId)
                        if (model) {
                          handleModelConfigChange('provider', model.provider)
                        }
                      }}
                      agentType={agentData.type}
                    />

                    <div className="space-y-4 pt-4 border-t">
                      <h4 className="font-medium">Model Parameters</h4>

                      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                        <div className="space-y-2">
                          <Label>Temperature: {modelConfig.temperature.toFixed(2)}</Label>
                          <Slider
                            min={0}
                            max={2}
                            step={0.01}
                            value={[modelConfig.temperature]}
                            onValueChange={([value]) => handleModelConfigChange('temperature', value)}
                            className="bg-secondary/50"
                          />
                          <p className="text-xs text-muted-foreground">
                            Controls randomness: 0 is focused, 2 is creative
                          </p>
                        </div>

                        <div className="space-y-2">
                          <Label>Max Output Tokens: {modelConfig.max_tokens}</Label>
                          <Slider
                            min={100}
                            max={8000}
                            step={100}
                            value={[modelConfig.max_tokens]}
                            onValueChange={([value]) => handleModelConfigChange('max_tokens', value)}
                            className="bg-secondary/50"
                          />
                        </div>

                        <div className="space-y-2">
                          <Label>Top P: {modelConfig.top_p.toFixed(2)}</Label>
                          <Slider
                            min={0}
                            max={1}
                            step={0.01}
                            value={[modelConfig.top_p]}
                            onValueChange={([value]) => handleModelConfigChange('top_p', value)}
                            className="bg-secondary/50"
                          />
                          <p className="text-xs text-muted-foreground">
                            Nucleus sampling threshold
                          </p>
                        </div>

                        <div className="space-y-2">
                          <Label>Frequency Penalty: {modelConfig.frequency_penalty.toFixed(2)}</Label>
                          <Slider
                            min={-2}
                            max={2}
                            step={0.01}
                            value={[modelConfig.frequency_penalty]}
                            onValueChange={([value]) => handleModelConfigChange('frequency_penalty', value)}
                            className="bg-secondary/50"
                          />
                          <p className="text-xs text-muted-foreground">
                            Reduces repetition of token sequences
                          </p>
                        </div>

                        <div className="space-y-2">
                          <Label>Presence Penalty: {modelConfig.presence_penalty.toFixed(2)}</Label>
                          <Slider
                            min={-2}
                            max={2}
                            step={0.01}
                            value={[modelConfig.presence_penalty]}
                            onValueChange={([value]) => handleModelConfigChange('presence_penalty', value)}
                            className="bg-secondary/50"
                          />
                          <p className="text-xs text-muted-foreground">
                            Encourages new topics
                          </p>
                        </div>

                        <div className="space-y-2">
                          <Label>Fallback Model (Optional)</Label>
                          <Select
                            value={modelConfig.fallback_model_id || 'none'}
                            onValueChange={(value) => handleModelConfigChange('fallback_model_id', value === 'none' ? null : value)}
                          >
                            <SelectTrigger className="bg-secondary/50">
                              <SelectValue placeholder="No fallback" />
                            </SelectTrigger>
                            <SelectContent>
                              <SelectItem value="none">No fallback</SelectItem>
                              {(models as any)?.filter((m: any) => m.model_id !== modelConfig.model_id).map((model: any) => (
                                <SelectItem key={model.model_id} value={model.model_id}>
                                  {model.display_name}
                                </SelectItem>
                              ))}
                            </SelectContent>
                          </Select>
                          <p className="text-xs text-muted-foreground">
                            Model to use if primary fails
                          </p>
                        </div>
                      </div>
                    </div>
                  </TabsContent>

                  {/* Step 4: Tool Selection */}
                  <TabsContent value="step-4" className="space-y-6 max-h-[50vh] overflow-y-auto">
                    <div>
                      <h3 className="text-lg font-semibold mb-2">Select Tools</h3>
                      <p className="text-muted-foreground mb-6">
                        Choose the tools this agent can use
                      </p>
                    </div>

                    {toolsLoading ? (
                      <div className="grid grid-cols-2 gap-4">
                        <div className="h-10 bg-secondary/20 animate-pulse rounded" />
                        <div className="h-10 bg-secondary/20 animate-pulse rounded" />
                      </div>
                    ) : availableTools.length === 0 ? (
                      <div className="text-center py-8 text-muted-foreground">
                        No active tools found. Enable tools in the Tools Dashboard first.
                      </div>
                    ) : (
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                        {(availableTools as any[]).map((tool: any) => {
                          const actualToolId = tool.id

                          return (
                            <motion.div
                              key={tool.id}
                              className={`p-3 rounded-lg border cursor-pointer flex items-center justify-between transition-all ${agentData.tools.includes(actualToolId)
                                ? 'border-primary bg-primary/10'
                                : 'border-border/50 hover:border-primary/30'
                                }`}
                              onClick={() => handleToolToggle(actualToolId)}
                              whileHover={{ scale: 1.01 }}
                              whileTap={{ scale: 0.99 }}
                            >
                              <div className="flex items-center gap-3">
                                <div className="flex items-center justify-center">
                                  <ToolLogo
                                    name={tool.name}
                                    logo={tool.icon}
                                    size={32}
                                  />
                                </div>
                                <div>
                                  <div className="font-medium">{tool.name}</div>
                                  <div className="text-xs text-muted-foreground">{tool.provider}</div>
                                </div>
                              </div>
                              {agentData.tools.includes(actualToolId) && (
                                <Badge className="bg-primary text-primary-foreground">Selected</Badge>
                              )}
                            </motion.div>
                          )
                        })}
                      </div>
                    )}
                  </TabsContent>

                  {/* Step 5: Skills & Settings */}
                  <TabsContent value="step-5" className="space-y-6 max-h-[50vh] overflow-y-auto">
                    <div>
                      <h3 className="text-lg font-semibold mb-2">Skills & Advanced Settings</h3>
                      <p className="text-muted-foreground mb-6">
                        Customize the agent's skills and advanced configuration
                      </p>
                    </div>

                    <div>
                      <Label className="text-base font-medium">Available Skills</Label>
                      <p className="text-sm text-muted-foreground mb-4">
                        Select the skills your agent should possess
                      </p>
                      {skillsLoading ? (
                        <div className="space-y-2">
                          <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-2">
                            {[1, 2, 3, 4, 5, 6].map((i) => (
                              <div key={i} className="p-3 rounded-lg border border-border/50 animate-pulse bg-secondary/20">
                                <div className="h-5 w-20 bg-secondary rounded" />
                              </div>
                            ))}
                          </div>
                        </div>
                      ) : (
                        <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-2">
                          {availableSkillsData.map((skill) => (
                            <motion.div
                              key={skill.id}
                              className={`p-3 rounded-lg border cursor-pointer transition-all duration-200 ${agentData.skills.includes(skill.id)
                                ? 'border-primary bg-primary/10'
                                : 'border-border/50 hover:border-primary/30'
                                }`}
                              onClick={() => handleSkillToggle(skill.id)}
                              whileHover={{ scale: 1.02 }}
                              whileTap={{ scale: 0.98 }}
                            >
                              <span className="text-sm font-medium">
                                {skill.name}
                              </span>
                            </motion.div>
                          ))}
                        </div>
                      )}
                    </div>

                    <div className="flex justify-between items-center pt-6 border-t border-border/30">
                      <div>
                        <p className="font-medium">Selected Skills: {agentData.skills.length}</p>
                        <p className="text-sm text-muted-foreground">
                          Agent will be created with these capabilities
                        </p>
                      </div>
                    </div>
                  </TabsContent>
                </Tabs>

                {/* Navigation */}
                <div className="flex justify-between items-center mt-8 pt-6 border-t border-border/30">
                  <Button
                    variant="outline"
                    onClick={() => setStep(Math.max(1, step - 1))}
                    disabled={step === 1}
                  >
                    Previous
                  </Button>

                  <div className="text-sm text-muted-foreground">
                    Step {step} of 5
                  </div>

                  {step < 5 ? (
                    <Button
                      onClick={() => setStep(Math.min(5, step + 1))}
                      disabled={step === 1 && !agentData.type}
                      className="gradient-accent hover:opacity-90"
                    >
                      Next
                    </Button>
                  ) : (
                    <Button
                      onClick={handleCreate}
                      disabled={!agentData.name || !agentData.type || (createAgentMutation as any).isLoading}
                      className="gradient-accent hover:opacity-90"
                    >
                      {(createAgentMutation as any).isLoading ? 'Creating...' : 'Create Agent'}
                    </Button>
                  )}
                </div>
              </CardContent>
            </Card>
          </motion.div>
        </>
      )}
    </AnimatePresence>
  )
}
