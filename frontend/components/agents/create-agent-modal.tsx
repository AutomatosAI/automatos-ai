
'use client'

import { useState, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import {
  X,
  Bot,
  ChevronDown,
  ChevronUp,
  User,
  PenLine,
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
import { apiClient } from '@/lib/api-client'

// Persona types
interface PersonaItem {
  id: string
  slug: string
  name: string
  description?: string
  system_prompt?: string
  voice_description?: string
  category?: string
  suggested_temperature: number
  scope: string
}

type PersonaMode = 'none' | 'predefined' | 'custom'

interface CreateAgentModalProps {
  open: boolean
  onClose: () => void
  onSuccess: () => void
}

import { AGENT_CATEGORIES, CATEGORY_TO_DB_MAP } from '@/lib/agent-constants'

// Skills will be loaded dynamically from the API

export function CreateAgentModal({ open, onClose, onSuccess }: CreateAgentModalProps) {
  const [step, setStep] = useState(1)
  const [agentData, setAgentData] = useState({
    name: '',
    category: '',  // Changed from 'type' to 'category'
    description: '',
    tags: '',
    skills: [] as string[],
    tools: [] as number[],
    specializations: [] as string[],
    // Marketplace field - just the toggle
    shareToMarketplace: false
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

  // Persona state (US-021)
  const [personaMode, setPersonaMode] = useState<PersonaMode>('none')
  const [selectedPersonaId, setSelectedPersonaId] = useState<string | null>(null)
  const [customPersonaPrompt, setCustomPersonaPrompt] = useState('')
  const [personas, setPersonas] = useState<PersonaItem[]>([])
  const [personasLoading, setPersonasLoading] = useState(false)
  const [personaCategoryFilter, setPersonaCategoryFilter] = useState<string>('all')
  const [expandedPersonaId, setExpandedPersonaId] = useState<string | null>(null)

  // API hooks
  const createAgentMutation = useCreateAgent()
  const { data: availableSkillsData = [], isLoading: skillsLoading } = useSkills()
  const { data: toolsResponse, isLoading: toolsLoading } = useTools({ status: 'active', limit: 100 })
  const availableTools = toolsResponse?.data || []
  const { data: models = [], isLoading: modelsLoading } = useModels()
  const updateModelConfigMutation = useUpdateAgentModelConfig()

  // Fetch personas when modal opens (US-021)
  useEffect(() => {
    if (!open) return
    setPersonasLoading(true)
    apiClient.request<any>('/api/personas')
      .then((data) => {
        const list = Array.isArray(data) ? data : (data?.personas || data?.data || [])
        setPersonas(list)
      })
      .catch((err) => {
        console.error('Failed to fetch personas:', err)
        setPersonas([])
      })
      .finally(() => setPersonasLoading(false))
  }, [open])

  // Pre-fill custom prompt when switching from predefined to custom (US-021)
  useEffect(() => {
    if (personaMode === 'custom' && selectedPersonaId && !customPersonaPrompt) {
      const persona = personas.find(p => p.id === selectedPersonaId)
      if (persona?.system_prompt) {
        setCustomPersonaPrompt(persona.system_prompt)
      }
    }
  }, [personaMode, selectedPersonaId, personas, customPersonaPrompt])

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

    if (!agentData.name || !agentData.category) {
      console.error('❌ Validation failed:', { name: agentData.name, category: agentData.category })
      toast.error('Please provide agent name and type')
      return
    }

    // No additional validation needed - uses same fields as main agent

    console.log('✅ Validation passed, creating agent...')
    try {
      // Prepare agent payload matching backend API expectations
      const tags = agentData.tags
        .split(',')
        .map(tag => tag.trim())
        .filter(Boolean)

      // Convert category name to database agent_type value
      const dbAgentType = CATEGORY_TO_DB_MAP[agentData.category] || 'custom'

      const agentPayload = {
        name: agentData.name,
        agent_type: dbAgentType,
        marketplace_category: agentData.category, // Preserve original UI category for round-trip
        description: agentData.description || '',
        skill_ids: agentData.skills, // Backend expects skill_ids array
        tool_ids: agentData.tools, // Agent app assignments (Composio apps)
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

        // US-021: Set persona if selected
        if (personaMode !== 'none') {
          try {
            const personaPayload: any = { use_custom: false }
            if (personaMode === 'predefined' && selectedPersonaId) {
              personaPayload.persona_id = selectedPersonaId
            } else if (personaMode === 'custom' && customPersonaPrompt) {
              personaPayload.custom_prompt = customPersonaPrompt
              personaPayload.use_custom = true
            }
            if (personaPayload.persona_id || personaPayload.custom_prompt) {
              await apiClient.request(`/api/agents/${newAgent.id}/persona`, {
                method: 'PUT',
                body: personaPayload,
              })
              console.log('Persona set successfully')
            }
          } catch (error) {
            console.error('Failed to set persona:', error)
          }
        }
      }

      toast.success(`Agent "${agentData.name}" created successfully!`)

      // Submit to marketplace if enabled - uses same fields as main agent
      if (agentData.shareToMarketplace && newAgent?.id) {
        try {
          const marketplacePayload = {
            type: 'agent',
            name: agentData.name,
            description: agentData.description,
            creator_name: 'You', // Backend will set actual username
            category: agentData.category,
            tags: tags,  // Use same tags as main agent
            metadata: {
              agent_id: newAgent.id,
              agent_type: agentData.category,
              skills: agentData.skills,
              tools: agentData.tools
            }
          }

          console.log('Submitting agent to marketplace:', marketplacePayload)
          const response = await fetch('/api/marketplace/submit', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(marketplacePayload)
          })

          if (response.ok) {
            toast.success('Agent submitted to marketplace for approval!')
          } else {
            const error = await response.json()
            console.error('Marketplace submission failed:', error)
            toast.error('Agent created but marketplace submission failed')
          }
        } catch (error) {
          console.error('Marketplace submission error:', error)
          toast.error('Agent created but marketplace submission failed')
        }
      }

      // Notify parent component and close modal
      onSuccess()
      onClose()

      // Reset form
      setAgentData({
        name: '',
        category: '',
        description: '',
        tags: '',
        skills: [],
        tools: [],
        specializations: [],
        shareToMarketplace: false
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
      setPersonaMode('none')
      setSelectedPersonaId(null)
      setCustomPersonaPrompt('')
      setPersonaCategoryFilter('all')
      setExpandedPersonaId(null)
      setStep(1)
    } catch (error: any) {
      console.error('❌ CREATE AGENT ERROR:', error)
      const errorMsg = error?.response?.data?.detail || error?.message || JSON.stringify(error)
      console.error('Error details:', errorMsg)
      alert('FAILED TO CREATE: ' + errorMsg)
      toast.error('Failed: ' + errorMsg)
    }
  }

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
                      1. Config
                    </TabsTrigger>
                    <TabsTrigger value="step-2" disabled={step < 2}>
                      2. Persona
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

                  {/* Step 1: Configuration */}
                  <TabsContent value="step-1" className="space-y-6 max-h-[50vh] overflow-y-auto">
                    <div>
                      <h3 className="text-lg font-semibold mb-2">Agent Configuration</h3>
                      <p className="text-muted-foreground mb-6">
                        Configure your agent's basic information
                      </p>
                    </div>

                    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                      <div className="space-y-4">
                        <div>
                          <Label htmlFor="agent-category">Category <span className="text-red-500">*</span></Label>
                          <Select
                            value={agentData.category}
                            onValueChange={(value) => setAgentData(prev => ({ ...prev, category: value }))}
                          >
                            <SelectTrigger id="agent-category" className="bg-secondary/50">
                              {agentData.category ? (
                                <div className="flex items-center gap-2">
                                  {(() => {
                                    const selected = AGENT_CATEGORIES.find(c => c.id === agentData.category)
                                    if (!selected) return <SelectValue placeholder="Select category..." />
                                    const Icon = selected.icon
                                    return (
                                      <>
                                        <Icon className={`w-4 h-4 ${selected.color}`} />
                                        <span>{selected.name}</span>
                                      </>
                                    )
                                  })()}
                                </div>
                              ) : (
                                <SelectValue placeholder="Select category..." />
                              )}
                            </SelectTrigger>
                            <SelectContent>
                              {AGENT_CATEGORIES.map(cat => {
                                const Icon = cat.icon
                                return (
                                  <SelectItem key={cat.id} value={cat.id}>
                                    <div className="flex items-center gap-2">
                                      <Icon className={`w-4 h-4 ${cat.color}`} />
                                      <span>{cat.name}</span>
                                    </div>
                                  </SelectItem>
                                )
                              })}
                            </SelectContent>
                          </Select>
                        </div>

                        <div>
                          <Label htmlFor="agent-name">Agent Name <span className="text-red-500">*</span></Label>
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
                            Lightweight keywords that describe the agent's strengths.
                          </p>
                        </div>
                      </div>

                      <div className="space-y-4">
                        {/* US-006: Marketplace Sharing */}
                        <div className="flex items-center justify-between mb-4">
                        <div>
                          <Label htmlFor="share-marketplace" className="text-base font-medium">
                            Share to <span className="text-orange-500">Marketplace</span>
                          </Label>
                          <p className="text-sm text-muted-foreground">
                            Make this agent available for others to discover and install
                          </p>
                        </div>
                        <Switch
                          id="share-marketplace"
                          checked={agentData.shareToMarketplace}
                          onCheckedChange={(checked) => setAgentData(prev => ({ ...prev, shareToMarketplace: checked }))}
                        />
                      </div>

                      {agentData.shareToMarketplace && (
                        <motion.div
                          initial={{ opacity: 0, height: 0 }}
                          animate={{ opacity: 1, height: 'auto' }}
                          exit={{ opacity: 0, height: 0 }}
                          className="mt-4"
                        >
                          <div className="bg-orange-500/10 border border-orange-500/30 rounded-lg p-3">
                            <p className="text-sm text-orange-200">
                              <strong>Note:</strong> Your agent will be submitted to the approval queue using the same name, description, category, and tags. Trusted users' submissions are auto-published.
                            </p>
                          </div>
                        </motion.div>
                      )}
                      </div>
                    </div>
                  </TabsContent>

                  {/* Step 2: Persona Selection (US-021) */}
                  <TabsContent value="step-2" className="space-y-6 max-h-[50vh] overflow-y-auto">
                    <div>
                      <h3 className="text-lg font-semibold mb-2">Agent Persona</h3>
                      <p className="text-muted-foreground mb-6">
                        Give your agent a personality and voice
                      </p>
                    </div>

                    {/* Persona Mode Selection */}
                    <div className="grid grid-cols-3 gap-3">
                      <motion.div
                        className={`p-4 rounded-lg border cursor-pointer transition-all text-center ${
                          personaMode === 'none'
                            ? 'border-primary bg-primary/10'
                            : 'border-border/50 hover:border-primary/30'
                        }`}
                        onClick={() => {
                          setPersonaMode('none')
                          setSelectedPersonaId(null)
                          setCustomPersonaPrompt('')
                        }}
                        whileHover={{ scale: 1.02 }}
                        whileTap={{ scale: 0.98 }}
                      >
                        <Bot className="w-6 h-6 mx-auto mb-2 text-muted-foreground" />
                        <div className="font-medium text-sm">No Persona</div>
                        <div className="text-xs text-muted-foreground mt-1">Default behavior</div>
                      </motion.div>

                      <motion.div
                        className={`p-4 rounded-lg border cursor-pointer transition-all text-center ${
                          personaMode === 'predefined'
                            ? 'border-primary bg-primary/10'
                            : 'border-border/50 hover:border-primary/30'
                        }`}
                        onClick={() => setPersonaMode('predefined')}
                        whileHover={{ scale: 1.02 }}
                        whileTap={{ scale: 0.98 }}
                      >
                        <User className="w-6 h-6 mx-auto mb-2 text-muted-foreground" />
                        <div className="font-medium text-sm">Predefined</div>
                        <div className="text-xs text-muted-foreground mt-1">Choose a persona</div>
                      </motion.div>

                      <motion.div
                        className={`p-4 rounded-lg border cursor-pointer transition-all text-center ${
                          personaMode === 'custom'
                            ? 'border-primary bg-primary/10'
                            : 'border-border/50 hover:border-primary/30'
                        }`}
                        onClick={() => setPersonaMode('custom')}
                        whileHover={{ scale: 1.02 }}
                        whileTap={{ scale: 0.98 }}
                      >
                        <PenLine className="w-6 h-6 mx-auto mb-2 text-muted-foreground" />
                        <div className="font-medium text-sm">Custom</div>
                        <div className="text-xs text-muted-foreground mt-1">Write your own</div>
                      </motion.div>
                    </div>

                    {/* Predefined Persona Selection */}
                    {personaMode === 'predefined' && (
                      <motion.div
                        initial={{ opacity: 0, height: 0 }}
                        animate={{ opacity: 1, height: 'auto' }}
                        exit={{ opacity: 0, height: 0 }}
                        className="space-y-4"
                      >
                        {/* Category Filter */}
                        <div>
                          <Label>Filter by Category</Label>
                          <Select
                            value={personaCategoryFilter}
                            onValueChange={setPersonaCategoryFilter}
                          >
                            <SelectTrigger className="bg-secondary/50">
                              <SelectValue placeholder="All categories" />
                            </SelectTrigger>
                            <SelectContent>
                              <SelectItem value="all">All Categories</SelectItem>
                              {[...new Set(personas.map(p => p.category).filter(Boolean))].map(cat => (
                                <SelectItem key={cat} value={cat!}>{cat}</SelectItem>
                              ))}
                            </SelectContent>
                          </Select>
                        </div>

                        {/* Persona List */}
                        {personasLoading ? (
                          <div className="space-y-2">
                            {[1, 2, 3].map(i => (
                              <div key={i} className="h-16 bg-secondary/20 animate-pulse rounded-lg" />
                            ))}
                          </div>
                        ) : (
                          <div className="space-y-2 max-h-[30vh] overflow-y-auto pr-1">
                            {personas
                              .filter(p => personaCategoryFilter === 'all' || p.category === personaCategoryFilter)
                              .map(persona => (
                                <motion.div
                                  key={persona.id}
                                  className={`p-3 rounded-lg border cursor-pointer transition-all ${
                                    selectedPersonaId === persona.id
                                      ? 'border-primary bg-primary/10'
                                      : 'border-border/50 hover:border-primary/30'
                                  }`}
                                  onClick={() => setSelectedPersonaId(persona.id)}
                                  whileHover={{ scale: 1.01 }}
                                  whileTap={{ scale: 0.99 }}
                                >
                                  <div className="flex items-center justify-between">
                                    <div className="flex-1 min-w-0">
                                      <div className="flex items-center gap-2">
                                        <span className="font-medium">{persona.name}</span>
                                        {persona.category && (
                                          <Badge variant="outline" className="text-xs">{persona.category}</Badge>
                                        )}
                                      </div>
                                      {persona.voice_description && (
                                        <p className="text-xs text-muted-foreground mt-1 truncate">
                                          {persona.voice_description}
                                        </p>
                                      )}
                                      <p className="text-xs text-muted-foreground mt-0.5">
                                        Temperature: {persona.suggested_temperature}
                                      </p>
                                    </div>
                                    <div className="flex items-center gap-2 ml-2">
                                      {selectedPersonaId === persona.id && (
                                        <Badge className="bg-primary text-primary-foreground">Selected</Badge>
                                      )}
                                      <Button
                                        variant="ghost"
                                        size="sm"
                                        onClick={(e) => {
                                          e.stopPropagation()
                                          setExpandedPersonaId(
                                            expandedPersonaId === persona.id ? null : persona.id
                                          )
                                        }}
                                      >
                                        {expandedPersonaId === persona.id ? (
                                          <ChevronUp className="w-4 h-4" />
                                        ) : (
                                          <ChevronDown className="w-4 h-4" />
                                        )}
                                      </Button>
                                    </div>
                                  </div>

                                  {/* Expandable System Prompt Preview */}
                                  {expandedPersonaId === persona.id && persona.system_prompt && (
                                    <motion.div
                                      initial={{ opacity: 0, height: 0 }}
                                      animate={{ opacity: 1, height: 'auto' }}
                                      className="mt-3 pt-3 border-t border-border/30"
                                    >
                                      <Label className="text-xs">System Prompt</Label>
                                      <pre className="text-xs text-muted-foreground mt-1 whitespace-pre-wrap bg-secondary/30 rounded p-2 max-h-[150px] overflow-y-auto">
                                        {persona.system_prompt}
                                      </pre>
                                    </motion.div>
                                  )}
                                </motion.div>
                              ))}
                            {personas.filter(p => personaCategoryFilter === 'all' || p.category === personaCategoryFilter).length === 0 && (
                              <div className="text-center py-6 text-muted-foreground">
                                No personas found{personaCategoryFilter !== 'all' ? ` in category "${personaCategoryFilter}"` : ''}.
                              </div>
                            )}
                          </div>
                        )}

                        {/* Tip: Switch to custom to pre-fill */}
                        {selectedPersonaId && (
                          <p className="text-xs text-muted-foreground italic">
                            Tip: Select a predefined persona and switch to &quot;Custom&quot; to pre-fill for editing.
                          </p>
                        )}
                      </motion.div>
                    )}

                    {/* Custom Persona */}
                    {personaMode === 'custom' && (
                      <motion.div
                        initial={{ opacity: 0, height: 0 }}
                        animate={{ opacity: 1, height: 'auto' }}
                        exit={{ opacity: 0, height: 0 }}
                        className="space-y-3"
                      >
                        <div>
                          <Label htmlFor="custom-persona">Custom Persona Prompt</Label>
                          <Textarea
                            id="custom-persona"
                            placeholder="Describe the agent's personality, communication style, expertise, and behavioral guidelines..."
                            value={customPersonaPrompt}
                            onChange={(e) => setCustomPersonaPrompt(e.target.value)}
                            className="bg-secondary/50 min-h-[200px] font-mono text-sm"
                          />
                          <p className="text-xs text-muted-foreground mt-1">
                            This prompt will be prepended to the agent&apos;s system message.
                          </p>
                        </div>
                      </motion.div>
                    )}

                    {/* Summary */}
                    <div className="flex justify-between items-center pt-4 border-t border-border/30">
                      <div>
                        <p className="font-medium">
                          {personaMode === 'none' && 'No persona selected'}
                          {personaMode === 'predefined' && (selectedPersonaId
                            ? `Persona: ${personas.find(p => p.id === selectedPersonaId)?.name || 'Selected'}`
                            : 'Select a persona above')}
                          {personaMode === 'custom' && (customPersonaPrompt
                            ? `Custom persona (${customPersonaPrompt.length} chars)`
                            : 'Write a custom persona above')}
                        </p>
                      </div>
                    </div>
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
                      agentType={agentData.category}
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
                      disabled={step === 1 && !agentData.category}
                      className="gradient-accent hover:opacity-90"
                    >
                      Next
                    </Button>
                  ) : (
                    <Button
                      onClick={handleCreate}
                      disabled={!agentData.name || !agentData.category || (createAgentMutation as any).isLoading}
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
