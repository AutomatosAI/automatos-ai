
'use client'

import { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { 
  Bot,
  X,
  Settings,
  Zap,
  MessageSquare,
  FileText,
  Database,
  Code
} from 'lucide-react'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Textarea } from '@/components/ui/textarea'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'
import { Card, CardContent } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Slider } from '@/components/ui/slider'
import { Switch } from '@/components/ui/switch'
import { useCreateAgent } from '@/hooks/use-agent-api'

interface RealAgentCreateModalProps {
  open: boolean
  onClose: () => void
}

interface AgentTemplate {
  id: string
  name: string
  type: string
  description: string
  icon: React.ComponentType<any>
  config: {
    model: string
    temperature: number
    max_tokens: number
    capabilities: string[]
  }
  badge?: string
}

export function RealAgentCreateModal({ open, onClose }: RealAgentCreateModalProps) {
  const [step, setStep] = useState(1)
  const [selectedTemplate, setSelectedTemplate] = useState<AgentTemplate | null>(null)
  const [formData, setFormData] = useState({
    name: '',
    description: '',
    type: '',
    model: 'gpt-4',
    temperature: 0.7,
    max_tokens: 2000,
    auto_start: true,
    capabilities: [] as string[],
  })

  const createAgentMutation = useCreateAgent()

  // Agent Templates
  const agentTemplates: AgentTemplate[] = [
    {
      id: 'general',
      name: 'General Assistant',
      type: 'general',
      description: 'Multi-purpose AI agent for general tasks and conversations',
      icon: Bot,
      config: {
        model: 'gpt-4',
        temperature: 0.7,
        max_tokens: 2000,
        capabilities: ['text_generation', 'conversation', 'reasoning']
      }
    },
    {
      id: 'document_processor',
      name: 'Document Processor',
      type: 'document',
      description: 'Specialized in analyzing, summarizing, and processing documents',
      icon: FileText,
      config: {
        model: 'gpt-4',
        temperature: 0.3,
        max_tokens: 4000,
        capabilities: ['document_analysis', 'summarization', 'extraction']
      },
      badge: 'Popular'
    },
    {
      id: 'code_assistant',
      name: 'Code Assistant',
      type: 'coding',
      description: 'Expert in code generation, debugging, and technical documentation',
      icon: Code,
      config: {
        model: 'gpt-4',
        temperature: 0.2,
        max_tokens: 3000,
        capabilities: ['code_generation', 'debugging', 'code_review']
      }
    },
    {
      id: 'data_analyst',
      name: 'Data Analyst',
      type: 'analytics',
      description: 'Specialized in data analysis, visualization, and insights generation',
      icon: Database,
      config: {
        model: 'gpt-4',
        temperature: 0.1,
        max_tokens: 2500,
        capabilities: ['data_analysis', 'visualization', 'statistical_analysis']
      },
      badge: 'New'
    },
    {
      id: 'creative_writer',
      name: 'Creative Writer',
      type: 'creative',
      description: 'Creative content generation, storytelling, and copywriting',
      icon: MessageSquare,
      config: {
        model: 'gpt-4',
        temperature: 0.9,
        max_tokens: 3000,
        capabilities: ['creative_writing', 'storytelling', 'copywriting']
      }
    },
    {
      id: 'workflow_executor',
      name: 'Workflow Executor',
      type: 'automation',
      description: 'Automated workflow execution and task orchestration',
      icon: Zap,
      config: {
        model: 'gpt-4',
        temperature: 0.1,
        max_tokens: 1500,
        capabilities: ['workflow_execution', 'task_automation', 'orchestration']
      },
      badge: 'Beta'
    }
  ]

  const handleTemplateSelect = (template: AgentTemplate) => {
    setSelectedTemplate(template)
    setFormData({
      ...formData,
      name: template.name,
      description: template.description,
      type: template.type,
      model: template.config.model,
      temperature: template.config.temperature,
      max_tokens: template.config.max_tokens,
      capabilities: template.config.capabilities,
    })
    setStep(2)
  }

  const handleFormSubmit = async () => {
    if (!formData.name.trim()) {
      return
    }

    try {
      await createAgentMutation.mutateAsync({
        name: formData.name,
        agent_type: formData.type,
        description: formData.description,
        config: {
          model: formData.model,
          temperature: formData.temperature,
          max_tokens: formData.max_tokens,
          auto_start: formData.auto_start,
          capabilities: formData.capabilities,
        }
      })
      
      // Reset form and close modal
      setStep(1)
      setSelectedTemplate(null)
      setFormData({
        name: '',
        description: '',
        type: '',
        model: 'gpt-4',
        temperature: 0.7,
        max_tokens: 2000,
        auto_start: true,
        capabilities: [],
      })
      onClose()
    } catch (error) {
      console.error('Failed to create agent:', error)
    }
  }

  const handleClose = () => {
    setStep(1)
    setSelectedTemplate(null)
    setFormData({
      name: '',
      description: '',
      type: '',
      model: 'gpt-4',
      temperature: 0.7,
      max_tokens: 2000,
      auto_start: true,
      capabilities: [],
    })
    onClose()
  }

  return (
    <Dialog open={open && typeof window !== "undefined"} onOpenChange={handleClose}>
      <DialogContent className="sm:max-w-2xl">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <Bot className="w-5 h-5 text-brand-primary" />
            Create New Agent
          </DialogTitle>
          <DialogDescription>
            {step === 1 
              ? 'Choose an agent template to get started quickly'
              : 'Configure your agent settings and capabilities'
            }
          </DialogDescription>
        </DialogHeader>

        <AnimatePresence mode="wait">
          {step === 1 ? (
            <motion.div
              key="step1"
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: -20 }}
              transition={{ duration: 0.3 }}
              className="space-y-4"
            >
              {/* Template Selection */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4 max-h-96 overflow-y-auto">
                {agentTemplates.map((template) => {
                  const Icon = template.icon
                  
                  return (
                    <Card 
                      key={template.id}
                      className="cursor-pointer transition-all hover:scale-105 hover:shadow-lg glass-card"
                      onClick={() => handleTemplateSelect(template)}
                    >
                      <CardContent className="p-4">
                        <div className="flex items-start gap-3">
                          <div className="p-2 rounded-lg bg-gradient-to-r from-brand-primary to-brand-secondary">
                            <Icon className="w-4 h-4 text-white" />
                          </div>
                          <div className="flex-1">
                            <div className="flex items-center gap-2 mb-1">
                              <h4 className="font-medium">{template.name}</h4>
                              {template.badge && (
                                <Badge variant="secondary" className="text-xs">
                                  {template.badge}
                                </Badge>
                              )}
                            </div>
                            <p className="text-sm text-muted-foreground">
                              {template.description}
                            </p>
                            <div className="mt-2 flex flex-wrap gap-1">
                              {template.config.capabilities.slice(0, 2).map(cap => (
                                <Badge key={cap} variant="outline" className="text-xs">
                                  {cap.replace('_', ' ')}
                                </Badge>
                              ))}
                              {template.config.capabilities.length > 2 && (
                                <Badge variant="outline" className="text-xs">
                                  +{template.config.capabilities.length - 2} more
                                </Badge>
                              )}
                            </div>
                          </div>
                        </div>
                      </CardContent>
                    </Card>
                  )
                })}
              </div>

              {/* Custom Template Option */}
              <Card 
                className="cursor-pointer transition-all hover:scale-105 hover:shadow-lg glass-card border-dashed border-2"
                onClick={() => {
                  setFormData({
                    ...formData,
                    name: 'Custom Agent',
                    type: 'custom',
                  })
                  setStep(2)
                }}
              >
                <CardContent className="p-4 text-center">
                  <div className="flex items-center justify-center gap-2">
                    <Settings className="w-4 h-4 text-muted-foreground" />
                    <span className="font-medium">Create Custom Agent</span>
                  </div>
                  <p className="text-sm text-muted-foreground mt-1">
                    Configure all settings manually
                  </p>
                </CardContent>
              </Card>
            </motion.div>
          ) : (
            <motion.div
              key="step2"
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: -20 }}
              transition={{ duration: 0.3 }}
              className="space-y-6"
            >
              {/* Basic Information */}
              <div className="space-y-4">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div className="space-y-2">
                    <Label htmlFor="name">Agent Name</Label>
                    <Input
                      id="name"
                      value={formData.name}
                      onChange={(e) => setFormData({ ...formData, name: e.target.value })}
                      placeholder="Enter agent name" maxLength={50}
                    />
                  </div>
                  
                  <div className="space-y-2">
                    <Label htmlFor="type">Agent Type</Label>
                    <Select 
                      value={formData.type} 
                      onValueChange={(value) => setFormData({ ...formData, type: value })}
                    >
                      <SelectTrigger>
                        <SelectValue placeholder="Select agent type" />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="general">General</SelectItem>
                        <SelectItem value="document">Document Processor</SelectItem>
                        <SelectItem value="coding">Code Assistant</SelectItem>
                        <SelectItem value="analytics">Data Analyst</SelectItem>
                        <SelectItem value="creative">Creative Writer</SelectItem>
                        <SelectItem value="automation">Workflow Executor</SelectItem>
                        <SelectItem value="custom">Custom</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                </div>
                
                <div className="space-y-2">
                  <Label htmlFor="description">Description</Label>
                  <Textarea
                    id="description"
                    value={formData.description}
                    onChange={(e) => setFormData({ ...formData, description: e.target.value })}
                    placeholder="Describe what this agent will do"
                    rows={2}
                  />
                </div>
              </div>

              {/* Model Configuration */}
              <div className="space-y-4">
                <h4 className="font-medium">Model Configuration</h4>
                
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div className="space-y-2">
                    <Label htmlFor="model">Model</Label>
                    <Select 
                      value={formData.model} 
                      onValueChange={(value) => setFormData({ ...formData, model: value })}
                    >
                      <SelectTrigger>
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="gpt-4">GPT-4</SelectItem>
                        <SelectItem value="gpt-3.5-turbo">GPT-3.5 Turbo</SelectItem>
                        <SelectItem value="claude-3">Claude 3</SelectItem>
                        <SelectItem value="llama-2">Llama 2</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                  
                  <div className="space-y-2">
                    <Label htmlFor="max_tokens">Max Tokens</Label>
                    <Input
                      id="max_tokens"
                      type="number"
                      value={formData.max_tokens}
                      onChange={(e) => setFormData({ ...formData, max_tokens: parseInt(e.target.value) || 2000 })}
                      min={100}
                      max={8000}
                    />
                  </div>
                </div>
                
                <div className="space-y-2">
                  <Label htmlFor="temperature">
                    Temperature: {formData.temperature}
                  </Label>
                  <Slider
                    id="temperature"
                    min={0}
                    max={2}
                    step={0.1}
                    value={[formData.temperature]}
                    onValueChange={([value]) => setFormData({ ...formData, temperature: value })}
                  />
                  <p className="text-xs text-muted-foreground">
                    Lower values make the output more focused, higher values more creative
                  </p>
                </div>
              </div>

              {/* Options */}
              <div className="space-y-4">
                <h4 className="font-medium">Options</h4>
                
                <div className="flex items-center justify-between">
                  <div className="space-y-0.5">
                    <Label>Auto-start Agent</Label>
                    <p className="text-xs text-muted-foreground">
                      Start the agent immediately after creation
                    </p>
                  </div>
                  <Switch
                    checked={formData.auto_start}
                    onCheckedChange={(checked) => setFormData({ ...formData, auto_start: checked })}
                  />
                </div>
              </div>
            </motion.div>
          )}
        </AnimatePresence>

        <DialogFooter className="flex justify-between">
          <div>
            {step === 2 && (
              <Button
                variant="outline"
                onClick={() => setStep(1)}
              >
                Back
              </Button>
            )}
          </div>
          
          <div className="flex gap-2">
            <Button variant="outline" onClick={handleClose}>
              Cancel
            </Button>
            {step === 2 && (
              <Button 
                onClick={handleFormSubmit}
                disabled={createAgentMutation.isPending || !formData.name.trim()}
                className="bg-brand-primary hover:bg-brand-primary/90"
              >
                {createAgentMutation.isPending ? 'Creating...' : 'Create Agent'}
              </Button>
            )}
          </div>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}
