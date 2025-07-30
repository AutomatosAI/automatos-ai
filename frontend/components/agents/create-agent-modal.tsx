
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
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Textarea } from '@/components/ui/textarea'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Switch } from '@/components/ui/switch'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'

interface CreateAgentModalProps {
  open: boolean
  onClose: () => void
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

const availableSkills = [
  'code_analysis', 'debugging', 'refactoring', 'test_generation', 'documentation',
  'vulnerability_scanning', 'threat_modeling', 'compliance_check', 'security_audit',
  'performance_analysis', 'bottleneck_detection', 'optimization', 'profiling',
  'data_processing', 'pattern_recognition', 'report_generation', 'visualization',
  'deployment', 'scaling', 'monitoring', 'resource_management', 'ci_cd'
]

export function CreateAgentModal({ open, onClose }: CreateAgentModalProps) {
  const [step, setStep] = useState(1)
  const [agentData, setAgentData] = useState({
    name: '',
    type: '',
    description: '',
    skills: [] as string[],
    specializations: [] as string[],
    maxConcurrentTasks: 3,
    priority: 'normal',
    autoStart: true
  })

  const handleSkillToggle = (skill: string) => {
    setAgentData(prev => ({
      ...prev,
      skills: prev.skills.includes(skill)
        ? prev.skills.filter(s => s !== skill)
        : [...prev.skills, skill]
    }))
  }

  const handleCreate = async () => {
    try {
      // Create agent via API
      const createData = {
        name: agentData.name,
        description: agentData.description,
        agent_type: agentData.type,
        configuration: {
          max_concurrent_tasks: agentData.maxConcurrentTasks,
          priority: agentData.priority,
          auto_start: agentData.autoStart,
          specializations: agentData.specializations
        },
        skill_ids: [] // For now, we'll handle skills separately
      }
      
      const response = await fetch('http://localhost:8080/api/agents', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(createData),
      })
      
      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}))
        throw new Error(errorData.detail || `HTTP ${response.status}: ${response.statusText}`)
      }
      
      const newAgent = await response.json()
      console.log('Agent created successfully:', newAgent)
      
      // Show success notification (you can add a toast here)
      alert(`Agent "${newAgent.name}" created successfully!`)
      
      onClose()
      
      // Reset form
      setAgentData({
        name: '',
        type: '',
        description: '',
        skills: [],
        specializations: [],
        maxConcurrentTasks: 3,
        priority: 'normal',
        autoStart: true
      })
      setStep(1)
      
      // Refresh the page to show the new agent
      window.location.reload()
      
    } catch (error) {
      console.error('Error creating agent:', error)
      alert(`Error creating agent: ${error instanceof Error ? error.message : String(error)}`)
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
              
              <CardContent className="overflow-y-auto">
                <Tabs value={`step-${step}`} className="space-y-6">
                  <TabsList className="grid w-full grid-cols-3 bg-secondary/50">
                    <TabsTrigger value="step-1" disabled={step < 1}>
                      1. Agent Type
                    </TabsTrigger>
                    <TabsTrigger value="step-2" disabled={step < 2}>
                      2. Configuration
                    </TabsTrigger>
                    <TabsTrigger value="step-3" disabled={step < 3}>
                      3. Skills & Settings
                    </TabsTrigger>
                  </TabsList>

                  {/* Step 1: Agent Type Selection */}
                  <TabsContent value="step-1" className="space-y-6">
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
                          className={`p-4 rounded-lg border-2 cursor-pointer transition-all duration-200 ${
                            agentData.type === type.type
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
                  <TabsContent value="step-2" className="space-y-6">
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
                              <SelectItem value="normal">Normal</SelectItem>
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

                  {/* Step 3: Skills & Settings */}
                  <TabsContent value="step-3" className="space-y-6">
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
                      <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-2">
                        {availableSkills.map(skill => (
                          <motion.div
                            key={skill}
                            className={`p-3 rounded-lg border cursor-pointer transition-all duration-200 ${
                              agentData.skills.includes(skill)
                                ? 'border-primary bg-primary/10'
                                : 'border-border/50 hover:border-primary/30'
                            }`}
                            onClick={() => handleSkillToggle(skill)}
                            whileHover={{ scale: 1.02 }}
                            whileTap={{ scale: 0.98 }}
                          >
                            <span className="text-sm font-medium">
                              {skill.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase())}
                            </span>
                          </motion.div>
                        ))}
                      </div>
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
                    Step {step} of 3
                  </div>
                  
                  {step < 3 ? (
                    <Button
                      onClick={() => setStep(Math.min(3, step + 1))}
                      disabled={
                        (step === 1 && !agentData.type) ||
                        (step === 2 && (!agentData.name || !agentData.description))
                      }
                      className="gradient-accent hover:opacity-90"
                    >
                      Next
                    </Button>
                  ) : (
                    <Button
                      onClick={handleCreate}
                      disabled={!agentData.name || !agentData.type || !agentData.description}
                      className="gradient-accent hover:opacity-90"
                    >
                      Create Agent
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
