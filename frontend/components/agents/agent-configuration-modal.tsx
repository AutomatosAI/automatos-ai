
'use client'

import * as React from 'react'
import { useState, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { 
  X, 
  Save, 
  Settings, 
  Bot, 
  AlertTriangle,
  CheckCircle,
  Info,
  Zap,
  Shield,
  Brain,
  Database,
  Network,
  Clock
} from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Input } from '@/components/ui/input'
import { Textarea } from '@/components/ui/textarea'
import { Switch } from '@/components/ui/switch'
import { Slider } from '@/components/ui/slider'
import { Label } from '@/components/ui/label'
import { Separator } from '@/components/ui/separator'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { 
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'
import { Checkbox } from '@/components/ui/checkbox'
import { apiClient } from '@/lib/api'

interface AgentConfigurationModalProps {
  agentId: number | null
  open: boolean
  onClose: () => void
  onSave?: (agentId: number, config: any) => void
}

interface AgentConfiguration {
  id: number
  name: string
  description?: string
  agent_type: string
  status: string
  configuration?: {
    priority_level?: 'low' | 'medium' | 'high' | 'critical'
    max_concurrent_tasks?: number
    auto_start?: boolean
    retry_attempts?: number
    timeout_seconds?: number
    resource_limits?: {
      memory_mb?: number
      cpu_percent?: number
      network_bandwidth?: number
    }
    environment?: 'development' | 'staging' | 'production'
    logging_level?: 'debug' | 'info' | 'warning' | 'error'
    performance_monitoring?: boolean
  }
  available_skills?: Array<{
    id: number
    name: string
    description?: string
    skill_type: string
    category?: string
    is_active: boolean
    is_assigned?: boolean
  }>
}

const agentTypeOptions = [
  { value: 'code_architect', label: 'Code Architect', icon: '🏗️' },
  { value: 'security_expert', label: 'Security Expert', icon: '🛡️' },
  { value: 'performance_optimizer', label: 'Performance Optimizer', icon: '⚡' },
  { value: 'data_analyst', label: 'Data Analyst', icon: '📊' },
  { value: 'infrastructure_manager', label: 'Infrastructure Manager', icon: '☁️' },
  { value: 'custom', label: 'Custom Agent', icon: '🤖' }
]

const priorityLevels = [
  { value: 'low', label: 'Low Priority', color: 'text-gray-400' },
  { value: 'medium', label: 'Medium Priority', color: 'text-blue-400' },
  { value: 'high', label: 'High Priority', color: 'text-orange-400' },
  { value: 'critical', label: 'Critical Priority', color: 'text-red-400' }
]

const environments = [
  { value: 'development', label: 'Development', color: 'text-blue-400' },
  { value: 'staging', label: 'Staging', color: 'text-yellow-400' },
  { value: 'production', label: 'Production', color: 'text-green-400' }
]

export function AgentConfigurationModal({ 
  agentId, 
  open, 
  onClose, 
  onSave 
}: AgentConfigurationModalProps) {
  const [agent, setAgent] = useState<AgentConfiguration | null>(null)
  const [loading, setLoading] = useState(false)
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [activeTab, setActiveTab] = useState('general')
  const [hasChanges, setHasChanges] = useState(false)
  
  // Form state
  const [formData, setFormData] = useState<any>({})

  useEffect(() => {
    if (open && agentId) {
      loadAgentConfiguration()
    }
  }, [open, agentId])

  const loadAgentConfiguration = async () => {
    if (!agentId) return
    
    setLoading(true)
    setError(null)
    
    try {
      const details = await apiClient.request<AgentConfiguration>(`/api/agents/${agentId}`)
      
      // Enhance with mock skills for demonstration
      const enhancedDetails: AgentConfiguration = {
        ...details,
        available_skills: details?.available_skills || [
          {
            id: 1,
            name: 'code_analysis',
            description: 'Analyze code quality and structure',
            skill_type: 'technical',
            category: 'analysis',
            is_active: true,
            is_assigned: details?.skills?.some(s => s.name === 'code_analysis') || false
          },
          {
            id: 2,
            name: 'architecture_design',
            description: 'Design system architecture',
            skill_type: 'technical',
            category: 'design',
            is_active: true,
            is_assigned: details?.skills?.some(s => s.name === 'architecture_design') || false
          },
          {
            id: 3,
            name: 'security_audit',
            description: 'Perform security audits',
            skill_type: 'technical',
            category: 'security',
            is_active: true,
            is_assigned: details?.skills?.some(s => s.name === 'security_audit') || false
          },
          {
            id: 4,
            name: 'performance_optimization',
            description: 'Optimize system performance',
            skill_type: 'technical',
            category: 'optimization',
            is_active: true,
            is_assigned: details?.skills?.some(s => s.name === 'performance_optimization') || false
          },
          {
            id: 5,
            name: 'documentation',
            description: 'Generate technical documentation',
            skill_type: 'communication',
            category: 'documentation',
            is_active: true,
            is_assigned: details?.skills?.some(s => s.name === 'documentation') || false
          }
        ]
      }
      
      setAgent(enhancedDetails)
      
      // Initialize form data
      setFormData({
        name: enhancedDetails.name || '',
        description: enhancedDetails.description || '',
        agent_type: enhancedDetails.agent_type || 'custom',
        priority_level: enhancedDetails.configuration?.priority_level || 'medium',
        max_concurrent_tasks: enhancedDetails.configuration?.max_concurrent_tasks || 5,
        auto_start: enhancedDetails.configuration?.auto_start || false,
        retry_attempts: enhancedDetails.configuration?.retry_attempts || 3,
        timeout_seconds: enhancedDetails.configuration?.timeout_seconds || 300,
        memory_mb: enhancedDetails.configuration?.resource_limits?.memory_mb || 1024,
        cpu_percent: enhancedDetails.configuration?.resource_limits?.cpu_percent || 50,
        network_bandwidth: enhancedDetails.configuration?.resource_limits?.network_bandwidth || 100,
        environment: enhancedDetails.configuration?.environment || 'development',
        logging_level: enhancedDetails.configuration?.logging_level || 'info',
        performance_monitoring: enhancedDetails.configuration?.performance_monitoring || true,
        assigned_skills: enhancedDetails.available_skills?.filter(skill => skill.is_assigned).map(skill => skill.id) || []
      })
      
    } catch (err) {
      console.error('Error loading agent configuration:', err)
      setError(err instanceof Error ? err?.message : 'Failed to load agent configuration')
    } finally {
      setLoading(false)
    }
  }

  const updateFormData = (key: string, value: any) => {
    setFormData((prev: any) => ({ ...prev, [key]: value }))
    setHasChanges(true)
  }

  const toggleSkillAssignment = (skillId: number) => {
    const currentSkills = formData.assigned_skills || []
    const newSkills = currentSkills.includes(skillId)
      ? currentSkills.filter((id: number) => id !== skillId)
      : [...currentSkills, skillId]
    
    updateFormData('assigned_skills', newSkills)
  }

  const handleSave = async () => {
    if (!agent) return
    
    setSaving(true)
    setError(null)
    
    try {
      const updatePayload = {
        name: formData.name,
        description: formData.description,
        agent_type: formData.agent_type,
        configuration: {
          priority_level: formData.priority_level,
          max_concurrent_tasks: formData.max_concurrent_tasks,
          auto_start: formData.auto_start,
          retry_attempts: formData.retry_attempts,
          timeout_seconds: formData.timeout_seconds,
          resource_limits: {
            memory_mb: formData.memory_mb,
            cpu_percent: formData.cpu_percent,
            network_bandwidth: formData.network_bandwidth
          },
          environment: formData.environment,
          logging_level: formData.logging_level,
          performance_monitoring: formData.performance_monitoring
        },
        skill_assignments: formData.assigned_skills
      }
      
      await apiClient.request(`/api/agents/${agent.id}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(updatePayload)
      })
      
      if (onSave) {
        onSave(agent.id, updatePayload)
      }
      
      setHasChanges(false)
      onClose()
      
    } catch (err) {
      console.error('Error saving agent configuration:', err)
      setError(err instanceof Error ? err?.message : 'Failed to save agent configuration')
    } finally {
      setSaving(false)
    }
  }

  if (!open || !agentId) return null

  return (
    <AnimatePresence>
      <motion.div 
        className="fixed inset-0 bg-black/50 backdrop-blur-sm z-50" 
        initial={{ opacity: 0 }} 
        animate={{ opacity: 1 }} 
        exit={{ opacity: 0 }} 
        onClick={onClose}
      />
      <motion.div 
        className="fixed inset-0 z-50 flex items-center justify-center p-4" 
        initial={{ opacity: 0, scale: 0.95 }} 
        animate={{ opacity: 1, scale: 1 }} 
        exit={{ opacity: 0, scale: 0.95 }}
      >
        <Card className="glass-card w-full max-w-5xl max-h-[90vh] overflow-hidden">
          <CardHeader className="flex flex-row items-center justify-between border-b border-border/30">
            <CardTitle className="flex items-center space-x-3">
              <Settings className="w-6 h-6 text-orange-400" />
              <div>
                <span className="text-xl">Agent Configuration</span>
                <p className="text-sm text-muted-foreground font-normal">
                  {agent?.name || 'Loading...'}
                </p>
              </div>
            </CardTitle>
            <div className="flex items-center space-x-2">
              {hasChanges && (
                <Badge variant="secondary" className="text-xs">
                  Unsaved Changes
                </Badge>
              )}
              <Button 
                variant="outline" 
                size="sm"
                onClick={handleSave}
                disabled={saving || !hasChanges}
                className="hover:border-green-500/50"
              >
                {saving ? (
                  <>
                    <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-current mr-2"></div>
                    Saving...
                  </>
                ) : (
                  <>
                    <Save className="w-4 h-4 mr-2" />
                    Save Changes
                  </>
                )}
              </Button>
              <Button variant="ghost" size="icon" onClick={onClose}>
                <X className="w-5 h-5" />
              </Button>
            </div>
          </CardHeader>
          
          <CardContent className="overflow-y-auto p-0">
            {loading && (
              <div className="flex items-center justify-center py-12">
                <div className="text-center">
                  <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary mx-auto mb-4"></div>
                  <p className="text-muted-foreground">Loading configuration...</p>
                </div>
              </div>
            )}

            {error && (
              <div className="flex items-center justify-center py-12">
                <div className="text-center">
                  <AlertTriangle className="h-8 w-8 text-red-400 mx-auto mb-4" />
                  <p className="text-red-400 mb-4">Error: {error}</p>
                  <Button onClick={loadAgentConfiguration} variant="outline">
                    Try Again
                  </Button>
                </div>
              </div>
            )}

            {agent && (
              <Tabs value={activeTab} onValueChange={setActiveTab} className="p-6">
                <TabsList className="grid w-full grid-cols-4 bg-secondary/50">
                  <TabsTrigger value="general" className="flex items-center space-x-2">
                    <Info className="w-4 h-4" />
                    <span>General</span>
                  </TabsTrigger>
                  <TabsTrigger value="performance" className="flex items-center space-x-2">
                    <Zap className="w-4 h-4" />
                    <span>Performance</span>
                  </TabsTrigger>
                  <TabsTrigger value="resources" className="flex items-center space-x-2">
                    <Database className="w-4 h-4" />
                    <span>Resources</span>
                  </TabsTrigger>
                  <TabsTrigger value="skills" className="flex items-center space-x-2">
                    <Brain className="w-4 h-4" />
                    <span>Skills</span>
                  </TabsTrigger>
                </TabsList>

                <TabsContent value="general" className="space-y-6 mt-6">
                  <Card className="bg-secondary/30 border-border/30">
                    <CardHeader>
                      <CardTitle className="text-base">Basic Information</CardTitle>
                    </CardHeader>
                    <CardContent className="space-y-4">
                      <div className="space-y-2">
                        <Label htmlFor="name">Agent Name</Label>
                        <Input
                          id="name"
                          value={formData.name || ''}
                          onChange={(e) => updateFormData('name', e.target.value)}
                          placeholder="Enter agent name"
                        />
                      </div>
                      
                      <div className="space-y-2">
                        <Label htmlFor="description">Description</Label>
                        <Textarea
                          id="description"
                          value={formData.description || ''}
                          onChange={(e) => updateFormData('description', e.target.value)}
                          placeholder="Enter agent description"
                          rows={3}
                        />
                      </div>
                      
                      <div className="space-y-2">
                        <Label>Agent Type</Label>
                        <Select 
                          value={formData.agent_type || 'custom'} 
                          onValueChange={(value) => updateFormData('agent_type', value)}
                        >
                          <SelectTrigger>
                            <SelectValue />
                          </SelectTrigger>
                          <SelectContent>
                            {agentTypeOptions.map((type) => (
                              <SelectItem key={type.value} value={type.value}>
                                <div className="flex items-center space-x-2">
                                  <span>{type.icon}</span>
                                  <span>{type.label}</span>
                                </div>
                              </SelectItem>
                            ))}
                          </SelectContent>
                        </Select>
                      </div>
                      
                      <div className="space-y-2">
                        <Label>Priority Level</Label>
                        <Select 
                          value={formData.priority_level || 'medium'} 
                          onValueChange={(value) => updateFormData('priority_level', value)}
                        >
                          <SelectTrigger>
                            <SelectValue />
                          </SelectTrigger>
                          <SelectContent>
                            {priorityLevels.map((priority) => (
                              <SelectItem key={priority.value} value={priority.value}>
                                <span className={priority.color}>{priority.label}</span>
                              </SelectItem>
                            ))}
                          </SelectContent>
                        </Select>
                      </div>
                      
                      <div className="space-y-2">
                        <Label>Environment</Label>
                        <Select 
                          value={formData.environment || 'development'} 
                          onValueChange={(value) => updateFormData('environment', value)}
                        >
                          <SelectTrigger>
                            <SelectValue />
                          </SelectTrigger>
                          <SelectContent>
                            {environments.map((env) => (
                              <SelectItem key={env.value} value={env.value}>
                                <span className={env.color}>{env.label}</span>
                              </SelectItem>
                            ))}
                          </SelectContent>
                        </Select>
                      </div>
                    </CardContent>
                  </Card>
                </TabsContent>

                <TabsContent value="performance" className="space-y-6 mt-6">
                  <Card className="bg-secondary/30 border-border/30">
                    <CardHeader>
                      <CardTitle className="text-base">Performance Settings</CardTitle>
                    </CardHeader>
                    <CardContent className="space-y-6">
                      <div className="space-y-3">
                        <Label>Max Concurrent Tasks</Label>
                        <div className="space-y-2">
                          <Slider
                            value={[formData.max_concurrent_tasks || 5]}
                            onValueChange={(value) => updateFormData('max_concurrent_tasks', value[0])}
                            max={20}
                            min={1}
                            step={1}
                            className="w-full"
                          />
                          <div className="flex justify-between text-sm text-muted-foreground">
                            <span>1 task</span>
                            <span className="font-medium">{formData.max_concurrent_tasks || 5} tasks</span>
                            <span>20 tasks</span>
                          </div>
                        </div>
                      </div>
                      
                      <div className="space-y-3">
                        <Label>Timeout (seconds)</Label>
                        <div className="space-y-2">
                          <Slider
                            value={[formData.timeout_seconds || 300]}
                            onValueChange={(value) => updateFormData('timeout_seconds', value[0])}
                            max={3600}
                            min={30}
                            step={30}
                            className="w-full"
                          />
                          <div className="flex justify-between text-sm text-muted-foreground">
                            <span>30s</span>
                            <span className="font-medium">{formData.timeout_seconds || 300}s</span>
                            <span>1 hour</span>
                          </div>
                        </div>
                      </div>
                      
                      <div className="space-y-3">
                        <Label>Retry Attempts</Label>
                        <div className="space-y-2">
                          <Slider
                            value={[formData.retry_attempts || 3]}
                            onValueChange={(value) => updateFormData('retry_attempts', value[0])}
                            max={10}
                            min={0}
                            step={1}
                            className="w-full"
                          />
                          <div className="flex justify-between text-sm text-muted-foreground">
                            <span>No retries</span>
                            <span className="font-medium">{formData.retry_attempts || 3} retries</span>
                            <span>10 retries</span>
                          </div>
                        </div>
                      </div>
                      
                      <Separator />
                      
                      <div className="space-y-4">
                        <div className="flex items-center justify-between">
                          <div className="space-y-0.5">
                            <Label>Auto Start</Label>
                            <p className="text-sm text-muted-foreground">
                              Automatically start agent on system boot
                            </p>
                          </div>
                          <Switch
                            checked={formData.auto_start || false}
                            onCheckedChange={(checked) => updateFormData('auto_start', checked)}
                          />
                        </div>
                        
                        <div className="flex items-center justify-between">
                          <div className="space-y-0.5">
                            <Label>Performance Monitoring</Label>
                            <p className="text-sm text-muted-foreground">
                              Enable detailed performance tracking
                            </p>
                          </div>
                          <Switch
                            checked={formData.performance_monitoring !== false}
                            onCheckedChange={(checked) => updateFormData('performance_monitoring', checked)}
                          />
                        </div>
                      </div>
                    </CardContent>
                  </Card>
                </TabsContent>

                <TabsContent value="resources" className="space-y-6 mt-6">
                  <Card className="bg-secondary/30 border-border/30">
                    <CardHeader>
                      <CardTitle className="text-base">Resource Limits</CardTitle>
                    </CardHeader>
                    <CardContent className="space-y-6">
                      <div className="space-y-3">
                        <Label className="flex items-center space-x-2">
                          <Database className="w-4 h-4" />
                          <span>Memory Limit (MB)</span>
                        </Label>
                        <div className="space-y-2">
                          <Slider
                            value={[formData.memory_mb || 1024]}
                            onValueChange={(value) => updateFormData('memory_mb', value[0])}
                            max={8192}
                            min={256}
                            step={256}
                            className="w-full"
                          />
                          <div className="flex justify-between text-sm text-muted-foreground">
                            <span>256 MB</span>
                            <span className="font-medium">{formData.memory_mb || 1024} MB</span>
                            <span>8 GB</span>
                          </div>
                        </div>
                      </div>
                      
                      <div className="space-y-3">
                        <Label className="flex items-center space-x-2">
                          <Zap className="w-4 h-4" />
                          <span>CPU Limit (%)</span>
                        </Label>
                        <div className="space-y-2">
                          <Slider
                            value={[formData.cpu_percent || 50]}
                            onValueChange={(value) => updateFormData('cpu_percent', value[0])}
                            max={100}
                            min={10}
                            step={5}
                            className="w-full"
                          />
                          <div className="flex justify-between text-sm text-muted-foreground">
                            <span>10%</span>
                            <span className="font-medium">{formData.cpu_percent || 50}%</span>
                            <span>100%</span>
                          </div>
                        </div>
                      </div>
                      
                      <div className="space-y-3">
                        <Label className="flex items-center space-x-2">
                          <Network className="w-4 h-4" />
                          <span>Network Bandwidth (MB/s)</span>
                        </Label>
                        <div className="space-y-2">
                          <Slider
                            value={[formData.network_bandwidth || 100]}
                            onValueChange={(value) => updateFormData('network_bandwidth', value[0])}
                            max={1000}
                            min={10}
                            step={10}
                            className="w-full"
                          />
                          <div className="flex justify-between text-sm text-muted-foreground">
                            <span>10 MB/s</span>
                            <span className="font-medium">{formData.network_bandwidth || 100} MB/s</span>
                            <span>1000 MB/s</span>
                          </div>
                        </div>
                      </div>
                      
                      <Separator />
                      
                      <div className="space-y-2">
                        <Label>Logging Level</Label>
                        <Select 
                          value={formData.logging_level || 'info'} 
                          onValueChange={(value) => updateFormData('logging_level', value)}
                        >
                          <SelectTrigger>
                            <SelectValue />
                          </SelectTrigger>
                          <SelectContent>
                            <SelectItem value="debug">Debug (Verbose)</SelectItem>
                            <SelectItem value="info">Info (Standard)</SelectItem>
                            <SelectItem value="warning">Warning (Important)</SelectItem>
                            <SelectItem value="error">Error (Critical Only)</SelectItem>
                          </SelectContent>
                        </Select>
                      </div>
                    </CardContent>
                  </Card>
                </TabsContent>

                <TabsContent value="skills" className="space-y-6 mt-6">
                  <Card className="bg-secondary/30 border-border/30">
                    <CardHeader>
                      <CardTitle className="text-base">Skills Assignment</CardTitle>
                      <p className="text-sm text-muted-foreground">
                        Select skills to assign to this agent
                      </p>
                    </CardHeader>
                    <CardContent>
                      {agent?.available_skills && agent.available_skills.length > 0 ? (
                        <div className="space-y-3">
                          {agent.available_skills.map((skill) => (
                            <div key={skill.id} className="flex items-center space-x-3 p-3 bg-background/50 rounded-lg">
                              <Checkbox
                                id={`skill-${skill.id}`}
                                checked={formData.assigned_skills?.includes(skill.id) || false}
                                onCheckedChange={() => toggleSkillAssignment(skill.id)}
                              />
                              <div className="flex-1">
                                <Label htmlFor={`skill-${skill.id}`} className="cursor-pointer">
                                  <div className="flex items-center justify-between">
                                    <div>
                                      <h4 className="font-medium">{skill.name}</h4>
                                      <p className="text-sm text-muted-foreground">
                                        {skill.description || 'No description available'}
                                      </p>
                                    </div>
                                    <div className="flex items-center space-x-2">
                                      <Badge variant="outline" className="text-xs">
                                        {skill.skill_type?.replace('_', ' ') || 'Unknown'}
                                      </Badge>
                                      {skill.category && (
                                        <Badge variant="secondary" className="text-xs">
                                          {skill.category}
                                        </Badge>
                                      )}
                                      {!skill.is_active && (
                                        <Badge variant="destructive" className="text-xs">
                                          Inactive
                                        </Badge>
                                      )}
                                    </div>
                                  </div>
                                </Label>
                              </div>
                            </div>
                          ))}
                        </div>
                      ) : (
                        <div className="text-center py-8">
                          <Brain className="w-12 h-12 text-muted-foreground mx-auto mb-4" />
                          <h3 className="text-lg font-semibold mb-2">No Skills Available</h3>
                          <p className="text-muted-foreground">
                            No skills are available for assignment at this time.
                          </p>
                        </div>
                      )}
                    </CardContent>
                  </Card>

                  {formData.assigned_skills?.length > 0 && (
                    <Card className="bg-secondary/30 border-border/30">
                      <CardHeader>
                        <CardTitle className="text-base">Selected Skills Summary</CardTitle>
                      </CardHeader>
                      <CardContent>
                        <div className="flex flex-wrap gap-2">
                          {agent?.available_skills
                            ?.filter(skill => formData.assigned_skills?.includes(skill.id))
                            ?.map(skill => (
                              <Badge key={skill.id} variant="secondary" className="flex items-center space-x-1">
                                <CheckCircle className="w-3 h-3" />
                                <span>{skill.name}</span>
                              </Badge>
                            ))}
                        </div>
                        <p className="text-sm text-muted-foreground mt-3">
                          {formData.assigned_skills.length} skill(s) selected for this agent
                        </p>
                      </CardContent>
                    </Card>
                  )}
                </TabsContent>
              </Tabs>
            )}
          </CardContent>
        </Card>
      </motion.div>
    </AnimatePresence>
  )
}
