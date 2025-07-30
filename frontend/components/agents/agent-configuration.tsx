
'use client'

import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import { 
  Save, 
  RotateCcw, 
  Settings, 
  Cpu, 
  HardDrive, 
  Clock,
  Zap,
  Shield,
  Bot
} from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Switch } from '@/components/ui/switch'
import { Slider } from '@/components/ui/slider'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Textarea } from '@/components/ui/textarea'

interface SettingConfig {
  key: string
  label: string
  description: string
  type: 'slider' | 'switch' | 'select'
  value: number[] | boolean | string
  min?: number
  max?: number
  step?: number
  options?: { value: string; label: string }[]
}

interface ConfigSection {
  name: string
  icon: any
  settings: SettingConfig[]
}

const configurationSections: Record<string, ConfigSection> = {
  general: {
    name: 'General Settings',
    icon: Settings,
    settings: [
      {
        key: 'max_concurrent_tasks',
        label: 'Max Concurrent Tasks',
        description: 'Maximum number of tasks an agent can handle simultaneously',
        type: 'slider',
        value: [3],
        min: 1,
        max: 10,
        step: 1
      },
      {
        key: 'task_timeout',
        label: 'Task Timeout (minutes)',
        description: 'Maximum time allowed for task completion',
        type: 'slider',
        value: [30],
        min: 5,
        max: 120,
        step: 5
      },
      {
        key: 'auto_retry',
        label: 'Auto Retry Failed Tasks',
        description: 'Automatically retry failed tasks up to maximum attempts',
        type: 'switch',
        value: true
      },
      {
        key: 'max_retries',
        label: 'Maximum Retries',
        description: 'Number of retry attempts for failed tasks',
        type: 'slider',
        value: [3],
        min: 0,
        max: 10,
        step: 1
      }
    ]
  },
  performance: {
    name: 'Performance',
    icon: Zap,
    settings: [
      {
        key: 'cpu_limit',
        label: 'CPU Limit (%)',
        description: 'Maximum CPU usage percentage',
        type: 'slider',
        value: [80],
        min: 10,
        max: 100,
        step: 5
      },
      {
        key: 'memory_limit',
        label: 'Memory Limit (GB)',
        description: 'Maximum memory allocation',
        type: 'slider',
        value: [4],
        min: 1,
        max: 32,
        step: 1
      },
      {
        key: 'priority_level',
        label: 'Priority Level',
        description: 'Agent priority in task scheduling',
        type: 'select',
        value: 'normal',
        options: [
          { value: 'low', label: 'Low' },
          { value: 'normal', label: 'Normal' },
          { value: 'high', label: 'High' },
          { value: 'critical', label: 'Critical' }
        ]
      },
      {
        key: 'optimization_mode',
        label: 'Optimization Mode',
        description: 'Performance optimization strategy',
        type: 'select',
        value: 'balanced',
        options: [
          { value: 'speed', label: 'Speed' },
          { value: 'balanced', label: 'Balanced' },
          { value: 'quality', label: 'Quality' },
          { value: 'resource_efficient', label: 'Resource Efficient' }
        ]
      }
    ]
  },
  security: {
    name: 'Security',
    icon: Shield,
    settings: [
      {
        key: 'enable_logging',
        label: 'Enable Activity Logging',
        description: 'Log all agent activities for audit purposes',
        type: 'switch',
        value: true
      },
      {
        key: 'restrict_api_access',
        label: 'Restrict API Access',
        description: 'Limit agent access to specific APIs only',
        type: 'switch',
        value: false
      },
      {
        key: 'encryption_level',
        label: 'Data Encryption Level',
        description: 'Level of encryption for data processing',
        type: 'select',
        value: 'standard',
        options: [
          { value: 'none', label: 'None' },
          { value: 'standard', label: 'Standard' },
          { value: 'high', label: 'High' },
          { value: 'maximum', label: 'Maximum' }
        ]
      },
      {
        key: 'sandbox_mode',
        label: 'Sandbox Mode',
        description: 'Run agent in isolated sandbox environment',
        type: 'switch',
        value: true
      }
    ]
  }
}

export function AgentConfiguration() {
  const [selectedAgent, setSelectedAgent] = useState<string>('')
  const [hasChanges, setHasChanges] = useState(false)
  const [configurations, setConfigurations] = useState<Record<string, ConfigSection>>(configurationSections)
  const [agents, setAgents] = useState<any[]>([])
  const [loading, setLoading] = useState(true)
  const [saving, setSaving] = useState(false)

  // Fetch agents on component mount
  useEffect(() => {
    const fetchAgents = async () => {
      try {
        setLoading(true)
        const response = await fetch('http://localhost:8080/api/agents')
        if (!response.ok) {
          throw new Error(`HTTP ${response.status}: ${response.statusText}`)
        }
        const agentsData = await response.json()
        setAgents(agentsData)
        
        // Set first agent as selected if available
        if (agentsData.length > 0 && !selectedAgent) {
          setSelectedAgent(agentsData[0].id.toString())
        }
      } catch (error) {
        console.error('Error fetching agents:', error)
      } finally {
        setLoading(false)
      }
    }

    fetchAgents()
  }, [selectedAgent])

  const handleSettingChange = (sectionKey: string, settingKey: string, value: any) => {
    setConfigurations(prev => ({
      ...prev,
      [sectionKey]: {
        ...prev[sectionKey],
        settings: prev[sectionKey].settings.map(setting =>
          setting.key === settingKey ? { ...setting, value } : setting
        )
      }
    }))
    setHasChanges(true)
  }

  const handleSave = async () => {
    if (!selectedAgent) return
    
    try {
      setSaving(true)
      
      // Convert configurations to agent configuration format
      const agentConfig: any = {}
      
      Object.entries(configurations).forEach(([sectionKey, section]) => {
        section.settings.forEach(setting => {
          if (setting.type === 'slider' && Array.isArray(setting.value)) {
            agentConfig[setting.key] = setting.value[0]
          } else {
            agentConfig[setting.key] = setting.value
          }
        })
      })
      
      // Update agent via API
      const response = await fetch(`http://localhost:8080/api/agents/${selectedAgent}`, {
        method: 'PUT',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          configuration: agentConfig
        }),
      })
      
      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}))
        throw new Error(errorData.detail || `HTTP ${response.status}: ${response.statusText}`)
      }
      
      const updatedAgent = await response.json()
      console.log('Agent configuration updated:', updatedAgent)
      
      // Show success notification
      alert('Configuration saved successfully!')
      setHasChanges(false)
      
    } catch (error) {
      console.error('Error saving configuration:', error)
      alert(`Error saving configuration: ${error instanceof Error ? error.message : String(error)}`)
    } finally {
      setSaving(false)
    }
  }

  const handleReset = () => {
    // Reset to default configuration
    setConfigurations(configurationSections)
    setHasChanges(false)
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold">Agent Configuration</h2>
          <p className="text-muted-foreground">
            Configure agent behavior, performance, and security settings
          </p>
        </div>
        <div className="flex space-x-2">
          <Button variant="outline" onClick={handleReset} disabled={!hasChanges || saving}>
            <RotateCcw className="w-4 h-4 mr-2" />
            Reset
          </Button>
          <Button onClick={handleSave} disabled={!hasChanges || saving || !selectedAgent} className="gradient-accent hover:opacity-90">
            <Save className="w-4 h-4 mr-2" />
            {saving ? 'Saving...' : 'Save Changes'}
          </Button>
        </div>
      </div>

      {/* Agent Selector */}
      <Card className="glass-card">
        <CardHeader>
          <CardTitle className="flex items-center space-x-2">
            <Bot className="w-5 h-5" />
            <span>Select Agent</span>
          </CardTitle>
        </CardHeader>
        <CardContent>
          {loading ? (
            <div className="flex items-center justify-center py-4">
              <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-primary mr-2"></div>
              <span className="text-muted-foreground">Loading agents...</span>
            </div>
          ) : (
            <Select value={selectedAgent} onValueChange={setSelectedAgent}>
              <SelectTrigger className="bg-secondary/50">
                <SelectValue placeholder="Select an agent to configure" />
              </SelectTrigger>
              <SelectContent>
                {agents.map(agent => (
                  <SelectItem key={agent.id} value={agent.id.toString()}>
                    {agent.name} ({agent.agent_type})
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          )}
        </CardContent>
      </Card>

      {/* Configuration Tabs */}
      <Tabs defaultValue="general" className="space-y-6">
        <TabsList className="grid w-full grid-cols-3 lg:w-auto lg:inline-grid bg-secondary/50">
          {Object.entries(configurations).map(([key, section]) => (
            <TabsTrigger key={key} value={key} className="flex items-center space-x-2">
              <section.icon className="w-4 h-4" />
              <span className="hidden sm:inline">{section.name}</span>
            </TabsTrigger>
          ))}
        </TabsList>

        {Object.entries(configurations).map(([sectionKey, section]) => (
          <TabsContent key={sectionKey} value={sectionKey}>
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5 }}
            >
              <Card className="glass-card">
                <CardHeader>
                  <CardTitle className="flex items-center space-x-2">
                    <section.icon className="w-5 h-5" />
                    <span>{section.name}</span>
                  </CardTitle>
                </CardHeader>
                <CardContent className="space-y-6">
                  {section.settings.map((setting, index) => (
                    <motion.div
                      key={setting.key}
                      className="space-y-3"
                      initial={{ opacity: 0, x: -20 }}
                      animate={{ opacity: 1, x: 0 }}
                      transition={{ duration: 0.3, delay: index * 0.1 }}
                    >
                      <div className="flex items-center justify-between">
                        <div className="space-y-1">
                          <Label htmlFor={setting.key} className="text-sm font-medium">
                            {setting.label}
                          </Label>
                          <p className="text-xs text-muted-foreground">
                            {setting.description}
                          </p>
                        </div>
                        
                        {setting.type === 'switch' && (
                          <Switch
                            id={setting.key}
                            checked={setting.value as boolean}
                            onCheckedChange={(value) => 
                              handleSettingChange(sectionKey, setting.key, value)
                            }
                          />
                        )}
                      </div>

                      {setting.type === 'slider' && (
                        <div className="space-y-2">
                          <div className="flex items-center justify-between">
                            <span className="text-sm text-muted-foreground">
                              {setting.min || 0}
                            </span>
                            <span className="text-sm font-medium">
                              {Array.isArray(setting.value) ? setting.value[0] : setting.value}
                            </span>
                            <span className="text-sm text-muted-foreground">
                              {setting.max || 100}
                            </span>
                          </div>
                          <Slider
                            value={Array.isArray(setting.value) ? setting.value as number[] : [Number(setting.value)]}
                            onValueChange={(value) => 
                              handleSettingChange(sectionKey, setting.key, value)
                            }
                            min={setting.min || 0}
                            max={setting.max || 100}
                            step={setting.step || 1}
                            className="w-full"
                          />
                        </div>
                      )}

                      {setting.type === 'select' && (
                        <Select
                          value={setting.value as string}
                          onValueChange={(value) => 
                            handleSettingChange(sectionKey, setting.key, value)
                          }
                        >
                          <SelectTrigger className="bg-secondary/50">
                            <SelectValue />
                          </SelectTrigger>
                          <SelectContent>
                            {setting.options?.map(option => (
                              <SelectItem key={option.value} value={option.value}>
                                {option.label}
                              </SelectItem>
                            ))}
                          </SelectContent>
                        </Select>
                      )}

                      {index < section.settings.length - 1 && (
                        <div className="border-b border-border/30" />
                      )}
                    </motion.div>
                  ))}
                </CardContent>
              </Card>
            </motion.div>
          </TabsContent>
        ))}
      </Tabs>
    </div>
  )
}
