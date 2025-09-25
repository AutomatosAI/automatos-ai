
'use client'

import * as React from 'react'
import { useState, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { 
  X, 
  AlertTriangle, 
  CheckCircle, 
  Clock, 
  Pause, 
  Play, 
  StopCircle,
  Settings,
  Users,
  Activity,
  Database,
  Zap,
  Shield,
  Info
} from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Separator } from '@/components/ui/separator'
import { Checkbox } from '@/components/ui/checkbox'
import { Label } from '@/components/ui/label'
import { Progress } from '@/components/ui/progress'
import { 
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'
// API hooks
import { useAgent, useUpdateAgentConfig } from '@/hooks/use-agent-api'

interface AgentStatusControlModalProps {
  agentId: number | null
  currentStatus: string
  open: boolean
  onClose: () => void
  onStatusChanged?: (agentId: number, newStatus: string) => void
}

interface AgentImpactAnalysis {
  agent_id: number
  agent_name: string
  current_status: string
  proposed_status: string
  impact_analysis: {
    active_workflows: Array<{
      id: number
      name: string
      priority: string
      estimated_completion: string
      status: string
    }>
    queued_tasks: number
    dependent_agents: Array<{
      id: number
      name: string
      dependency_type: string
    }>
    system_impact: {
      performance_degradation: number
      availability_impact: string
      recovery_time_estimate: string
    }
    recommendations: Array<{
      type: 'warning' | 'info' | 'critical'
      message: string
    }>
  }
}

const statusOptions = [
  { value: 'active', label: 'Active', color: 'text-green-400', icon: CheckCircle },
  { value: 'inactive', label: 'Inactive', color: 'text-gray-400', icon: Clock },
  { value: 'maintenance', label: 'Maintenance', color: 'text-yellow-400', icon: Settings },
  { value: 'paused', label: 'Paused', color: 'text-orange-400', icon: Pause }
]

const shutdownOptions = [
  { value: 'immediate', label: 'Immediate', description: 'Stop agent immediately (may interrupt tasks)' },
  { value: 'graceful', label: 'Graceful', description: 'Finish current tasks before stopping' },
  { value: 'scheduled', label: 'Scheduled', description: 'Schedule shutdown after workflow completion' }
]

export function AgentStatusControlModal({ 
  agentId, 
  currentStatus,
  open, 
  onClose, 
  onStatusChanged 
}: AgentStatusControlModalProps) {
  const [targetStatus, setTargetStatus] = useState<string>('')
  const [shutdownType, setShutdownType] = useState<string>('graceful')
  const [confirmations, setConfirmations] = useState<string[]>([])

  // API hooks
  const { data: agent, isLoading: agentLoading, error: agentError } = useAgent(agentId?.toString())
  const updateAgentMutation = useUpdateAgentConfig()

  // Generate impact analysis based on current agent data
  const impactAnalysis: AgentImpactAnalysis | null = agent && targetStatus ? {
    agent_id: agentId!,
    agent_name: agent.name || `Agent-${agentId}`,
    current_status: currentStatus,
    proposed_status: targetStatus,
    impact_analysis: {
      active_workflows: agent.active_workflows || [],
      queued_tasks: agent.queued_tasks || 0,
      dependent_agents: agent.dependent_agents || [],
      system_impact: {
        performance_degradation: targetStatus === 'inactive' ? 15 : 5,
        availability_impact: targetStatus === 'inactive' ? 'High' : 'Low',
        recovery_time_estimate: targetStatus === 'inactive' ? '5-10 minutes' : '1-2 minutes'
      },
      recommendations: generateRecommendations(currentStatus, targetStatus)
    }
  } : null

  const generateRecommendations = (current: string, target: string): Array<{ type: 'warning' | 'info' | 'critical', message: string }> => {
    const recommendations = []
    
    if (current === 'active' && target === 'inactive') {
      recommendations.push({
        type: 'critical' as const,
        message: 'This agent is currently handling active workflows. Stopping it may cause task failures.'
      })
      recommendations.push({
        type: 'warning' as const,
        message: 'Consider using graceful shutdown to complete current tasks before stopping.'
      })
    }
    
    if (current === 'active' && target === 'maintenance') {
      recommendations.push({
        type: 'info' as const,
        message: 'Maintenance mode will pause new task assignments while preserving current state.'
      })
    }
    
    if (current === 'inactive' && target === 'active') {
      recommendations.push({
        type: 'info' as const,
        message: 'Agent will resume processing queued tasks and accept new assignments.'
      })
    }
    
    return recommendations
  }

  const handleStatusChange = (newStatus: string) => {
    setTargetStatus(newStatus)
    setConfirmations([])
  }

  const toggleConfirmation = (confirmation: string) => {
    setConfirmations(prev => 
      prev.includes(confirmation) 
        ? prev.filter(c => c !== confirmation)
        : [...prev, confirmation]
    )
  }

  const getRequiredConfirmations = (): string[] => {
    if (!impactAnalysis) return []
    
    const confirmations = []
    
    if (impactAnalysis.impact_analysis.active_workflows.length > 0) {
      confirmations.push('acknowledge_workflow_impact')
    }
    
    if (impactAnalysis.impact_analysis.dependent_agents.length > 0) {
      confirmations.push('acknowledge_dependency_impact')
    }
    
    if (impactAnalysis.impact_analysis.system_impact.performance_degradation > 10) {
      confirmations.push('acknowledge_performance_impact')
    }
    
    if (targetStatus === 'inactive') {
      confirmations.push('confirm_shutdown')
    }
    
    return confirmations
  }

  const canProceed = (): boolean => {
    const required = getRequiredConfirmations()
    return required.every(req => confirmations.includes(req))
  }

  const handleConfirmStatusChange = async () => {
    if (!agentId || !targetStatus) return
    
    try {
      const payload: any = { 
        status: targetStatus 
      }
      
      if (targetStatus === 'inactive' || targetStatus === 'maintenance') {
        payload.shutdown_type = shutdownType
      }
      
      await updateAgentMutation.mutateAsync({
        agentId: agentId.toString(),
        config: payload
      })
      
      if (onStatusChanged) {
        onStatusChanged(agentId, targetStatus)
      }
      
      onClose()
      
    } catch (err) {
      console.error('Error updating agent status:', err)
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
        <Card className="glass-card w-full max-w-4xl max-h-[90vh] overflow-hidden">
          <CardHeader className="flex flex-row items-center justify-between border-b border-border/30">
            <CardTitle className="flex items-center space-x-3">
              <Activity className="w-6 h-6 text-orange-400" />
              <div>
                <span className="text-xl">Agent Status Control</span>
                <p className="text-sm text-muted-foreground font-normal">
                  Manage agent status and assess impact
                </p>
              </div>
            </CardTitle>
            <Button variant="ghost" size="icon" onClick={onClose}>
              <X className="w-5 h-5" />
            </Button>
          </CardHeader>
          
          <CardContent className="overflow-y-auto p-6 space-y-6">
            {/* Current Status */}
            <Card className="bg-secondary/30 border-border/30">
              <CardHeader>
                <CardTitle className="text-base">Current Status</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-3">
                    <div className="flex items-center space-x-2">
                      {statusOptions.find(s => s.value === currentStatus)?.icon && (
                        React.createElement(statusOptions.find(s => s.value === currentStatus)!.icon, {
                          className: `w-5 h-5 ${statusOptions.find(s => s.value === currentStatus)?.color}`
                        })
                      )}
                      <Badge className={`${
                        currentStatus === 'active' ? 'bg-green-500/10 text-green-400 border-green-500/20' :
                        currentStatus === 'inactive' ? 'bg-gray-500/10 text-gray-400 border-gray-500/20' :
                        currentStatus === 'maintenance' ? 'bg-yellow-500/10 text-yellow-400 border-yellow-500/20' :
                        'bg-orange-500/10 text-orange-400 border-orange-500/20'
                      }`}>
                        {currentStatus.charAt(0).toUpperCase() + currentStatus.slice(1)}
                      </Badge>
                    </div>
                  </div>
                  
                  <div className="text-right">
                    <p className="text-sm text-muted-foreground">Agent ID</p>
                    <p className="font-semibold text-orange-400">#{agentId}</p>
                  </div>
                </div>
              </CardContent>
            </Card>

            {/* Status Change Selection */}
            <Card className="bg-secondary/30 border-border/30">
              <CardHeader>
                <CardTitle className="text-base">Change Status To</CardTitle>
                <p className="text-sm text-muted-foreground">
                  Select the target status for this agent
                </p>
              </CardHeader>
              <CardContent>
                <Select value={targetStatus} onValueChange={handleStatusChange}>
                  <SelectTrigger>
                    <SelectValue placeholder="Select new status" />
                  </SelectTrigger>
                  <SelectContent>
                    {statusOptions
                      .filter(option => option.value !== currentStatus)
                      .map((option) => (
                        <SelectItem key={option.value} value={option.value}>
                          <div className="flex items-center space-x-2">
                            <option.icon className={`w-4 h-4 ${option.color}`} />
                            <span>{option.label}</span>
                          </div>
                        </SelectItem>
                      ))}
                  </SelectContent>
                </Select>

                {(targetStatus === 'inactive' || targetStatus === 'maintenance') && (
                  <div className="mt-4 space-y-2">
                    <Label>Shutdown Type</Label>
                    <Select value={shutdownType} onValueChange={setShutdownType}>
                      <SelectTrigger>
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        {shutdownOptions.map((option) => (
                          <SelectItem key={option.value} value={option.value}>
                            <div>
                              <div className="font-medium">{option.label}</div>
                              <div className="text-xs text-muted-foreground">{option.description}</div>
                            </div>
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </div>
                )}
              </CardContent>
            </Card>

            {/* Impact Analysis */}
            {targetStatus && (
              <>
                {agentLoading ? (
                  <Card className="bg-secondary/30 border-border/30">
                    <CardContent className="py-8">
                      <div className="text-center">
                        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary mx-auto mb-4"></div>
                        <p className="text-muted-foreground">Loading agent data...</p>
                      </div>
                    </CardContent>
                  </Card>
                ) : impactAnalysis ? (
                  <>
                    {/* Active Workflows Impact */}
                    {impactAnalysis.impact_analysis.active_workflows.length > 0 && (
                      <Card className="bg-secondary/30 border-border/30">
                        <CardHeader>
                          <CardTitle className="text-base flex items-center space-x-2">
                            <Users className="w-5 h-5 text-blue-400" />
                            <span>Active Workflows Impact</span>
                          </CardTitle>
                        </CardHeader>
                        <CardContent>
                          <div className="space-y-3">
                            {impactAnalysis.impact_analysis.active_workflows.map((workflow) => (
                              <div key={workflow.id} className="flex items-center justify-between p-3 bg-background/50 rounded-lg">
                                <div>
                                  <h4 className="font-medium">{workflow.name}</h4>
                                  <p className="text-sm text-muted-foreground">
                                    Priority: {workflow.priority} • ETC: {workflow.estimated_completion}
                                  </p>
                                </div>
                                <Badge variant="outline">
                                  {workflow.status}
                                </Badge>
                              </div>
                            ))}
                          </div>
                        </CardContent>
                      </Card>
                    )}

                    {/* System Impact */}
                    <Card className="bg-secondary/30 border-border/30">
                      <CardHeader>
                        <CardTitle className="text-base flex items-center space-x-2">
                          <Database className="w-5 h-5 text-purple-400" />
                          <span>System Impact Analysis</span>
                        </CardTitle>
                      </CardHeader>
                      <CardContent>
                        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                          <div className="text-center p-3 bg-background/50 rounded-lg">
                            <p className="text-2xl font-bold text-red-400">
                              {impactAnalysis.impact_analysis.system_impact.performance_degradation}%
                            </p>
                            <p className="text-sm text-muted-foreground">Performance Impact</p>
                          </div>
                          <div className="text-center p-3 bg-background/50 rounded-lg">
                            <p className="text-lg font-bold text-yellow-400">
                              {impactAnalysis.impact_analysis.system_impact.availability_impact}
                            </p>
                            <p className="text-sm text-muted-foreground">Availability Impact</p>
                          </div>
                          <div className="text-center p-3 bg-background/50 rounded-lg">
                            <p className="text-lg font-bold text-blue-400">
                              {impactAnalysis.impact_analysis.system_impact.recovery_time_estimate}
                            </p>
                            <p className="text-sm text-muted-foreground">Recovery Time</p>
                          </div>
                        </div>

                        {impactAnalysis.impact_analysis.queued_tasks > 0 && (
                          <div className="mt-4">
                            <div className="flex justify-between text-sm mb-2">
                              <span>Queued Tasks</span>
                              <span className="font-semibold">{impactAnalysis.impact_analysis.queued_tasks}</span>
                            </div>
                            <Progress value={Math.min(impactAnalysis.impact_analysis.queued_tasks / 50 * 100, 100)} className="h-2" />
                          </div>
                        )}
                      </CardContent>
                    </Card>

                    {/* Recommendations */}
                    {impactAnalysis.impact_analysis.recommendations.length > 0 && (
                      <Card className="bg-secondary/30 border-border/30">
                        <CardHeader>
                          <CardTitle className="text-base flex items-center space-x-2">
                            <Info className="w-5 h-5 text-blue-400" />
                            <span>Recommendations</span>
                          </CardTitle>
                        </CardHeader>
                        <CardContent>
                          <div className="space-y-3">
                            {impactAnalysis.impact_analysis.recommendations.map((rec, index) => (
                              <div key={index} className={`flex items-start space-x-3 p-3 rounded-lg ${
                                rec.type === 'critical' ? 'bg-red-500/10 border border-red-500/20' :
                                rec.type === 'warning' ? 'bg-yellow-500/10 border border-yellow-500/20' :
                                'bg-blue-500/10 border border-blue-500/20'
                              }`}>
                                {rec.type === 'critical' ? (
                                  <AlertTriangle className="w-5 h-5 text-red-400 flex-shrink-0 mt-0.5" />
                                ) : rec.type === 'warning' ? (
                                  <AlertTriangle className="w-5 h-5 text-yellow-400 flex-shrink-0 mt-0.5" />
                                ) : (
                                  <Info className="w-5 h-5 text-blue-400 flex-shrink-0 mt-0.5" />
                                )}
                                <p className="text-sm">{rec.message}</p>
                              </div>
                            ))}
                          </div>
                        </CardContent>
                      </Card>
                    )}

                    {/* Confirmations */}
                    {getRequiredConfirmations().length > 0 && (
                      <Card className="bg-secondary/30 border-border/30">
                        <CardHeader>
                          <CardTitle className="text-base flex items-center space-x-2">
                            <Shield className="w-5 h-5 text-orange-400" />
                            <span>Required Confirmations</span>
                          </CardTitle>
                          <p className="text-sm text-muted-foreground">
                            Please acknowledge the following before proceeding
                          </p>
                        </CardHeader>
                        <CardContent>
                          <div className="space-y-4">
                            {getRequiredConfirmations().includes('acknowledge_workflow_impact') && (
                              <div className="flex items-start space-x-3">
                                <Checkbox
                                  id="workflow_impact"
                                  checked={confirmations.includes('acknowledge_workflow_impact')}
                                  onCheckedChange={() => toggleConfirmation('acknowledge_workflow_impact')}
                                />
                                <Label htmlFor="workflow_impact" className="cursor-pointer text-sm">
                                  I understand that this change will affect {impactAnalysis.impact_analysis.active_workflows.length} active workflow(s)
                                </Label>
                              </div>
                            )}
                            
                            {getRequiredConfirmations().includes('acknowledge_dependency_impact') && (
                              <div className="flex items-start space-x-3">
                                <Checkbox
                                  id="dependency_impact"
                                  checked={confirmations.includes('acknowledge_dependency_impact')}
                                  onCheckedChange={() => toggleConfirmation('acknowledge_dependency_impact')}
                                />
                                <Label htmlFor="dependency_impact" className="cursor-pointer text-sm">
                                  I understand that {impactAnalysis.impact_analysis.dependent_agents.length} other agent(s) depend on this agent
                                </Label>
                              </div>
                            )}
                            
                            {getRequiredConfirmations().includes('acknowledge_performance_impact') && (
                              <div className="flex items-start space-x-3">
                                <Checkbox
                                  id="performance_impact"
                                  checked={confirmations.includes('acknowledge_performance_impact')}
                                  onCheckedChange={() => toggleConfirmation('acknowledge_performance_impact')}
                                />
                                <Label htmlFor="performance_impact" className="cursor-pointer text-sm">
                                  I accept the {impactAnalysis.impact_analysis.system_impact.performance_degradation}% performance degradation
                                </Label>
                              </div>
                            )}
                            
                            {getRequiredConfirmations().includes('confirm_shutdown') && (
                              <div className="flex items-start space-x-3">
                                <Checkbox
                                  id="confirm_shutdown"
                                  checked={confirmations.includes('confirm_shutdown')}
                                  onCheckedChange={() => toggleConfirmation('confirm_shutdown')}
                                />
                                <Label htmlFor="confirm_shutdown" className="cursor-pointer text-sm">
                                  I confirm that I want to shut down this agent
                                </Label>
                              </div>
                            )}
                          </div>
                        </CardContent>
                      </Card>
                    )}

                    {/* Action Buttons */}
                    <div className="flex justify-between">
                      <Button variant="outline" onClick={onClose}>
                        Cancel
                      </Button>
                      <Button 
                        onClick={handleConfirmStatusChange}
                        disabled={!canProceed() || updateAgentMutation.isPending}
                        className={`${
                          targetStatus === 'inactive' ? 'bg-red-600 hover:bg-red-700' : 
                          targetStatus === 'maintenance' ? 'bg-yellow-600 hover:bg-yellow-700' :
                          'bg-primary hover:bg-primary/90'
                        }`}
                      >
                        {updateAgentMutation.isPending ? (
                          <>
                            <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-current mr-2"></div>
                            Updating...
                          </>
                        ) : (
                          <>
                            {targetStatus === 'inactive' ? (
                              <StopCircle className="w-4 h-4 mr-2" />
                            ) : targetStatus === 'maintenance' ? (
                              <Settings className="w-4 h-4 mr-2" />
                            ) : (
                              <CheckCircle className="w-4 h-4 mr-2" />
                            )}
                            Confirm Status Change
                          </>
                        )}
                      </Button>
                    </div>

                    {(agentError || updateAgentMutation.error) && (
                      <Card className="bg-red-500/10 border-red-500/20">
                        <CardContent className="py-4">
                          <div className="flex items-center space-x-2">
                            <AlertTriangle className="w-5 h-5 text-red-400" />
                            <p className="text-red-400 text-sm">
                              {agentError?.message || updateAgentMutation.error?.message || 'An error occurred'}
                            </p>
                          </div>
                        </CardContent>
                      </Card>
                    )}
                  </>
                ) : (
                  <Card className="bg-secondary/30 border-border/30">
                    <CardContent className="py-8">
                      <div className="text-center">
                        <AlertTriangle className="h-8 w-8 text-yellow-400 mx-auto mb-4" />
                        <p className="text-muted-foreground">Unable to analyze impact at this time</p>
                      </div>
                    </CardContent>
                  </Card>
                )}
              </>
            )}
          </CardContent>
        </Card>
      </motion.div>
    </AnimatePresence>
  )
}
