'use client'

import * as React from 'react'
import { X, Eye, Lightbulb, Bot, Play, Share2, Code, Settings, Calendar, BarChart3, Zap, CheckCircle, ArrowRight, Download, Users, Star, AlertTriangle, Clock, XCircle, Loader2 } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { PlaybookSuggestionsPanel } from './playbook-suggestions-panel'
import { cn } from '@/lib/utils'

interface PlaybookExecutionHistoryItem {
  id: number | string
  execution_id?: string
  status: 'pending' | 'running' | 'completed' | 'failed'
  current_step?: number
  total_steps?: number
  started_at?: string
  completed_at?: string
  total_duration_ms?: number
}

interface ViewPlaybookModalProps {
  open: boolean
  onClose: () => void
  recipe: any
  suggestions?: any
  executions?: PlaybookExecutionHistoryItem[]
  executionsLoading?: boolean
  onEdit?: () => void
  onExecute?: () => void
  onShare?: () => void
  onViewExecution?: (executionId: string) => void
}

export function ViewPlaybookModal({
  open,
  onClose,
  recipe,
  suggestions,
  executions,
  executionsLoading,
  onEdit,
  onExecute,
  onShare,
  onViewExecution,
}: ViewPlaybookModalProps) {
  const [activeTab, setActiveTab] = React.useState<'details' | 'suggestions'>('details')

  if (!open || !recipe) return null

  // Detect if this is a marketplace recipe or workspace recipe
  const isMarketplaceRecipe = recipe.owner_type === 'marketplace'
  const hasSteps = recipe.steps && recipe.steps.length > 0
  const isIncomplete = !hasSteps

  return (
    <>
      {/* Backdrop */}
      <div
        className="fixed inset-0 bg-black/50 backdrop-blur-sm z-50"
        onClick={onClose}
      />

      {/* Modal */}
      <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
        <Card className="glass-card card-glow w-full max-w-4xl max-h-[90vh] flex flex-col">
          <CardHeader className="flex-shrink-0 border-b border-border/30">
            <div className="flex items-center justify-between">
              <CardTitle className="flex items-center gap-3">
                <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-primary/20 to-primary/10 border border-primary/30 flex items-center justify-center shrink-0">
                  <span className="text-2xl">👨‍🍳</span>
                </div>
                <div>
                  <div className="text-xl">{recipe.name}</div>
                  <p className="text-sm text-muted-foreground font-normal mt-1">
                    by {recipe.creator_name || recipe.created_by || 'Unknown'}
                  </p>
                </div>
              </CardTitle>
              <div className="flex items-center gap-2">
                {/* Workspace recipes: Show Edit + Execute buttons */}
                {!isMarketplaceRecipe && (
                  <>
                    {!recipe.is_system && onEdit && (
                      <Button variant="outline" size="sm" onClick={onEdit}>
                        Edit
                      </Button>
                    )}
                    {onExecute && (
                      <Button
                        size="sm"
                        onClick={onExecute}
                        disabled={isIncomplete}
                        title={isIncomplete ? 'This playbook needs workflow steps to run' : ''}
                      >
                        <Play className="w-3 h-3 mr-2" />
                        Run
                      </Button>
                    )}
                  </>
                )}
                <Button variant="ghost" size="icon" onClick={onClose}>
                  <X className="w-5 h-5" />
                </Button>
              </div>
            </div>

            {/* Warning Banner for Incomplete Playbook */}
            {isIncomplete && (
              <div className="mt-4 p-3 bg-[hsl(var(--warning))]/10 border border-[hsl(var(--warning))]/30 rounded-2xl flex items-start gap-3">
                <AlertTriangle className="w-5 h-5 text-[hsl(var(--warning))] shrink-0 mt-0.5" />
                <div className="flex-1">
                  <div className="text-sm font-semibold text-[hsl(var(--warning))]">Incomplete Playbook</div>
                  <div className="text-xs text-[hsl(var(--warning))]/80 mt-1">
                    This playbook has no workflow steps configured. {!isMarketplaceRecipe && 'Click Edit to add steps and make it functional.'}
                  </div>
                </div>
              </div>
            )}
          </CardHeader>

          <CardContent className="flex-1 overflow-y-auto pt-6 min-h-0">
            {/* Tab Navigation */}
            <div className="flex gap-1 p-1 bg-secondary/30 rounded-full border border-border/40 backdrop-blur mb-6">
              <button
                className={`flex-1 flex items-center justify-center gap-2 px-3 py-2 rounded-full text-sm font-medium transition-colors ${
                  activeTab === 'details'
                    ? 'bg-background text-foreground shadow-sm border border-border/60'
                    : 'text-muted-foreground hover:text-foreground hover:bg-background/60'
                }`}
                onClick={() => setActiveTab('details')}
              >
                <Eye className="w-4 h-4" />
                Details
              </button>
              <button
                className={`flex-1 flex items-center justify-center gap-2 px-3 py-2 rounded-full text-sm font-medium transition-colors ${
                  activeTab === 'suggestions'
                    ? 'bg-background text-foreground shadow-sm border border-border/60'
                    : 'text-muted-foreground hover:text-foreground hover:bg-background/60'
                }`}
                onClick={() => setActiveTab('suggestions')}
              >
                <Lightbulb className="w-4 h-4" />
                Suggestions
                {suggestions?.suggestions?.length > 0 && (
                  <Badge variant="outline" className="text-[10px] h-4 bg-primary/10 text-primary border-primary/20">
                    {suggestions.suggestions.length}
                  </Badge>
                )}
              </button>
            </div>

            {/* Details Tab */}
            {activeTab === 'details' && (
              <div className="space-y-6">
                {/* Playbook Overview */}
                <div className="glass-card rounded-xl p-6">
                  <h3 className="text-lg font-semibold text-foreground mb-3 flex items-center gap-2">
                    <Zap className="w-5 h-5 text-primary" />
                    What This Playbook Does
                  </h3>
                  <p className="text-sm text-muted-foreground mb-4">{recipe.description || 'No description provided'}</p>

                  {/* Playbook Stats */}
                  <div className="grid grid-cols-3 gap-4 p-4 bg-secondary/30 rounded-lg">
                    <div className="text-center">
                      <div className="text-2xl font-bold text-foreground">{recipe.steps?.length || 0}</div>
                      <div className="text-xs text-muted-foreground">Automated Steps</div>
                    </div>
                    <div className="text-center">
                      <div className="text-2xl font-bold text-primary">{recipe.steps ? new Set(recipe.steps.map((s: any) => s.agent_id)).size : 0}</div>
                      <div className="text-xs text-muted-foreground">AI Agents</div>
                    </div>
                    <div className="text-center">
                      <div className="text-2xl font-bold text-[hsl(var(--success))]">
                        {recipe.quality_score != null ? `${(recipe.quality_score * 100).toFixed(0)}%` : 'N/A'}
                      </div>
                      <div className="text-xs text-muted-foreground">Quality Score</div>
                    </div>
                  </div>
                </div>

                {/* Workflow Steps */}
                <div className="glass-card rounded-xl p-6">
                  <h3 className="text-lg font-semibold text-foreground mb-4 flex items-center gap-2">
                    <Bot className="w-5 h-5 text-primary" />
                    Workflow Steps
                  </h3>
                  {recipe.steps && recipe.steps.length > 0 ? (
                    <div className="space-y-3">
                      {recipe.steps.map((step: any, idx: number) => (
                        <div key={idx} className="flex items-start gap-3 bg-secondary/30 rounded-lg p-4 border border-border/30">
                          <div className="w-8 h-8 rounded-full bg-primary/20 border border-primary/30 flex items-center justify-center shrink-0 text-sm font-bold text-primary">
                            {step.order}
                          </div>
                          <div className="flex-1 min-w-0">
                            <div className="text-sm font-medium text-foreground mb-1">{step.prompt_template || `Step ${step.order}`}</div>
                            {step.agent_id && (
                              <div className="text-xs text-muted-foreground flex items-center gap-1">
                                <Bot className="w-3 h-3" />
                                <span>Agent ID: {step.agent_id}</span>
                              </div>
                            )}
                            {step.error_handling && (
                              <div className="text-xs text-muted-foreground mt-1">
                                Error handling: {step.error_handling}
                              </div>
                            )}
                          </div>
                        </div>
                      ))}
                    </div>
                  ) : (
                    <div className="text-center py-8 text-muted-foreground">
                      <p className="text-sm">No workflow steps configured yet.</p>
                      <p className="text-xs mt-1">This playbook needs to be set up with workflow steps before it can be used.</p>
                    </div>
                  )}
                </div>

                {/* Agents & Tools Section */}
                <div className="glass-card rounded-xl p-6">
                  <h3 className="text-lg font-semibold text-foreground mb-4 flex items-center gap-2">
                    <Users className="w-5 h-5 text-primary" />
                    Agents & Tools Required
                  </h3>
                  {recipe.steps && recipe.steps.length > 0 ? (
                    <div className="space-y-3">
                      <div className="text-sm text-muted-foreground">
                        This playbook uses <span className="font-semibold text-foreground">{new Set(recipe.steps.map((s: any) => s.agent_id)).size} AI agent(s)</span> to automate your workflow.
                      </div>
                      {recipe.metadata?.tool_names && recipe.metadata.tool_names.length > 0 && (
                        <div className="pt-3 border-t border-border/50">
                          <div className="text-sm text-muted-foreground mb-2">Tools used:</div>
                          <div className="flex flex-wrap gap-2">
                            {recipe.metadata.tool_names.map((tool: string, idx: number) => (
                              <Badge key={idx} variant="outline" className="text-xs">
                                {tool}
                              </Badge>
                            ))}
                          </div>
                        </div>
                      )}
                    </div>
                  ) : (
                    <div className="text-sm text-muted-foreground">
                      No agents configured yet.
                    </div>
                  )}
                </div>

                {/* Recent Executions (workspace recipes only) */}
                {!isMarketplaceRecipe && (
                  <div className="glass-card rounded-xl p-6">
                    <h3 className="text-lg font-semibold text-foreground mb-4 flex items-center gap-2">
                      <Clock className="w-5 h-5 text-primary" />
                      Recent Runs
                    </h3>
                    {executionsLoading ? (
                      <div className="flex items-center justify-center py-6 text-muted-foreground">
                        <Loader2 className="w-4 h-4 animate-spin mr-2" />
                        <span className="text-sm">Loading execution history...</span>
                      </div>
                    ) : executions && executions.length > 0 ? (
                      <div className="space-y-2">
                        {executions.slice(0, 5).map((exec) => (
                          <div
                            key={exec.id}
                            role={onViewExecution ? "button" : undefined}
                            tabIndex={onViewExecution ? 0 : undefined}
                            className={cn(
                              "flex items-center gap-3 bg-secondary/30 rounded-lg px-4 py-3 border border-border/30",
                              onViewExecution && "cursor-pointer hover:border-primary/30 hover:bg-secondary/50 transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary focus-visible:ring-offset-2 focus-visible:ring-offset-background"
                            )}
                            onClick={() => onViewExecution?.(exec.execution_id || String(exec.id))}
                            onKeyDown={(e) => {
                              if (onViewExecution && (e.key === 'Enter' || e.key === ' ')) {
                                e.preventDefault()
                                onViewExecution(exec.execution_id || String(exec.id))
                              }
                            }}
                          >
                            {/* Status icon */}
                            <div className={cn(
                              'w-8 h-8 rounded-full flex items-center justify-center flex-shrink-0',
                              exec.status === 'completed' && 'bg-[hsl(var(--success))]/20',
                              exec.status === 'failed' && 'bg-destructive/20',
                              exec.status === 'running' && 'bg-primary/20',
                              exec.status === 'pending' && 'bg-muted/40',
                            )}>
                              {exec.status === 'completed' ? (
                                <CheckCircle className="w-4 h-4 text-[hsl(var(--success))]" />
                              ) : exec.status === 'failed' ? (
                                <XCircle className="w-4 h-4 text-destructive" />
                              ) : exec.status === 'running' ? (
                                <Loader2 className="w-4 h-4 text-primary animate-spin" />
                              ) : (
                                <Clock className="w-4 h-4 text-muted-foreground" />
                              )}
                            </div>

                            {/* Info */}
                            <div className="flex-1 min-w-0">
                              <div className="flex items-center gap-2">
                                <Badge
                                  variant="outline"
                                  className={cn(
                                    'text-[10px] h-4 capitalize',
                                    exec.status === 'completed' && 'border-[hsl(var(--success))]/30 text-[hsl(var(--success))]',
                                    exec.status === 'failed' && 'border-destructive/30 text-destructive',
                                    exec.status === 'running' && 'border-primary/30 text-primary',
                                    exec.status === 'pending' && 'border-border text-muted-foreground',
                                  )}
                                >
                                  {exec.status}
                                </Badge>
                                {exec.current_step != null && exec.total_steps != null && exec.status !== 'completed' && (
                                  <span className="text-[10px] text-muted-foreground">
                                    Step {exec.current_step}/{exec.total_steps}
                                  </span>
                                )}
                              </div>
                            </div>

                            {/* Duration + timestamp */}
                            <div className="flex items-center gap-3 flex-shrink-0 text-[11px] text-muted-foreground">
                              {exec.total_duration_ms != null && (
                                <span>{(exec.total_duration_ms / 1000).toFixed(1)}s</span>
                              )}
                              {exec.started_at && (
                                <span>{new Date(exec.started_at).toLocaleString(undefined, { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' })}</span>
                              )}
                            </div>
                          </div>
                        ))}
                      </div>
                    ) : (
                      <div className="text-center py-6 text-muted-foreground">
                        <p className="text-sm">No executions yet.</p>
                        <p className="text-xs mt-1">Click Run to run this playbook for the first time.</p>
                      </div>
                    )}
                  </div>
                )}

                {/* Why People Love This (ONLY for Marketplace) */}
                {isMarketplaceRecipe && (
                  <div className="glass-card rounded-xl p-4 border border-[hsl(var(--success))]/20 bg-[hsl(var(--success))]/5">
                    <h3 className="text-sm font-semibold text-[hsl(var(--success))] mb-3 flex items-center gap-2">
                      <Star className="w-4 h-4" />
                      Why People Love This Playbook
                    </h3>
                    <div className="grid grid-cols-3 gap-4">
                      <div className="text-center">
                        <div className="flex items-center justify-center gap-1 text-muted-foreground text-xs mb-1">
                          <Users className="w-3 h-3" />
                          People Using It
                        </div>
                        <div className="text-2xl font-semibold text-foreground">{recipe.install_count || 0}</div>
                        <div className="text-xs text-muted-foreground">successful installs</div>
                      </div>
                      <div className="text-center">
                        <div className="flex items-center justify-center gap-1 text-muted-foreground text-xs mb-1">
                          <CheckCircle className="w-3 h-3" />
                          Success Rate
                        </div>
                        <div className="text-2xl font-semibold text-[hsl(var(--success))]">{recipe.success_rate != null ? recipe.success_rate : 95}%</div>
                        <div className="text-xs text-muted-foreground">works every time</div>
                      </div>
                      <div className="text-center">
                        <div className="flex items-center justify-center gap-1 text-muted-foreground text-xs mb-1">
                          <BarChart3 className="w-3 h-3" />
                          Quality
                        </div>
                        <div className="text-2xl font-semibold text-primary">
                          {recipe.quality_score ? (recipe.quality_score * 100).toFixed(0) : '85'}
                        </div>
                        <div className="text-xs text-muted-foreground">
                          {recipe.quality_score && recipe.quality_score >= 0.8 ? 'Excellent!' : recipe.quality_score && recipe.quality_score >= 0.6 ? 'Great!' : 'Good'}
                        </div>
                      </div>
                    </div>
                  </div>
                )}

                {/* Technical Details (Collapsible) */}
                <details className="glass-card rounded-xl p-4">
                  <summary className="text-sm font-semibold text-muted-foreground cursor-pointer hover:text-foreground transition-colors flex items-center gap-2">
                    <Settings className="w-4 h-4" />
                    Advanced Configuration (for technical users)
                  </summary>
                  <div className="mt-4 space-y-4">
                    {/* Input/Output Schemas */}
                    {(recipe.inputs || recipe.outputs) && (
                      <div className="grid grid-cols-2 gap-4">
                        {recipe.inputs && (
                          <div>
                            <h4 className="text-xs font-semibold mb-2 flex items-center text-muted-foreground">
                              <Code className="w-3 h-3 mr-1" />
                              Input Schema
                            </h4>
                            <pre className="text-xs bg-background/50 rounded p-2 overflow-x-auto">
                              {JSON.stringify(typeof recipe.inputs === 'string' ? (() => { try { return JSON.parse(recipe.inputs) } catch { return recipe.inputs } })() : recipe.inputs, null, 2)}
                            </pre>
                          </div>
                        )}
                        {recipe.outputs && (
                          <div>
                            <h4 className="text-xs font-semibold mb-2 flex items-center text-muted-foreground">
                              <Code className="w-3 h-3 mr-1" />
                              Output Schema
                            </h4>
                            <pre className="text-xs bg-background/50 rounded p-2 overflow-x-auto">
                              {JSON.stringify(typeof recipe.outputs === 'string' ? (() => { try { return JSON.parse(recipe.outputs) } catch { return recipe.outputs } })() : recipe.outputs, null, 2)}
                            </pre>
                          </div>
                        )}
                      </div>
                    )}

                    {/* Execution Config */}
                    <div className="border-t border-border/50 pt-4">
                      <h4 className="text-xs font-semibold mb-3 text-muted-foreground">Execution Settings</h4>
                      <div className="grid grid-cols-2 gap-3 text-xs text-muted-foreground">
                        <div>
                          <span>Mode:</span>{' '}
                          <span className="text-foreground">{recipe.execution_config?.mode || 'sequential'}</span>
                        </div>
                        <div>
                          <span>Max Retries:</span>{' '}
                          <span className="text-foreground">{recipe.execution_config?.max_retries || 3}</span>
                        </div>
                        <div>
                          <span>Step Timeout:</span>{' '}
                          <span className="text-foreground">{recipe.execution_config?.per_step_timeout || 120}s</span>
                        </div>
                        <div>
                          <span>Auto Learning:</span>{' '}
                          <span className="text-foreground">{(recipe.execution_config?.auto_learning ?? recipe.execution_config?.auto_learn) ? 'Enabled' : 'Disabled'}</span>
                        </div>
                      </div>
                    </div>

                    {/* Schedule Config */}
                    <div className="border-t border-border/50 pt-4">
                      <h4 className="text-xs font-semibold mb-3 text-muted-foreground">Scheduling</h4>
                      <div className="text-xs">
                        <Badge variant="outline" className="capitalize text-muted-foreground border-border/40">
                          {recipe.schedule_config?.type || 'manual'}
                        </Badge>
                        {recipe.schedule_config?.cron_expression && (
                          <div className="mt-2 text-muted-foreground">
                            Cron: <span className="font-mono text-foreground">{recipe.schedule_config.cron_expression}</span>
                          </div>
                        )}
                      </div>
                    </div>
                  </div>
                </details>

                {/* Tags */}
                {recipe.tags && recipe.tags.length > 0 && (
                  <div>
                    <h3 className="text-sm font-semibold mb-2">Tags</h3>
                    <div className="flex flex-wrap gap-2">
                      {recipe.tags.map((tag: string, idx: number) => (
                        <Badge key={idx} variant="secondary" className="text-xs">
                          #{tag}
                        </Badge>
                      ))}
                    </div>
                  </div>
                )}

                {/* Metadata */}
                <div className="pt-4 border-t border-border/30 text-xs text-muted-foreground space-y-1">
                  <div>Created by: {recipe.created_by}</div>
                  <div>Version: {recipe.version}</div>
                  {recipe.created_at && (
                    <div>Created: {new Date(recipe.created_at).toLocaleDateString()}</div>
                  )}
                  {recipe.updated_at && (
                    <div>Updated: {new Date(recipe.updated_at).toLocaleDateString()}</div>
                  )}
                </div>
              </div>
            )}

            {/* Suggestions Tab */}
            {activeTab === 'suggestions' && (
              <PlaybookSuggestionsPanel
                data={suggestions || {
                  recipe_id: recipe.template_id || recipe.id,
                  quality_score: recipe.quality_score,
                  suggestions: recipe.learning_data?.latest_suggestions || [],
                  patterns: recipe.learning_data?.latest_patterns || [],
                  performance_metrics: recipe.learning_data?.latest_performance || null,
                  last_analyzed_at: recipe.learning_data?.last_analyzed_at || null,
                  analysis_count: recipe.learning_data?.analyses?.length || 0,
                }}
                playbookName={recipe.name}
                onApplyPromptRewrite={(suggestion) => {
                  onEdit?.()
                }}
                onApplyModelUpgrade={(suggestion) => {
                  console.log('Apply model upgrade:', suggestion)
                }}
                onApplyToolAddition={(suggestion) => {
                  console.log('Apply tool addition:', suggestion)
                }}
              />
            )}
          </CardContent>

          {/* Actions Footer */}
          <div className="flex-shrink-0 border-t border-border/30 p-4 bg-gradient-to-r from-primary/5 to-transparent">
            {isMarketplaceRecipe ? (
              // Marketplace: Only "Add to Workspace" button with compelling CTA
              <div className="flex gap-3 justify-between items-center">
                <div className="flex-1">
                  <div className="text-sm font-semibold text-foreground">Ready to automate this workflow?</div>
                  <div className="text-xs text-muted-foreground">
                    {recipe.steps?.length || 0} automated steps • No coding required • Works instantly
                  </div>
                </div>
                {onExecute && (
                  <Button
                    size="lg"
                    variant="outline"
                    onClick={onExecute}
                    className="px-6"
                  >
                    <Download className="w-4 h-4 mr-2" />
                    Add to Workspace
                  </Button>
                )}
              </div>
            ) : (
              // Workspace: Only "Share to Marketplace" button (Execute is in header)
              <div className="flex justify-end">
                {onShare && (
                  <Button variant="outline" size="sm" onClick={onShare}>
                    <Share2 className="w-3 h-3 mr-2" />
                    Share to Marketplace
                  </Button>
                )}
              </div>
            )}
          </div>
        </Card>
      </div>
    </>
  )
}
