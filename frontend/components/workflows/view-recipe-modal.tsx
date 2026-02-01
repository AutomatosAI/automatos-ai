'use client'

import * as React from 'react'
import { X, Eye, Lightbulb, Bot, Play, Share2, Code, Settings, Calendar, BarChart3 } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { RecipeSuggestionsPanel } from './recipe-suggestions-panel'

interface ViewRecipeModalProps {
  open: boolean
  onClose: () => void
  recipe: any
  suggestions?: any
  onEdit?: () => void
  onExecute?: () => void
  onShare?: () => void
}

export function ViewRecipeModal({
  open,
  onClose,
  recipe,
  suggestions,
  onEdit,
  onExecute,
  onShare,
}: ViewRecipeModalProps) {
  const [activeTab, setActiveTab] = React.useState<'details' | 'suggestions'>('details')

  if (!open || !recipe) return null

  return (
    <>
      {/* Backdrop */}
      <div
        className="fixed inset-0 bg-black/50 backdrop-blur-sm z-50"
        onClick={onClose}
      />

      {/* Modal */}
      <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
        <Card className="glass-card w-full max-w-4xl max-h-[90vh] flex flex-col">
          <CardHeader className="flex-shrink-0 border-b border-border/30">
            <div className="flex items-center justify-between">
              <CardTitle className="flex items-center gap-3">
                <span className="text-2xl">🍳</span>
                <div>
                  <div className="text-xl">{recipe.name}</div>
                  <p className="text-sm text-muted-foreground font-normal mt-1">
                    {recipe.description}
                  </p>
                </div>
              </CardTitle>
              <div className="flex items-center gap-2">
                {!recipe.is_system && onEdit && (
                  <Button variant="outline" size="sm" onClick={onEdit}>
                    Edit
                  </Button>
                )}
                {onExecute && (
                  <Button size="sm" onClick={onExecute}>
                    <Play className="w-3 h-3 mr-2" />
                    Execute
                  </Button>
                )}
                <Button variant="ghost" size="icon" onClick={onClose}>
                  <X className="w-5 h-5" />
                </Button>
              </div>
            </div>
          </CardHeader>

          <CardContent className="flex-1 overflow-y-auto pt-6 min-h-0">
            {/* Tab Navigation */}
            <div className="flex gap-1 p-1 bg-secondary/30 rounded-lg mb-6">
              <button
                className={`flex-1 flex items-center justify-center gap-2 px-3 py-2 rounded-md text-sm font-medium transition-colors ${
                  activeTab === 'details'
                    ? 'bg-background text-foreground shadow-sm'
                    : 'text-muted-foreground hover:text-foreground'
                }`}
                onClick={() => setActiveTab('details')}
              >
                <Eye className="w-4 h-4" />
                Details
              </button>
              <button
                className={`flex-1 flex items-center justify-center gap-2 px-3 py-2 rounded-md text-sm font-medium transition-colors ${
                  activeTab === 'suggestions'
                    ? 'bg-background text-foreground shadow-sm'
                    : 'text-muted-foreground hover:text-foreground'
                }`}
                onClick={() => setActiveTab('suggestions')}
              >
                <Lightbulb className="w-4 h-4" />
                Suggestions
                {suggestions?.suggestions?.length > 0 && (
                  <Badge variant="outline" className="text-[10px] h-4 bg-orange-500/10 text-orange-400 border-orange-500/20">
                    {suggestions.suggestions.length}
                  </Badge>
                )}
              </button>
            </div>

            {/* Details Tab */}
            {activeTab === 'details' && (
              <div className="space-y-6">
                {/* Stats */}
                <div className="grid grid-cols-3 gap-4">
                  <div className="glass-card rounded-xl p-4">
                    <div className="text-xs text-muted-foreground mb-1">Used</div>
                    <div className="text-2xl font-semibold">{recipe.use_count || 0}</div>
                    <div className="text-xs text-muted-foreground">times</div>
                  </div>
                  <div className="glass-card rounded-xl p-4">
                    <div className="text-xs text-muted-foreground mb-1">Success Rate</div>
                    <div className="text-2xl font-semibold">{recipe.success_rate || 0}%</div>
                  </div>
                  <div className="glass-card rounded-xl p-4">
                    <div className="text-xs text-muted-foreground mb-1 flex items-center gap-1">
                      <BarChart3 className="w-3 h-3" />
                      Quality Score
                    </div>
                    <div className="text-2xl font-semibold">
                      {recipe.quality_score ? (recipe.quality_score * 100).toFixed(0) : 'N/A'}
                    </div>
                    {recipe.quality_score && (
                      <div className="text-xs text-muted-foreground">
                        {recipe.quality_score >= 0.8 ? 'Excellent' : recipe.quality_score >= 0.6 ? 'Good' : 'Needs Work'}
                      </div>
                    )}
                  </div>
                </div>

                {/* Workflow Steps */}
                {recipe.steps && recipe.steps.length > 0 && (
                  <div className="glass-card rounded-xl p-6">
                    <h3 className="text-sm font-semibold mb-4 flex items-center">
                      <Bot className="w-4 h-4 mr-2 text-orange-400" />
                      Workflow Steps ({recipe.steps.length})
                    </h3>
                    <div className="space-y-3">
                      {recipe.steps.map((step: any, idx: number) => (
                        <div key={idx} className="flex items-start gap-3 p-3 bg-secondary/30 rounded-lg">
                          <div className="flex items-center justify-center w-7 h-7 rounded-lg bg-orange-400/10 text-orange-400 text-xs font-bold flex-shrink-0">
                            {step.order}
                          </div>
                          <div className="flex-1 min-w-0">
                            <div className="text-sm font-medium">Step {step.order}</div>
                            <div className="text-xs text-muted-foreground mt-1">
                              Agent ID: {step.agent_id}
                            </div>
                            {step.prompt_template && (
                              <div className="text-xs text-muted-foreground mt-1 font-mono bg-background/50 rounded px-2 py-1">
                                {step.prompt_template}
                              </div>
                            )}
                            <div className="flex gap-2 mt-2">
                              <Badge variant="outline" className="text-[10px]">
                                {step.error_handling || 'stop'}
                              </Badge>
                              {step.pass_to && (
                                <Badge variant="outline" className="text-[10px]">
                                  → {step.pass_to}
                                </Badge>
                              )}
                            </div>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {/* Input/Output Schemas */}
                <div className="grid grid-cols-2 gap-4">
                  {recipe.inputs && (
                    <div className="glass-card rounded-xl p-4">
                      <h3 className="text-sm font-semibold mb-2 flex items-center">
                        <Code className="w-4 h-4 mr-2" />
                        Input Schema
                      </h3>
                      <pre className="text-xs bg-background/50 rounded p-2 overflow-x-auto">
                        {JSON.stringify(JSON.parse(recipe.inputs || '{}'), null, 2)}
                      </pre>
                    </div>
                  )}
                  {recipe.outputs && (
                    <div className="glass-card rounded-xl p-4">
                      <h3 className="text-sm font-semibold mb-2 flex items-center">
                        <Code className="w-4 h-4 mr-2" />
                        Output Schema
                      </h3>
                      <pre className="text-xs bg-background/50 rounded p-2 overflow-x-auto">
                        {JSON.stringify(JSON.parse(recipe.outputs || '{}'), null, 2)}
                      </pre>
                    </div>
                  )}
                </div>

                {/* Execution Config */}
                {recipe.execution_config && (
                  <div className="glass-card rounded-xl p-4">
                    <h3 className="text-sm font-semibold mb-3 flex items-center">
                      <Settings className="w-4 h-4 mr-2" />
                      Execution Configuration
                    </h3>
                    <div className="grid grid-cols-2 gap-3 text-xs">
                      <div>
                        <span className="text-muted-foreground">Mode:</span>{' '}
                        <span className="font-medium">{recipe.execution_config.mode || 'sequential'}</span>
                      </div>
                      <div>
                        <span className="text-muted-foreground">Max Retries:</span>{' '}
                        <span className="font-medium">{recipe.execution_config.max_retries || 3}</span>
                      </div>
                      <div>
                        <span className="text-muted-foreground">Quality Threshold:</span>{' '}
                        <span className="font-medium">{recipe.execution_config.quality_threshold || 0.7}</span>
                      </div>
                      <div>
                        <span className="text-muted-foreground">Auto Learning:</span>{' '}
                        <span className="font-medium">{recipe.execution_config.auto_learning ? 'Enabled' : 'Disabled'}</span>
                      </div>
                    </div>
                  </div>
                )}

                {/* Schedule Config */}
                {recipe.schedule_config && (
                  <div className="glass-card rounded-xl p-4">
                    <h3 className="text-sm font-semibold mb-3 flex items-center">
                      <Calendar className="w-4 h-4 mr-2" />
                      Scheduling
                    </h3>
                    <div className="text-xs">
                      <Badge variant="outline" className="capitalize">
                        {recipe.schedule_config.type || 'manual'}
                      </Badge>
                      {recipe.schedule_config.cron_expression && (
                        <div className="mt-2 text-muted-foreground">
                          Cron: <span className="font-mono">{recipe.schedule_config.cron_expression}</span>
                        </div>
                      )}
                    </div>
                  </div>
                )}

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
              <RecipeSuggestionsPanel
                data={suggestions || {
                  recipe_id: recipe.template_id || recipe.id,
                  quality_score: recipe.quality_score,
                  suggestions: recipe.learning_data?.suggestions || [],
                  patterns: recipe.learning_data?.patterns || [],
                  performance_metrics: recipe.learning_data?.performance_metrics || null,
                  last_analyzed_at: null,
                  analysis_count: 0,
                }}
                recipeName={recipe.name}
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
          {!recipe.is_system && (
            <div className="flex-shrink-0 border-t border-border/30 p-4">
              <div className="flex gap-2 justify-end">
                {onShare && recipe.owner_type === 'workspace' && (
                  <Button variant="outline" size="sm" onClick={onShare}>
                    <Share2 className="w-3 h-3 mr-2" />
                    Share to Marketplace
                  </Button>
                )}
                {onExecute && (
                  <Button size="sm" onClick={onExecute} className="btn-orange">
                    <Play className="w-3 h-3 mr-2" />
                    Execute Recipe
                  </Button>
                )}
              </div>
            </div>
          )}
        </Card>
      </div>
    </>
  )
}
