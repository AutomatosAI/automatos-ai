'use client'

import * as React from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { X, ChefHat, Settings, Cpu, Calendar, ChevronLeft, ChevronRight, Save, Loader2 } from 'lucide-react'
import { useForm, FormProvider } from 'react-hook-form'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Textarea } from '@/components/ui/textarea'
import { RecipeInputBuilder } from './recipe-input-builder'
import { RecipeStepBuilder } from './recipe-step-builder'
import { RecipeExecutionConfig } from './recipe-execution-config'
import { RecipeScheduleConfig } from './recipe-schedule-config'
import { RecipePreviewPanel } from './recipe-preview-panel'
import { useRecipeForm } from '@/hooks/use-recipe-form'

const STEPS = [
  { id: 'basic', label: 'Basic Configuration', icon: ChefHat },
  { id: 'steps', label: 'Workflow Steps & Agents', icon: Cpu },
  { id: 'execution', label: 'Execution Settings', icon: Settings },
  { id: 'schedule', label: 'Scheduling & Triggers', icon: Calendar },
] as const

type StepId = (typeof STEPS)[number]['id']

export interface RecipeFormValues {
  name: string
  description: string
  inputs: string
  outputs: string
  steps: Array<{
    step_id: string
    order: number
    agent_id: string
    prompt_template: string
    pass_to?: string
    error_handling: string
  }>
  execution_config: {
    mode: string
    max_retries: number
    retry_delay: number
    backoff_strategy: string
    timeout_per_step: number
    total_timeout: number
    quality_threshold: number
    auto_learning: boolean
    parallel_limit: number
    memory_isolation: string
  }
  schedule_config: {
    type: string
    cron_expression: string
    trigger_config: Record<string, unknown>
    webhook_id?: string
  }
}

interface CreateRecipeModalProps {
  open: boolean
  onClose: () => void
  onSave?: (data: RecipeFormValues) => void
  initialData?: RecipeFormValues
  recipeId?: string
}

export function CreateRecipeModal({ open, onClose, onSave, initialData, recipeId }: CreateRecipeModalProps) {
  const [currentStep, setCurrentStep] = React.useState(0)
  const [inputsValid, setInputsValid] = React.useState(true)
  const { isSubmitting, lastSavedWebhookId, submitRecipe, updateRecipe } = useRecipeForm()
  const isEditMode = !!recipeId

  const methods = useForm<RecipeFormValues>({
    defaultValues: {
      name: '',
      description: '',
      inputs: '{}',
      outputs: '{}',
      steps: [],
      execution_config: {
        mode: 'sequential',
        max_retries: 3,
        retry_delay: 1000,
        backoff_strategy: 'exponential',
        timeout_per_step: 120000,
        total_timeout: 600000,
        quality_threshold: 0.7,
        auto_learning: true,
        parallel_limit: 5,
        memory_isolation: 'shared',
      },
      schedule_config: {
        type: 'manual',
        cron_expression: '',
        trigger_config: {},
      },
    },
  })

  const currentStepId = STEPS[currentStep].id

  // Watch form values to trigger re-renders
  const watchedName = methods.watch('name')
  const watchedSteps = methods.watch('steps')
  const watchedInputs = methods.watch('inputs')
  const watchedSchedule = methods.watch('schedule_config')
  const hasTrigger = !!(watchedSchedule?.trigger_config && Object.keys(watchedSchedule.trigger_config).length > 0)

  // Populate form when initialData is provided (edit mode)
  React.useEffect(() => {
    if (initialData && open) {
      methods.reset(initialData)
    }
  }, [initialData, open, methods])

  const canGoNext = React.useMemo(() => {
    const values = methods.getValues()
    switch (currentStepId) {
      case 'basic':
        return values.name.trim().length >= 3 && inputsValid
      case 'steps':
        return (
          values.steps.length > 0 &&
          values.steps.every((s) => s.agent_id)
        )
      case 'execution':
        return true
      case 'schedule':
        return true
      default:
        return false
    }
  }, [currentStepId, methods, inputsValid, watchedName, watchedSteps, watchedInputs])

  const handleNext = () => {
    if (currentStep < STEPS.length - 1) {
      setCurrentStep(currentStep + 1)
    }
  }

  const handleBack = () => {
    if (currentStep > 0) {
      setCurrentStep(currentStep - 1)
    }
  }

  const handleSave = async () => {
    const data = methods.getValues()
    if (isEditMode && recipeId) {
      await updateRecipe(recipeId, data, () => {
        onSave?.(data)
        handleClose()
      })
    } else {
      await submitRecipe(data, () => {
        onSave?.(data)
        // For trigger recipes, stay open so user can see/copy the webhook URL
        if (data.schedule_config.type !== 'trigger') {
          handleClose()
        }
      })
    }
  }

  const handleClose = () => {
    setCurrentStep(0)
    methods.reset()
    onClose()
  }

  const isLastStep = currentStep === STEPS.length - 1

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
            onClick={handleClose}
          />

          {/* Modal */}
          <motion.div
            className="fixed inset-0 z-50 flex items-center justify-center p-4"
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.95 }}
            transition={{ duration: 0.2 }}
          >
            <Card className="glass-card card-glow w-full max-w-6xl flex flex-col max-h-[90vh]">
              <CardHeader className="flex flex-row items-center justify-between border-b border-border/30 pb-4 flex-shrink-0">
                <CardTitle className="flex items-center space-x-2">
                  <ChefHat className="w-6 h-6 text-primary" />
                  <span>{isEditMode ? 'Edit' : 'Create'} <span className="gradient-text">Recipe</span></span>
                </CardTitle>
                <Button variant="ghost" size="icon" onClick={handleClose}>
                  <X className="w-5 h-5" />
                </Button>
              </CardHeader>

              <CardContent className="overflow-y-auto pt-6 flex-1 min-h-0">
                <FormProvider {...methods}>
                  <div className="flex gap-6">
                    {/* Left: Form Content */}
                    <div className="flex-1 min-w-0">
                      {/* Step Tabs */}
                      <Tabs value={currentStepId} className="space-y-6">
                        <TabsList className="grid w-full grid-cols-4 bg-secondary/50">
                          {STEPS.map((step, index) => {
                            const Icon = step.icon
                            return (
                              <TabsTrigger
                                key={step.id}
                                value={step.id}
                                disabled={index > currentStep}
                                onClick={() => {
                                  if (index <= currentStep) {
                                    setCurrentStep(index)
                                  }
                                }}
                                className="text-xs sm:text-sm gap-1.5"
                              >
                                <Icon className="w-3.5 h-3.5 hidden sm:block" />
                                <span className="hidden md:inline">{step.label}</span>
                                <span className="md:hidden">{index + 1}. {step.id === 'basic' ? 'Basic' : step.id === 'steps' ? 'Steps' : step.id === 'execution' ? 'Config' : 'Schedule'}</span>
                              </TabsTrigger>
                            )
                          })}
                        </TabsList>

                        {/* Step 1: Basic Configuration */}
                        <TabsContent value="basic">
                          <motion.div
                            initial={{ opacity: 0, y: 20 }}
                            animate={{ opacity: 1, y: 0 }}
                            transition={{ duration: 0.3 }}
                            className="space-y-4"
                          >
                            <div className="glass-card rounded-2xl p-6 space-y-4">
                              <h3 className="text-sm font-semibold text-muted-foreground uppercase tracking-wider">Basic Information</h3>
                              <div>
                                <Label htmlFor="recipe-name">
                                  Recipe Name <span className="text-red-400">*</span>
                                </Label>
                                <Input
                                  id="recipe-name"
                                  {...methods.register('name', { required: true, minLength: 3 })}
                                  placeholder="My Workflow Recipe"
                                  className="bg-secondary/50 rounded-2xl"
                                />
                                {watchedName.length > 0 && watchedName.length < 3 && (
                                  <p className="text-xs text-[hsl(var(--destructive))] mt-1">Name must be at least 3 characters</p>
                                )}
                              </div>
                              <div>
                                <Label htmlFor="recipe-description">Description</Label>
                                <Textarea
                                  id="recipe-description"
                                  {...methods.register('description')}
                                  placeholder="Describe what this recipe does..."
                                  className="bg-secondary/50 rounded-2xl min-h-[80px]"
                                />
                              </div>
                            </div>

                            <div className="glass-card rounded-2xl p-6 space-y-4">
                              <RecipeInputBuilder
                                value={watchedInputs}
                                onChange={(val) => methods.setValue('inputs', val)}
                                onValidation={(valid) => setInputsValid(valid)}
                                hasTrigger={hasTrigger}
                              />
                            </div>
                          </motion.div>
                        </TabsContent>

                        {/* Step 2: Workflow Steps & Agents */}
                        <TabsContent value="steps">
                          <motion.div
                            initial={{ opacity: 0, y: 20 }}
                            animate={{ opacity: 1, y: 0 }}
                            transition={{ duration: 0.3 }}
                            className="space-y-4"
                          >
                            <div className="glass-card rounded-2xl p-6">
                              <RecipeStepBuilder />
                            </div>
                          </motion.div>
                        </TabsContent>

                        {/* Step 3: Execution Settings */}
                        <TabsContent value="execution">
                          <motion.div
                            initial={{ opacity: 0, y: 20 }}
                            animate={{ opacity: 1, y: 0 }}
                            transition={{ duration: 0.3 }}
                            className="space-y-4"
                          >
                            <RecipeExecutionConfig />
                          </motion.div>
                        </TabsContent>

                        {/* Step 4: Scheduling & Triggers */}
                        <TabsContent value="schedule">
                          <motion.div
                            initial={{ opacity: 0, y: 20 }}
                            animate={{ opacity: 1, y: 0 }}
                            transition={{ duration: 0.3 }}
                            className="space-y-4"
                          >
                            <RecipeScheduleConfig webhookId={lastSavedWebhookId || (initialData?.schedule_config?.webhook_id as string | undefined)} />
                          </motion.div>
                        </TabsContent>
                      </Tabs>

                      {/* Navigation Footer */}
                      <div className="flex justify-between items-center mt-8 pt-6 border-t border-border/30">
                        <Button variant="outline" onClick={handleClose}>
                          Cancel
                        </Button>

                        <div className="text-sm text-muted-foreground">
                          Step {currentStep + 1} of {STEPS.length}
                        </div>

                        <div className="flex gap-2">
                          {currentStep > 0 && (
                            <motion.div whileHover={{ scale: 1.02 }} whileTap={{ scale: 0.98 }}>
                              <Button variant="outline" onClick={handleBack}>
                                <ChevronLeft className="w-4 h-4 mr-1" />
                                Back
                              </Button>
                            </motion.div>
                          )}

                          {!isLastStep ? (
                            <motion.div whileHover={{ scale: 1.02 }} whileTap={{ scale: 0.98 }}>
                              <Button
                                variant="outline"
                                onClick={handleNext}
                                disabled={!canGoNext}
                              >
                                Next
                                <ChevronRight className="w-4 h-4 ml-1" />
                              </Button>
                            </motion.div>
                          ) : (
                            <motion.div whileHover={{ scale: isSubmitting ? 1 : 1.05 }} whileTap={{ scale: isSubmitting ? 1 : 0.95 }}>
                              <Button
                                onClick={handleSave}
                                disabled={isSubmitting}
                              >
                                {isSubmitting ? (
                                  <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                                ) : (
                                  <Save className="w-4 h-4 mr-2" />
                                )}
                                {isSubmitting ? (isEditMode ? 'Updating...' : 'Saving...') : (isEditMode ? 'Update Recipe' : 'Save Recipe')}
                              </Button>
                            </motion.div>
                          )}
                        </div>
                      </div>
                    </div>

                    {/* Right: Preview Panel */}
                    <div className="hidden lg:block w-72 flex-shrink-0 border-l border-border/20 pl-6 overflow-y-auto max-h-[calc(90vh-8rem)]">
                      <RecipePreviewPanel />
                    </div>
                  </div>
                </FormProvider>
              </CardContent>
            </Card>
          </motion.div>
        </>
      )}
    </AnimatePresence>
  )
}
