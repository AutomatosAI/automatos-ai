'use client'

import { motion } from 'framer-motion'
import { CheckCircle, Circle, Loader2 } from 'lucide-react'

interface ProcessingStepsProps {
  steps: string[]
  currentStep: string
}

export function ProcessingSteps({ steps, currentStep }: ProcessingStepsProps) {
  const stepMap: Record<string, number> = {
    'upload': 0,
    'text_extraction': 1,
    'multimodal_extraction': 2,
    'chunking': 3,
    'embedding': 4,
    'storage': 5,
    'finalizing': 6,
  }

  const currentStepIndex = stepMap[currentStep] ?? 0

  const getStepStatus = (index: number) => {
    if (index < currentStepIndex) return 'completed'
    if (index === currentStepIndex) return 'active'
    return 'pending'
  }

  return (
    <div className="flex items-center justify-between py-3">
      {steps.map((step, index) => {
        const status = getStepStatus(index)
        
        return (
          <div key={step} className="flex items-center flex-1">
            {/* Step Circle */}
            <div className="flex flex-col items-center">
              <motion.div
                initial={{ scale: 0.8 }}
                animate={{ 
                  scale: status === 'active' ? [1, 1.1, 1] : 1,
                }}
                transition={{ 
                  repeat: status === 'active' ? Infinity : 0,
                  duration: 1.5 
                }}
                className="relative"
              >
                {status === 'completed' && (
                  <CheckCircle className="w-6 h-6 text-success" />
                )}
                {status === 'active' && (
                  <Loader2 className="w-6 h-6 text-info animate-spin" />
                )}
                {status === 'pending' && (
                  <Circle className="w-6 h-6 text-muted-foreground" />
                )}
              </motion.div>
              
              {/* Step Label */}
              <span className={`text-xs mt-1 text-center max-w-[80px] ${
                status === 'completed' ? 'text-success' :
                status === 'active' ? 'text-info' :
                'text-muted-foreground'
              }`}>
                {step}
              </span>
            </div>

            {/* Connector Line */}
            {index < steps.length - 1 && (
              <div className="flex-1 h-[2px] mx-2 relative">
                <div className="absolute inset-0 bg-secondary" />
                {status === 'completed' && (
                  <motion.div
                    initial={{ width: '0%' }}
                    animate={{ width: '100%' }}
                    transition={{ duration: 0.5 }}
                    className="absolute inset-0 bg-success"
                  />
                )}
              </div>
            )}
          </div>
        )
      })}
    </div>
  )
}


