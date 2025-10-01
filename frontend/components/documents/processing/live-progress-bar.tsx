'use client'

import { motion } from 'framer-motion'
import { Progress } from '@/components/ui/progress'
import { Clock } from 'lucide-react'

interface LiveProgressBarProps {
  progress: number
  step?: string
  eta_seconds?: number
}

export function LiveProgressBar({ progress, step, eta_seconds }: LiveProgressBarProps) {
  const formatETA = (seconds: number) => {
    if (seconds < 60) {
      return `~${seconds}s remaining`
    }
    const minutes = Math.floor(seconds / 60)
    const secs = seconds % 60
    return `~${minutes}m ${secs}s remaining`
  }

  return (
    <div className="space-y-2">
      {/* Progress Bar */}
      <div className="relative">
        <Progress value={progress} className="h-3" />
        
        {/* Animated Pulse on Progress Bar */}
        {progress < 100 && (
          <motion.div
            className="absolute top-0 left-0 h-full bg-gradient-to-r from-transparent via-white/20 to-transparent"
            animate={{
              left: ['-20%', '120%'],
            }}
            transition={{
              repeat: Infinity,
              duration: 2,
              ease: 'linear',
            }}
            style={{
              width: '20%',
            }}
          />
        )}
      </div>

      {/* Progress Info */}
      <div className="flex justify-between items-center text-sm">
        <div className="flex items-center space-x-2">
          <span className="text-muted-foreground">Progress:</span>
          <motion.span
            key={progress}
            initial={{ scale: 1.2 }}
            animate={{ scale: 1 }}
            className="font-medium text-foreground"
          >
            {progress}%
          </motion.span>
        </div>

        {eta_seconds !== undefined && eta_seconds > 0 && (
          <div className="flex items-center space-x-2 text-muted-foreground">
            <Clock className="w-4 h-4" />
            <span className="text-xs">{formatETA(eta_seconds)}</span>
          </div>
        )}
      </div>

      {/* Current Step */}
      {step && (
        <div className="text-xs text-muted-foreground">
          Current step: <span className="text-blue-400">{step}</span>
        </div>
      )}
    </div>
  )
}


