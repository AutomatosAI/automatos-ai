'use client'

import { motion } from 'framer-motion'
import { Wifi, WifiOff } from 'lucide-react'
import { Badge } from '@/components/ui/badge'

interface LiveIndicatorProps {
  isConnected: boolean
  lastUpdate?: string | null
}

export function LiveIndicator({ isConnected, lastUpdate }: LiveIndicatorProps) {
  return (
    <Badge 
      variant="outline" 
      className={`flex items-center space-x-2 ${
        isConnected 
          ? 'text-success border-success/20 bg-success/10' 
          : 'text-muted-foreground border-border/30 bg-secondary/50'
      }`}
    >
      {isConnected ? (
        <>
          <motion.div
            animate={{ scale: [1, 1.2, 1] }}
            transition={{ repeat: Infinity, duration: 2 }}
          >
            <Wifi className="w-3 h-3" />
          </motion.div>
          <span className="text-xs">Live</span>
          {lastUpdate && (
            <span className="text-xs opacity-70">
              {new Date(lastUpdate).toLocaleTimeString()}
            </span>
          )}
        </>
      ) : (
        <>
          <WifiOff className="w-3 h-3" />
          <span className="text-xs">Disconnected</span>
        </>
      )}
    </Badge>
  )
}


