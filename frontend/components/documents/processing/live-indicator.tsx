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
          ? 'text-green-400 border-green-500/20 bg-green-500/10' 
          : 'text-gray-400 border-gray-500/20 bg-gray-500/10'
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


