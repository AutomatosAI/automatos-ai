
'use client'

import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts'
import { apiClient } from '@/lib/api'

interface ActivityData {
  time: string;
  agents: number;
  workflows: number;
  documents: number;
}

export function ActivityChart() {
  const [data, setData] = useState<ActivityData[]>([])
  const [loading, setLoading] = useState(true)

  const generateRealTimeData = async () => {
    try {
      // Get current system metrics
      const systemMetrics = await apiClient.getSystemMetrics()
      
      // Get current counts
      let agentCount = 0
      let workflowCount = 0
      let documentCount = 0
      
      try {
        const agents = await apiClient.getAgents()
        agentCount = agents.filter(agent => agent.status === 'active').length
      } catch (err) {
        agentCount = 0  // NO FAKE DATA
      }
      
      try {
        const workflows = await apiClient.getWorkflows()
        workflowCount = workflows.filter(workflow => workflow.status === 'active').length
      } catch (err) {
        workflowCount = 0  // NO FAKE DATA
      }
      
      try {
        const documents = await apiClient.getDocuments()
        documentCount = documents.length
      } catch (err) {
        documentCount = 0  // NO FAKE DATA  
      }

      const now = new Date()
      const newDataPoint: ActivityData = {
        time: now.toLocaleTimeString('en-US', { 
          hour12: false, 
          hour: '2-digit', 
          minute: '2-digit' 
        }),
        agents: agentCount,
        workflows: workflowCount,
        documents: documentCount
      }

      setData(prevData => {
        const newData = [...prevData, newDataPoint]
        // Keep only last 20 data points
        return newData.slice(-20)
      })
      
    } catch (error) {
      console.error('Error generating activity data:', error)
      
      // Fallback to simulated data if API fails
      const now = new Date()
      const newDataPoint: ActivityData = {
        time: now.toLocaleTimeString('en-US', { 
          hour12: false, 
          hour: '2-digit', 
          minute: '2-digit' 
        }),
        agents: 0,
        workflows: 0,
        documents: 0
      }

      setData(prevData => {
        const newData = [...prevData, newDataPoint]
        return newData.slice(-20)
      })
    }
  }

  useEffect(() => {
    // Initialize with some historical data
    const initializeData = async () => {
      setLoading(true)
      
      // Generate initial data points
      const initialData: ActivityData[] = []
      const now = new Date()
      
      for (let i = 19; i >= 0; i--) {
        const time = new Date(now.getTime() - i * 60000) // Every minute
        initialData.push({
          time: time.toLocaleTimeString('en-US', { 
            hour12: false, 
            hour: '2-digit', 
            minute: '2-digit' 
          }),
        agents: 0,
        workflows: 0,
        documents: 0
        })
      }
      
      setData(initialData)
      setLoading(false)
    }

    initializeData()

    // Update data every 30 seconds
    const interval = setInterval(generateRealTimeData, 30000)
    
    return () => clearInterval(interval)
  }, [])

  if (loading) {
    return (
      <div className="glass-card p-6">
        <h3 className="text-lg font-semibold mb-4">Activity Overview</h3>
        <div className="h-64 flex items-center justify-center">
          <div className="animate-pulse text-muted-foreground">Loading activity data...</div>
        </div>
      </div>
    )
  }

  return (
    <motion.div
      className="glass-card p-6"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.8 }}
    >
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold">Activity Overview</h3>
        <div className="text-xs text-muted-foreground">
          Real-time • Updates every 30s
        </div>
      </div>
      
      <div className="h-64">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={data}>
            <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
            <XAxis 
              dataKey="time" 
              stroke="hsl(var(--muted-foreground))"
              fontSize={12}
            />
            <YAxis 
              stroke="hsl(var(--muted-foreground))"
              fontSize={12}
            />
            <Tooltip 
              contentStyle={{
                backgroundColor: 'hsl(var(--card))',
                border: '1px solid hsl(var(--border))',
                borderRadius: '8px',
                color: 'hsl(var(--card-foreground))'
              }}
            />
            <Line 
              type="monotone" 
              dataKey="agents" 
              stroke="#f97316" 
              strokeWidth={2}
              dot={{ fill: '#f97316', strokeWidth: 2, r: 3 }}
              name="Active Agents"
            />
            <Line 
              type="monotone" 
              dataKey="workflows" 
              stroke="#3b82f6" 
              strokeWidth={2}
              dot={{ fill: '#3b82f6', strokeWidth: 2, r: 3 }}
              name="Running Workflows"
            />
            <Line 
              type="monotone" 
              dataKey="documents" 
              stroke="#10b981" 
              strokeWidth={2}
              dot={{ fill: '#10b981', strokeWidth: 2, r: 3 }}
              name="Documents"
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
      
      <div className="flex items-center justify-center space-x-6 mt-4 text-sm">
        <div className="flex items-center space-x-2">
          <div className="w-3 h-3 bg-orange-500 rounded-full"></div>
          <span className="text-muted-foreground">Active Agents</span>
        </div>
        <div className="flex items-center space-x-2">
          <div className="w-3 h-3 bg-blue-500 rounded-full"></div>
          <span className="text-muted-foreground">Running Workflows</span>
        </div>
        <div className="flex items-center space-x-2">
          <div className="w-3 h-3 bg-green-500 rounded-full"></div>
          <span className="text-muted-foreground">Documents</span>
        </div>
      </div>
    </motion.div>
  )
}
