'use client'

import { motion } from 'framer-motion'
import { 
  Bot, 
  FileText, 
  GitBranch, 
  Activity,
  TrendingUp,
  Clock,
  CheckCircle,
  AlertTriangle,
  Zap,
  Users,
  Database,
  Cpu,
  MemoryStick,
  HardDrive,
  Wifi,
  Brain,
  Target,
  BarChart3
} from 'lucide-react'

export function SimpleDashboard() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 to-gray-800 p-8">
      <div className="max-w-7xl mx-auto space-y-8">
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
          className="text-center"
        >
          <h1 className="text-4xl font-bold text-white mb-2">
            🚀 <span className="gradient-text">Automatos AI</span> Dashboard
          </h1>
          <p className="text-gray-300 text-lg">
            Enterprise AI automation and agent management platform
          </p>
        </motion.div>

        {/* Quick Stats */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.1 }}
          className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6"
        >
          <div className="glass-card p-6 card-glow hover:border-primary/20 transition-all duration-300">
            <div className="flex items-center justify-between mb-4">
              <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-orange-500 to-red-500 flex items-center justify-center">
                <Bot className="w-5 h-5 text-white" />
              </div>
            </div>
            <div className="space-y-1">
              <h3 className="text-2xl font-bold text-white">15</h3>
              <p className="text-muted-foreground text-sm">Active Agents</p>
              <p className="text-xs text-green-400">+3 this week</p>
            </div>
          </div>

          <div className="glass-card p-6 card-glow hover:border-primary/20 transition-all duration-300">
            <div className="flex items-center justify-between mb-4">
              <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-orange-500 to-red-500 flex items-center justify-center">
                <Activity className="w-5 h-5 text-white" />
              </div>
            </div>
            <div className="space-y-1">
              <h3 className="text-2xl font-bold text-white">94.5%</h3>
              <p className="text-muted-foreground text-sm">Success Rate</p>
              <p className="text-xs text-green-400">↑ 2.1% this month</p>
            </div>
          </div>

          <div className="glass-card p-6 card-glow hover:border-primary/20 transition-all duration-300">
            <div className="flex items-center justify-between mb-4">
              <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-orange-500 to-red-500 flex items-center justify-center">
                <GitBranch className="w-5 h-5 text-white" />
              </div>
            </div>
            <div className="space-y-1">
              <h3 className="text-2xl font-bold text-white">8</h3>
              <p className="text-muted-foreground text-sm">Workflows</p>
              <p className="text-xs text-green-400">6 completed</p>
            </div>
          </div>

          <div className="glass-card p-6 card-glow hover:border-primary/20 transition-all duration-300">
            <div className="flex items-center justify-between mb-4">
              <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-orange-500 to-red-500 flex items-center justify-center">
                <Database className="w-5 h-5 text-white" />
              </div>
            </div>
            <div className="space-y-1">
              <h3 className="text-2xl font-bold text-white">234</h3>
              <p className="text-muted-foreground text-sm">Documents</p>
              <p className="text-xs text-green-400">+12 this week</p>
            </div>
          </div>
        </motion.div>

        {/* System Health */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.2 }}
          className="grid grid-cols-1 lg:grid-cols-2 gap-6"
        >
          <div className="glass-card p-6">
            <h3 className="text-xl font-semibold text-white mb-4 flex items-center">
              <Cpu className="w-5 h-5 mr-2 text-orange-400" />
              System Health
            </h3>
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <span className="text-muted-foreground">CPU Usage</span>
                <span className="text-white font-medium">45%</span>
              </div>
              <div className="w-full bg-gray-700 rounded-full h-2">
                <div className="bg-gradient-to-r from-orange-500 to-red-500 h-2 rounded-full" style={{ width: '45%' }}></div>
              </div>
              
              <div className="flex items-center justify-between">
                <span className="text-muted-foreground">Memory Usage</span>
                <span className="text-white font-medium">67%</span>
              </div>
              <div className="w-full bg-gray-700 rounded-full h-2">
                <div className="bg-gradient-to-r from-orange-500 to-red-500 h-2 rounded-full" style={{ width: '67%' }}></div>
              </div>
              
              <div className="flex items-center justify-between">
                <span className="text-muted-foreground">Disk Usage</span>
                <span className="text-white font-medium">23%</span>
              </div>
              <div className="w-full bg-gray-700 rounded-full h-2">
                <div className="bg-gradient-to-r from-orange-500 to-red-500 h-2 rounded-full" style={{ width: '23%' }}></div>
              </div>
            </div>
          </div>

          <div className="glass-card p-6">
            <h3 className="text-xl font-semibold text-white mb-4 flex items-center">
              <TrendingUp className="w-5 h-5 mr-2 text-orange-400" />
              Recent Activity
            </h3>
            <div className="space-y-3">
              <div className="flex items-center space-x-3">
                <div className="w-2 h-2 bg-green-400 rounded-full"></div>
                <span className="text-sm text-muted-foreground">Agent "DataProcessor" completed task</span>
                <span className="text-xs text-gray-500">2m ago</span>
              </div>
              <div className="flex items-center space-x-3">
                <div className="w-2 h-2 bg-orange-400 rounded-full"></div>
                <span className="text-sm text-muted-foreground">Workflow "EmailAnalysis" started</span>
                <span className="text-xs text-gray-500">5m ago</span>
              </div>
              <div className="flex items-center space-x-3">
                <div className="w-2 h-2 bg-blue-400 rounded-full"></div>
                <span className="text-sm text-muted-foreground">New document uploaded</span>
                <span className="text-xs text-gray-500">12m ago</span>
              </div>
              <div className="flex items-center space-x-3">
                <div className="w-2 h-2 bg-green-400 rounded-full"></div>
                <span className="text-sm text-muted-foreground">System optimization completed</span>
                <span className="text-xs text-gray-500">1h ago</span>
              </div>
            </div>
          </div>
        </motion.div>

        {/* Quick Actions */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.3 }}
          className="glass-card p-6"
        >
          <h3 className="text-xl font-semibold text-white mb-4 flex items-center">
            <Zap className="w-5 h-5 mr-2 text-orange-400" />
            Quick Actions
          </h3>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <button className="flex items-center space-x-3 p-4 rounded-lg bg-secondary/30 hover:bg-secondary/50 transition-colors">
              <Bot className="w-5 h-5 text-orange-400" />
              <span className="text-white">Create Agent</span>
            </button>
            <button className="flex items-center space-x-3 p-4 rounded-lg bg-secondary/30 hover:bg-secondary/50 transition-colors">
              <GitBranch className="w-5 h-5 text-orange-400" />
              <span className="text-white">New Workflow</span>
            </button>
            <button className="flex items-center space-x-3 p-4 rounded-lg bg-secondary/30 hover:bg-secondary/50 transition-colors">
              <FileText className="w-5 h-5 text-orange-400" />
              <span className="text-white">Upload Document</span>
            </button>
          </div>
        </motion.div>
      </div>
    </div>
  )
}
