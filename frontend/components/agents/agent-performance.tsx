
'use client'

import { useState } from 'react'
import { motion } from 'framer-motion'
import { 
  BarChart, 
  TrendingUp, 
  Clock, 
  Target, 
  Zap,
  Activity,
  CheckCircle,
  AlertTriangle
} from 'lucide-react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { ResponsiveContainer, LineChart, Line, XAxis, YAxis, Tooltip, BarChart as RechartsBarChart, Bar } from 'recharts'

const performanceData = [
  { time: '00:00', CodeArchitect: 95, BugHunter: 92, SecurityGuard: 98, TestMaster: 89 },
  { time: '04:00', CodeArchitect: 87, BugHunter: 94, SecurityGuard: 96, TestMaster: 91 },
  { time: '08:00', CodeArchitect: 96, BugHunter: 88, SecurityGuard: 99, TestMaster: 94 },
  { time: '12:00', CodeArchitect: 98, BugHunter: 95, SecurityGuard: 97, TestMaster: 92 },
  { time: '16:00', CodeArchitect: 94, BugHunter: 91, SecurityGuard: 98, TestMaster: 88 },
  { time: '20:00', CodeArchitect: 91, BugHunter: 93, SecurityGuard: 96, TestMaster: 90 }
]

const taskCompletionData = [
  { agent: 'CodeArchitect', completed: 1247, failed: 23, success_rate: 98.2 },
  { agent: 'BugHunter', completed: 892, failed: 18, success_rate: 98.0 },
  { agent: 'SecurityGuard', completed: 567, failed: 8, success_rate: 98.6 },
  { agent: 'TestMaster', completed: 1156, failed: 44, success_rate: 96.3 },
  { agent: 'PerformanceOptimizer', completed: 234, failed: 12, success_rate: 95.1 },
  { agent: 'DocuMentor', completed: 678, failed: 15, success_rate: 97.8 }
]

const agentMetrics = [
  {
    id: 'agent-001',
    name: 'CodeArchitect',
    performance: 96.5,
    uptime: 99.8,
    avgResponseTime: 1.2,
    tasksPerHour: 12.5,
    errorRate: 1.8,
    trend: 'up',
    status: 'excellent'
  },
  {
    id: 'agent-002',
    name: 'BugHunter',
    performance: 94.2,
    uptime: 98.9,
    avgResponseTime: 2.1,
    tasksPerHour: 8.3,
    errorRate: 2.0,
    trend: 'stable',
    status: 'good'
  },
  {
    id: 'agent-003',
    name: 'SecurityGuard',
    performance: 98.1,
    uptime: 99.9,
    avgResponseTime: 0.8,
    tasksPerHour: 6.7,
    errorRate: 1.4,
    trend: 'up',
    status: 'excellent'
  },
  {
    id: 'agent-004',
    name: 'TestMaster',
    performance: 91.8,
    uptime: 97.8,
    avgResponseTime: 3.2,
    tasksPerHour: 15.2,
    errorRate: 3.7,
    trend: 'down',
    status: 'warning'
  }
]

const statusStyles: Record<string, string> = {
  excellent: 'bg-green-500/10 text-green-400 border-green-500/20',
  good: 'bg-blue-500/10 text-blue-400 border-blue-500/20',
  warning: 'bg-yellow-500/10 text-yellow-400 border-yellow-500/20',
  critical: 'bg-red-500/10 text-red-400 border-red-500/20'
}

export function AgentPerformance() {
  const [timeRange, setTimeRange] = useState('24h')
  const [selectedMetric, setSelectedMetric] = useState('performance')

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold">Agent Performance Analytics</h2>
          <p className="text-muted-foreground">
            Monitor and analyze agent performance metrics and trends
          </p>
        </div>
        <div className="flex space-x-2">
          <Select value={timeRange} onValueChange={setTimeRange}>
            <SelectTrigger className="w-32 bg-secondary/50">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="1h">Last Hour</SelectItem>
              <SelectItem value="24h">Last 24h</SelectItem>
              <SelectItem value="7d">Last 7 days</SelectItem>
              <SelectItem value="30d">Last 30 days</SelectItem>
            </SelectContent>
          </Select>
        </div>
      </div>

      {/* Performance Overview */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        {[
          { 
            label: 'Avg Performance', 
            value: '94.7%', 
            change: '+2.3%', 
            icon: BarChart,
            color: 'text-green-400'
          },
          { 
            label: 'Total Tasks', 
            value: '4,774', 
            change: '+156', 
            icon: Target,
            color: 'text-blue-400'
          },
          { 
            label: 'Avg Response Time', 
            value: '1.8s', 
            change: '-0.2s', 
            icon: Clock,
            color: 'text-orange-400'
          },
          { 
            label: 'System Uptime', 
            value: '99.2%', 
            change: '+0.1%', 
            icon: Activity,
            color: 'text-purple-400'
          }
        ].map((metric, index) => (
          <motion.div
            key={metric.label}
            className="glass-card p-6 card-glow hover:border-primary/20 transition-all duration-300"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: index * 0.1 }}
          >
            <div className="flex items-center justify-between mb-4">
              <div className="w-10 h-10 rounded-lg bg-secondary/50 flex items-center justify-center">
                <metric.icon className={`w-5 h-5 ${metric.color}`} />
              </div>
              <Badge variant="secondary" className="text-xs">
                {metric.change}
              </Badge>
            </div>
            <div className="space-y-1">
              <h3 className="text-2xl font-bold">{metric.value}</h3>
              <p className="text-muted-foreground text-sm">{metric.label}</p>
            </div>
          </motion.div>
        ))}
      </div>

      {/* Performance Chart */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5, delay: 0.4 }}
      >
        <Card className="glass-card">
          <CardHeader>
            <CardTitle>Performance Trends</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="h-80">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={performanceData}>
                  <XAxis 
                    dataKey="time" 
                    axisLine={false}
                    tickLine={false}
                    tick={{ fontSize: 12, fill: 'hsl(var(--muted-foreground))' }}
                  />
                  <YAxis 
                    axisLine={false}
                    tickLine={false}
                    tick={{ fontSize: 12, fill: 'hsl(var(--muted-foreground))' }}
                  />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: 'hsl(var(--card))',
                      border: '1px solid hsl(var(--border))',
                      borderRadius: '8px',
                      fontSize: '12px'
                    }}
                  />
                  <Line 
                    type="monotone" 
                    dataKey="CodeArchitect" 
                    stroke="#ff6b35" 
                    strokeWidth={2}
                    name="CodeArchitect"
                  />
                  <Line 
                    type="monotone" 
                    dataKey="BugHunter" 
                    stroke="#60B5FF" 
                    strokeWidth={2}
                    name="BugHunter"
                  />
                  <Line 
                    type="monotone" 
                    dataKey="SecurityGuard" 
                    stroke="#72BF78" 
                    strokeWidth={2}
                    name="SecurityGuard"
                  />
                  <Line 
                    type="monotone" 
                    dataKey="TestMaster" 
                    stroke="#A19AD3" 
                    strokeWidth={2}
                    name="TestMaster"
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </CardContent>
        </Card>
      </motion.div>

      {/* Agent Metrics Table */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5, delay: 0.6 }}
      >
        <Card className="glass-card">
          <CardHeader>
            <CardTitle>Detailed Agent Metrics</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="overflow-x-auto">
              <table className="data-table">
                <thead>
                  <tr>
                    <th>Agent</th>
                    <th>Performance</th>
                    <th>Uptime</th>
                    <th>Response Time</th>
                    <th>Tasks/Hour</th>
                    <th>Error Rate</th>
                    <th>Status</th>
                    <th>Trend</th>
                  </tr>
                </thead>
                <tbody>
                  {agentMetrics.map(agent => (
                    <tr key={agent.id}>
                      <td className="font-medium">{agent.name}</td>
                      <td>
                        <div className="flex items-center space-x-2">
                          <span>{agent.performance}%</span>
                          <div className="w-16 bg-secondary rounded-full h-2">
                            <div 
                              className="bg-gradient-to-r from-orange-500 to-red-500 h-2 rounded-full"
                              style={{ width: `${agent.performance}%` }}
                            />
                          </div>
                        </div>
                      </td>
                      <td>{agent.uptime}%</td>
                      <td>{agent.avgResponseTime}s</td>
                      <td>{agent.tasksPerHour}</td>
                      <td>{agent.errorRate}%</td>
                      <td>
                        <Badge className={statusStyles[agent.status]}>
                          {agent.status === 'excellent' && <CheckCircle className="w-3 h-3 mr-1" />}
                          {agent.status === 'warning' && <AlertTriangle className="w-3 h-3 mr-1" />}
                          {agent.status}
                        </Badge>
                      </td>
                      <td>
                        <div className="flex items-center">
                          {agent.trend === 'up' && <TrendingUp className="w-4 h-4 text-green-400" />}
                          {agent.trend === 'down' && <TrendingUp className="w-4 h-4 text-red-400 rotate-180" />}
                          {agent.trend === 'stable' && <div className="w-4 h-0.5 bg-muted-foreground" />}
                        </div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </CardContent>
        </Card>
      </motion.div>

      {/* Task Completion Chart */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5, delay: 0.8 }}
      >
        <Card className="glass-card">
          <CardHeader>
            <CardTitle>Task Completion Statistics</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <RechartsBarChart data={taskCompletionData}>
                  <XAxis 
                    dataKey="agent" 
                    axisLine={false}
                    tickLine={false}
                    tick={{ fontSize: 12, fill: 'hsl(var(--muted-foreground))' }}
                    angle={-45}
                    textAnchor="end"
                    height={80}
                  />
                  <YAxis 
                    axisLine={false}
                    tickLine={false}
                    tick={{ fontSize: 12, fill: 'hsl(var(--muted-foreground))' }}
                  />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: 'hsl(var(--card))',
                      border: '1px solid hsl(var(--border))',
                      borderRadius: '8px',
                      fontSize: '12px'
                    }}
                  />
                  <Bar 
                    dataKey="completed" 
                    fill="#ff6b35" 
                    name="Completed Tasks"
                    radius={[4, 4, 0, 0]}
                  />
                </RechartsBarChart>
              </ResponsiveContainer>
            </div>
          </CardContent>
        </Card>
      </motion.div>
    </div>
  )
}
