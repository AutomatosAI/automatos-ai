
'use client'

import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import { useInView } from 'react-intersection-observer'
import { 
  BarChart, 
  TrendingUp, 
  DollarSign, 
  Clock, 
  Cpu,
  Database,
  Network,
  Zap,
  Activity,
  Users,
  Target,
  AlertTriangle,
  CheckCircle,
  RefreshCw,
  Download,
  Calendar
} from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { ResponsiveContainer, LineChart, Line, XAxis, YAxis, Tooltip, Legend, AreaChart, Area, BarChart as RechartsBarChart, Bar } from 'recharts'
import { performanceService, type PerformanceMetrics, type SystemPerformanceData, type CostAnalysisData, type AgentUtilizationData, type SystemAlert } from '@/lib/performance-service'

// Real data will be loaded from backend
const initialPerformanceMetrics = [
  {
    label: 'System Uptime',
    value: '0%',
    change: 'Loading...',
    icon: Activity,
    color: 'text-green-400',
    trend: 'up'
  },
  {
    label: 'Total Cost',
    value: '$0',
    change: 'Loading...',
    icon: DollarSign,
    color: 'text-blue-400',
    trend: 'down'
  },
  {
    label: 'Token Usage',
    value: '0',
    change: 'Loading...',
    icon: Zap,
    color: 'text-orange-400',
    trend: 'up'
  },
  {
    label: 'Avg Response',
    value: '0s',
    change: 'Loading...',
    icon: Clock,
    color: 'text-purple-400',
    trend: 'down'
  }
]

// Hardcoded data removed - will be loaded from backend

const alertStyles: Record<string, string> = {
  warning: 'bg-yellow-500/10 text-yellow-400 border-yellow-500/20',
  info: 'bg-blue-500/10 text-blue-400 border-blue-500/20',
  success: 'bg-green-500/10 text-green-400 border-green-500/20',
  error: 'bg-red-500/10 text-red-400 border-red-500/20'
}

const alertIcons: Record<string, any> = {
  warning: AlertTriangle,
  info: Activity,
  success: CheckCircle,
  error: AlertTriangle
}

export function PerformanceAnalytics() {
  const [selectedTimeRange, setSelectedTimeRange] = useState('7d')
  const [selectedMetric, setSelectedMetric] = useState('all')
  const [performanceMetrics, setPerformanceMetrics] = useState(initialPerformanceMetrics)
  const [systemPerformanceData, setSystemPerformanceData] = useState<SystemPerformanceData[]>([])
  const [costAnalysisData, setCostAnalysisData] = useState<CostAnalysisData[]>([])
  const [agentUtilizationData, setAgentUtilizationData] = useState<AgentUtilizationData[]>([])
  const [systemAlerts, setSystemAlerts] = useState<SystemAlert[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [ref, inView] = useInView({
    triggerOnce: true,
    threshold: 0.1,
  })

  // Load real data from backend
  useEffect(() => {
    loadPerformanceData()
  }, [])

  const loadPerformanceData = async () => {
    try {
      setLoading(true)
      setError(null)

      // Load all performance data in parallel
      const [
        metricsData,
        systemData,
        costData,
        utilizationData,
        alertsData
      ] = await Promise.all([
        performanceService.getPerformanceMetrics(),
        performanceService.getSystemPerformanceData(),
        performanceService.getCostAnalysisData(),
        performanceService.getAgentUtilizationData(),
        performanceService.getSystemAlerts()
      ])

      // Update metrics with real data
      setPerformanceMetrics([
        {
          label: 'System Uptime',
          value: `${metricsData.systemUptime.toFixed(1)}%`,
          change: metricsData.systemUptime > 99 ? 'Excellent' : 'Monitoring',
          icon: Activity,
          color: 'text-green-400',
          trend: 'up'
        },
        {
          label: 'Total Cost',
          value: `$${metricsData.totalCost.toLocaleString()}`,
          change: 'Real-time estimate',
          icon: DollarSign,
          color: 'text-blue-400',
          trend: 'down'
        },
        {
          label: 'Token Usage',
          value: metricsData.tokenUsage.toLocaleString(),
          change: 'Based on CPU usage',
          icon: Zap,
          color: 'text-orange-400',
          trend: 'up'
        },
        {
          label: 'Avg Response',
          value: `${metricsData.avgResponse.toFixed(1)}s`,
          change: 'Real-time measurement',
          icon: Clock,
          color: 'text-purple-400',
          trend: 'down'
        }
      ])

      setSystemPerformanceData(systemData)
      setCostAnalysisData(costData)
      setAgentUtilizationData(utilizationData)
      setSystemAlerts(alertsData)

    } catch (err) {
      console.error('Error loading performance data:', err)
      setError(err instanceof Error ? err.message : 'Failed to load performance data')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="space-y-8">
      {/* Header */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.8 }}
        className="flex items-center justify-between"
      >
        <div>
          <h1 className="text-3xl font-bold mb-2">
            Performance <span className="gradient-text">Analytics</span>
          </h1>
          <p className="text-muted-foreground text-lg">
            Monitor system performance, costs, and optimization opportunities
          </p>
        </div>
        
        <div className="flex space-x-2">
          <Button variant="outline">
            <RefreshCw className="w-4 h-4 mr-2" />
            Refresh
          </Button>
          <Button variant="outline">
            <Download className="w-4 h-4 mr-2" />
            Export
          </Button>
          <Button className="gradient-accent hover:opacity-90">
            <Calendar className="w-4 h-4 mr-2" />
            Custom Report
          </Button>
        </div>
      </motion.div>

      {/* Loading State */}
      {loading && (
        <div className="flex items-center justify-center py-12">
          <div className="text-center">
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary mx-auto mb-4"></div>
            <p className="text-muted-foreground">Loading performance data...</p>
          </div>
        </div>
      )}

      {/* Error State */}
      {error && !loading && (
        <div className="flex items-center justify-center py-12">
          <div className="text-center">
            <BarChart className="h-8 w-8 text-red-400 mx-auto mb-4" />
            <p className="text-red-400 mb-4">Error loading performance data: {error}</p>
            <Button onClick={loadPerformanceData} variant="outline">
              Try Again
            </Button>
          </div>
        </div>
      )}

      {/* Performance Metrics */}
      {!loading && !error && (
        <motion.div
          ref={ref}
          className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6"
          initial={{ opacity: 0, y: 20 }}
          animate={inView ? { opacity: 1, y: 0 } : {}}
          transition={{ duration: 0.8, delay: 0.2 }}
        >
          {performanceMetrics.map((metric, index) => (
          <motion.div
            key={metric.label}
            className="glass-card p-6 card-glow hover:border-primary/20 transition-all duration-300"
            initial={{ opacity: 0, y: 20 }}
            animate={inView ? { opacity: 1, y: 0 } : {}}
            transition={{ duration: 0.8, delay: index * 0.1 }}
          >
            <div className="flex items-center justify-between mb-4">
              <div className="w-10 h-10 rounded-lg bg-secondary/50 flex items-center justify-center">
                <metric.icon className={`w-5 h-5 ${metric.color}`} />
              </div>
              <div className={`flex items-center space-x-1 text-sm ${
                metric.trend === 'up' ? 'text-green-400' : 'text-red-400'
              }`}>
                <TrendingUp className={`w-4 h-4 ${metric.trend === 'down' ? 'rotate-180' : ''}`} />
                <span>{metric.change}</span>
              </div>
            </div>
            <div className="space-y-1">
              <h3 className="text-2xl font-bold">{metric.value}</h3>
              <p className="text-muted-foreground text-sm">{metric.label}</p>
            </div>
          </motion.div>
        ))}
        </motion.div>
      )}

      {/* Analytics Tabs */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={inView ? { opacity: 1, y: 0 } : {}}
        transition={{ duration: 0.8, delay: 0.4 }}
      >
        <Tabs defaultValue="system" className="space-y-6">
          <TabsList className="grid w-full grid-cols-5 lg:w-auto lg:inline-grid bg-secondary/50">
            <TabsTrigger value="system" className="flex items-center space-x-2">
              <Cpu className="w-4 h-4" />
              <span className="hidden sm:inline">System</span>
            </TabsTrigger>
            <TabsTrigger value="costs" className="flex items-center space-x-2">
              <DollarSign className="w-4 h-4" />
              <span className="hidden sm:inline">Costs</span>
            </TabsTrigger>
            <TabsTrigger value="agents" className="flex items-center space-x-2">
              <Users className="w-4 h-4" />
              <span className="hidden sm:inline">Agent Metrics</span>
            </TabsTrigger>
            <TabsTrigger value="alerts" className="flex items-center space-x-2">
              <AlertTriangle className="w-4 h-4" />
              <span className="hidden sm:inline">Alerts</span>
            </TabsTrigger>
            <TabsTrigger value="optimization" className="flex items-center space-x-2">
              <Target className="w-4 h-4" />
              <span className="hidden sm:inline">Optimization</span>
            </TabsTrigger>
          </TabsList>

          <TabsContent value="system" className="space-y-6">
            {/* System Performance Chart */}
            <Card className="glass-card">
              <CardHeader>
                <CardTitle className="flex items-center justify-between">
                  <span>System Performance Metrics</span>
                  <Select value={selectedTimeRange} onValueChange={setSelectedTimeRange}>
                    <SelectTrigger className="w-24 bg-secondary/50">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="1h">1h</SelectItem>
                      <SelectItem value="24h">24h</SelectItem>
                      <SelectItem value="7d">7d</SelectItem>
                      <SelectItem value="30d">30d</SelectItem>
                    </SelectContent>
                  </Select>
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="h-80">
                  <ResponsiveContainer width="100%" height="100%">
                    <AreaChart data={systemPerformanceData}>
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
                      <Legend />
                      <Area 
                        type="monotone" 
                        dataKey="cpu" 
                        stackId="1"
                        stroke="#ff6b35" 
                        fill="#ff6b35"
                        fillOpacity={0.3}
                        name="CPU Usage (%)"
                      />
                      <Area 
                        type="monotone" 
                        dataKey="memory" 
                        stackId="2"
                        stroke="#60B5FF" 
                        fill="#60B5FF"
                        fillOpacity={0.3}
                        name="Memory Usage (%)"
                      />
                      <Area 
                        type="monotone" 
                        dataKey="network" 
                        stackId="3"
                        stroke="#72BF78" 
                        fill="#72BF78"
                        fillOpacity={0.3}
                        name="Network I/O (%)"
                      />
                    </AreaChart>
                  </ResponsiveContainer>
                </div>
              </CardContent>
            </Card>

            {/* Resource Usage Cards */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
              {[
                { name: 'CPU Usage', value: 68, icon: Cpu, color: 'text-orange-400' },
                { name: 'Memory Usage', value: 82, icon: Database, color: 'text-blue-400' },
                { name: 'Network I/O', value: 32, icon: Network, color: 'text-green-400' },
                { name: 'Storage Usage', value: 58, icon: Database, color: 'text-purple-400' }
              ].map((resource, index) => (
                <motion.div
                  key={resource.name}
                  className="glass-card p-6"
                  initial={{ opacity: 0, scale: 0.95 }}
                  animate={{ opacity: 1, scale: 1 }}
                  transition={{ duration: 0.5, delay: index * 0.1 }}
                >
                  <div className="flex items-center justify-between mb-4">
                    <resource.icon className={`w-6 h-6 ${resource.color}`} />
                    <span className="text-2xl font-bold">{resource.value}%</span>
                  </div>
                  <p className="text-sm text-muted-foreground mb-2">{resource.name}</p>
                  <div className="w-full bg-secondary rounded-full h-2">
                    <div 
                      className="bg-gradient-to-r from-orange-500 to-red-500 h-2 rounded-full transition-all duration-300"
                      style={{ width: `${resource.value}%` }}
                    />
                  </div>
                </motion.div>
              ))}
            </div>
          </TabsContent>

          <TabsContent value="costs" className="space-y-6">
            {/* Cost Analysis Chart */}
            <Card className="glass-card">
              <CardHeader>
                <CardTitle>Cost Analysis & Token Usage</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="h-80">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={costAnalysisData}>
                      <XAxis 
                        dataKey="date" 
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
                      <Legend />
                      <Line 
                        type="monotone" 
                        dataKey="cost" 
                        stroke="#ff6b35" 
                        strokeWidth={3}
                        name="Daily Cost ($)"
                      />
                      <Line 
                        type="monotone" 
                        dataKey="api_calls" 
                        stroke="#60B5FF" 
                        strokeWidth={2}
                        name="API Calls (000s)"
                      />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </CardContent>
            </Card>

            {/* Cost Breakdown */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              {[
                { label: 'API Costs', value: '$1,847', percentage: 65, color: 'from-orange-500 to-red-500' },
                { label: 'Infrastructure', value: '$745', percentage: 26, color: 'from-blue-500 to-cyan-500' },
                { label: 'Storage', value: '$255', percentage: 9, color: 'from-green-500 to-emerald-500' }
              ].map((cost, index) => (
                <motion.div
                  key={cost.label}
                  className="glass-card p-6"
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.5, delay: index * 0.1 }}
                >
                  <div className="text-center">
                    <h3 className="text-2xl font-bold mb-1">{cost.value}</h3>
                    <p className="text-muted-foreground text-sm mb-4">{cost.label}</p>
                    <div className="w-full bg-secondary rounded-full h-2 mb-2">
                      <div 
                        className={`bg-gradient-to-r ${cost.color} h-2 rounded-full transition-all duration-300`}
                        style={{ width: `${cost.percentage}%` }}
                      />
                    </div>
                    <p className="text-xs text-muted-foreground">{cost.percentage}% of total</p>
                  </div>
                </motion.div>
              ))}
            </div>
          </TabsContent>

          <TabsContent value="agents" className="space-y-6">
            {/* Agent Utilization Chart */}
            <Card className="glass-card">
              <CardHeader>
                <CardTitle>Agent Utilization & Efficiency</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="h-80">
                  <ResponsiveContainer width="100%" height="100%">
                    <RechartsBarChart data={agentUtilizationData}>
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
                      <Legend />
                      <Bar 
                        dataKey="utilization" 
                        fill="#60B5FF" 
                        name="Utilization (%)"
                        radius={[4, 4, 0, 0]}
                      />
                      <Bar 
                        dataKey="efficiency" 
                        fill="#72BF78" 
                        name="Efficiency (%)"
                        radius={[4, 4, 0, 0]}
                      />
                    </RechartsBarChart>
                  </ResponsiveContainer>
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="alerts" className="space-y-6">
            {/* System Alerts */}
            <Card className="glass-card">
              <CardHeader>
                <CardTitle>System Alerts & Notifications</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  {systemAlerts.map((alert, index) => {
                    const AlertIcon = alertIcons[alert.type]
                    
                    return (
                      <motion.div
                        key={alert.id}
                        className={`p-4 rounded-lg border ${alertStyles[alert.type]}`}
                        initial={{ opacity: 0, x: -20 }}
                        animate={{ opacity: 1, x: 0 }}
                        transition={{ duration: 0.3, delay: index * 0.1 }}
                      >
                        <div className="flex items-start space-x-3">
                          <AlertIcon className="w-5 h-5 mt-0.5" />
                          <div className="flex-1">
                            <div className="flex items-center justify-between mb-1">
                              <h4 className="font-semibold text-sm">{alert.title}</h4>
                              <Badge variant="outline" className="text-xs">
                                {alert.severity}
                              </Badge>
                            </div>
                            <p className="text-sm opacity-90 mb-2">{alert.description}</p>
                            <div className="flex items-center justify-between text-xs opacity-75">
                              <span>Component: {alert.component}</span>
                              <span>{alert.timestamp}</span>
                            </div>
                          </div>
                        </div>
                      </motion.div>
                    )
                  })}
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="optimization" className="space-y-6">
            <Card className="glass-card">
              <CardHeader>
                <CardTitle>Performance Optimization Recommendations</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-6">
                  <div className="text-center text-muted-foreground">
                    Performance optimization recommendations and insights will be displayed here
                  </div>
                </div>
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
      </motion.div>
    </div>
  )
}
