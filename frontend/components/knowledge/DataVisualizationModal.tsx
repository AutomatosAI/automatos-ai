"use client"

import React, { useState } from 'react'
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle } from '@/components/ui/dialog'
import { Button } from '@/components/ui/button'
import {
  BarChart as BarChartIcon,
  LineChart as LineChartIcon,
  PieChart as PieChartIcon,
  AreaChart as AreaChartIcon,
  ScatterChart as ScatterChartIcon
} from 'lucide-react'
import {
  BarChart,
  Bar,
  LineChart,
  Line,
  PieChart,
  Pie,
  Cell,
  AreaChart,
  Area,
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer
} from 'recharts'

interface DataVisualizationModalProps {
  isOpen: boolean
  onClose: () => void
  data: any[]
  suggestedChartType?: 'bar' | 'line' | 'pie' | 'area' | 'scatter'
}

export function DataVisualizationModal({ isOpen, onClose, data, suggestedChartType = 'bar' }: DataVisualizationModalProps) {
  const [selectedChartType, setSelectedChartType] = useState<'bar' | 'line' | 'pie' | 'area' | 'scatter'>(suggestedChartType)

  console.log('[DataVisualizationModal] Rendered with:', { isOpen, dataLength: data?.length, suggestedChartType })

  // Helper to check if value is numeric
  const isNumeric = (value: any): boolean => {
    if (typeof value === 'number') return true
    if (typeof value === 'string') return !isNaN(parseFloat(value)) && isFinite(parseFloat(value))
    return false
  }

  // Convert string numbers to actual numbers
  const prepareChartData = (rawData: any[]) => {
    return rawData.map(row => {
      const newRow: any = {}
      Object.keys(row).forEach(key => {
        const value = row[key]
        newRow[key] = isNumeric(value) ? parseFloat(value) : value
      })
      return newRow
    })
  }

  if (!data || data.length === 0) {
    console.log('[DataVisualizationModal] No data, returning null')
    return null
  }

  let chartData, firstRow, columns, numericColumns, labelKey
  
  try {
    chartData = prepareChartData(data)
    firstRow = chartData[0]
    columns = Object.keys(firstRow)
    numericColumns = columns.filter(col => typeof firstRow[col] === 'number')
    labelKey = columns[0]
    
    console.log('[DataVisualizationModal] Prepared data:', { 
      dataPoints: chartData.length, 
      columns, 
      numericColumns,
      labelKey 
    })
  } catch (error) {
    console.error('[DataVisualizationModal] Data preparation error:', error)
    return (
      <Dialog open={isOpen} onOpenChange={onClose}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Visualization Error</DialogTitle>
          </DialogHeader>
          <div className="text-destructive p-4">
            <p>Failed to prepare data: {error instanceof Error ? error.message : 'Unknown error'}</p>
          </div>
          <Button onClick={onClose}>Close</Button>
        </DialogContent>
      </Dialog>
    )
  }

  const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042', '#8884D8', '#82CA9D']

  return (
    <Dialog open={isOpen} onOpenChange={onClose}>
      <DialogContent className="max-w-4xl max-h-[90vh] overflow-y-auto">
        <DialogHeader>
          <DialogTitle>Data Visualization</DialogTitle>
          <DialogDescription>
            Choose a chart type to visualize your query results
          </DialogDescription>
        </DialogHeader>

        {/* Chart Type Selector */}
        <div className="flex gap-2 flex-wrap mb-4">
          <Button
            variant={selectedChartType === 'bar' ? 'default' : 'outline'}
            size="sm"
            onClick={() => setSelectedChartType('bar')}
          >
            <BarChartIcon className="h-4 w-4 mr-2" />
            Bar Chart
          </Button>
          <Button
            variant={selectedChartType === 'line' ? 'default' : 'outline'}
            size="sm"
            onClick={() => setSelectedChartType('line')}
          >
            <LineChartIcon className="h-4 w-4 mr-2" />
            Line Chart
          </Button>
          <Button
            variant={selectedChartType === 'pie' ? 'default' : 'outline'}
            size="sm"
            onClick={() => setSelectedChartType('pie')}
          >
            <PieChartIcon className="h-4 w-4 mr-2" />
            Pie Chart
          </Button>
          <Button
            variant={selectedChartType === 'area' ? 'default' : 'outline'}
            size="sm"
            onClick={() => setSelectedChartType('area')}
          >
            <AreaChartIcon className="h-4 w-4 mr-2" />
            Area Chart
          </Button>
          <Button
            variant={selectedChartType === 'scatter' ? 'default' : 'outline'}
            size="sm"
            onClick={() => setSelectedChartType('scatter')}
          >
            <ScatterChartIcon className="h-4 w-4 mr-2" />
            Scatter Plot
          </Button>
        </div>

        {/* Chart Rendering */}
        {numericColumns.length === 0 ? (
          <div className="text-center text-warning py-8">
            No numeric columns found in the data. Charts require at least one numeric column.
          </div>
        ) : (
          <div className="w-full h-96 mt-4">
            <ResponsiveContainer width="100%" height="100%">
              {selectedChartType === 'bar' && (
                <BarChart data={chartData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
                  <XAxis 
                    dataKey={labelKey} 
                    tick={{ fill: 'hsl(var(--foreground))' }}
                  />
                  <YAxis tick={{ fill: 'hsl(var(--foreground))' }} />
                  <Tooltip 
                    contentStyle={{ 
                      backgroundColor: 'hsl(var(--card))', 
                      border: '1px solid hsl(var(--border))',
                      borderRadius: '8px'
                    }}
                  />
                  <Legend />
                  {numericColumns.map((key, idx) => (
                    <Bar key={key} dataKey={key} fill={COLORS[idx % COLORS.length]} />
                  ))}
                </BarChart>
              )}

              {selectedChartType === 'line' && (
                <LineChart data={chartData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
                  <XAxis 
                    dataKey={labelKey}
                    tick={{ fill: 'hsl(var(--foreground))' }}
                  />
                  <YAxis tick={{ fill: 'hsl(var(--foreground))' }} />
                  <Tooltip 
                    contentStyle={{ 
                      backgroundColor: 'hsl(var(--card))', 
                      border: '1px solid hsl(var(--border))',
                      borderRadius: '8px'
                    }}
                  />
                  <Legend />
                  {numericColumns.map((key, idx) => (
                    <Line key={key} type="monotone" dataKey={key} stroke={COLORS[idx % COLORS.length]} strokeWidth={2} />
                  ))}
                </LineChart>
              )}

              {selectedChartType === 'pie' && (
                <PieChart>
                  <Pie
                    data={chartData}
                    dataKey={numericColumns[0]}
                    nameKey={labelKey}
                    cx="50%"
                    cy="50%"
                    outerRadius={120}
                    label={(entry) => `${entry[labelKey]}: ${parseFloat(entry[numericColumns[0]]).toFixed(2)}`}
                  >
                    {chartData.map((entry: any, index: number) => (
                      <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                    ))}
                  </Pie>
                  <Tooltip />
                  <Legend />
                </PieChart>
              )}

              {selectedChartType === 'area' && (
                <AreaChart data={chartData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
                  <XAxis 
                    dataKey={labelKey}
                    tick={{ fill: 'hsl(var(--foreground))' }}
                  />
                  <YAxis tick={{ fill: 'hsl(var(--foreground))' }} />
                  <Tooltip 
                    contentStyle={{ 
                      backgroundColor: 'hsl(var(--card))', 
                      border: '1px solid hsl(var(--border))',
                      borderRadius: '8px'
                    }}
                  />
                  <Legend />
                  {numericColumns.map((key, idx) => (
                    <Area 
                      key={key} 
                      type="monotone" 
                      dataKey={key} 
                      stroke={COLORS[idx % COLORS.length]} 
                      fill={COLORS[idx % COLORS.length]} 
                      fillOpacity={0.6} 
                    />
                  ))}
                </AreaChart>
              )}

              {selectedChartType === 'scatter' && (
                <ScatterChart>
                  <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
                  <XAxis 
                    type="number"
                    dataKey={numericColumns[0]} 
                    name={numericColumns[0]}
                    tick={{ fill: 'hsl(var(--foreground))' }}
                  />
                  <YAxis 
                    type="number"
                    dataKey={numericColumns[1] || numericColumns[0]} 
                    name={numericColumns[1] || numericColumns[0]}
                    tick={{ fill: 'hsl(var(--foreground))' }}
                  />
                  <Tooltip cursor={{ strokeDasharray: '3 3' }} />
                  <Legend />
                  <Scatter name="Data Points" data={chartData} fill="#8884d8" />
                </ScatterChart>
              )}
            </ResponsiveContainer>
          </div>
        )}

        <div className="flex justify-between mt-4">
          <p className="text-sm text-muted-foreground">
            Showing {data.length} data points • {numericColumns.length} numeric columns
          </p>
          <Button variant="outline" onClick={onClose}>
            Close
          </Button>
        </div>
      </DialogContent>
    </Dialog>
  )
}

