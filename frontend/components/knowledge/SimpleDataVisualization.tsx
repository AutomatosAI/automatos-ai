"use client"

import React, { useState, useEffect } from 'react'
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle } from '@/components/ui/dialog'
import { Button } from '@/components/ui/button'
import {
  BarChart as BarChartIcon,
  Table as TableIcon
} from 'lucide-react'

interface SimpleDataVisualizationProps {
  isOpen: boolean
  onClose: () => void
  data: any[]
}

export function SimpleDataVisualization({ isOpen, onClose, data }: SimpleDataVisualizationProps) {
  const [chartType, setChartType] = useState<'bar' | 'table'>('bar')
  const [mounted, setMounted] = useState(false)

  useEffect(() => {
    setMounted(true)
  }, [])

  if (!mounted || !data || data.length === 0) {
    return null
  }

  const firstRow = data[0]
  const columns = Object.keys(firstRow)
  
  // Find numeric columns
  const numericCols = columns.filter(col => {
    const val = firstRow[col]
    return typeof val === 'number' || (!isNaN(parseFloat(val)) && isFinite(parseFloat(val)))
  })

  const labelCol = columns[0]
  const valueCol = numericCols[0] || columns[1]

  // Find max value for scaling
  const maxValue = Math.max(...data.map(row => {
    const val = row[valueCol]
    return parseFloat(val) || 0
  }))

  return (
    <Dialog open={isOpen} onOpenChange={onClose}>
      <DialogContent className="max-w-4xl">
        <DialogHeader>
          <DialogTitle>Data Visualization</DialogTitle>
          <DialogDescription>
            Visualizing {data.length} rows of data
          </DialogDescription>
        </DialogHeader>

        {/* Chart Type Toggle */}
        <div className="flex gap-2 mb-4">
          <Button
            variant={chartType === 'bar' ? 'default' : 'outline'}
            size="sm"
            onClick={() => setChartType('bar')}
          >
            <BarChartIcon className="h-4 w-4 mr-2" />
            Bar Chart
          </Button>
          <Button
            variant={chartType === 'table' ? 'default' : 'outline'}
            size="sm"
            onClick={() => setChartType('table')}
          >
            <TableIcon className="h-4 w-4 mr-2" />
            Table View
          </Button>
        </div>

        {/* Chart */}
        {chartType === 'bar' && numericCols.length > 0 ? (
          <div className="space-y-3 p-4 bg-card rounded-lg border">
            {data.map((row, idx) => {
              const value = parseFloat(row[valueCol]) || 0
              const percentage = (value / maxValue) * 100
              
              return (
                <div key={idx} className="space-y-1">
                  <div className="flex justify-between text-sm">
                    <span className="font-medium truncate max-w-[200px]">{row[labelCol]}</span>
                    <span className="text-muted-foreground">{value.toFixed(2)}</span>
                  </div>
                  <div className="w-full bg-secondary rounded-full h-6 overflow-hidden">
                    <div
                      className="h-full bg-gradient-to-r from-blue-500 to-purple-500 transition-all duration-500 flex items-center justify-end px-2"
                      style={{ width: `${percentage}%` }}
                    >
                      <span className="text-xs text-white font-medium">
                        {percentage.toFixed(0)}%
                      </span>
                    </div>
                  </div>
                </div>
              )
            })}
          </div>
        ) : (
          <div className="overflow-x-auto border rounded-lg">
            <table className="w-full text-sm">
              <thead className="bg-secondary">
                <tr>
                  {columns.map(col => (
                    <th key={col} className="px-4 py-2 text-left font-medium">
                      {col}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {data.map((row, idx) => (
                  <tr key={idx} className="border-t hover:bg-secondary/50">
                    {columns.map(col => (
                      <td key={col} className="px-4 py-2">
                        {row[col]}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}

        <div className="flex justify-between mt-4">
          <p className="text-sm text-muted-foreground">
            {data.length} rows • {numericCols.length} numeric columns
          </p>
          <Button onClick={onClose}>Close</Button>
        </div>
      </DialogContent>
    </Dialog>
  )
}

