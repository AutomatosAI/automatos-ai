'use client'

import React, { useState, useEffect } from 'react'
import Link from 'next/link'
import { useAllApiToggles } from '@/hooks/use-api-toggle'

export default function ApiControlPanel() {
  const { apiToggles, toggleApi, resetAllApis } = useAllApiToggles()
  const [dashboardStatus, setDashboardStatus] = useState('stable')

  const apis = [
    { key: 'systemHealth', name: 'System Health', endpoint: '/api/system/health' },
    { key: 'agents', name: 'Agents', endpoint: '/api/agents' },
    { key: 'documents', name: 'Documents', endpoint: '/api/documents' },
    { key: 'workflows', name: 'Workflows', endpoint: '/api/workflows' },
    { key: 'analytics', name: 'Analytics', endpoint: '/api/analytics' }
  ]

  // Check dashboard status when API toggles change
  useEffect(() => {
    const enabledCount = Object.values(apiToggles).filter(v => v).length
    
    if (enabledCount === 0) {
      setDashboardStatus('stable')
    } else if (enabledCount <= 2) {
      setDashboardStatus('good')
    } else if (enabledCount <= 4) {
      setDashboardStatus('moderate')
    } else {
      setDashboardStatus('at risk')
    }
  }, [apiToggles])

  return (
    <div className='container mx-auto py-8 px-4'>
      <h1 className='text-2xl font-bold mb-6'>API Control Panel</h1>
      
      <div className='mb-8 p-4 border rounded-lg'>
        <div className='flex justify-between items-center mb-4'>
          <h2 className='text-xl font-medium'>
            Dashboard Status: 
            <span className={
              dashboardStatus === 'stable' ? ' text-green-600' : 
              dashboardStatus === 'good' ? ' text-green-500' :
              dashboardStatus === 'moderate' ? ' text-yellow-500' :
              ' text-red-500'
            }> {dashboardStatus}</span>
          </h2>
          <div className='flex gap-4'>
            <button 
              onClick={resetAllApis}
              className='px-3 py-1 bg-red-100 text-red-800 rounded hover:bg-red-200'
            >
              Disable All APIs
            </button>
            <Link href='/' className='text-blue-600 hover:underline'>
              View Dashboard
            </Link>
          </div>
        </div>
        <p className='text-gray-500 mb-2'>
          Enable API endpoints one at a time to test their impact on dashboard stability.
        </p>
        <p className='text-sm text-gray-400'>
          APIs enabled: {Object.values(apiToggles).filter(v => v).length}/{Object.values(apiToggles).length}
        </p>
      </div>
      
      <div className='space-y-4'>
        {apis.map((api) => (
          <div key={api.key} className={'border rounded-lg p-4 ' + (apiToggles[api.key] ? 'border-blue-500' : '')}>
            <div className='flex justify-between items-center'>
              <div>
                <h3 className='font-medium'>{api.name}</h3>
                <p className='text-sm text-gray-500'>{api.endpoint}</p>
              </div>
              <div className='flex items-center gap-3'>
                <span>
                  {apiToggles[api.key] ? '✅ Active' : '❌ Disabled'}
                </span>
                <button 
                  onClick={() => toggleApi(api.key)}
                  className={'px-3 py-1 rounded ' + (
                    apiToggles[api.key] 
                      ? 'bg-red-100 text-red-800 hover:bg-red-200' 
                      : 'bg-green-100 text-green-800 hover:bg-green-200'
                  )}
                >
                  {apiToggles[api.key] ? 'Disable' : 'Enable'}
                </button>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}
