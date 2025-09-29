'use client'

/**
 * Minimal Mock Context Provider
 * Provides React context for mock status visibility
 */

import React, { createContext, useContext, useEffect, useState } from 'react'

interface MockContextValue {
  isMockEnabled: boolean
  mockEndpoints: Record<string, boolean>
  lastMockUsed: string | null
}

const MockContext = createContext<MockContextValue>({
  isMockEnabled: false,
  mockEndpoints: {},
  lastMockUsed: null
})

export function MockProvider({ children }: { children: React.ReactNode }) {
  const [mockState, setMockState] = useState<MockContextValue>({
    isMockEnabled: false,
    mockEndpoints: {},
    lastMockUsed: null
  })

  // Listen for mock usage from window events
  useEffect(() => {
    if (typeof window === 'undefined') return

    // Get initial state
    const automatos = (window as any).automatos
    if (automatos?.mocks) {
      const status = automatos.mocks.status()
      setMockState({
        isMockEnabled: status.global,
        mockEndpoints: status.endpoints,
        lastMockUsed: null
      })
    }

    // Listen for mock usage events (we'll emit these from api-client)
    const handleMockUsed = (event: CustomEvent) => {
      setMockState(prev => ({
        ...prev,
        lastMockUsed: event.detail.endpoint
      }))
      
      // Clear after 3 seconds
      setTimeout(() => {
        setMockState(prev => ({
          ...prev,
          lastMockUsed: null
        }))
      }, 3000)
    }

    window.addEventListener('mock-used' as any, handleMockUsed as any)
    
    return () => {
      window.removeEventListener('mock-used' as any, handleMockUsed as any)
    }
  }, [])

  return (
    <MockContext.Provider value={mockState}>
      {children}
      {/* Optional: Show a small indicator when mock is used */}
      {mockState.lastMockUsed && (
        <div className="fixed bottom-4 right-4 bg-orange-500/10 border border-orange-500/30 text-orange-600 px-3 py-2 rounded-lg text-sm z-50 animate-in fade-in slide-in-from-bottom-2">
          🎭 Mock: {mockState.lastMockUsed}
        </div>
      )}
    </MockContext.Provider>
  )
}

export function useMockStatus() {
  return useContext(MockContext)
}
