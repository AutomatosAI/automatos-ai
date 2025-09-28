'use client'

import { useState, useEffect } from 'react'
import useLocalStorageState from 'use-local-storage-state'

export function useApiToggle(api) {
  const [apiToggles, setApiToggles] = useLocalStorageState('api-toggles', {
    defaultValue: {
      systemHealth: false,
      agents: false, 
      documents: false,
      workflows: false,
      analytics: false
    }
  })
  
  const isEnabled = apiToggles[api] || false
  
  const toggle = () => {
    setApiToggles(prev => ({
      ...prev,
      [api]: !prev[api]
    }))
  }
  
  return { isEnabled, toggle }
}

export function useAllApiToggles() {
  const [apiToggles, setApiToggles] = useLocalStorageState('api-toggles', {
    defaultValue: {
      systemHealth: false,
      agents: false, 
      documents: false,
      workflows: false,
      analytics: false
    }
  })
  
  const toggleApi = (api) => {
    setApiToggles(prev => ({
      ...prev,
      [api]: !prev[api]
    }))
  }
  
  const resetAllApis = () => {
    setApiToggles({
      systemHealth: false,
      agents: false, 
      documents: false,
      workflows: false,
      analytics: false
    })
  }
  
  return { 
    apiToggles,
    toggleApi,
    resetAllApis
  }
}
