'use client'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'

const BASE = (process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000').replace(/\/$/, '')
const PREFIX = (process.env.NEXT_PUBLIC_API_PREFIX || '/api').replace(/\/$/, '')
const API_KEY = process.env.NEXT_PUBLIC_API_KEY || ''
const AUTH = process.env.NEXT_PUBLIC_AUTH_TOKEN || ''

function headers(extra?: Record<string,string>) {
  const h: Record<string,string> = { 'Content-Type':'application/json', ...(extra||{}) }
  if (API_KEY && !h['X-API-Key']) h['X-API-Key'] = API_KEY
  if (AUTH && !h['Authorization']) h['Authorization'] = `Bearer ${AUTH}`
  return h
}

async function http<T=any>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${PREFIX}${path}`, { ...init, headers: headers(init?.headers as any), cache: 'no-store' })
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`)
  return res.json().catch(()=> (null as T))
}

export function usePolicy(policy_id: string){
  return useQuery({ queryKey:['policy', policy_id], queryFn:()=>http(`/policy/${encodeURIComponent(policy_id)}`) })
}

export function useAssemble(policy_id: string){
  return useMutation({ mutationFn:(body:{ q:string; slot_values?: any; tenant_id?: string })=>http(`/policy/${encodeURIComponent(policy_id)}/assemble`, { method:'POST', body: JSON.stringify(body) }) })
}

export function useUpsertPolicy(policy_id: string){
  const qc = useQueryClient()
  return useMutation({ mutationFn:(body:any)=>http(`/policy/${encodeURIComponent(policy_id)}`, { method:'PUT', body: JSON.stringify(body) }), onSuccess:()=>qc.invalidateQueries({ queryKey:['policy', policy_id] }) })
}


