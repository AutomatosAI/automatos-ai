'use client'
import * as React from 'react'

export function AssemblyPreview({ result }: { result: any }){
  if(!result) return <div className="text-sm text-muted-foreground">Run an assembly preview…</div>
  return (
    <div className="rounded-xl border p-4">
      <div className="font-medium mb-2">Assembled Prompt</div>
      <pre className="whitespace-pre-wrap text-sm">{result?.prompt || JSON.stringify(result,null,2)}</pre>
      {result?.stats && (
        <div className="mt-3 text-sm">
          <div className="font-medium mb-1">Slot Stats</div>
          <pre>{JSON.stringify(result.stats, null, 2)}</pre>
        </div>
      )}
    </div>
  )
}


