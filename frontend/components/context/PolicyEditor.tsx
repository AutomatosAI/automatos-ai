'use client'
import * as React from 'react'

type SlotKey = 'INSTRUCTION'|'MEMORY'|'RETRIEVAL'|'CODE'|'TOOLS'|'CONSTRAINTS'

export function PolicyEditor({ policy, onChange }: { policy: any; onChange: (p:any)=>void }){
  const slots: SlotKey[] = ['INSTRUCTION','MEMORY','RETRIEVAL','CODE','TOOLS','CONSTRAINTS']
  return (
    <div className="grid gap-4">
      {slots.map(s => (
        <div key={s} className="rounded-xl border p-4">
          <div className="font-medium mb-2">{s}</div>
          <div className="flex gap-4 items-center">
            <label className="text-sm w-24">enabled</label>
            <input type="checkbox" checked={policy?.slots?.[s]?.enabled ?? true}
              onChange={e=>onChange({ ...policy, slots:{...policy.slots, [s]:{...policy.slots?.[s], enabled:e.target.checked}} })} />
          </div>
          <div className="flex gap-4 items-center mt-2">
            <label className="text-sm w-24">weight</label>
            <input className="border rounded px-2 py-1 w-28" type="number" step="0.1"
              value={policy?.slots?.[s]?.weight ?? 1}
              onChange={e=>onChange({ ...policy, slots:{...policy.slots, [s]:{...policy.slots?.[s], weight:Number(e.target.value)}} })} />
          </div>
          <div className="flex gap-4 items-center mt-2">
            <label className="text-sm w-24">max_chars</label>
            <input className="border rounded px-2 py-1 w-28" type="number"
              value={policy?.slots?.[s]?.max_chars ?? 2000}
              onChange={e=>onChange({ ...policy, slots:{...policy.slots, [s]:{...policy.slots?.[s], max_chars:Number(e.target.value)}} })} />
          </div>
        </div>
      ))}
    </div>
  )
}


