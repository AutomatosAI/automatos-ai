'use client'

import { Check } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { WIZARD_GOALS } from '@/hooks/use-wizard-api'

interface Step1Props {
  selected: string[]
  onChange: (goals: string[]) => void
  onNext: () => void
}

export function Step1Goals({ selected, onChange, onNext }: Step1Props) {
  const toggle = (id: string) => {
    if (selected.includes(id)) {
      onChange(selected.filter(g => g !== id))
    } else {
      onChange([...selected, id])
    }
  }

  return (
    <Card className="bg-secondary/30 border-border/30">
      <CardHeader>
        <CardTitle className="text-base">What do you want Automatos to do for your business?</CardTitle>
        <p className="text-sm text-muted-foreground">
          Pick one or more goals. We&apos;ll use these to recommend the right team of agents.
        </p>
      </CardHeader>
      <CardContent className="space-y-3">
        {WIZARD_GOALS.map(goal => {
          const isSelected = selected.includes(goal.id)
          return (
            <button
              key={goal.id}
              type="button"
              onClick={() => toggle(goal.id)}
              className={`w-full text-left p-4 rounded-lg border transition-all ${
                isSelected
                  ? 'border-primary bg-primary/10'
                  : 'border-border/30 bg-secondary/20 hover:border-border/60'
              }`}
            >
              <div className="flex items-start gap-3">
                <div
                  className={`mt-0.5 w-5 h-5 rounded border flex items-center justify-center flex-shrink-0 ${
                    isSelected ? 'bg-primary border-primary' : 'border-border/60'
                  }`}
                >
                  {isSelected && <Check className="w-3 h-3 text-primary-foreground" />}
                </div>
                <div className="flex-1">
                  <div className="font-medium">{goal.label}</div>
                  <div className="text-sm text-muted-foreground">{goal.description}</div>
                </div>
              </div>
            </button>
          )
        })}

        <div className="flex justify-end pt-2">
          <Button onClick={onNext} disabled={selected.length === 0}>
            Continue
          </Button>
        </div>
      </CardContent>
    </Card>
  )
}
