'use client'

import { useState } from 'react'
import { Star } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Textarea } from '@/components/ui/textarea'
import { cn } from '@/lib/utils'

interface ReportGradeFormProps {
  currentGrade?: number | null
  currentNotes?: string | null
  onSubmit: (grade: number, notes?: string) => void
  isSubmitting?: boolean
}

export function ReportGradeForm({
  currentGrade,
  currentNotes,
  onSubmit,
  isSubmitting,
}: ReportGradeFormProps) {
  const [grade, setGrade] = useState<number>(currentGrade || 0)
  const [hoveredStar, setHoveredStar] = useState<number>(0)
  const [notes, setNotes] = useState(currentNotes || '')

  const handleSubmit = () => {
    if (grade < 1) return
    onSubmit(grade, notes.trim() || undefined)
  }

  return (
    <div className="space-y-3">
      <div className="flex items-center gap-1">
        <span className="text-sm text-muted-foreground mr-2">Rate:</span>
        {[1, 2, 3, 4, 5].map((s) => (
          <button
            key={s}
            type="button"
            aria-label={`Rate ${s} out of 5`}
            className="p-0.5 hover:scale-110 transition-transform"
            onMouseEnter={() => setHoveredStar(s)}
            onMouseLeave={() => setHoveredStar(0)}
            onClick={() => setGrade(s)}
          >
            <Star
              className={cn(
                'w-5 h-5 transition-colors',
                s <= (hoveredStar || grade)
                  ? 'text-[hsl(var(--warning))] fill-[hsl(var(--warning))]'
                  : 'text-muted-foreground/30'
              )}
            />
          </button>
        ))}
        {grade > 0 && (
          <span className="text-sm font-medium ml-2">{grade}/5</span>
        )}
      </div>

      <Textarea
        value={notes}
        onChange={(e) => setNotes(e.target.value)}
        placeholder="Optional notes (what could be improved?)"
        className="min-h-[60px] text-sm resize-none"
        maxLength={1000}
      />

      <div className="flex justify-end">
        <Button
          size="sm"
          onClick={handleSubmit}
          disabled={grade < 1 || isSubmitting}
        >
          {isSubmitting ? 'Submitting...' : currentGrade ? 'Update Grade' : 'Submit Grade'}
        </Button>
      </div>
    </div>
  )
}
