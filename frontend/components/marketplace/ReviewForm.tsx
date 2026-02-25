"use client";

import { useState, useEffect } from "react";
import { Star } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";

interface Review {
  id: string;
  rating: number;
  title: string | null;
  body: string | null;
  is_verified_purchase: boolean;
  created_at: string;
  user_name?: string;
}

interface ReviewFormProps {
  widgetId: string;
  existingReview?: Review | null;
  isInstalled: boolean;
  onSubmit: (review: { rating: number; title: string; body: string }) => void;
  onUpdate?: (review: { rating: number; title: string; body: string }) => void;
}

export function ReviewForm({
  widgetId,
  existingReview,
  isInstalled,
  onSubmit,
  onUpdate,
}: ReviewFormProps) {
  const isEditing = !!existingReview;

  const [rating, setRating] = useState(existingReview?.rating ?? 0);
  const [hoveredStar, setHoveredStar] = useState(0);
  const [title, setTitle] = useState(existingReview?.title ?? "");
  const [body, setBody] = useState(existingReview?.body ?? "");
  const [errors, setErrors] = useState<{ rating?: string; title?: string }>({});

  // Sync form when existingReview changes (e.g. switching to edit mode)
  useEffect(() => {
    if (existingReview) {
      setRating(existingReview.rating);
      setTitle(existingReview.title ?? "");
      setBody(existingReview.body ?? "");
    }
  }, [existingReview]);

  // Don't render if not installed, or if there's already a review and we're not editing
  if (!isInstalled) return null;
  if (existingReview && !onUpdate) return null;

  function validate(): boolean {
    const newErrors: { rating?: string; title?: string } = {};
    if (rating < 1) newErrors.rating = "Please select a rating";
    if (!title.trim()) newErrors.title = "Title is required";
    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  }

  function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!validate()) return;

    const payload = { rating, title: title.trim(), body: body.trim() };

    if (isEditing && onUpdate) {
      onUpdate(payload);
    } else {
      onSubmit(payload);
    }
  }

  return (
    <form
      onSubmit={handleSubmit}
      className="glass-card rounded-2xl border border-border/50 p-5 space-y-4"
    >
      <h3 className="text-sm font-semibold text-foreground">
        {isEditing ? "Update Your Review" : "Write a Review"}
      </h3>

      {/* Star selector */}
      <div className="space-y-1">
        <label className="text-xs text-muted-foreground">Rating</label>
        <div className="flex items-center gap-1">
          {Array.from({ length: 5 }, (_, i) => {
            const starValue = i + 1;
            const isFilled = starValue <= (hoveredStar || rating);
            return (
              <button
                key={i}
                type="button"
                className="p-0.5 rounded transition-transform hover:scale-110 focus:outline-none focus:ring-2 focus:ring-primary/50 focus:ring-offset-1 focus:ring-offset-background"
                onClick={() => {
                  setRating(starValue);
                  setErrors((prev) => ({ ...prev, rating: undefined }));
                }}
                onMouseEnter={() => setHoveredStar(starValue)}
                onMouseLeave={() => setHoveredStar(0)}
                aria-label={`Rate ${starValue} star${starValue !== 1 ? "s" : ""}`}
              >
                <Star
                  className={`h-6 w-6 transition-colors ${
                    isFilled
                      ? "fill-primary text-primary"
                      : "fill-none text-muted-foreground/40"
                  }`}
                />
              </button>
            );
          })}
          {rating > 0 && (
            <span className="ml-2 text-xs text-muted-foreground">
              {rating}/5
            </span>
          )}
        </div>
        {errors.rating && (
          <p className="text-xs text-[hsl(var(--destructive))]">{errors.rating}</p>
        )}
      </div>

      {/* Title input */}
      <div className="space-y-1">
        <label htmlFor={`review-title-${widgetId}`} className="text-xs text-muted-foreground">
          Title
        </label>
        <Input
          id={`review-title-${widgetId}`}
          placeholder="Summarize your experience"
          value={title}
          onChange={(e) => {
            setTitle(e.target.value);
            if (e.target.value.trim()) {
              setErrors((prev) => ({ ...prev, title: undefined }));
            }
          }}
          maxLength={120}
        />
        {errors.title && (
          <p className="text-xs text-[hsl(var(--destructive))]">{errors.title}</p>
        )}
      </div>

      {/* Body textarea */}
      <div className="space-y-1">
        <label htmlFor={`review-body-${widgetId}`} className="text-xs text-muted-foreground">
          Review (optional)
        </label>
        <Textarea
          id={`review-body-${widgetId}`}
          placeholder="Share more details about your experience..."
          value={body}
          onChange={(e) => setBody(e.target.value)}
          rows={4}
          maxLength={2000}
        />
      </div>

      {/* Submit */}
      <div className="flex justify-end">
        <Button type="submit">
          {isEditing ? "Update Review" : "Write Review"}
        </Button>
      </div>
    </form>
  );
}
