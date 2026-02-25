"use client";

import { Star, CheckCircle, ChevronLeft, ChevronRight, User } from "lucide-react";
import { Button } from "@/components/ui/button";

interface Review {
  id: string;
  rating: number;
  title: string | null;
  body: string | null;
  is_verified_purchase: boolean;
  created_at: string;
  user_name?: string;
}

interface ReviewListProps {
  reviews: Review[];
  total: number;
  page: number;
  onPageChange: (page: number) => void;
}

function StarRating({ rating }: { rating: number }) {
  return (
    <div className="flex items-center gap-0.5">
      {Array.from({ length: 5 }, (_, i) => (
        <Star
          key={i}
          className={`h-4 w-4 ${
            i < rating
              ? "fill-primary text-primary"
              : "fill-none text-muted-foreground/40"
          }`}
        />
      ))}
    </div>
  );
}

function formatDate(dateStr: string): string {
  const date = new Date(dateStr);
  return date.toLocaleDateString("en-US", {
    year: "numeric",
    month: "short",
    day: "numeric",
  });
}

export function ReviewList({ reviews, total, page, onPageChange }: ReviewListProps) {
  const perPage = 10;
  const totalPages = Math.max(1, Math.ceil(total / perPage));
  const hasPrev = page > 1;
  const hasNext = page < totalPages;

  if (reviews.length === 0 && total === 0) {
    return (
      <div className="glass-card rounded-2xl border border-border/50 p-8 text-center">
        <p className="text-muted-foreground">No reviews yet</p>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      {reviews.map((review) => (
        <div
          key={review.id}
          className="glass-card rounded-2xl border border-border/50 p-4 space-y-3"
        >
          <div className="flex items-start gap-3">
            {/* Avatar placeholder */}
            <div className="flex-shrink-0 h-9 w-9 rounded-full bg-secondary/50 border border-border/30 flex items-center justify-center">
              <User className="h-4 w-4 text-muted-foreground" />
            </div>

            <div className="flex-1 min-w-0">
              {/* Header row: name + date */}
              <div className="flex items-center justify-between gap-2">
                <div className="flex items-center gap-2">
                  <span className="text-sm font-medium text-foreground">
                    {review.user_name || "Anonymous"}
                  </span>
                  {review.is_verified_purchase && (
                    <span className="inline-flex items-center gap-1 text-xs text-[hsl(var(--success))]">
                      <CheckCircle className="h-3 w-3" />
                      Verified
                    </span>
                  )}
                </div>
                <span className="text-xs text-muted-foreground flex-shrink-0">
                  {formatDate(review.created_at)}
                </span>
              </div>

              {/* Star rating */}
              <div className="mt-1">
                <StarRating rating={review.rating} />
              </div>

              {/* Title */}
              {review.title && (
                <h4 className="mt-2 text-sm font-semibold text-foreground">
                  {review.title}
                </h4>
              )}

              {/* Body */}
              {review.body && (
                <p className="mt-1 text-sm text-muted-foreground leading-relaxed">
                  {review.body}
                </p>
              )}
            </div>
          </div>
        </div>
      ))}

      {/* Pagination */}
      {totalPages > 1 && (
        <div className="flex items-center justify-between pt-2">
          <span className="text-xs text-muted-foreground">
            Page {page} of {totalPages} ({total} review{total !== 1 ? "s" : ""})
          </span>
          <div className="flex items-center gap-2">
            <Button
              variant="outline"
              size="sm"
              disabled={!hasPrev}
              onClick={() => onPageChange(page - 1)}
            >
              <ChevronLeft className="h-4 w-4 mr-1" />
              Prev
            </Button>
            <Button
              variant="outline"
              size="sm"
              disabled={!hasNext}
              onClick={() => onPageChange(page + 1)}
            >
              Next
              <ChevronRight className="h-4 w-4 ml-1" />
            </Button>
          </div>
        </div>
      )}
    </div>
  );
}
