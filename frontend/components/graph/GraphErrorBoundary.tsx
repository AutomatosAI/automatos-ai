"use client";

import React from "react";
import { AlertTriangle } from "lucide-react";

interface State {
  hasError: boolean;
  message: string | null;
}

interface Props {
  children: React.ReactNode;
}

/**
 * Localised error boundary for any graph surface on the GraphView shell
 * (PRD-165 S1 — generalised from the KG-only boundary). Without this, an
 * exception thrown by a renderer (a malformed node, a missing peer dep at
 * runtime, a non-finite coordinate) bubbles to Next's root boundary and shows
 * the opaque "Application error: a client-side exception has occurred".
 * Catching it here keeps the rest of the dashboard usable and surfaces a real
 * message — and reassures the user the agent can still query the graph.
 */
export class GraphErrorBoundary extends React.Component<Props, State> {
  constructor(props: Props) {
    super(props);
    this.state = { hasError: false, message: null };
  }

  static getDerivedStateFromError(error: unknown): State {
    return {
      hasError: true,
      message: error instanceof Error ? error.message : "Unknown render error",
    };
  }

  componentDidCatch(error: unknown, info: unknown) {
    // eslint-disable-next-line no-console
    console.error("[GraphView] render error:", error, info);
  }

  render() {
    if (this.state.hasError) {
      return (
        <div className="flex h-full min-h-[400px] flex-col items-center justify-center gap-2 text-center p-6">
          <AlertTriangle className="w-6 h-6 text-amber-400" />
          <div className="text-sm text-foreground">
            Couldn't render the graph view.
          </div>
          <div className="text-xs text-muted-foreground max-w-md">
            {this.state.message ?? "An exception was thrown by the renderer."}
            {" "}The agent can still query this graph via tools — only the
            browser visualisation is affected.
          </div>
        </div>
      );
    }
    return this.props.children;
  }
}
