"use client";

/**
 * Business Graph Visualization Component
 * Canvas/WebGL force-directed graph for the workspace knowledge graph.
 *
 * Renderer: react-force-graph-2d (same d3-force physics under the hood,
 * Canvas-based rendering — handles 25k+ nodes smoothly where SVG d3 chokes
 * around 5k). Matches Obsidian's approach for large knowledge graphs.
 *
 * Features kept from the previous SVG implementation:
 *  - Community-based coloring
 *  - Node size proportional to degree ("god-node scaling")
 *  - Click-to-select with selected-community highlighting
 *  - Confidence-score edge filtering
 */

import React, { useCallback, useMemo, useRef, useEffect, useState } from "react";
import dynamic from "next/dynamic";

// SSR-disabled dynamic import — react-force-graph needs window/canvas.
const ForceGraph2D = dynamic(
  () => import("react-force-graph-2d").then((m) => m.default),
  { ssr: false },
) as any;

// ---------------------------------------------------------------------------
// Types — same shape the existing BusinessGraphPanel feeds in.
// ---------------------------------------------------------------------------

interface GraphNode {
  id: string;
  label: string;
  file_type: string;
  community?: number;
  source_file?: string;
}

interface GraphLink {
  source: string;
  target: string;
  relation: string;
  confidence: string;
  confidence_score: number;
}

interface BusinessGraphVisualizationProps {
  graphData: {
    nodes: GraphNode[];
    links: GraphLink[];
  };
  onNodeSelect?: (node: GraphNode) => void;
  selectedCommunity?: number | null;
  minConfidence?: number;
}

// Runtime-augmented node — degree added per render for sizing.
interface VizNode extends GraphNode {
  degree: number;
}

// ---------------------------------------------------------------------------
// Visual constants
// ---------------------------------------------------------------------------

const COMMUNITY_COLORS = [
  "#e6194b", "#f58231", "#ffe119", "#3cb44b", "#42d4f4", "#4363d8",
  "#911eb4", "#f032e6", "#fabebe", "#9a6324", "#800000", "#aaffc3",
  "#469990", "#bfef45", "#fffac8", "#dcbeff", "#9A6324", "#fffac8",
];

const communityColor = (community: number | undefined): string => {
  if (community == null) return "#6b7280";
  return COMMUNITY_COLORS[community % COMMUNITY_COLORS.length];
};

const DIMMED_OPACITY = 0.18;

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

const BusinessGraphVisualization: React.FC<BusinessGraphVisualizationProps> = ({
  graphData,
  onNodeSelect,
  selectedCommunity = null,
  minConfidence = 0,
}) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const fgRef = useRef<any>(null);
  const [size, setSize] = useState<{ w: number; h: number }>({ w: 800, h: 600 });

  // ── Filter links + compute degree per node ──────────────────────────────

  const data = useMemo(() => {
    const filteredLinks = graphData.links.filter(
      (l) => l.confidence_score >= minConfidence,
    );
    const degree = new Map<string, number>();
    for (const l of filteredLinks) {
      const s = typeof l.source === "string" ? l.source : (l.source as any).id;
      const t = typeof l.target === "string" ? l.target : (l.target as any).id;
      degree.set(s, (degree.get(s) ?? 0) + 1);
      degree.set(t, (degree.get(t) ?? 0) + 1);
    }
    const nodes: VizNode[] = graphData.nodes.map((n) => ({
      ...n,
      degree: degree.get(n.id) ?? 0,
    }));
    return { nodes, links: filteredLinks };
  }, [graphData, minConfidence]);

  // For node sizing — relative to the max-degree node in the current view.
  const maxDegree = useMemo(
    () => Math.max(1, ...data.nodes.map((n) => n.degree)),
    [data.nodes],
  );

  // ── Resize observer keeps the canvas filling its container ─────────────

  useEffect(() => {
    if (!containerRef.current) return;
    const el = containerRef.current;
    const update = () => setSize({ w: el.clientWidth, h: el.clientHeight || 500 });
    update();
    const ro = new ResizeObserver(update);
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  // ── Node visual: filled circle, color by community, size by degree.
  // For dense graphs (25k+) drawing per-node labels is too noisy — only
  // show labels for high-degree nodes ("god nodes") and when zoomed in.

  const nodeCanvasObject = useCallback(
    (node: any, ctx: CanvasRenderingContext2D, globalScale: number) => {
      const baseR = 2 + 6 * (node.degree / maxDegree);  // 2–8px radius
      const dim = selectedCommunity != null && node.community !== selectedCommunity;
      ctx.globalAlpha = dim ? DIMMED_OPACITY : 0.95;

      ctx.beginPath();
      ctx.arc(node.x, node.y, baseR, 0, 2 * Math.PI);
      ctx.fillStyle = communityColor(node.community);
      ctx.fill();
      ctx.lineWidth = 0.5 / globalScale;
      ctx.strokeStyle = "rgba(255,255,255,0.4)";
      ctx.stroke();

      // Only label when zoomed enough AND node is significant — avoids
      // unreadable text-spaghetti at low zoom on big graphs.
      const showLabel = globalScale > 1.5 || node.degree > maxDegree * 0.6;
      if (showLabel && node.label) {
        const fontSize = Math.max(8, 12 / globalScale);
        ctx.font = `${fontSize}px Inter, ui-sans-serif`;
        ctx.fillStyle = "#e5e7eb";
        ctx.textBaseline = "middle";
        ctx.fillText(node.label, node.x + baseR + 2, node.y);
      }
      ctx.globalAlpha = 1;
    },
    [maxDegree, selectedCommunity],
  );

  // Click hit-area: extend slightly past the visible radius so it's easy
  // to click small nodes on a dense graph.
  const nodePointerAreaPaint = useCallback(
    (node: any, color: string, ctx: CanvasRenderingContext2D) => {
      const baseR = 2 + 6 * (node.degree / maxDegree);
      ctx.fillStyle = color;
      ctx.beginPath();
      ctx.arc(node.x, node.y, baseR + 3, 0, 2 * Math.PI);
      ctx.fill();
    },
    [maxDegree],
  );

  const linkColor = useCallback(
    (l: any) => {
      // Dim links that don't touch the selected community.
      if (selectedCommunity != null) {
        const s = typeof l.source === "object" ? l.source : null;
        const t = typeof l.target === "object" ? l.target : null;
        const touches = s?.community === selectedCommunity || t?.community === selectedCommunity;
        return touches ? "rgba(180,180,200,0.45)" : `rgba(180,180,200,${DIMMED_OPACITY})`;
      }
      return "rgba(180,180,200,0.30)";
    },
    [selectedCommunity],
  );

  const handleNodeClick = useCallback(
    (node: any) => {
      if (!onNodeSelect) return;
      const { degree, ...plain } = node;
      onNodeSelect(plain as GraphNode);
    },
    [onNodeSelect],
  );

  // ── Zoom to fit on first load / when data changes substantially ────────

  useEffect(() => {
    if (!fgRef.current || data.nodes.length === 0) return;
    // Let the simulation settle a bit before zooming.
    const timer = setTimeout(() => {
      try {
        fgRef.current.zoomToFit(400, 40);
      } catch {
        /* ignore — graph not mounted */
      }
    }, 600);
    return () => clearTimeout(timer);
  }, [data.nodes.length]);

  // ── Render ─────────────────────────────────────────────────────────────

  return (
    <div ref={containerRef} className="w-full h-full min-h-[500px]">
      {data.nodes.length === 0 ? (
        <div className="flex items-center justify-center h-full text-sm text-muted-foreground">
          No nodes match the current filters.
        </div>
      ) : (
        <ForceGraph2D
          ref={fgRef}
          graphData={data}
          width={size.w}
          height={size.h}
          backgroundColor="#0f1117"
          nodeRelSize={4}
          nodeCanvasObject={nodeCanvasObject}
          nodePointerAreaPaint={nodePointerAreaPaint}
          linkColor={linkColor}
          linkWidth={0.5}
          // Force config tuned for large graphs — quicker cooldown so we
          // don't burn CPU rebalancing 25k nodes forever.
          cooldownTicks={120}
          d3AlphaDecay={0.03}
          d3VelocityDecay={0.35}
          warmupTicks={20}
          enableNodeDrag={false}
          onNodeClick={handleNodeClick}
        />
      )}
    </div>
  );
};

export default BusinessGraphVisualization;
