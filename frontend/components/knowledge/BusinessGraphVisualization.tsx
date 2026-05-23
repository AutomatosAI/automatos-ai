"use client";

/**
 * Business Graph Visualization — WebGL/Canvas force-directed graph.
 *
 * Features (PRD-009 Phase-2 polish):
 *  - Color modes: by node type OR by community (toggleable from panel)
 *  - Per-type visibility filtering (panel-controlled)
 *  - Click-to-focus: clicked node becomes the center, the rest dims unless
 *    it's in the 1-hop neighbourhood. ESC / second click clears focus.
 *  - Hover tooltip: label + type + degree
 *  - Edge directional particles ("data flow" feel) on focused subgraph
 *  - God-node halo for the top-5 highest-degree nodes
 *  - Adaptive labels: only top-degree nodes labelled at low zoom; expand
 *    as the user zooms in
 *  - Imperative ref forwarded so the panel can trigger zoomToFit / focus
 */

import React, {
  useCallback,
  useMemo,
  useRef,
  useEffect,
  useState,
  forwardRef,
  useImperativeHandle,
} from "react";
import dynamic from "next/dynamic";

const ForceGraph2D = dynamic(() => import("react-force-graph-2d"), {
  ssr: false,
  loading: () => (
    <div className="flex items-center justify-center h-full text-sm text-muted-foreground">
      Loading graph renderer…
    </div>
  ),
}) as any;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface GraphNode {
  id: string;
  label: string;
  file_type: string;
  community?: number;
  source_file?: string;
}

export interface GraphLink {
  source: string;
  target: string;
  relation: string;
  confidence: string;
  confidence_score: number;
}

export type ColorMode = "type" | "community";

export interface BusinessGraphVisualizationProps {
  graphData: { nodes: GraphNode[]; links: GraphLink[] };
  onNodeSelect?: (node: GraphNode | null) => void;
  selectedCommunity?: number | null;
  minConfidence?: number;
  /** Which node `file_type` values to display. Undefined = show all. */
  visibleTypes?: Set<string>;
  /** Coloring strategy. */
  colorMode?: ColorMode;
}

export interface BusinessGraphHandle {
  zoomToFit: () => void;
  resetFocus: () => void;
}

interface VizNode extends GraphNode {
  degree: number;
  x?: number;
  y?: number;
  vx?: number;
  vy?: number;
}

// ---------------------------------------------------------------------------
// Palette — Automatos-aligned, picked to maximise contrast across the
// node types we see in a Shopify catalog graph. Stable color per type.
// ---------------------------------------------------------------------------

const TYPE_COLORS: Record<string, string> = {
  shopify_product: "#ff7849",      // brand orange
  shopify_variant: "#fdba74",      // light orange
  shopify_vendor: "#a78bfa",       // violet
  shopify_collection: "#34d399",   // emerald
  shopify_metafield: "#60a5fa",    // sky blue
  // Catch-alls
  product: "#ff7849",
  variant: "#fdba74",
  vendor: "#a78bfa",
  collection: "#34d399",
  metafield: "#60a5fa",
};

// Soft, distinguishable cluster palette (12 colors cycled).
const COMMUNITY_COLORS = [
  "#ff7849", "#34d399", "#60a5fa", "#a78bfa",
  "#fbbf24", "#f472b6", "#22d3ee", "#fb7185",
  "#84cc16", "#818cf8", "#facc15", "#2dd4bf",
];

const colorByType = (t: string | undefined): string =>
  TYPE_COLORS[t ?? ""] ?? "#94a3b8";

const colorByCommunity = (c: number | undefined): string =>
  c == null ? "#64748b" : COMMUNITY_COLORS[c % COMMUNITY_COLORS.length];

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

const BusinessGraphVisualization = forwardRef<
  BusinessGraphHandle,
  BusinessGraphVisualizationProps
>(function BusinessGraphVisualization(
  {
    graphData,
    onNodeSelect,
    selectedCommunity = null,
    minConfidence = 0,
    visibleTypes,
    colorMode = "community",
  },
  ref,
) {
  const containerRef = useRef<HTMLDivElement>(null);
  const fgRef = useRef<any>(null);
  const [size, setSize] = useState({ w: 800, h: 600 });
  const [hoverNode, setHoverNode] = useState<VizNode | null>(null);
  const [focusNodeId, setFocusNodeId] = useState<string | null>(null);

  // ── Resize observer ────────────────────────────────────────────────────

  useEffect(() => {
    if (!containerRef.current) return;
    const el = containerRef.current;
    const update = () => setSize({ w: el.clientWidth, h: el.clientHeight || 500 });
    update();
    const ro = new ResizeObserver(update);
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  // ── ESC clears focus ───────────────────────────────────────────────────

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") setFocusNodeId(null);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  // ── Filter + augment ───────────────────────────────────────────────────

  const data = useMemo(() => {
    const wantedTypes = visibleTypes;
    const allowed = (n: GraphNode) =>
      !wantedTypes || wantedTypes.size === 0 || wantedTypes.has(n.file_type);

    const nodes = graphData.nodes.filter(allowed);
    const nodeIds = new Set(nodes.map((n) => n.id));
    const links = graphData.links.filter(
      (l) =>
        l.confidence_score >= minConfidence &&
        nodeIds.has(typeof l.source === "string" ? l.source : (l.source as any).id) &&
        nodeIds.has(typeof l.target === "string" ? l.target : (l.target as any).id),
    );

    const degree = new Map<string, number>();
    for (const l of links) {
      const s = typeof l.source === "string" ? l.source : (l.source as any).id;
      const t = typeof l.target === "string" ? l.target : (l.target as any).id;
      degree.set(s, (degree.get(s) ?? 0) + 1);
      degree.set(t, (degree.get(t) ?? 0) + 1);
    }
    const vizNodes: VizNode[] = nodes.map((n) => ({
      ...n,
      degree: degree.get(n.id) ?? 0,
    }));
    return { nodes: vizNodes, links };
  }, [graphData, visibleTypes, minConfidence]);

  const maxDegree = useMemo(
    () => Math.max(1, ...data.nodes.map((n) => n.degree)),
    [data.nodes],
  );

  // ── 1-hop neighbourhood of the focused node ────────────────────────────

  const focusNeighbourhood = useMemo(() => {
    if (!focusNodeId) return null;
    const neighbours = new Set<string>([focusNodeId]);
    for (const l of data.links) {
      const s = typeof l.source === "object" ? (l.source as any).id : l.source;
      const t = typeof l.target === "object" ? (l.target as any).id : l.target;
      if (s === focusNodeId) neighbours.add(t);
      if (t === focusNodeId) neighbours.add(s);
    }
    return neighbours;
  }, [focusNodeId, data.links]);

  // Top-5 god nodes by degree — get the halo.
  const godNodeIds = useMemo(() => {
    const sorted = [...data.nodes].sort((a, b) => b.degree - a.degree);
    return new Set(sorted.slice(0, 5).map((n) => n.id));
  }, [data.nodes]);

  // ── Coloring ───────────────────────────────────────────────────────────

  const getColor = useCallback(
    (n: VizNode): string =>
      colorMode === "type" ? colorByType(n.file_type) : colorByCommunity(n.community),
    [colorMode],
  );

  // ── Per-node canvas draw ───────────────────────────────────────────────

  const nodeCanvasObject = useCallback(
    (node: any, ctx: CanvasRenderingContext2D, globalScale: number) => {
      const n = node as VizNode;
      const baseR = 2 + 7 * (n.degree / maxDegree);

      // Dim if community filtered OR outside focus neighbourhood
      const dimmedByCommunity =
        selectedCommunity != null && n.community !== selectedCommunity;
      const dimmedByFocus =
        focusNeighbourhood != null && !focusNeighbourhood.has(n.id);
      const dimmed = dimmedByCommunity || dimmedByFocus;
      ctx.globalAlpha = dimmed ? 0.12 : 0.95;

      const fill = getColor(n);

      // God-node halo (top-5 by degree)
      if (godNodeIds.has(n.id) && !dimmed) {
        ctx.beginPath();
        ctx.arc(node.x, node.y, baseR + 6, 0, 2 * Math.PI);
        const halo = ctx.createRadialGradient(
          node.x, node.y, baseR,
          node.x, node.y, baseR + 6,
        );
        halo.addColorStop(0, fill + "AA");
        halo.addColorStop(1, fill + "00");
        ctx.fillStyle = halo;
        ctx.fill();
      }

      // Hover/focus emphasis ring
      const isHover = hoverNode?.id === n.id;
      const isFocus = focusNodeId === n.id;
      if (isHover || isFocus) {
        ctx.beginPath();
        ctx.arc(node.x, node.y, baseR + 3, 0, 2 * Math.PI);
        ctx.lineWidth = 2 / globalScale;
        ctx.strokeStyle = "#ffffff";
        ctx.stroke();
      }

      // The node itself
      ctx.beginPath();
      ctx.arc(node.x, node.y, baseR, 0, 2 * Math.PI);
      ctx.fillStyle = fill;
      ctx.fill();
      ctx.lineWidth = 0.5 / globalScale;
      ctx.strokeStyle = "rgba(255,255,255,0.35)";
      ctx.stroke();

      // Label — only at higher zoom OR for god-nodes OR for hovered/focused
      const showLabel =
        isHover || isFocus || godNodeIds.has(n.id) || globalScale > 1.8;
      if (showLabel && n.label) {
        const fontSize = Math.max(9, 13 / globalScale);
        ctx.font = `${fontSize}px Inter, ui-sans-serif`;
        ctx.fillStyle = "#f1f5f9";
        ctx.textBaseline = "middle";
        // Subtle text shadow for readability over dense edge fields
        ctx.shadowColor = "rgba(0,0,0,0.85)";
        ctx.shadowBlur = 3;
        ctx.fillText(n.label, node.x + baseR + 3, node.y);
        ctx.shadowBlur = 0;
      }
      ctx.globalAlpha = 1;
    },
    [maxDegree, selectedCommunity, focusNeighbourhood, getColor, godNodeIds, hoverNode, focusNodeId],
  );

  const nodePointerAreaPaint = useCallback(
    (node: any, color: string, ctx: CanvasRenderingContext2D) => {
      const baseR = 2 + 7 * ((node as VizNode).degree / maxDegree);
      ctx.fillStyle = color;
      ctx.beginPath();
      ctx.arc(node.x, node.y, baseR + 4, 0, 2 * Math.PI);
      ctx.fill();
    },
    [maxDegree],
  );

  // ── Link styling ───────────────────────────────────────────────────────

  const linkColor = useCallback(
    (l: any) => {
      const s = typeof l.source === "object" ? l.source : null;
      const t = typeof l.target === "object" ? l.target : null;

      // Focus dim
      if (focusNeighbourhood) {
        const touches =
          focusNeighbourhood.has(s?.id) && focusNeighbourhood.has(t?.id);
        return touches ? "rgba(180,200,255,0.55)" : "rgba(120,130,150,0.06)";
      }
      // Community dim
      if (selectedCommunity != null) {
        const touches =
          s?.community === selectedCommunity || t?.community === selectedCommunity;
        return touches ? "rgba(180,200,255,0.45)" : "rgba(120,130,150,0.10)";
      }
      return "rgba(160,170,190,0.28)";
    },
    [selectedCommunity, focusNeighbourhood],
  );

  // Directional particles only on the focused neighbourhood edges — keeps
  // the "data flow" feel without overwhelming a 30k-edge graph.
  const linkParticleCount = useCallback(
    (l: any) => {
      if (!focusNeighbourhood) return 0;
      const s = typeof l.source === "object" ? l.source : null;
      const t = typeof l.target === "object" ? l.target : null;
      return s && t && focusNeighbourhood.has(s.id) && focusNeighbourhood.has(t.id)
        ? 2
        : 0;
    },
    [focusNeighbourhood],
  );

  // ── Click handlers ─────────────────────────────────────────────────────

  const handleNodeClick = useCallback(
    (node: any) => {
      const n = node as VizNode;
      // Click same node again = clear focus.
      const next = focusNodeId === n.id ? null : n.id;
      setFocusNodeId(next);

      if (next && fgRef.current?.centerAt && typeof n.x === "number") {
        fgRef.current.centerAt(n.x, n.y, 600);
        fgRef.current.zoom(Math.max(2.5, fgRef.current.zoom() ?? 1), 600);
      }

      onNodeSelect?.(next ? (n as GraphNode) : null);
    },
    [focusNodeId, onNodeSelect],
  );

  const handleBgClick = useCallback(() => {
    setFocusNodeId(null);
    onNodeSelect?.(null);
  }, [onNodeSelect]);

  // ── Imperative API ─────────────────────────────────────────────────────

  useImperativeHandle(
    ref,
    () => ({
      zoomToFit: () => fgRef.current?.zoomToFit?.(500, 40),
      resetFocus: () => setFocusNodeId(null),
    }),
    [],
  );

  // ── Auto zoom-to-fit on data change ────────────────────────────────────

  useEffect(() => {
    if (!fgRef.current || data.nodes.length === 0) return;
    const timer = setTimeout(() => {
      try {
        fgRef.current.zoomToFit(500, 40);
      } catch {
        /* ignore */
      }
    }, 600);
    return () => clearTimeout(timer);
  }, [data.nodes.length]);

  // ── Render ─────────────────────────────────────────────────────────────

  return (
    <div ref={containerRef} className="relative w-full h-full min-h-[500px]">
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
          backgroundColor="#0a0d14"
          nodeRelSize={4}
          nodeCanvasObject={nodeCanvasObject}
          nodePointerAreaPaint={nodePointerAreaPaint}
          linkColor={linkColor}
          linkWidth={(l: any) => {
            if (!focusNeighbourhood) return 0.5;
            const s = typeof l.source === "object" ? l.source : null;
            const t = typeof l.target === "object" ? l.target : null;
            return s && t && focusNeighbourhood.has(s.id) && focusNeighbourhood.has(t.id) ? 1.5 : 0.4;
          }}
          linkDirectionalParticles={linkParticleCount}
          linkDirectionalParticleSpeed={0.006}
          linkDirectionalParticleWidth={2}
          linkDirectionalParticleColor={() => "#ffffff"}
          cooldownTicks={140}
          d3AlphaDecay={0.025}
          d3VelocityDecay={0.35}
          warmupTicks={20}
          enableNodeDrag={false}
          onNodeClick={handleNodeClick}
          onNodeHover={(n: any) => setHoverNode(n)}
          onBackgroundClick={handleBgClick}
        />
      )}

      {/* Hover tooltip */}
      {hoverNode && (
        <div className="pointer-events-none absolute top-3 left-3 max-w-xs rounded-md border border-white/10 bg-black/80 backdrop-blur-sm px-3 py-2 text-xs text-foreground shadow-lg">
          <div className="font-medium text-sm text-white">{hoverNode.label}</div>
          <div className="text-muted-foreground mt-0.5">
            {hoverNode.file_type.replace(/_/g, " ")}
          </div>
          <div className="text-muted-foreground">
            {hoverNode.degree} relation{hoverNode.degree === 1 ? "" : "s"}
            {hoverNode.community != null && (
              <span className="ml-2">· cluster {hoverNode.community}</span>
            )}
          </div>
        </div>
      )}

      {/* Focus hint */}
      {focusNodeId && (
        <div className="absolute bottom-3 left-3 rounded-md border border-white/10 bg-black/70 backdrop-blur-sm px-3 py-1.5 text-xs text-muted-foreground">
          Focused on neighbourhood — press <kbd className="px-1 rounded bg-white/10 text-white">Esc</kbd> or click background to clear
        </div>
      )}
    </div>
  );
});

export default BusinessGraphVisualization;
