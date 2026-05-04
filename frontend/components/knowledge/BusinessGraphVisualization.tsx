"use client";

/**
 * Business Graph Visualization Component
 * D3 force-directed graph for business entity relationships
 * Features: community coloring, god-node scaling, confidence-based edge styling
 */

import React, { useEffect, useRef, useCallback, useMemo } from "react";
import * as d3 from "d3";

// ---------------------------------------------------------------------------
// Types
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

// D3 simulation node (extends GraphNode with x/y/fx/fy)
interface SimNode extends d3.SimulationNodeDatum {
  id: string;
  label: string;
  file_type: string;
  community?: number;
  source_file?: string;
  degree: number;
}

interface SimLink extends d3.SimulationLinkDatum<SimNode> {
  relation: string;
  confidence: string;
  confidence_score: number;
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const ORANGE_ACCENT = "#FF6B35";
const BG_COLOR = "#0f1117";
const NODE_STROKE = "#ffffff";
const LABEL_COLOR = "#e5e7eb";
const DIMMED_OPACITY = 0.15;

// Warm palette for community coloring (up to 12 distinct communities)
const COMMUNITY_COLORS = [
  "#e6194b", // red
  "#f58231", // orange
  "#ffe119", // yellow
  "#3cb44b", // green
  "#42d4f4", // cyan
  "#4363d8", // blue
  "#911eb4", // purple
  "#f032e6", // magenta
  "#fabebe", // pink
  "#9a6324", // brown
  "#800000", // maroon
  "#aaffc3", // mint
];

const communityColor = (community: number | undefined): string => {
  if (community == null) return "#6b7280";
  return COMMUNITY_COLORS[community % COMMUNITY_COLORS.length];
};

/** Map confidence label to SVG stroke-dasharray */
const confidenceDash = (confidence: string): string => {
  switch (confidence.toUpperCase()) {
    case "EXTRACTED":
      return "none"; // solid
    case "INFERRED":
      return "6,4"; // dashed
    case "AMBIGUOUS":
      return "2,3"; // dotted
    default:
      return "none";
  }
};

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

const BusinessGraphVisualization: React.FC<BusinessGraphVisualizationProps> = ({
  graphData,
  onNodeSelect,
  selectedCommunity = null,
  minConfidence = 0,
}) => {
  const svgRef = useRef<SVGSVGElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const simulationRef = useRef<d3.Simulation<SimNode, SimLink> | null>(null);

  // ---- Derived data (filtered, with degree counts) -----------------------

  const { nodes, links } = useMemo(() => {
    // Filter links by minConfidence
    const filteredLinks: SimLink[] = graphData.links
      .filter((l) => l.confidence_score >= minConfidence)
      .map((l) => ({
        source: l.source,
        target: l.target,
        relation: l.relation,
        confidence: l.confidence,
        confidence_score: l.confidence_score,
      }));

    // Compute degree per node from filtered links
    const degreeMap = new Map<string, number>();
    for (const l of filteredLinks) {
      const src = typeof l.source === "string" ? l.source : (l.source as SimNode).id;
      const tgt = typeof l.target === "string" ? l.target : (l.target as SimNode).id;
      degreeMap.set(src, (degreeMap.get(src) ?? 0) + 1);
      degreeMap.set(tgt, (degreeMap.get(tgt) ?? 0) + 1);
    }

    const simNodes: SimNode[] = graphData.nodes.map((n) => ({
      ...n,
      degree: degreeMap.get(n.id) ?? 0,
    }));

    return { nodes: simNodes, links: filteredLinks };
  }, [graphData, minConfidence]);

  // ---- Node radius (god-node scaling) ------------------------------------

  const nodeRadius = useCallback(
    (d: SimNode): number => {
      const maxDegree = Math.max(1, ...nodes.map((n) => n.degree));
      const normalized = d.degree / maxDegree;
      return 6 + normalized * 20; // 6px min, 26px max
    },
    [nodes],
  );

  // ---- Render D3 graph ---------------------------------------------------

  useEffect(() => {
    if (!svgRef.current || nodes.length === 0) return;

    const svg = d3.select(svgRef.current);
    svg.selectAll("*").remove();

    const width = svgRef.current.clientWidth || 800;
    const height = svgRef.current.clientHeight || 600;

    // Background
    svg
      .append("rect")
      .attr("width", width)
      .attr("height", height)
      .attr("fill", BG_COLOR);

    // Zoom container
    const g = svg.append("g");

    const zoomBehavior = d3
      .zoom<SVGSVGElement, unknown>()
      .scaleExtent([0.1, 6])
      .on("zoom", (event) => {
        g.attr("transform", event.transform);
      });

    svg.call(zoomBehavior);

    // ---- Simulation -------------------------------------------------------

    const simulation = d3
      .forceSimulation<SimNode>(nodes)
      .force(
        "link",
        d3
          .forceLink<SimNode, SimLink>(links)
          .id((d) => d.id)
          .distance(100),
      )
      .force("charge", d3.forceManyBody().strength(-250))
      .force("center", d3.forceCenter(width / 2, height / 2))
      .force(
        "collision",
        d3.forceCollide<SimNode>().radius((d) => nodeRadius(d) + 4),
      );

    simulationRef.current = simulation;

    // ---- Links ------------------------------------------------------------

    const linkGroup = g.append("g").attr("class", "links");

    const linkSelection = linkGroup
      .selectAll<SVGLineElement, SimLink>("line")
      .data(links)
      .enter()
      .append("line")
      .attr("stroke", "#8b8b8b")
      .attr("stroke-width", (d) => 1 + d.confidence_score * 2)
      .attr("stroke-opacity", 0.6)
      .attr("stroke-dasharray", (d) => confidenceDash(d.confidence));

    // Link labels (relation)
    const linkLabelGroup = g.append("g").attr("class", "link-labels");

    const linkLabels = linkLabelGroup
      .selectAll<SVGTextElement, SimLink>("text")
      .data(links)
      .enter()
      .append("text")
      .attr("font-size", 9)
      .attr("fill", "#9ca3af")
      .attr("text-anchor", "middle")
      .attr("pointer-events", "none")
      .text((d) => d.relation);

    // ---- Nodes ------------------------------------------------------------

    const nodeGroup = g.append("g").attr("class", "nodes");

    const nodeSelection = nodeGroup
      .selectAll<SVGCircleElement, SimNode>("circle")
      .data(nodes)
      .enter()
      .append("circle")
      .attr("r", (d) => nodeRadius(d))
      .attr("fill", (d) => communityColor(d.community))
      .attr("fill-opacity", (d) => {
        if (selectedCommunity != null && d.community !== selectedCommunity) {
          return DIMMED_OPACITY;
        }
        return 0.85;
      })
      .attr("stroke", NODE_STROKE)
      .attr("stroke-width", 1.5)
      .style("cursor", "pointer")
      .call(
        d3
          .drag<SVGCircleElement, SimNode>()
          .on("start", (event, d) => {
            if (!event.active) simulation.alphaTarget(0.3).restart();
            d.fx = d.x;
            d.fy = d.y;
          })
          .on("drag", (event, d) => {
            d.fx = event.x;
            d.fy = event.y;
          })
          .on("end", (event, d) => {
            if (!event.active) simulation.alphaTarget(0);
            d.fx = null;
            d.fy = null;
          }),
      );

    // Hover effects
    nodeSelection
      .on("mouseenter", function (_event, d) {
        d3.select(this)
          .attr("stroke", ORANGE_ACCENT)
          .attr("stroke-width", 3);
      })
      .on("mouseleave", function (_event, _d) {
        d3.select(this)
          .attr("stroke", NODE_STROKE)
          .attr("stroke-width", 1.5);
      });

    // Click handler
    nodeSelection.on("click", (_event, d) => {
      // Highlight selected node
      nodeSelection
        .attr("stroke", NODE_STROKE)
        .attr("stroke-width", 1.5);
      d3.select(_event.currentTarget as SVGCircleElement)
        .attr("stroke", ORANGE_ACCENT)
        .attr("stroke-width", 3);

      if (onNodeSelect) {
        onNodeSelect({
          id: d.id,
          label: d.label,
          file_type: d.file_type,
          community: d.community,
          source_file: d.source_file,
        });
      }
    });

    // Tooltips
    nodeSelection
      .append("title")
      .text(
        (d) =>
          `${d.label}\nType: ${d.file_type}\nCommunity: ${d.community ?? "none"}\nConnections: ${d.degree}`,
      );

    // ---- Node labels ------------------------------------------------------

    const nodeLabelGroup = g.append("g").attr("class", "node-labels");

    const nodeLabels = nodeLabelGroup
      .selectAll<SVGTextElement, SimNode>("text")
      .data(nodes)
      .enter()
      .append("text")
      .attr("font-size", 11)
      .attr("fill", (d) => {
        if (selectedCommunity != null && d.community !== selectedCommunity) {
          return `rgba(229,231,235,${DIMMED_OPACITY})`;
        }
        return LABEL_COLOR;
      })
      .attr("text-anchor", "middle")
      .attr("dy", (d) => nodeRadius(d) + 14)
      .attr("pointer-events", "none")
      .text((d) => {
        // Truncate long labels
        return d.label.length > 20 ? d.label.slice(0, 18) + "\u2026" : d.label;
      });

    // ---- Tick -------------------------------------------------------------

    simulation.on("tick", () => {
      linkSelection
        .attr("x1", (d) => (d.source as SimNode).x ?? 0)
        .attr("y1", (d) => (d.source as SimNode).y ?? 0)
        .attr("x2", (d) => (d.target as SimNode).x ?? 0)
        .attr("y2", (d) => (d.target as SimNode).y ?? 0);

      linkLabels
        .attr(
          "x",
          (d) =>
            (((d.source as SimNode).x ?? 0) + ((d.target as SimNode).x ?? 0)) /
            2,
        )
        .attr(
          "y",
          (d) =>
            (((d.source as SimNode).y ?? 0) + ((d.target as SimNode).y ?? 0)) /
            2,
        );

      nodeSelection
        .attr("cx", (d) => d.x ?? 0)
        .attr("cy", (d) => d.y ?? 0);

      nodeLabels
        .attr("x", (d) => d.x ?? 0)
        .attr("y", (d) => d.y ?? 0);
    });

    // ---- Cleanup ----------------------------------------------------------

    return () => {
      simulation.stop();
      simulationRef.current = null;
    };
  }, [nodes, links, selectedCommunity, nodeRadius, onNodeSelect]);

  // ---- Render -------------------------------------------------------------

  return (
    <div ref={containerRef} className="relative w-full h-full min-h-[400px]">
      <svg
        ref={svgRef}
        className="w-full h-full"
        style={{ backgroundColor: BG_COLOR }}
      />

      {/* Legend overlay — bottom-left */}
      {nodes.length > 0 && (
        <div
          className="absolute bottom-3 left-3 px-3 py-2 rounded-lg text-xs"
          style={{
            background: "rgba(15, 17, 23, 0.8)",
            backdropFilter: "blur(8px)",
            border: "1px solid rgba(255,255,255,0.1)",
          }}
        >
          <div className="flex items-center gap-4 flex-wrap">
            <span className="text-muted-foreground font-medium">Edges:</span>
            <span className="text-foreground/90 flex items-center gap-1">
              <svg width="24" height="8">
                <line
                  x1="0"
                  y1="4"
                  x2="24"
                  y2="4"
                  stroke="#9ca3af"
                  strokeWidth="2"
                />
              </svg>
              Extracted
            </span>
            <span className="text-foreground/90 flex items-center gap-1">
              <svg width="24" height="8">
                <line
                  x1="0"
                  y1="4"
                  x2="24"
                  y2="4"
                  stroke="#9ca3af"
                  strokeWidth="2"
                  strokeDasharray="6,4"
                />
              </svg>
              Inferred
            </span>
            <span className="text-foreground/90 flex items-center gap-1">
              <svg width="24" height="8">
                <line
                  x1="0"
                  y1="4"
                  x2="24"
                  y2="4"
                  stroke="#9ca3af"
                  strokeWidth="2"
                  strokeDasharray="2,3"
                />
              </svg>
              Ambiguous
            </span>
          </div>
        </div>
      )}

      {/* Empty state */}
      {nodes.length === 0 && (
        <div className="absolute inset-0 flex items-center justify-center">
          <p className="text-muted-foreground text-sm">
            No graph data available
          </p>
        </div>
      )}
    </div>
  );
};

export { BusinessGraphVisualization };
export default BusinessGraphVisualization;
