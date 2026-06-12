"use client";

/**
 * Knowledge Graph Visualizer Component
 * Interactive network graph visualization for entity relationships
 * Uses D3.js for force-directed graph layout
 */

import React, { useEffect, useRef, useState } from 'react';
import * as d3 from 'd3';
import { Search, ZoomIn, ZoomOut, Maximize2, Download, Info } from 'lucide-react';

interface GraphNode {
  id: number;
  label: string;
  type: string;
  importance: number;
  mention_count: number;
}

interface GraphEdge {
  source: number;
  target: number;
  label: string;
  strength: number;
}

interface GraphData {
  nodes: GraphNode[];
  edges: GraphEdge[];
  center_entity?: {
    id: number;
    entity_name: string;
    entity_type: string;
  };
  metadata?: {
    depth: number;
    total_nodes: number;
    total_edges: number;
  };
}

interface Entity {
  id: number;
  entity_name: string;
  entity_type: string;
  mention_count: number;
  importance_score: number;
}

const KnowledgeGraphVisualizer: React.FC = () => {
  const svgRef = useRef<SVGSVGElement>(null);
  const [graphData, setGraphData] = useState<GraphData | null>(null);
  const [selectedEntity, setSelectedEntity] = useState<number | null>(null);
  const [searchQuery, setSearchQuery] = useState('');
  const [searchResults, setSearchResults] = useState<Entity[]>([]);
  const [loading, setLoading] = useState(false);
  const [depth, setDepth] = useState(2);
  const [zoom, setZoom] = useState(1);

  // Entity type colors - proper colors
  const typeColors: Record<string, string> = {
    technology: '#3b82f6',    // blue
    concept: '#8b5cf6',       // purple
    person: '#10b981',         // green
    organization: '#f59e0b',   // orange
    location: '#ef4444',       // red
    event: '#ec4899',          // pink
  };

  // Search entities
  const handleSearch = async (query: string) => {
    if (query.length < 2) {
      setSearchResults([]);
      return;
    }

    try {
      const response = await fetch(`/api/knowledge/entities/search?query=${encodeURIComponent(query)}&limit=10`);
      if (response.ok) {
        const data = await response.json();
        setSearchResults(data);
      }
    } catch (error) {
      console.error('Search failed:', error);
    }
  };

  // Load graph for selected entity
  const loadGraph = async (entityId: number) => {
    setLoading(true);
    setSelectedEntity(entityId);

    try {
      console.log(`Loading graph for entity ID: ${entityId}`);
      const response = await fetch(`/api/knowledge/entities/${entityId}/graph?depth=${depth}&max_nodes=50`);
      if (response.ok) {
        const data: GraphData = await response.json();
        console.log('Graph data received:', data);
        setGraphData(data);
        renderGraph(data);
      } else {
        console.error('Graph API error:', response.status, response.statusText);
      }
    } catch (error) {
      console.error('Failed to load graph:', error);
    } finally {
      setLoading(false);
    }
  };

  // Render D3 force-directed graph
  const renderGraph = (data: GraphData) => {
    console.log('renderGraph called with:', data);
    if (!svgRef.current || !data.nodes.length) {
      console.log('No SVG ref or no nodes:', { svgRef: !!svgRef.current, nodes: data.nodes.length });
      return;
    }

    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove(); // Clear previous graph

    // PRD-169 S3: resolve theme tokens to concrete colors so the graph renders
    // correctly in BOTH light and dark themes (previously hardcoded dark #111827
    // + white text, which was invisible/wrong in the light theme).
    const css = getComputedStyle(svgRef.current);
    const token = (name: string) => `hsl(${css.getPropertyValue(name).trim()})`;
    const surfaceColor = token('--card');
    const fgColor = token('--foreground');
    const edgeColor = token('--primary');
    const haloColor = token('--background');
    const mutedColor = token('--muted-foreground');

    svg.attr('style', `background-color: ${surfaceColor};`);

    const width = svgRef.current.clientWidth;
    const height = svgRef.current.clientHeight;
    console.log('SVG dimensions:', { width, height });

    // Create zoom behavior
    const zoomBehavior = d3.zoom<SVGSVGElement, unknown>()
      .scaleExtent([0.1, 4])
      .on('zoom', (event) => {
        g.attr('transform', event.transform);
        setZoom(event.transform.k);
      });

    svg.call(zoomBehavior);

    // Add background rectangle to force gray background
    svg.append('rect')
      .attr('width', width)
      .attr('height', height)
      .attr('fill', surfaceColor)
      .attr('opacity', 1);

    // Main group for zooming/panning
    const g = svg.append('g');

    // Define arrow markers for directed edges
    svg.append('defs').selectAll('marker')
      .data(['end'])
      .enter().append('marker')
      .attr('id', 'arrowhead')
      .attr('viewBox', '0 -5 10 10')
      .attr('refX', 20)
      .attr('refY', 0)
      .attr('markerWidth', 6)
      .attr('markerHeight', 6)
      .attr('orient', 'auto')
      .append('path')
      .attr('d', 'M0,-5L10,0L0,5')
      .attr('fill', edgeColor);

    // Prepare node data with D3 force simulation types
    const nodes = data.nodes.map((n) => ({
      ...n,
      x: width / 2,
      y: height / 2,
    }));

    const edges = data.edges.map((e) => ({
      ...e,
      source: nodes.find((n) => n.id === e.source)!,
      target: nodes.find((n) => n.id === e.target)!,
    }));

    console.log('Prepared nodes:', nodes);
    console.log('Prepared edges:', edges);

    // Create force simulation
    const simulation = d3.forceSimulation(nodes)
      .force('link', d3.forceLink(edges)
        .id((d: any) => d.id)
        .distance(100)
        .strength((d: any) => d.strength))
      .force('charge', d3.forceManyBody().strength(-300))
      .force('center', d3.forceCenter(width / 2, height / 2))
      .force('collision', d3.forceCollide().radius(30));

    // Draw edges
    const link = g.append('g')
      .selectAll('line')
      .data(edges)
      .enter().append('line')
      .attr('stroke', edgeColor)
      .attr('stroke-width', (d: any) => 2 + d.strength * 2)
      .attr('stroke-opacity', 0.8)
      .attr('marker-end', 'url(#arrowhead)');

    console.log('Links created:', link.size());

    // Draw edge labels
    const linkLabel = g.append('g')
      .selectAll('text')
      .data(edges)
      .enter().append('text')
      .attr('font-size', 12)
      .attr('fill', fgColor)
      .attr('text-anchor', 'middle')
      .text((d: any) => d.label);

    console.log('Link labels created:', linkLabel.size());

    // Draw nodes
    const node = g.append('g')
      .selectAll('circle')
      .data(nodes)
      .enter().append('circle')
      .attr('r', (d: any) => 10 + (d.importance * 15))
      .attr('fill', (d: any) => typeColors[d.type] || mutedColor)
      .attr('stroke', haloColor)
      .attr('stroke-width', 2)
      .attr('opacity', 0.9)
      .style('cursor', 'pointer')
      .call(d3.drag<SVGCircleElement, any>()
        .on('start', dragstarted)
        .on('drag', dragged)
        .on('end', dragended) as any);

    console.log('Nodes created:', node.size());

    // Node labels
    const nodeLabel = g.append('g')
      .selectAll('text')
      .data(nodes)
      .enter().append('text')
      .attr('font-size', 14)
      .attr('font-weight', 'bold')
      .attr('fill', fgColor)
      .attr('text-anchor', 'middle')
      .attr('dy', '0.35em')
      .text((d: any) => d.label)
      .style('pointer-events', 'none');

    // Node click handler
    node.on('click', function(event, d: any) {
      event.stopPropagation();
      loadGraph(d.id);
    });

    // Tooltips
    node.append('title')
      .text((d: any) => `${d.label}\nType: ${d.type}\nMentions: ${d.mention_count}\nImportance: ${d.importance.toFixed(2)}`);

    // Update positions on simulation tick
    simulation.on('tick', () => {
      link
        .attr('x1', (d: any) => d.source.x)
        .attr('y1', (d: any) => d.source.y)
        .attr('x2', (d: any) => d.target.x)
        .attr('y2', (d: any) => d.target.y);

      linkLabel
        .attr('x', (d: any) => (d.source.x + d.target.x) / 2)
        .attr('y', (d: any) => (d.source.y + d.target.y) / 2);

      node
        .attr('cx', (d: any) => d.x)
        .attr('cy', (d: any) => d.y);

      nodeLabel
        .attr('x', (d: any) => d.x)
        .attr('y', (d: any) => d.y);
    });

    // Drag functions
    function dragstarted(event: any) {
      if (!event.active) simulation.alphaTarget(0.3).restart();
      event.subject.fx = event.subject.x;
      event.subject.fy = event.subject.y;
    }

    function dragged(event: any) {
      event.subject.fx = event.x;
      event.subject.fy = event.y;
    }

    function dragended(event: any) {
      if (!event.active) simulation.alphaTarget(0);
      event.subject.fx = null;
      event.subject.fy = null;
    }
  };

  // Zoom controls
  const handleZoomIn = () => {
    const svg = d3.select(svgRef.current);
    svg.transition().call(
      d3.zoom<SVGSVGElement, unknown>().scaleBy as any,
      1.3
    );
  };

  const handleZoomOut = () => {
    const svg = d3.select(svgRef.current);
    svg.transition().call(
      d3.zoom<SVGSVGElement, unknown>().scaleBy as any,
      0.7
    );
  };

  const handleReset = () => {
    const svg = d3.select(svgRef.current);
    svg.transition().call(
      d3.zoom<SVGSVGElement, unknown>().transform as any,
      d3.zoomIdentity
    );
  };

  // Export graph as PNG
  const handleExport = () => {
    if (!svgRef.current) return;

    const svgData = new XMLSerializer().serializeToString(svgRef.current);
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    const img = new Image();

    img.onload = () => {
      canvas.width = img.width;
      canvas.height = img.height;
      ctx?.drawImage(img, 0, 0);
      const pngFile = canvas.toDataURL('image/png');
      const downloadLink = document.createElement('a');
      downloadLink.download = 'knowledge-graph.png';
      downloadLink.href = pngFile;
      downloadLink.click();
    };

    img.src = 'data:image/svg+xml;base64,' + btoa(svgData);
  };

  return (
    <div className="flex flex-col h-full bg-background rounded-lg shadow-lg">
      {/* Header */}
      <div className="flex items-center justify-between p-4 border-b border-border">
        <h2 className="text-xl font-semibold text-white">Knowledge Graph Explorer</h2>
        
        {/* Search */}
        <div className="flex items-center gap-4 flex-1 max-w-md ml-4">
          <div className="relative flex-1">
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-muted-foreground" size={18} />
            <input
              type="text"
              placeholder="Search entities..."
              value={searchQuery}
              onChange={(e) => {
                setSearchQuery(e.target.value);
                handleSearch(e.target.value);
              }}
              onKeyDown={(e) => {
                if (e.key === 'Enter') {
                  if (searchResults.length > 0) {
                    // Auto-select first result
                    loadGraph(searchResults[0].id);
                    setSearchResults([]);
                    setSearchQuery('');
                  } else {
                    handleSearch(searchQuery);
                  }
                }
              }}
              className="w-full pl-10 pr-4 py-2 bg-secondary border border-border rounded-lg text-white placeholder-muted-foreground focus:ring-2 focus:ring-ring focus:outline-none focus:border-border"
              autoFocus
              tabIndex={0}
            />
          </div>
        </div>

        {/* Controls */}
        <div className="flex items-center gap-2">
          <select
            value={depth}
            onChange={(e) => setDepth(Number(e.target.value))}
            className="px-3 py-2 bg-secondary border border-border rounded-lg text-sm text-white"
          >
            <option value={1}>1 hop</option>
            <option value={2}>2 hops</option>
            <option value={3}>3 hops</option>
          </select>
          
          <button onClick={handleZoomIn} className="p-2 hover:bg-secondary rounded text-foreground/90 hover:text-white" title="Zoom in">
            <ZoomIn size={18} />
          </button>
          <button onClick={handleZoomOut} className="p-2 hover:bg-secondary rounded text-foreground/90 hover:text-white" title="Zoom out">
            <ZoomOut size={18} />
          </button>
          <button onClick={handleReset} className="p-2 hover:bg-secondary rounded text-foreground/90 hover:text-white" title="Reset view">
            <Maximize2 size={18} />
          </button>
          <button onClick={handleExport} className="p-2 hover:bg-secondary rounded text-foreground/90 hover:text-white" title="Export as PNG">
            <Download size={18} />
          </button>
        </div>
      </div>

      {/* Search Results Dropdown */}
      {searchResults.length > 0 && (
        <div className="absolute top-16 right-4 w-96 bg-secondary border border-border rounded-lg shadow-xl z-10 max-h-96 overflow-y-auto">
          {searchResults.map((entity) => (
            <div
              key={entity.id}
              onClick={() => {
                loadGraph(entity.id);
                setSearchResults([]);
                setSearchQuery('');
              }}
              className="p-3 hover:bg-secondary cursor-pointer border-b border-border last:border-b-0"
            >
              <div className="flex items-center justify-between">
                <div>
                  <div className="font-medium text-white">{entity.entity_name}</div>
                  <div className="text-sm text-muted-foreground capitalize">{entity.entity_type}</div>
                </div>
                <div className="text-right">
                  <div className="text-xs text-muted-foreground">{entity.mention_count} mentions</div>
                  <div className="text-xs text-muted-foreground">Score: {entity.importance_score.toFixed(2)}</div>
                </div>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Graph Canvas */}
      <div className="flex-1 relative">
        {loading && (
          <div className="absolute inset-0 flex items-center justify-center bg-background bg-opacity-75 z-10">
            <div className="text-center">
              <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-border mx-auto"></div>
              <p className="mt-4 text-foreground/90">Loading knowledge graph...</p>
            </div>
          </div>
        )}

        {!graphData && !loading && (
          <div className="absolute inset-0 flex items-center justify-center">
            <div className="text-center text-muted-foreground">
              <Info size={48} className="mx-auto mb-4" />
              <p className="text-lg mb-2 text-foreground/90">Search for an entity to explore the knowledge graph</p>
              <p className="text-sm">Try searching for technologies, concepts, or organizations</p>
            </div>
          </div>
        )}

        <svg
          ref={svgRef}
          className="w-full h-full bg-background"
          style={{ minHeight: '600px', backgroundColor: 'hsl(var(--card))' }}
        />
      </div>

      {/* Legend */}
      {graphData && (
        <div className="p-4 border-t border-border bg-secondary">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-6">
              <div className="text-sm font-medium text-foreground/90">Legend:</div>
              {Object.entries(typeColors).map(([type, color]) => (
                <div key={type} className="flex items-center gap-2">
                  <div 
                    className="w-4 h-4 rounded-full" 
                    style={{ backgroundColor: color }}
                  ></div>
                  <span className="text-sm text-muted-foreground capitalize">{type}</span>
                </div>
              ))}
            </div>

            {graphData.metadata && (
              <div className="text-sm text-muted-foreground">
                <span className="font-medium">{graphData.metadata.total_nodes}</span> nodes · 
                <span className="font-medium ml-1">{graphData.metadata.total_edges}</span> relationships · 
                <span className="font-medium ml-1">{graphData.metadata.depth}</span> hops
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
};

export default KnowledgeGraphVisualizer;

