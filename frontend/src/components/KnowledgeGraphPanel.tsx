import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import type { KnowledgeGraph, KGNode } from '../types';

interface Props {
  graph: KnowledgeGraph;
  onClose: () => void;
}

// ---------------------------------------------------------------------------
// Layout constants
// ---------------------------------------------------------------------------

const BASE_W = 900;
const BASE_H = 620;
const MIN_NODE_GAP = 34;

const NODE_COLORS: Record<KGNode['type'], string> = {
  query: '#0c0a09',
  community: '#292524',
  concept: '#57534e',
  paper: '#a16207',
};

const NODE_RADIUS: Record<KGNode['type'], number> = {
  query: 22,
  community: 16,
  concept: 10,
  paper: 12,
};

function toRad(deg: number) { return (deg * Math.PI) / 180; }

interface NodePos { id: string; x: number; y: number; node: KGNode }
interface GraphLayout { positions: Map<string, NodePos>; width: number; height: number }

function computeLayout(graph: KnowledgeGraph): GraphLayout {
  const pos = new Map<string, NodePos>();
  const paperCount = graph.nodes.filter((n) => n.type === 'paper').length;
  const layoutWidth = Math.max(BASE_W, 520 + Math.ceil(Math.sqrt(Math.max(1, paperCount))) * 190);
  const layoutHeight = Math.max(BASE_H, 420 + Math.ceil(Math.sqrt(Math.max(1, paperCount))) * 150);
  const cx = layoutWidth / 2;
  const cy = layoutHeight / 2;

  const queryNode = graph.nodes.find((n) => n.type === 'query');
  if (queryNode) pos.set(queryNode.id, { id: queryNode.id, x: cx, y: cy, node: queryNode });

  const communities = graph.nodes.filter((n) => n.type === 'community');

  if (communities.length > 0) {
    // Community-based layout (mock / structured data)
    const communityR = Math.min(layoutWidth, layoutHeight) * 0.25;
    const conceptR = 90;
    const paperROffset = 125;

    communities.forEach((c, i) => {
      const angle = toRad((360 / communities.length) * i - 90);
      pos.set(c.id, {
        id: c.id,
        x: cx + communityR * Math.cos(angle),
        y: cy + communityR * Math.sin(angle),
        node: c,
      });
    });

    communities.forEach((c, ci) => {
      const communityAngle = toRad((360 / communities.length) * ci - 90);
      const children = graph.edges
        .filter((e) => e.source === c.id)
        .map((e) => graph.nodes.find((n) => n.id === e.target))
        .filter((n): n is KGNode => n !== undefined && n.type !== 'community' && n.type !== 'query');

      const comPos = pos.get(c.id);
      if (!comPos) return;

      children.forEach((child, j) => {
        if (pos.has(child.id)) return;
        const spread = Math.min(80, 20 * children.length);
        const startAngle = communityAngle - toRad(spread / 2);
        const step = children.length > 1 ? toRad(spread / (children.length - 1)) : 0;
        const angle = startAngle + step * j;
        const r = child.type === 'paper' ? paperROffset : conceptR;
        pos.set(child.id, {
          id: child.id,
          x: comPos.x + r * Math.cos(angle),
          y: comPos.y + r * Math.sin(angle),
          node: child,
        });
      });
    });
  } else {
    // Flat layout for real Neo4j data: query → source papers → walked papers
    const queryTargets = new Set(
      graph.edges.filter((e) => e.source === 'query').map((e) => e.target)
    );
    const sourcePapers = graph.nodes.filter((n) => n.type === 'paper' && queryTargets.has(n.id));

    // Scale rings to number of papers so they don't crowd
    const innerR = Math.max(160, sourcePapers.length * 28);
    const outerR = Math.min(layoutWidth, layoutHeight) * 0.43;

    // Inner ring: source papers evenly distributed
    sourcePapers.forEach((p, i) => {
      const angle = toRad((360 / Math.max(1, sourcePapers.length)) * i - 90);
      pos.set(p.id, {
        id: p.id,
        x: cx + innerR * Math.cos(angle),
        y: cy + innerR * Math.sin(angle),
        node: p,
      });
    });

    // Outer ring: walked papers grouped behind their source
    const childrenBySource = new Map<string, KGNode[]>();
    graph.edges.forEach((e) => {
      if (e.source === 'query') return;
      const target = graph.nodes.find((n) => n.id === e.target);
      if (!target) return;
      if (!childrenBySource.has(e.source)) childrenBySource.set(e.source, []);
      childrenBySource.get(e.source)!.push(target);
    });

    childrenBySource.forEach((children, sourceId) => {
      const srcPos = pos.get(sourceId);
      if (!srcPos) return;
      const baseAngle = Math.atan2(srcPos.y - cy, srcPos.x - cx);
      const spread = toRad(Math.min(85, 18 * children.length));
      children.forEach((child, j) => {
        if (pos.has(child.id)) return;
        const step = children.length > 1 ? (2 * spread) / (children.length - 1) : 0;
        const angle = baseAngle - spread + step * j;
        pos.set(child.id, {
          id: child.id,
          x: cx + outerR * Math.cos(angle),
          y: cy + outerR * Math.sin(angle),
          node: child,
        });
      });
    });
  }

  // Fallback for any node still without a position
  let fallbackIdx = 0;
  graph.nodes.forEach((n) => {
    if (!pos.has(n.id)) {
      const angle = toRad(fallbackIdx * 47);
      pos.set(n.id, {
        id: n.id,
        x: cx + 230 * Math.cos(angle),
        y: cy + 230 * Math.sin(angle),
        node: n,
      });
      fallbackIdx++;
    }
  });

  relaxCollisions(pos, layoutWidth, layoutHeight);

  return { positions: pos, width: layoutWidth, height: layoutHeight };
}

function nodeClearance(node: KGNode) {
  if (node.type === 'query') return 70;
  if (node.type === 'community') return 58;
  return NODE_RADIUS[node.type] + MIN_NODE_GAP;
}

function relaxCollisions(pos: Map<string, NodePos>, width: number, height: number) {
  const nodes = Array.from(pos.values());
  const margin = 46;

  for (let iter = 0; iter < 80; iter++) {
    for (let i = 0; i < nodes.length; i++) {
      for (let j = i + 1; j < nodes.length; j++) {
        const a = nodes[i];
        const b = nodes[j];
        let dx = b.x - a.x;
        let dy = b.y - a.y;
        let dist = Math.hypot(dx, dy);
        if (dist < 0.001) {
          dx = Math.cos(i + j);
          dy = Math.sin(i + j);
          dist = 1;
        }

        const minDist = nodeClearance(a.node) + nodeClearance(b.node);
        if (dist >= minDist) continue;

        const push = (minDist - dist) / 2;
        const nx = dx / dist;
        const ny = dy / dist;
        if (a.node.type !== 'query') {
          a.x -= nx * push;
          a.y -= ny * push;
        }
        if (b.node.type !== 'query') {
          b.x += nx * push;
          b.y += ny * push;
        }
      }
    }

    nodes.forEach((n) => {
      if (n.node.type === 'query') return;
      n.x = Math.min(width - margin, Math.max(margin, n.x));
      n.y = Math.min(height - margin, Math.max(margin, n.y));
    });
  }
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export default function KnowledgeGraphPanel({ graph, onClose }: Props) {
  const { positions, width, height } = useMemo(() => computeLayout(graph), [graph]);
  const homeOffsetX = (BASE_W - width) / 2;
  const homeOffsetY = (BASE_H - height) / 2;

  // Pan / zoom state
  const [offsetX, setOffsetX] = useState(homeOffsetX);
  const [offsetY, setOffsetY] = useState(homeOffsetY);
  const [scale, setScale] = useState(1);
  const dragging = useRef(false);
  const dragStart = useRef({ x: 0, y: 0, ox: 0, oy: 0 });
  const svgRef = useRef<SVGSVGElement>(null);
  const nodePointerStart = useRef<{ id: string; x: number; y: number } | null>(null);

  // Selected node for detail panel
  const [selectedNode, setSelectedNode] = useState<KGNode | null>(null);

  // Track whether mouse has moved since mousedown (to distinguish click from drag)
  const hasDragged = useRef(false);

  // Non-passive wheel listener for zoom-toward-cursor
  useEffect(() => {
    const svg = svgRef.current;
    if (!svg) return;
    const handler = (e: WheelEvent) => {
      e.preventDefault();
      const rect = svg.getBoundingClientRect();
      const mx = e.clientX - rect.left;
      const my = e.clientY - rect.top;
      const factor = e.deltaY > 0 ? 0.9 : 1.1;
      setScale((s) => {
        const ns = Math.max(0.25, Math.min(4, s * factor));
        setOffsetX((ox) => mx - (mx - ox) * (ns / s));
        setOffsetY((oy) => my - (my - oy) * (ns / s));
        return ns;
      });
    };
    svg.addEventListener('wheel', handler, { passive: false });
    return () => svg.removeEventListener('wheel', handler);
  }, []);

  const handleMouseDown = useCallback((e: React.MouseEvent) => {
    if (e.button !== 0) return;
    dragging.current = true;
    hasDragged.current = false;
    dragStart.current = { x: e.clientX, y: e.clientY, ox: offsetX, oy: offsetY };
    e.currentTarget.addEventListener('mousemove', onMouseMove as EventListener);
    e.currentTarget.addEventListener('mouseup', onMouseUp as EventListener);
  }, [offsetX, offsetY]);

  function onMouseMove(e: MouseEvent) {
    if (!dragging.current) return;
    const dx = e.clientX - dragStart.current.x;
    const dy = e.clientY - dragStart.current.y;
    if (Math.abs(dx) > 3 || Math.abs(dy) > 3) hasDragged.current = true;
    setOffsetX(dragStart.current.ox + dx);
    setOffsetY(dragStart.current.oy + dy);
  }

  function onMouseUp(e: MouseEvent) {
    dragging.current = false;
    (e.currentTarget as SVGElement).removeEventListener('mousemove', onMouseMove as EventListener);
    (e.currentTarget as SVGElement).removeEventListener('mouseup', onMouseUp as EventListener);
  }

  function handleNodePointerDown(e: React.PointerEvent, node: KGNode) {
    e.stopPropagation();
    nodePointerStart.current = { id: node.id, x: e.clientX, y: e.clientY };
  }

  function handleNodePointerUp(e: React.PointerEvent, node: KGNode) {
    e.stopPropagation();
    const start = nodePointerStart.current;
    nodePointerStart.current = null;
    if (!start || start.id !== node.id) return;
    if (Math.hypot(e.clientX - start.x, e.clientY - start.y) > 4) return;
    setSelectedNode((prev) => (prev?.id === node.id ? null : node));
  }

  // Background rect click — only fires when clicking empty canvas area
  function handleBgClick() {
    if (hasDragged.current) return;
    setSelectedNode(null);
  }

  function resetView() {
    setOffsetX(homeOffsetX);
    setOffsetY(homeOffsetY);
    setScale(1);
  }

  const transform = `translate(${offsetX} ${offsetY}) scale(${scale})`;
  const selectedBody = selectedNode?.summary || selectedNode?.abstract;

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-panel modal-wide kg-panel-outer" onClick={(e) => e.stopPropagation()}>
        <div className="modal-header">
          <div>
            <div className="modal-title">Knowledge graph</div>
            <div className="modal-subtitle">
              {graph.nodes.length} nodes · {graph.edges.length} edges
              <span className="kg-hint"> · scroll to zoom · drag to pan · click node for details</span>
            </div>
          </div>
          <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
            <button type="button" className="ghost-btn kg-reset-btn" onClick={resetView} title="Reset view">
              ⟲
            </button>
            <button type="button" className="modal-close" onClick={onClose} aria-label="Close">✕</button>
          </div>
        </div>

        <div className="kg-body kg-body-split">
          {/* Graph */}
          <div className="kg-graph-area">
            <svg
              ref={svgRef}
              viewBox={`0 0 ${BASE_W} ${BASE_H}`}
              className="kg-svg"
              style={{ cursor: 'grab' }}
              aria-label="Knowledge graph visualisation"
              onMouseDown={handleMouseDown}
            >
              <g transform={transform}>
                {/* Background click target — separate sibling from nodes so
                    node clicks never propagate here */}
                <rect
                  x={-9999} y={-9999} width={99999} height={99999}
                  fill="transparent"
                  onClick={handleBgClick}
                  style={{ cursor: 'grab' }}
                />
                {/* Edges */}
                {graph.edges.map((edge, i) => {
                  const from = positions.get(edge.source);
                  const to = positions.get(edge.target);
                  if (!from || !to) return null;
                  return (
                    <line
                      key={i}
                      x1={from.x} y1={from.y}
                      x2={to.x} y2={to.y}
                      stroke="#d6d3d1"
                      strokeWidth={1.2}
                      strokeOpacity={0.6}
                    />
                  );
                })}

                {/* Nodes */}
                {Array.from(positions.values()).map(({ id, x, y, node }) => {
                  const r = NODE_RADIUS[node.type];
                  const fill = NODE_COLORS[node.type];
                  const isQuery = node.type === 'query';
                  const isCommunity = node.type === 'community';
                  const isPaper = node.type === 'paper';
                  const isSelected = selectedNode?.id === id;

                  return (
                    <g
                      key={id}
                      onPointerDown={(e) => handleNodePointerDown(e, node)}
                      onPointerUp={(e) => handleNodePointerUp(e, node)}
                      style={{ cursor: 'pointer' }}
                    >
                      {isSelected && (
                        <circle cx={x} cy={y} r={r + 5} fill="none" stroke="#d97706" strokeWidth={2} />
                      )}
                      <circle
                        cx={x} cy={y} r={r}
                        fill={fill}
                        fillOpacity={isQuery ? 1 : isCommunity ? 0.9 : 0.7}
                      />
                      {isPaper && (
                        <text
                          x={x} y={y + 1}
                          textAnchor="middle"
                          dominantBaseline="middle"
                          fontSize={7}
                          fill="white"
                          fontFamily="var(--font-mono)"
                          style={{ pointerEvents: 'none', userSelect: 'none' }}
                        >
                          P
                        </text>
                      )}
                      {(isQuery || isCommunity) && (
                        <text
                          x={x} y={y + r + 13}
                          textAnchor="middle"
                          fontSize={isQuery ? 9.5 : 8.5}
                          fill="#44403c"
                          fontFamily="var(--font-mono)"
                          style={{ pointerEvents: 'none', userSelect: 'none' }}
                        >
                          {node.label.length > 22 ? node.label.slice(0, 20) + '…' : node.label}
                        </text>
                      )}
                      {isPaper && isSelected && (
                        <text
                          x={x} y={y + r + 11}
                          textAnchor="middle"
                          fontSize={7.5}
                          fill="#92400e"
                          fontFamily="var(--font-mono)"
                          style={{ pointerEvents: 'none', userSelect: 'none' }}
                        >
                          {node.label.length > 22 ? node.label.slice(0, 20) + '…' : node.label}
                        </text>
                      )}
                    </g>
                  );
                })}
              </g>
            </svg>

            <div className="kg-legend">
              {(['query', 'community', 'concept', 'paper'] as const)
                .filter((t) => graph.nodes.some((n) => n.type === t))
                .map((t) => (
                  <div key={t} className="kg-legend-item">
                    <span className="kg-legend-dot" style={{ background: NODE_COLORS[t] }} />
                    <span>{t}</span>
                  </div>
                ))}
            </div>
          </div>

          {/* Node details panel */}
          <div className={`kg-details-panel${selectedNode ? ' kg-details-visible' : ''}`}>
            {selectedNode ? (
              <>
                <div className="kg-details-type">{selectedNode.type}</div>
                <div className="kg-details-label">{selectedNode.label}</div>
                {selectedNode.paperId && (
                  <div className="kg-details-paperid">{selectedNode.paperId}</div>
                )}
                {selectedNode.summary && (
                  <>
                    <div className="kg-details-section">Summary</div>
                    <p className="kg-details-text">{selectedNode.summary}</p>
                  </>
                )}
                {!selectedNode.summary && selectedNode.abstract && (
                  <>
                    <div className="kg-details-section">Abstract</div>
                    <p className="kg-details-text">{selectedNode.abstract}</p>
                  </>
                )}
                {selectedNode.summary && selectedNode.abstract && selectedNode.abstract !== selectedNode.summary && (
                  <>
                    <div className="kg-details-section">Abstract</div>
                    <p className="kg-details-text">{selectedNode.abstract}</p>
                  </>
                )}
                {!selectedBody && (
                  <p className="kg-details-text kg-details-empty">
                    No additional metadata for this node.
                  </p>
                )}
                <button
                  type="button"
                  className="ghost-btn kg-details-close"
                  onClick={() => setSelectedNode(null)}
                >
                  Dismiss
                </button>
              </>
            ) : (
              <p className="kg-details-empty kg-details-hint">
                Click any node to see its details here.
              </p>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
