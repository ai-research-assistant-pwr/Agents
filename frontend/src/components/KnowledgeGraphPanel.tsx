import { useCallback, useEffect, useRef, useState } from 'react';
import type { KnowledgeGraph, KGNode } from '../types';

interface Props {
  graph: KnowledgeGraph;
  onClose: () => void;
}

// ---------------------------------------------------------------------------
// Layout constants
// ---------------------------------------------------------------------------

const W = 700;
const H = 460;
const CX = W / 2;
const CY = H / 2;
const COMMUNITY_R = 160;
const CONCEPT_R = 72;
const PAPER_R_OFFSET = 90; // distance from parent community

function toRad(deg: number) { return (deg * Math.PI) / 180; }

interface NodePos { id: string; x: number; y: number; node: KGNode }

function computePositions(graph: KnowledgeGraph): Map<string, NodePos> {
  const pos = new Map<string, NodePos>();

  const queryNode = graph.nodes.find((n) => n.type === 'query');
  if (queryNode) pos.set(queryNode.id, { id: queryNode.id, x: CX, y: CY, node: queryNode });

  const communities = graph.nodes.filter((n) => n.type === 'community');
  communities.forEach((c, i) => {
    const angle = toRad((360 / communities.length) * i - 90);
    pos.set(c.id, {
      id: c.id,
      x: CX + COMMUNITY_R * Math.cos(angle),
      y: CY + COMMUNITY_R * Math.sin(angle),
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
      const r = child.type === 'paper' ? PAPER_R_OFFSET : CONCEPT_R;
      pos.set(child.id, {
        id: child.id,
        x: comPos.x + r * Math.cos(angle),
        y: comPos.y + r * Math.sin(angle),
        node: child,
      });
    });
  });

  graph.nodes.forEach((n) => {
    if (!pos.has(n.id)) pos.set(n.id, { id: n.id, x: CX, y: CY + 20, node: n });
  });

  return pos;
}

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

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export default function KnowledgeGraphPanel({ graph, onClose }: Props) {
  const positions = computePositions(graph);

  // Pan / zoom state
  const [offsetX, setOffsetX] = useState(0);
  const [offsetY, setOffsetY] = useState(0);
  const [scale, setScale] = useState(1);
  const dragging = useRef(false);
  const dragStart = useRef({ x: 0, y: 0, ox: 0, oy: 0 });
  const svgRef = useRef<SVGSVGElement>(null);

  // Selected node for detail panel
  const [selectedNode, setSelectedNode] = useState<KGNode | null>(null);

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
    dragStart.current = { x: e.clientX, y: e.clientY, ox: offsetX, oy: offsetY };
    e.currentTarget.addEventListener('mousemove', onMouseMove as EventListener);
    e.currentTarget.addEventListener('mouseup', onMouseUp as EventListener);
  }, [offsetX, offsetY]);

  function onMouseMove(e: MouseEvent) {
    if (!dragging.current) return;
    setOffsetX(dragStart.current.ox + (e.clientX - dragStart.current.x));
    setOffsetY(dragStart.current.oy + (e.clientY - dragStart.current.y));
  }

  function onMouseUp(e: MouseEvent) {
    dragging.current = false;
    (e.currentTarget as SVGElement).removeEventListener('mousemove', onMouseMove as EventListener);
    (e.currentTarget as SVGElement).removeEventListener('mouseup', onMouseUp as EventListener);
  }

  function handleNodeClick(e: React.MouseEvent, node: KGNode) {
    e.stopPropagation();
    setSelectedNode((prev) => (prev?.id === node.id ? null : node));
  }

  function handleBgClick() {
    setSelectedNode(null);
  }

  function resetView() {
    setOffsetX(0);
    setOffsetY(0);
    setScale(1);
  }

  const transform = `translate(${offsetX} ${offsetY}) scale(${scale})`;

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
              viewBox={`0 0 ${W} ${H}`}
              className="kg-svg"
              style={{ cursor: dragging.current ? 'grabbing' : 'grab' }}
              aria-label="Knowledge graph visualisation"
              onMouseDown={handleMouseDown}
              onClick={handleBgClick}
            >
              <g transform={transform}>
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
                      onClick={(e) => handleNodeClick(e, node)}
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
                      {(isPaper) && (
                        <text
                          x={x} y={y + r + 11}
                          textAnchor="middle"
                          fontSize={7.5}
                          fill="#92400e"
                          fontFamily="var(--font-mono)"
                          style={{ pointerEvents: 'none', userSelect: 'none' }}
                        >
                          {node.label.length > 18 ? node.label.slice(0, 16) + '…' : node.label}
                        </text>
                      )}
                    </g>
                  );
                })}
              </g>
            </svg>

            <div className="kg-legend">
              {(['query', 'community', 'concept', 'paper'] as const).map((t) => (
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
                {selectedNode.abstract && (
                  <>
                    <div className="kg-details-section">Abstract</div>
                    <p className="kg-details-text">{selectedNode.abstract}</p>
                  </>
                )}
                {selectedNode.summary && (
                  <>
                    <div className="kg-details-section">Summary</div>
                    <p className="kg-details-text">{selectedNode.summary}</p>
                  </>
                )}
                {!selectedNode.abstract && !selectedNode.summary && (
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
