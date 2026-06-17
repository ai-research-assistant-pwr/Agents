import type { ExplorerResult } from '../types';
import { ArrowUpRight } from './icons';

interface Props {
  exploration: ExplorerResult | null;
  onViewTrace?: () => void;
}

export default function ExplorerCard({ exploration, onViewTrace }: Props) {
  if (!exploration) {
    return (
      <div className="explorer-card">
        <div className="explorer-head">
          <span className="status-row">
            <span className="status-dot pulsing" />
            <span>Traversing knowledge graph…</span>
          </span>
        </div>
        <div className="explorer-stats">
          {Array.from({ length: 4 }).map((_, i) => (
            <div key={i} className="stat">
              <div className="skel skel-num" />
              <div className="skel skel-label" />
            </div>
          ))}
        </div>
      </div>
    );
  }

  return (
    <div className="explorer-card">
      <div className="explorer-head">
        <span className="status-row">
          <span className="status-dot green" />
          <span>Knowledge graph traversed</span>
        </span>
        <button type="button" className="link-btn" onClick={onViewTrace}>
          View reasoning trace <ArrowUpRight />
        </button>
      </div>
      <div className="explorer-stats">
        <Stat n={exploration.nodesTraversed} label="nodes traversed" />
        <Stat n={exploration.relations} label="typed relations" />
        <Stat n={exploration.sourcePapers} label="source papers" />
        <Stat n={exploration.clusters} label="subgraph clusters" />
      </div>
      {exploration.communities.length > 0 && (
        <div className="explorer-communities">
          <div className="micro-label">Communities surfaced</div>
          <div className="chip-row">
            {exploration.communities.map((c) => (
              <span key={c} className="chip chip-soft">{c}</span>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

function Stat({ n, label }: { n: number; label: string }) {
  return (
    <div className="stat">
      <div className="stat-num">{n}</div>
      <div className="stat-label">{label}</div>
    </div>
  );
}
