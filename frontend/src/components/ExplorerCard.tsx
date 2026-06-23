import type { ExplorerResult } from '../types';
import { ArrowUpRight } from './icons';

interface Props {
  exploration: ExplorerResult | null;
  hasGraph?: boolean;
  onViewTrace?: () => void;
  onViewGraph?: () => void;
}

export default function ExplorerCard({ exploration, hasGraph, onViewTrace, onViewGraph }: Props) {
  if (!exploration) {
    return (
      <div className="explorer-card">
        <div className="explorer-head">
          <span className="status-row">
            <span className="status-dot pulsing" />
            <span>Traversing knowledge graph…</span>
          </span>
        </div>
      </div>
    );
  }

  return (
    <div className="explorer-card">
      <div className="explorer-head">
        <span className="status-row">
          <span className="status-dot green" />
          <span>
            Knowledge graph traversed
            {exploration.nodesTraversed > 0 && ` · ${exploration.nodesTraversed} nodes`}
          </span>
        </span>
        <span className="explorer-actions">
          {hasGraph && (
            <button type="button" className="link-btn" onClick={onViewGraph}>
              View graph <ArrowUpRight />
            </button>
          )}
          <button type="button" className="link-btn" onClick={onViewTrace}>
            View reasoning trace <ArrowUpRight />
          </button>
        </span>
      </div>
    </div>
  );
}
