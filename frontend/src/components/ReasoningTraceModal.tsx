import { useState } from 'react';
import type { ReasoningStep } from '../types';

interface Props {
  steps: ReasoningStep[];
  onClose: () => void;
}

export default function ReasoningTraceModal({ steps, onClose }: Props) {
  const [expanded, setExpanded] = useState<number | null>(null);

  const total = steps.reduce((s, step) => s + step.durationSec, 0);

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-panel" onClick={(e) => e.stopPropagation()}>
        <div className="modal-header">
          <div>
            <div className="modal-title">Reasoning trace</div>
            <div className="modal-subtitle">
              {steps.length} pipeline steps · {total.toFixed(2)}s total
            </div>
          </div>
          <button type="button" className="modal-close" onClick={onClose} aria-label="Close">
            ✕
          </button>
        </div>

        <div className="trace-list">
          {steps.map((step, i) => {
            const isOpen = expanded === i;
            return (
              <div key={i} className="trace-item">
                <button
                  type="button"
                  className="trace-header"
                  onClick={() => setExpanded(isOpen ? null : i)}
                >
                  <span className="trace-index">{i + 1}</span>
                  <span className="trace-step-name">{step.step}</span>
                  <span className="trace-desc">{step.description}</span>
                  <span className="trace-dur">{step.durationSec.toFixed(2)}s</span>
                  <span className="trace-chevron">{isOpen ? '▲' : '▼'}</span>
                </button>
                {isOpen && step.details && (
                  <div className="trace-details">{step.details}</div>
                )}
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
