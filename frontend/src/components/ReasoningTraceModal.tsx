import type { Hypothesis, ReasoningStep } from '../types';

interface Props {
  hypotheses: [Hypothesis, Hypothesis];
  steps: ReasoningStep[];
  onClose: () => void;
}

export default function ReasoningTraceModal({ hypotheses, steps, onClose }: Props) {
  const hasReasoning = hypotheses.some((h) => h.reasoning);

  const generatorStep = steps.find((s) => s.step === 'Generator');
  const total = steps.reduce((s, step) => s + step.durationSec, 0);

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-panel" onClick={(e) => e.stopPropagation()}>
        <div className="modal-header">
          <div>
            <div className="modal-title">Generator reasoning</div>
            <div className="modal-subtitle">
              How each hypothesis was derived from the knowledge graph
              {generatorStep && ` · ${generatorStep.durationSec.toFixed(2)}s · ${total.toFixed(2)}s total`}
            </div>
          </div>
          <button type="button" className="modal-close" onClick={onClose} aria-label="Close">
            ✕
          </button>
        </div>

        <div className="trace-list">
          {hasReasoning ? (
            hypotheses.map((h) => (
              <div key={h.id} className="trace-item">
                <div className="trace-header" style={{ cursor: 'default' }}>
                  <span className="trace-index">{h.id}</span>
                  <span className="trace-step-name">{h.headline}</span>
                  <span className="trace-desc">{h.classification}</span>
                </div>
                <div className="trace-details" style={{ whiteSpace: 'pre-wrap' }}>
                  {h.reasoning ?? '(No reasoning captured for this hypothesis.)'}
                </div>
              </div>
            ))
          ) : (
            <div className="trace-details" style={{ padding: '1rem' }}>
              {generatorStep?.details ?? 'No generator reasoning available.'}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
