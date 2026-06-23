import type { Hypothesis, ReasoningStep } from '../types';

interface Props {
  hypotheses: [Hypothesis, Hypothesis];
  steps: ReasoningStep[];
  onClose: () => void;
}

export default function ReasoningTraceModal({ hypotheses, steps, onClose }: Props) {
  const reasoning = hypotheses.find((h) => h.reasoning)?.reasoning;
  const generatorStep = steps.find((s) => s.step === 'Generator');
  const total = steps.reduce((s, step) => s + step.durationSec, 0);

  const displayText = reasoning ?? generatorStep?.details ?? 'No generator reasoning available.';

  const timing = generatorStep
    ? `${generatorStep.durationSec.toFixed(2)}s generator · ${total.toFixed(2)}s total`
    : `${total.toFixed(2)}s total`;

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-panel modal-wide" onClick={(e) => e.stopPropagation()}>
        <div className="modal-header">
          <div>
            <div className="modal-title">Generator reasoning</div>
            <div className="modal-subtitle">{timing}</div>
          </div>
          <button type="button" className="modal-close" onClick={onClose} aria-label="Close">
            ✕
          </button>
        </div>

        <div className="reasoning-body">
          {displayText}
        </div>
      </div>
    </div>
  );
}
