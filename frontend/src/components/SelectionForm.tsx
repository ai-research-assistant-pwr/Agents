import type { Hypothesis, HypothesisId } from '../types';

interface Props {
  hypotheses: [Hypothesis, Hypothesis];
  selected: HypothesisId | undefined;
  rationale: string;
  submitting: boolean;
  submitted: boolean;
  onSelect: (id: HypothesisId) => void;
  onRationaleChange: (text: string) => void;
  onSubmit: () => void;
  onRegenerate: () => void;
}

const MAX_RATIONALE = 600;

export default function SelectionForm({
  hypotheses,
  selected,
  rationale,
  submitting,
  submitted,
  onSelect,
  onRationaleChange,
  onSubmit,
  onRegenerate,
}: Props) {
  const canSubmit = selected != null && !submitted && !submitting;
  const statusText = submitted ? 'Submitted' : selected ? 'Ready' : 'Pick one to continue';
  const statusClass = submitted || selected ? 'status-green' : 'status-muted';

  return (
    <div className="form-card">
      <div className="form-head">
        <div className="form-title">Which hypothesis should we develop further?</div>
        <div className={`form-status ${statusClass}`}>
          {(submitted || selected) && <span className="status-dot green" />}
          {statusText}
        </div>
      </div>
      <div className="form-helper">
        Select one to continue. A brief rationale is optional but valuable — when you leave one,
        it calibrates how future hypotheses are generated for you.
      </div>

      <div className="seg-control" role="radiogroup" aria-label="Choose hypothesis">
        {hypotheses.map((h) => {
          const isOn = selected === h.id;
          return (
            <button
              key={h.id}
              type="button"
              role="radio"
              aria-checked={isOn}
              className={`seg${isOn ? ' on' : ''}`}
              onClick={() => !submitted && !submitting && onSelect(h.id)}
              disabled={submitted || submitting}
            >
              <span className={`radio${isOn ? ' on' : ''}`} aria-hidden="true" />
              <span className="seg-text">
                <span className="seg-id">{h.id} ·</span> {h.headline}
              </span>
            </button>
          );
        })}
      </div>

      <div className="rationale-field">
        <div className="rationale-head">
          <label className="micro-label" htmlFor="rationale">Rationale · optional</label>
          <span className="rationale-counter">{rationale.length} / {MAX_RATIONALE}</span>
        </div>
        <textarea
          id="rationale"
          rows={4}
          maxLength={MAX_RATIONALE}
          placeholder="Optional: what made this one more compelling? A sentence or two is enough — mechanism, falsifiability, fit with prior evidence…"
          value={rationale}
          onChange={(e) => onRationaleChange(e.target.value)}
          disabled={submitted || submitting}
        />
      </div>

      <div className="form-footer">
        <button
          type="button"
          className="ghost-link"
          onClick={onRegenerate}
          disabled={submitted || submitting}
        >
          ↻ Regenerate two new hypotheses
        </button>
        <button
          type="button"
          className="primary-btn primary-btn-inline"
          onClick={onSubmit}
          disabled={!canSubmit}
        >
          {submitted ? 'Submitted ✓' : submitting ? 'Submitting…' : 'Submit'}
        </button>
      </div>
    </div>
  );
}
