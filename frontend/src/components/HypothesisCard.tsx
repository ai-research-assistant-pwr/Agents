import type { Hypothesis } from '../types';
import Md from './Md';

interface Props {
  hypothesis: Hypothesis;
  selected: boolean;
  dimmed: boolean;
  disabled: boolean;
  onSelect: () => void;
}

export default function HypothesisCard({ hypothesis, selected, dimmed, disabled, onSelect }: Props) {
  const cls = [
    'hyp-card',
    selected ? 'selected' : '',
    dimmed ? 'dimmed' : '',
    disabled ? 'disabled' : '',
  ]
    .filter(Boolean)
    .join(' ');

  return (
    <button
      type="button"
      className={cls}
      onClick={onSelect}
      disabled={disabled}
      aria-pressed={selected}
    >
      {selected && <span className="hyp-card-badge">Selected</span>}

      <div className="hyp-card-head">
        <div className="hyp-card-class">{hypothesis.classification}</div>
        <span className={`radio${selected ? ' on' : ''}`} aria-hidden="true" />
      </div>

      <div className="hyp-card-statement"><Md>{hypothesis.statement}</Md></div>

      <div className="hyp-card-bottom">
        {hypothesis.drawnFrom.length > 0 && (
          <>
            <div className="micro-label">Drawn from</div>
            <div className="chip-row drawn-from">
              {hypothesis.drawnFrom.map((c) => (
                <span key={c} className="chip chip-outline">{c}</span>
              ))}
            </div>
          </>
        )}
        {hypothesis.falsifiablePrediction && (
          <div className={`prediction${selected ? ' prediction-selected' : ''}`}>
            <div className="micro-label">Falsifiable prediction</div>
            <div className="prediction-text"><Md>{hypothesis.falsifiablePrediction}</Md></div>
          </div>
        )}
      </div>
    </button>
  );
}
