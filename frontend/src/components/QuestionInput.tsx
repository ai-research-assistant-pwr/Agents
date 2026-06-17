import { useState, type FormEvent, type KeyboardEvent } from 'react';

interface Props {
  onSubmit: (question: string) => void;
  isMock: boolean;
}

const EXAMPLES = [
  'Why do transformer models exhibit emergent in-context learning at certain scales?',
  'What mechanism explains reward-model overoptimization under RLHF?',
  'Can sparse autoencoder feature density predict downstream task transfer?',
];

export default function QuestionInput({ onSubmit, isMock }: Props) {
  const [q, setQ] = useState('');

  function submit(e?: FormEvent) {
    e?.preventDefault();
    const v = q.trim();
    if (v) onSubmit(v);
  }

  function handleKey(e: KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) submit();
  }

  return (
    <form className="qinput" onSubmit={submit}>
      <div className="qinput-eyebrow">Pose a research question</div>
      <h1 className="qinput-title">What would you like a hypothesis for?</h1>
      <div className="qinput-helper">
        The explorer agent traverses the knowledge graph from your question and the generator
        proposes two competing hypotheses. You pick one to develop further.
      </div>

      <textarea
        className="qinput-area"
        rows={4}
        placeholder="e.g. Why do transformer models exhibit emergent in-context learning at certain scales rather than improving smoothly with compute?"
        value={q}
        onChange={(e) => setQ(e.target.value)}
        onKeyDown={handleKey}
        autoFocus
      />

      <div className="qinput-footer">
        <span className="qinput-hint">⌘+Enter to submit{isMock ? ' · running in mock mode' : ''}</span>
        <button type="submit" className="primary-btn primary-btn-inline" disabled={!q.trim()}>
          Explore knowledge graph →
        </button>
      </div>

      <div className="qinput-examples">
        <div className="micro-label">Try one</div>
        <div className="example-list">
          {EXAMPLES.map((ex) => (
            <button
              key={ex}
              type="button"
              className="example-chip"
              onClick={() => setQ(ex)}
            >
              {ex}
            </button>
          ))}
        </div>
      </div>
    </form>
  );
}
