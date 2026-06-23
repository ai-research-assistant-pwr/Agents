import { useState, type FormEvent, type KeyboardEvent } from 'react';
import type { ModelList, Persona } from '../types';

interface Props {
  onSubmit: (question: string) => void;
  isMock: boolean;
  models: ModelList;
  retrieverModel: string;
  generatorModel: string;
  modelsLoading: boolean;
  onRetrieverModelChange: (model: string) => void;
  onGeneratorModelChange: (model: string) => void;
  personas: Persona[];
  selectedPersonaId: string;
  onPersonaChange: (id: string) => void;
}

const EXAMPLES = [
  'Why do transformer models exhibit emergent in-context learning at certain scales?',
  'What mechanism explains reward-model overoptimization under RLHF?',
  'Can sparse autoencoder feature density predict downstream task transfer?',
];

export default function QuestionInput({
  onSubmit,
  isMock,
  models,
  retrieverModel,
  generatorModel,
  modelsLoading,
  onRetrieverModelChange,
  onGeneratorModelChange,
  personas,
  selectedPersonaId,
  onPersonaChange,
}: Props) {
  const [q, setQ] = useState('');
  const canSubmit = !!q.trim() && !!retrieverModel && !!generatorModel;

  function submit(e?: FormEvent) {
    e?.preventDefault();
    const v = q.trim();
    if (v && retrieverModel && generatorModel) onSubmit(v);
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

      <div className="model-controls">
        <label className="model-field">
          <span className="micro-label">Retriever model</span>
          <select
            value={retrieverModel}
            onChange={(e) => onRetrieverModelChange(e.target.value)}
            disabled={modelsLoading || models.retrieverModels.length === 0}
          >
            {models.retrieverModels.length === 0 && (
              <option value="">{modelsLoading ? 'Loading models...' : 'No models available'}</option>
            )}
            {models.retrieverModels.map((model) => (
              <option key={model} value={model}>{model}</option>
            ))}
          </select>
        </label>
        <label className="model-field">
          <span className="micro-label">Generator model</span>
          <select
            value={generatorModel}
            onChange={(e) => onGeneratorModelChange(e.target.value)}
            disabled={modelsLoading || models.generatorModels.length === 0}
          >
            {models.generatorModels.length === 0 && (
              <option value="">{modelsLoading ? 'Loading models...' : 'No models available'}</option>
            )}
            {models.generatorModels.map((model) => (
              <option key={model} value={model}>{model}</option>
            ))}
          </select>
        </label>
        <label className="model-field">
          <span className="micro-label">Scientific Persona · optional</span>
          <select
            value={selectedPersonaId}
            onChange={(e) => onPersonaChange(e.target.value)}
            disabled={personas.length === 0}
          >
            <option value="">No persona (default)</option>
            {personas.map((p) => (
              <option key={p.personaId} value={p.personaId}>{p.displayName}</option>
            ))}
          </select>
        </label>
      </div>
      {selectedPersonaId && (() => {
        const p = personas.find((x) => x.personaId === selectedPersonaId);
        return p ? (
          <div className="persona-hint">{p.corePhilosophy}</div>
        ) : null;
      })()}

      <div className="qinput-footer">
        <span className="qinput-hint">⌘+Enter to submit{isMock ? ' · running in mock mode' : ''}</span>
        <button type="submit" className="primary-btn primary-btn-inline" disabled={!canSubmit}>
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
