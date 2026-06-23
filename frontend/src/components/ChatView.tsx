import type { HypothesisId, Message, Session } from '../types';
import ExplorerCard from './ExplorerCard';
import HypothesisCard from './HypothesisCard';
import SelectionForm from './SelectionForm';
import ConversationInput from './ConversationInput';
import { AgentMark } from './icons';

interface ExploringPhase {
  kind: 'exploring';
  question: string;
}
interface GeneratedPhase {
  kind: 'generated';
  session: Session;
  selected?: HypothesisId;
  rationale: string;
  submitting: boolean;
}
interface SubmittedPhase {
  kind: 'submitted';
  session: Session;
  selected: HypothesisId;
  rationale: string;
  convInput: string;
  convPending: boolean;
}

export type ChatPhase = ExploringPhase | GeneratedPhase | SubmittedPhase;

interface Props {
  phase: ChatPhase;
  onSelect: (id: HypothesisId) => void;
  onRationaleChange: (text: string) => void;
  onSubmit: () => void;
  onRegenerate: () => void;
  onConvInputChange: (text: string) => void;
  onConvSend: () => void;
  onViewTrace: () => void;
  onViewGraph: () => void;
}

export default function ChatView({
  phase,
  onSelect,
  onRationaleChange,
  onSubmit,
  onRegenerate,
  onConvInputChange,
  onConvSend,
  onViewTrace,
  onViewGraph,
}: Props) {
  const question = phase.kind === 'exploring' ? phase.question : phase.session.question;
  const session: Session | null = phase.kind !== 'exploring' ? phase.session : null;
  const selected = phase.kind === 'generated' ? phase.selected
    : phase.kind === 'submitted' ? phase.selected : undefined;
  const rationale = (phase.kind === 'generated' || phase.kind === 'submitted') ? phase.rationale : '';
  const submitting = phase.kind === 'generated' ? phase.submitting : false;
  const submitted = phase.kind === 'submitted';
  const convInput = phase.kind === 'submitted' ? phase.convInput : '';
  const convPending = phase.kind === 'submitted' ? phase.convPending : false;
  const messages: Message[] = session?.messages ?? [];

  return (
    <div className="chat">
      {/* User message */}
      <div className="msg msg-user">
        <div className="avatar avatar-user">ER</div>
        <div className="msg-body">
          <div className="msg-meta">You · {fmtTime()}</div>
          <div className="msg-text">{question}</div>
        </div>
      </div>

      {/* Explorer card */}
      <div className="msg msg-agent">
        <div className="avatar avatar-agent"><AgentMark /></div>
        <div className="msg-body">
          <div className="msg-meta">
            Explorer agent
            {session
              ? ` · traversal complete · ${session.exploration.durationSec.toFixed(1)}s`
              : ' · running…'}
          </div>
          <ExplorerCard
            exploration={session?.exploration ?? null}
            hasGraph={!!session?.knowledgeGraph}
            onViewTrace={onViewTrace}
            onViewGraph={onViewGraph}
          />
        </div>
      </div>

      {session && (
        <>
          <div className="msg msg-agent">
            <div className="avatar avatar-agent"><AgentMark /></div>
            <div className="msg-body">
              <div className="msg-meta">Generator · 2 candidates</div>
              <div className="msg-text">
                Two hypotheses, each grounded in a distinct subgraph of the explored knowledge.
                Pick the one you want to develop further.
              </div>
            </div>
          </div>

          <div className="cards-row">
            {session.hypotheses.map((h) => (
              <HypothesisCard
                key={h.id}
                hypothesis={h}
                selected={selected === h.id}
                dimmed={selected != null && selected !== h.id}
                disabled={submitted || submitting}
                onSelect={() => onSelect(h.id)}
              />
            ))}
          </div>

          <div className="form-wrap">
            <SelectionForm
              hypotheses={session.hypotheses}
              selected={selected}
              rationale={rationale}
              submitting={submitting}
              submitted={submitted}
              onSelect={onSelect}
              onRationaleChange={onRationaleChange}
              onSubmit={onSubmit}
              onRegenerate={onRegenerate}
            />
          </div>

          {submitted && (
            <div className="conv-wrap">
              <ConversationInput
                messages={messages}
                input={convInput}
                pending={convPending}
                onInputChange={onConvInputChange}
                onSend={onConvSend}
              />
            </div>
          )}
        </>
      )}
    </div>
  );
}

function fmtTime() {
  return new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', hour12: false });
}
