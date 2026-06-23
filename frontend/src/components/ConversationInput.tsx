import { useRef, useEffect } from 'react';
import type { Message } from '../types';
import Md from './Md';

interface Props {
  messages: Message[];
  input: string;
  pending: boolean;
  onInputChange: (v: string) => void;
  onSend: () => void;
}

const MAX_CHARS = 1000;

export default function ConversationInput({
  messages,
  input,
  pending,
  onInputChange,
  onSend,
}: Props) {
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages.length]);

  function handleKeyDown(e: React.KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      if (canSend) onSend();
    }
  }

  const canSend = input.trim().length > 0 && !pending;

  return (
    <div className="conversation">
      {(messages.length > 0 || pending) && (
        <div className="conversation-messages">
          {messages.map((m, i) => (
            <div key={i} className={`conv-msg conv-msg-${m.role}`}>
              <div className="conv-msg-role">
                {m.role === 'user' ? 'You' : 'Assistant'}
              </div>
              <div className="conv-msg-text">
                {m.role === 'assistant' ? <Md>{m.content}</Md> : m.content}
              </div>
            </div>
          ))}
          {pending && (
            <div className="conv-msg conv-msg-assistant">
              <div className="conv-msg-role">Assistant</div>
              <div className="conv-msg-text conv-thinking">
                <span className="thinking-dot" /><span className="thinking-dot" /><span className="thinking-dot" />
              </div>
            </div>
          )}
          <div ref={bottomRef} />
        </div>
      )}

      <div className="conv-input-wrap">
        <div className="conv-input-eyebrow">
          Continue the conversation based on your chosen hypothesis
        </div>
        <div className="conv-input-box">
          <textarea
            className="conv-textarea"
            rows={3}
            maxLength={MAX_CHARS}
            placeholder="Ask a follow-up question, request experimental design ideas, challenge the prediction…  (Enter to send, Shift+Enter for new line)"
            value={input}
            onChange={(e) => onInputChange(e.target.value)}
            onKeyDown={handleKeyDown}
            disabled={pending}
          />
          <div className="conv-input-footer">
            <span className="conv-char-count">{input.length} / {MAX_CHARS}</span>
            <button
              type="button"
              className="primary-btn primary-btn-inline"
              onClick={onSend}
              disabled={!canSend}
            >
              {pending ? 'Thinking…' : 'Send →'}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
