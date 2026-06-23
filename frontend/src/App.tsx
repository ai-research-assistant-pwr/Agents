import { useEffect, useRef, useState } from 'react';
import { api, authLogout, exportSessionAsMarkdown, setMockUserId } from './api';
import { getAuthUser } from './auth';
import type { AuthUser } from './auth';
import type { HypothesisId, ModelList, Persona, Session, SessionListItem } from './types';
import Sidebar from './components/Sidebar';
import ChatView, { type ChatPhase } from './components/ChatView';
import QuestionInput from './components/QuestionInput';
import ReasoningTraceModal from './components/ReasoningTraceModal';
import KnowledgeGraphPanel from './components/KnowledgeGraphPanel';
import AuthModal from './components/AuthModal';
import { GraphIcon } from './components/icons';

type AppPhase = { kind: 'idle' } | ChatPhase;

const EMPTY_MODELS: ModelList = { retrieverModels: [], generatorModels: [] };

export default function App() {
  const [phase, setPhase] = useState<AppPhase>({ kind: 'idle' });
  const [sessions, setSessions] = useState<SessionListItem[]>([]);
  const [models, setModels] = useState<ModelList>(EMPTY_MODELS);
  const [personas, setPersonas] = useState<Persona[]>([]);
  const [selectedPersonaId, setSelectedPersonaId] = useState<string>('');
  const [selectedRetrieverModel, setSelectedRetrieverModel] = useState('');
  const [selectedGeneratorModel, setSelectedGeneratorModel] = useState('');
  const [modelsLoading, setModelsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [traceOpen, setTraceOpen] = useState(false);
  const [kgOpen, setKgOpen] = useState(false);
  const [loadingSession, setLoadingSession] = useState(false);
  const [authUser, setAuthUser] = useState<AuthUser | null>(getAuthUser);
  const [authModalOpen, setAuthModalOpen] = useState(false);
  const [userMenuOpen, setUserMenuOpen] = useState(false);
  const userMenuRef = useRef<HTMLDivElement>(null);

  // Close user menu on outside click
  useEffect(() => {
    function onMouseDown(e: MouseEvent) {
      if (userMenuRef.current && !userMenuRef.current.contains(e.target as Node)) {
        setUserMenuOpen(false);
      }
    }
    document.addEventListener('mousedown', onMouseDown);
    return () => document.removeEventListener('mousedown', onMouseDown);
  }, []);

  // Reload sessions when auth state changes
  useEffect(() => {
    const userId = authUser?.id ?? '__anonymous__';
    setMockUserId(userId);
    api.listSessions().then(setSessions).catch(() => setSessions([]));
  }, [authUser]);

  useEffect(() => {
    let cancelled = false;
    setModelsLoading(true);
    api.listModels()
      .then((nextModels) => {
        if (cancelled) return;
        setModels(nextModels);
        setSelectedRetrieverModel((current) => (
          current && nextModels.retrieverModels.includes(current)
            ? current
            : nextModels.retrieverModels[0] ?? ''
        ));
        setSelectedGeneratorModel((current) => (
          current && nextModels.generatorModels.includes(current)
            ? current
            : nextModels.generatorModels[0] ?? ''
        ));
      })
      .catch((e) => {
        if (cancelled) return;
        setModels(EMPTY_MODELS);
        setError(e instanceof Error ? e.message : 'Failed to load models');
      })
      .finally(() => {
        if (!cancelled) setModelsLoading(false);
      });
    return () => { cancelled = true; };
  }, []);

  useEffect(() => {
    api.listPersonas().then(setPersonas).catch(() => setPersonas([]));
  }, []);

  function refreshSidebar() {
    api.listSessions().then(setSessions).catch(() => {});
  }

  // ---------------------------------------------------------------------------
  // Auth
  // ---------------------------------------------------------------------------

  function handleAuthSuccess(user: AuthUser) {
    setAuthUser(user);
    setAuthModalOpen(false);
    setPhase({ kind: 'idle' });
    setSessions([]);
  }

  function handleLogout() {
    authLogout();
    setAuthUser(null);
    setUserMenuOpen(false);
    setSessions([]);
    setPhase({ kind: 'idle' });
    setMockUserId('__anonymous__');
  }

  // ---------------------------------------------------------------------------
  // Helpers
  // ---------------------------------------------------------------------------

  function currentSession(): Session | null {
    if (phase.kind === 'idle' || phase.kind === 'exploring') return null;
    return phase.session;
  }

  function activeSessionId(): string | null {
    const s = currentSession();
    return s ? s.sessionId : null;
  }

  function getQuestion(): string {
    if (phase.kind === 'idle') return '';
    if (phase.kind === 'exploring') return phase.question;
    return phase.session.question;
  }

  // ---------------------------------------------------------------------------
  // Actions
  // ---------------------------------------------------------------------------

  async function handleAsk(question: string) {
    if (!selectedRetrieverModel || !selectedGeneratorModel) {
      setError('Choose a retriever and generator model before starting.');
      return;
    }
    setError(null);
    setPhase({ kind: 'exploring', question });
    try {
      const session = await api.generate({
        question,
        retrieverModelName: selectedRetrieverModel,
        generatorModelName: selectedGeneratorModel,
        personaId: selectedPersonaId || undefined,
      });
      setPhase({ kind: 'generated', session, rationale: '', submitting: false });
      refreshSidebar();
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to generate hypotheses');
      setPhase({ kind: 'idle' });
    }
  }

  function handleSelect(id: HypothesisId) {
    if (phase.kind !== 'generated') return;
    setPhase({ ...phase, selected: id });
  }

  function handleRationaleChange(rationale: string) {
    if (phase.kind !== 'generated') return;
    setPhase({ ...phase, rationale });
  }

  async function handleSubmit() {
    if (phase.kind !== 'generated' || !phase.selected) return;
    const { session, selected, rationale } = phase;
    setError(null);
    setPhase({ ...phase, submitting: true });
    try {
      await api.select(session.sessionId, {
        selectedId: selected,
        rationale: rationale.trim() ? rationale.trim() : undefined,
      });
      const updatedSession: Session = {
        ...session,
        selectedId: selected,
        rationale: rationale.trim() || undefined,
      };
      setPhase({
        kind: 'submitted',
        session: updatedSession,
        selected,
        rationale,
        convInput: '',
        convPending: false,
      });
      refreshSidebar();
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to submit selection');
      setPhase({ ...phase, submitting: false });
    }
  }

  function handleRegenerate() {
    const q = getQuestion();
    if (q) handleAsk(q);
  }

  function handleNewHypothesis() {
    setError(null);
    setTraceOpen(false);
    setKgOpen(false);
    setPhase({ kind: 'idle' });
  }

  async function handleDeleteSession(id: string) {
    try {
      await api.deleteSession(id);
      setSessions((prev) => prev.filter((s) => s.id !== id));
      if (activeSessionId() === id) {
        setPhase({ kind: 'idle' });
        setTraceOpen(false);
        setKgOpen(false);
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to delete session');
    }
  }

  // ---------------------------------------------------------------------------
  // Conversation
  // ---------------------------------------------------------------------------

  function handleConvInputChange(text: string) {
    if (phase.kind !== 'submitted') return;
    setPhase({ ...phase, convInput: text });
  }

  async function handleConvSend() {
    if (phase.kind !== 'submitted' || !phase.convInput.trim()) return;
    const message = phase.convInput.trim();
    setPhase({ ...phase, convInput: '', convPending: true });
    try {
      const reply = await api.sendMessage(phase.session.sessionId, message);
      const now = new Date().toISOString();
      const updatedSession: Session = {
        ...phase.session,
        messages: [
          ...phase.session.messages,
          { role: 'user', content: message, createdAt: now },
          { role: 'assistant', content: reply, createdAt: now },
        ],
      };
      setPhase({ ...phase, session: updatedSession, convInput: '', convPending: false });
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to send message');
      setPhase({ ...phase, convPending: false });
    }
  }

  // ---------------------------------------------------------------------------
  // Sidebar session loading
  // ---------------------------------------------------------------------------

  async function handleSessionClick(id: string) {
    if (id === activeSessionId()) return;
    setError(null);
    setTraceOpen(false);
    setKgOpen(false);
    setLoadingSession(true);
    try {
      const session = await api.getSession(id);
      if (session.selectedId) {
        setPhase({
          kind: 'submitted',
          session,
          selected: session.selectedId as HypothesisId,
          rationale: session.rationale ?? '',
          convInput: '',
          convPending: false,
        });
      } else {
        setPhase({ kind: 'generated', session, rationale: '', submitting: false });
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to load session');
    } finally {
      setLoadingSession(false);
    }
  }

  // ---------------------------------------------------------------------------
  // Export
  // ---------------------------------------------------------------------------

  function handleExport() {
    const s = currentSession();
    if (!s) return;
    exportSessionAsMarkdown(s);
  }

  // ---------------------------------------------------------------------------
  // Topbar title
  // ---------------------------------------------------------------------------

  function topbarTitle(): string {
    if (phase.kind === 'idle') return 'New hypothesis';
    const q = getQuestion();
    return q.length > 55 ? q.slice(0, 53) + '…' : q;
  }

  // ---------------------------------------------------------------------------
  // Render
  // ---------------------------------------------------------------------------

  const session = currentSession();
  const hasKG = !!session?.knowledgeGraph;
  const canExport = !!session;

  return (
    <div className="app">
      <Sidebar
        sessions={sessions}
        activeId={activeSessionId()}
        authUser={authUser}
        userMenuOpen={userMenuOpen}
        userMenuRef={userMenuRef}
        onNewHypothesis={handleNewHypothesis}
        onSessionClick={handleSessionClick}
        onDeleteSession={handleDeleteSession}
        onOpenAuthModal={() => setAuthModalOpen(true)}
        onToggleUserMenu={() => setUserMenuOpen((v) => !v)}
        onLogout={handleLogout}
      />

      <main className="main">
        <header className="topbar">
          <div className="topbar-title">
            <strong>{topbarTitle()}</strong>
            <span className="topbar-sep" />
            <span className="topbar-meta">{topbarMeta(phase)}</span>
          </div>
          <div className="topbar-actions">
            <button
              type="button"
              className="ghost-btn"
              onClick={() => setKgOpen(true)}
              disabled={!hasKG || loadingSession}
              title={hasKG ? 'View knowledge graph' : 'No graph yet'}
            >
              <GraphIcon />
              <span>Knowledge graph</span>
            </button>
            <button
              type="button"
              className="ghost-btn"
              onClick={handleExport}
              disabled={!canExport}
              title={canExport ? 'Export session as Markdown' : 'No session yet'}
            >
              Export
            </button>
          </div>
        </header>

        {error && <div className="error-banner" role="alert">{error}</div>}
        {loadingSession && <div className="loading-banner">Loading session…</div>}

        <div className="chat-region">
          {phase.kind === 'idle' && (
            <QuestionInput
              onSubmit={handleAsk}
              isMock={api.isMock}
              models={models}
              retrieverModel={selectedRetrieverModel}
              generatorModel={selectedGeneratorModel}
              modelsLoading={modelsLoading}
              onRetrieverModelChange={setSelectedRetrieverModel}
              onGeneratorModelChange={setSelectedGeneratorModel}
              personas={personas}
              selectedPersonaId={selectedPersonaId}
              onPersonaChange={setSelectedPersonaId}
            />
          )}
          {phase.kind !== 'idle' && (
            <ChatView
              phase={phase}
              onSelect={handleSelect}
              onRationaleChange={handleRationaleChange}
              onSubmit={handleSubmit}
              onRegenerate={handleRegenerate}
              onConvInputChange={handleConvInputChange}
              onConvSend={handleConvSend}
              onViewTrace={() => setTraceOpen(true)}
              onViewGraph={() => setKgOpen(true)}
            />
          )}
        </div>
      </main>

      {traceOpen && session?.reasoningTrace && (
        <ReasoningTraceModal
          steps={session.reasoningTrace}
          onClose={() => setTraceOpen(false)}
        />
      )}

      {kgOpen && session?.knowledgeGraph && (
        <KnowledgeGraphPanel
          graph={session.knowledgeGraph}
          onClose={() => setKgOpen(false)}
        />
      )}

      {authModalOpen && (
        <AuthModal
          onSuccess={handleAuthSuccess}
          onClose={() => setAuthModalOpen(false)}
        />
      )}
    </div>
  );
}

function topbarMeta(phase: AppPhase): string {
  if (phase.kind === 'idle') return 'pose a research question to begin';
  if (phase.kind === 'exploring') return 'exploring knowledge graph…';
  if (phase.kind === 'generated') {
    return phase.selected
      ? `hypothesis ${phase.selected} chosen · ready to develop`
      : '2 candidates · awaiting choice';
  }
  const msgCount = phase.session.messages.length / 2;
  return `hypothesis ${phase.selected} developed${msgCount > 0 ? ` · ${msgCount} exchange${msgCount > 1 ? 's' : ''}` : ''}`;
}
