import type { AuthUser } from './auth';
import { getAuthHeader, saveAuth, clearAuth } from './auth';
import type { Session, SelectPayload, SessionListItem } from './types';

const USE_MOCK = import.meta.env.VITE_USE_MOCK !== 'false';
const API_BASE = '/api';

// ---------------------------------------------------------------------------
// HTTP helper
// ---------------------------------------------------------------------------

async function jsonFetch<T>(input: string, init?: RequestInit): Promise<T> {
  const res = await fetch(input, {
    ...init,
    headers: {
      'Content-Type': 'application/json',
      ...getAuthHeader(),
      ...(init?.headers ?? {}),
    },
  });
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = body.detail ?? body.message ?? detail;
    } catch { /* ignore */ }
    throw new Error(`${res.status} ${detail}`);
  }
  return (await res.json()) as T;
}

// ---------------------------------------------------------------------------
// Auth API calls (always real — no mock needed for auth)
// ---------------------------------------------------------------------------

export async function authRegister(
  username: string,
  email: string,
  password: string,
): Promise<{ token: string; user: AuthUser }> {
  const result = await jsonFetch<{ token: string; user: AuthUser }>(
    `${API_BASE}/auth/register`,
    { method: 'POST', body: JSON.stringify({ username, email, password }) },
  );
  saveAuth(result.token, result.user);
  return result;
}

export async function authLogin(
  username_or_email: string,
  password: string,
): Promise<{ token: string; user: AuthUser }> {
  const result = await jsonFetch<{ token: string; user: AuthUser }>(
    `${API_BASE}/auth/login`,
    { method: 'POST', body: JSON.stringify({ username_or_email, password }) },
  );
  saveAuth(result.token, result.user);
  return result;
}

export async function authMe(): Promise<AuthUser> {
  return jsonFetch<AuthUser>(`${API_BASE}/auth/me`);
}

export function authLogout(): void {
  clearAuth();
}

// ---------------------------------------------------------------------------
// Real REST calls
// ---------------------------------------------------------------------------

async function realGenerate(question: string): Promise<Session> {
  return jsonFetch<Session>(`${API_BASE}/sessions`, {
    method: 'POST',
    body: JSON.stringify({ question }),
  });
}

async function realSelect(sessionId: string, payload: SelectPayload): Promise<void> {
  await jsonFetch<{ ok: true }>(`${API_BASE}/sessions/${sessionId}/select`, {
    method: 'POST',
    body: JSON.stringify(payload),
  });
}

async function realListSessions(): Promise<SessionListItem[]> {
  const r = await jsonFetch<{ sessions: SessionListItem[] }>(`${API_BASE}/sessions`);
  return r.sessions;
}

async function realGetSession(sessionId: string): Promise<Session> {
  return jsonFetch<Session>(`${API_BASE}/sessions/${sessionId}`);
}

async function realSendMessage(sessionId: string, message: string): Promise<string> {
  const r = await jsonFetch<{ reply: string }>(`${API_BASE}/sessions/${sessionId}/chat`, {
    method: 'POST',
    body: JSON.stringify({ message }),
  });
  return r.reply;
}

// ---------------------------------------------------------------------------
// Mock implementations
// ---------------------------------------------------------------------------

const MOCK_DELAY_MS = 1600;
const sleep = (ms: number) => new Promise<void>((r) => setTimeout(r, ms));

// In-memory store for mock sessions (scoped to the current auth user)
const _mockSessions = new Map<string, Session>();

// User-keyed index so sessions switch when auth changes
let _mockUserId = '__anonymous__';
const _userSessionIds = new Map<string, Set<string>>();

function _mockUserSessions(): Session[] {
  const ids = _userSessionIds.get(_mockUserId) ?? new Set<string>();
  return Array.from(ids)
    .map((id) => _mockSessions.get(id))
    .filter((s): s is Session => s !== undefined)
    .reverse();
}

// Called by App when auth state changes so mock sessions respect user scoping
export function setMockUserId(id: string): void {
  _mockUserId = id;
}

const MOCK_TRACE = [
  { step: 'Search', description: 'Semantic search over 50K paper embeddings', durationSec: 0.12, details: 'Found 10 candidate papers matching the query via Weaviate.' },
  { step: 'Explorer', description: 'BFS traversal of the Neo4j knowledge graph', durationSec: 2.31, details: 'Traversed 127 nodes and 43 typed relations. Discovered 4 communities via Louvain clustering.' },
  { step: 'Retriever', description: 'LLM filtering of retrieved context', durationSec: 1.85, details: 'Condensed 127 nodes into a 2 400-token structured context summary.' },
  { step: 'Generator', description: 'Hypothesis synthesis from curated context', durationSec: 3.42, details: 'Generated 2 competing hypotheses grounded in distinct subgraphs.' },
];

const MOCK_KG = {
  nodes: [
    { id: 'query', label: 'Research Query', type: 'query' as const },
    { id: 'c_mech', label: 'Mechanistic Interpretability', type: 'community' as const },
    { id: 'c_scale', label: 'Scaling Laws', type: 'community' as const },
    { id: 'c_train', label: 'Training Dynamics', type: 'community' as const },
    { id: 'c_eval', label: 'Evaluation Methods', type: 'community' as const },
    { id: 'n_ind', label: 'Induction Heads', type: 'concept' as const },
    { id: 'n_circ', label: 'Circuit Formation', type: 'concept' as const },
    { id: 'n_phase', label: 'Phase Transitions', type: 'concept' as const },
    { id: 'n_grok', label: 'Grokking', type: 'concept' as const },
    { id: 'n_icl', label: 'Emergent ICL', type: 'concept' as const },
    { id: 'n_metric', label: 'Metric Continuity', type: 'concept' as const },
    { id: 'n_data', label: 'Data Composition', type: 'concept' as const },
    { id: 'n_dist', label: 'Distributional Learning', type: 'concept' as const },
    {
      id: 'p_olsson', label: 'Olsson et al. 2022', type: 'paper' as const,
      paperId: 'arXiv:2209.11895',
      abstract: "We present evidence that 'induction heads' are the mechanism by which transformers perform in-context learning. We find a striking correlation between the phase change in loss curves and the formation of induction heads.",
      summary: 'Shows induction heads form during a sharp phase transition and are the core circuit enabling in-context learning.',
    },
    {
      id: 'p_wei', label: 'Wei et al. 2022', type: 'paper' as const,
      paperId: 'arXiv:2206.07682',
      abstract: 'We investigate emergent abilities of large language models — abilities not present in smaller-scale models that appear when model scale reaches a critical threshold.',
      summary: 'Documents emergent abilities that appear unpredictably at scale and proposes scale as the primary driver.',
    },
    {
      id: 'p_schaeffer', label: 'Schaeffer et al. 2023', type: 'paper' as const,
      paperId: 'arXiv:2304.15004',
      abstract: 'Are emergent abilities a mirage? We show apparent emergent abilities are a consequence of the researcher\'s choice of metric rather than fundamental changes in model behaviour.',
      summary: 'Argues emergence is an artifact of discontinuous metrics; continuous metrics reveal smooth, predictable capability scaling.',
    },
  ],
  edges: [
    { source: 'query', target: 'c_mech', relation: 'explores' },
    { source: 'query', target: 'c_scale', relation: 'explores' },
    { source: 'query', target: 'c_train', relation: 'explores' },
    { source: 'query', target: 'c_eval', relation: 'explores' },
    { source: 'c_mech', target: 'n_ind', relation: 'contains' },
    { source: 'c_mech', target: 'n_circ', relation: 'contains' },
    { source: 'c_mech', target: 'n_phase', relation: 'contains' },
    { source: 'c_mech', target: 'p_olsson', relation: 'cites' },
    { source: 'c_scale', target: 'n_grok', relation: 'contains' },
    { source: 'c_scale', target: 'n_icl', relation: 'contains' },
    { source: 'c_scale', target: 'p_wei', relation: 'cites' },
    { source: 'c_eval', target: 'n_metric', relation: 'contains' },
    { source: 'c_eval', target: 'n_data', relation: 'contains' },
    { source: 'c_eval', target: 'p_schaeffer', relation: 'cites' },
    { source: 'n_ind', target: 'n_icl', relation: 'causes' },
    { source: 'n_phase', target: 'n_grok', relation: 'predicts' },
    { source: 'n_metric', target: 'n_icl', relation: 'explains' },
    { source: 'p_olsson', target: 'n_ind', relation: 'studies' },
    { source: 'p_wei', target: 'n_icl', relation: 'studies' },
    { source: 'p_schaeffer', target: 'n_metric', relation: 'studies' },
  ],
};

async function mockGenerate(question: string): Promise<Session> {
  await sleep(MOCK_DELAY_MS);
  const session: Session = {
    sessionId: 'mock-' + Math.random().toString(36).slice(2, 8),
    question,
    createdAt: new Date().toISOString(),
    exploration: {
      nodesTraversed: 127, relations: 43, sourcePapers: 8, clusters: 4,
      communities: ['Mechanistic interpretability', 'Scaling laws', 'Training dynamics', 'Evaluation methods'],
      durationSec: 14.2,
    },
    hypotheses: [
      {
        id: 'A', classification: 'Mechanistic · capacity-side',
        headline: 'Phase transition in induction-head circuit formation',
        statement: 'Emergent in-context learning corresponds to a sharp phase transition in induction-head circuit formation. Below a critical product of depth and width, multi-step attention compositions are sample-inefficient and decoupled; above it, they crystallise into a stable circuit that supports in-context pattern matching at near-Bayes-optimal rates on synthetic tasks.',
        drawnFrom: ['induction heads', 'grokking', 'circuit formation', 'phase transitions'],
        falsifiablePrediction: 'Synthetic-task ICL accuracy shows a sharp discontinuity locatable within a ±5% parameter band across model families.',
      },
      {
        id: 'B', classification: 'Statistical · measurement-side',
        headline: 'Apparent emergence as evaluation discontinuity',
        statement: 'The emergence pattern is an artefact of discontinuous evaluation. Pretraining corpora cross a coverage-density threshold at which few-shot meta-patterns become reliably retrievable; underlying capacity itself improves smoothly. Replacing exact-match metrics with continuous proxies should reveal a smooth curve and no critical scale.',
        drawnFrom: ['emergence critique', 'metric continuity', 'distributional learning', 'data composition'],
        falsifiablePrediction: 'No smooth continuous metric reveals a discontinuity at the same critical scale across at least three model families.',
      },
    ],
    messages: [],
    reasoningTrace: MOCK_TRACE,
    knowledgeGraph: MOCK_KG,
  };
  _mockSessions.set(session.sessionId, session);
  const ids = _userSessionIds.get(_mockUserId) ?? new Set<string>();
  ids.add(session.sessionId);
  _userSessionIds.set(_mockUserId, ids);
  return session;
}

async function mockSelect(sessionId: string, payload: SelectPayload): Promise<void> {
  await sleep(400);
  const s = _mockSessions.get(sessionId);
  if (s) { s.selectedId = payload.selectedId; s.rationale = payload.rationale; }
}

async function mockListSessions(): Promise<SessionListItem[]> {
  await sleep(200);
  return _mockUserSessions().map((s) => ({
    id: s.sessionId,
    title: s.question.slice(0, 60),
    createdAt: s.createdAt,
    status: s.selectedId ? 'developed' : 'awaiting',
  }));
}

async function mockGetSession(sessionId: string): Promise<Session> {
  await sleep(300);
  const s = _mockSessions.get(sessionId);
  if (!s) throw new Error(`Session ${sessionId} not found`);
  return { ...s };
}

async function mockSendMessage(sessionId: string, message: string): Promise<string> {
  await sleep(800);
  const s = _mockSessions.get(sessionId);
  const hyp = s?.hypotheses.find((h) => h.id === s?.selectedId);
  const turn = s ? Math.floor(s.messages.length / 2) + 1 : 1;
  const now = new Date().toISOString();
  const reply = hyp
    ? `(Turn ${turn}) Building on **${hyp.headline}**: your question — "${message.slice(0, 60)}" — points toward an important extension. The key circuits identified in the explorer subgraph suggest a deeper mechanistic link.\n\n*[Mock response — set VITE_USE_MOCK=false for real AI continuations.]*`
    : `Thank you for your question. *[Mock response — set VITE_USE_MOCK=false for real AI continuations.]*`;
  if (s) {
    s.messages = [
      ...s.messages,
      { role: 'user', content: message, createdAt: now },
      { role: 'assistant', content: reply, createdAt: now },
    ];
  }
  return reply;
}

// ---------------------------------------------------------------------------
// Exports
// ---------------------------------------------------------------------------

export const api = {
  generate:     USE_MOCK ? mockGenerate     : realGenerate,
  select:       USE_MOCK ? mockSelect       : realSelect,
  listSessions: USE_MOCK ? mockListSessions : realListSessions,
  getSession:   USE_MOCK ? mockGetSession   : realGetSession,
  sendMessage:  USE_MOCK ? mockSendMessage  : realSendMessage,
  isMock: USE_MOCK,
};

// ---------------------------------------------------------------------------
// Export helper
// ---------------------------------------------------------------------------

export function exportSessionAsMarkdown(session: Session): void {
  const lines: string[] = [
    `# Hypothesis Forge — Session Export`,
    ``,
    `**Question:** ${session.question}`,
    `**Created:** ${new Date(session.createdAt).toLocaleString()}`,
    ``,
    `---`,
    ``,
    `## Hypotheses`,
    ``,
  ];

  for (const h of session.hypotheses) {
    lines.push(`### Hypothesis ${h.id} — ${h.headline}`);
    lines.push(`**Classification:** ${h.classification}`);
    lines.push(``);
    lines.push(h.statement);
    if (h.drawnFrom.length) {
      lines.push(``);
      lines.push(`**Drawn from:** ${h.drawnFrom.join(', ')}`);
    }
    if (h.falsifiablePrediction) {
      lines.push(``);
      lines.push(`**Falsifiable prediction:** ${h.falsifiablePrediction}`);
    }
    lines.push(``);
  }

  if (session.selectedId) {
    lines.push(`---`);
    lines.push(``);
    lines.push(`## Selection`);
    lines.push(``);
    lines.push(`**Selected:** Hypothesis ${session.selectedId}`);
    if (session.rationale) {
      lines.push(``);
      lines.push(`**Rationale:** ${session.rationale}`);
    }
    lines.push(``);
  }

  if (session.messages.length) {
    lines.push(`---`);
    lines.push(``);
    lines.push(`## Conversation`);
    lines.push(``);
    for (const m of session.messages) {
      const who = m.role === 'user' ? 'You' : 'Assistant';
      lines.push(`**${who}:** ${m.content}`);
      lines.push(``);
    }
  }

  const blob = new Blob([lines.join('\n')], { type: 'text/markdown' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `hypothesis-forge-${session.sessionId.slice(0, 8)}.md`;
  a.click();
  URL.revokeObjectURL(url);
}
