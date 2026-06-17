// Shared types — must match the JSON shapes returned by the backend.
// See src/api/models.py for the Python counterparts.

export type HypothesisId = 'A' | 'B';

export interface ExplorerResult {
  nodesTraversed: number;
  relations: number;
  sourcePapers: number;
  clusters: number;
  communities: string[];
  durationSec: number;
}

export interface Hypothesis {
  id: HypothesisId;
  classification: string;
  headline: string;
  statement: string;
  drawnFrom: string[];
  falsifiablePrediction: string;
}

export interface ReasoningStep {
  step: string;
  description: string;
  durationSec: number;
  details: string;
}

export interface Message {
  role: 'user' | 'assistant';
  content: string;
  createdAt: string;
}

export interface KGNode {
  id: string;
  label: string;
  type: 'query' | 'community' | 'concept' | 'paper';
  // Paper-specific metadata (optional)
  paperId?: string;
  abstract?: string;
  summary?: string;
}

export interface KGEdge {
  source: string;
  target: string;
  relation: string;
}

export interface KnowledgeGraph {
  nodes: KGNode[];
  edges: KGEdge[];
}

export interface Session {
  sessionId: string;
  question: string;
  createdAt: string;
  exploration: ExplorerResult;
  hypotheses: [Hypothesis, Hypothesis];
  selectedId?: HypothesisId;
  rationale?: string;
  messages: Message[];
  reasoningTrace: ReasoningStep[];
  knowledgeGraph?: KnowledgeGraph;
}

export interface SessionListItem {
  id: string;
  title: string;
  createdAt: string;
  status: 'awaiting' | 'developed' | 'archived';
}

export interface SelectPayload {
  selectedId: HypothesisId;
  rationale?: string;
}
