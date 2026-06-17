# Hypothesis Forge — Frontend

React + TypeScript chat-style frontend for the agentic hypothesis-generation system.
Pose a research question → an explorer agent traverses the knowledge graph → a generator
proposes two competing hypotheses → the user picks one and (optionally) leaves a rationale.

Built to be a weekend-doable implementation: small surface area, no extra libraries beyond
React + Vite, mock mode so you can demo before the backend is finished.

## Quick start

```bash
cd frontend
npm install
cp .env.example .env
npm run dev
```

Open <http://localhost:5173>.

By default `VITE_USE_MOCK=true` — the app runs against an in-memory mock and you can click
through the whole flow without a backend.

When your backend is reachable:

```bash
# .env
VITE_USE_MOCK=false
VITE_API_PROXY_TARGET=http://localhost:8000
```

Vite proxies `/api/*` to that target during `npm run dev`. For production, serve the static
build (`npm run build` → `dist/`) behind the same origin as your API, or set up your own
reverse proxy.

## REST API contract

The frontend expects three endpoints. Shapes live in `src/types.ts` — keep them in sync with
your backend models.

### `POST /api/sessions` — create a session

Runs the explorer agent and generator. Returns synchronously when both finish.

**Request**

```json
{ "question": "Why do transformer models exhibit emergent ICL at certain scales?" }
```

**Response 200**

```json
{
  "sessionId": "abc123",
  "question": "Why do transformer models exhibit emergent ICL at certain scales?",
  "createdAt": "2026-06-17T14:32:00Z",
  "exploration": {
    "nodesTraversed": 127,
    "relations": 43,
    "sourcePapers": 8,
    "clusters": 4,
    "communities": ["Mechanistic interpretability", "Scaling laws"],
    "durationSec": 14.2
  },
  "hypotheses": [
    {
      "id": "A",
      "classification": "Mechanistic · capacity-side",
      "headline": "Phase transition in induction-head circuit formation",
      "statement": "Emergent in-context learning corresponds to …",
      "drawnFrom": ["induction heads", "grokking", "phase transitions"],
      "falsifiablePrediction": "Synthetic-task ICL accuracy shows a sharp discontinuity …"
    },
    { "id": "B", "...": "..." }
  ]
}
```

The frontend renders whatever exploration stats and chip lists the backend supplies — if a
field is empty (e.g. `communities: []`) the section just hides.

### `POST /api/sessions/{sessionId}/select` — record the choice

**Request**

```json
{
  "selectedId": "A",
  "rationale": "Optional free text, up to 600 chars."
}
```

`rationale` is omitted when the user leaves the field empty.

**Response 200**

```json
{ "ok": true }
```

### `GET /api/sessions` — sidebar list (optional)

If you skip this for v1, leave `VITE_USE_MOCK=true` for the sidebar only, or have the
endpoint return `{ "sessions": [] }`. The sidebar is non-functional in v1 — past sessions
display but aren't yet clickable.

```json
{
  "sessions": [
    {
      "id": "abc123",
      "title": "ICL emergence mechanism",
      "createdAt": "2026-06-17T14:32:00Z",
      "status": "awaiting"
    }
  ]
}
```

`status` is one of `awaiting | developed | archived`.

## Errors

Any non-2xx response is shown as a banner with `{ "detail": "..." }` or `{ "message": "..." }`
extracted from the JSON body, falling back to HTTP status text. Network errors are caught
and surfaced the same way.

## Project layout

```
frontend/
├── index.html
├── package.json
├── vite.config.ts
├── tsconfig.json
├── .env.example
└── src/
    ├── main.tsx              ← Vite entry
    ├── App.tsx               ← top-level shell + phase state machine
    ├── api.ts                ← fetch client + mock implementations
    ├── types.ts              ← shared TS types
    ├── styles.css            ← single stylesheet
    └── components/
        ├── Sidebar.tsx
        ├── QuestionInput.tsx ← idle state: pose a question
        ├── ChatView.tsx      ← conversation timeline
        ├── ExplorerCard.tsx  ← KG-traversal result + skeleton
        ├── HypothesisCard.tsx
        ├── SelectionForm.tsx ← radio + optional rationale + submit
        └── icons.tsx
```

## Phase state machine

`App.tsx` owns a single `phase` value:

```
idle  ── ask ──▶  exploring  ── ok ──▶  generated  ── select+submit ──▶  submitted
                       │                     │
                       │ error               │ regenerate
                       ▼                     ▼
                     idle                 exploring
```

The user can submit without filling the rationale — the rationale field is optional
end-to-end and is only sent if non-empty.

## Putting this on a feature branch in your repo

From your repo root:

```bash
git checkout -b feature/hypothesis-forge-ui

# Copy this folder into your repo. If your backend lives in the same repo, drop
# this whole `frontend/` directory at the root. If it lives elsewhere, pick a
# different name (e.g. `web/` or `ui/`).
cp -r /path/to/this/frontend ./frontend

cd frontend && npm install && npm run typecheck

git add frontend
git commit -m "feat(frontend): chat-style UI for hypothesis generation"
git push -u origin feature/hypothesis-forge-ui
```

Open a PR from `feature/hypothesis-forge-ui` to your default branch.

## Scope shipped this weekend

What's in:

- Full flow: question → exploration → two hypotheses → optional rationale → submit
- Loading skeleton for the explorer
- Mock mode so the UI demos without a backend
- Light theme; CSS variables make a dark theme a one-file diff later
- Keyboard shortcut: ⌘/Ctrl+Enter to submit a question
- 600-char rationale counter
- Type-safe API contract

What's stubbed (do later):

- Sidebar past sessions are listed but not yet clickable
- "Knowledge graph", "Export", "View reasoning trace" buttons are visual only
- No auth, no streaming, no session sharing

## License

Whatever your project uses. No third-party UI or styling libraries to attribute.
