# GraphPulse UI — Next.js Dashboard

> **Week 8 deliverable.** The frontend is planned but not yet scaffolded.
> This README documents the intended stack, pages, and how to bootstrap the app.

---

## Stack

| Layer | Technology |
|---|---|
| Framework | Next.js 15 (App Router) |
| Styling | Tailwind CSS |
| Components | shadcn/ui |
| Charts | Recharts |
| Real-time | Server-Sent Events (SSE) |
| API | GraphPulse FastAPI at `localhost:8000` |

---

## Planned Pages

### 1. Live Feed (`/`)
Streams fraud scoring decisions in real time via SSE from the FastAPI backend.
Each incoming score is rendered as a coloured card (red = fraud, green = legit)
with transaction ID, amount, fraud score, and latency.

### 2. Alert Dashboard (`/alerts`)
Filterable table of high-risk transactions (`is_fraud = true`). Sortable by
fraud score, timestamp, and model. Supports CSV export.

### 3. Model Comparison (`/models`)
Side-by-side PR-AUC / ROC-AUC table for all available model artefacts.
Pulls from `GET /health` (model list) and `GET /metrics` (Prometheus counters).

### 4. SHAP Explanations (`/explain`)
Waterfall chart of the top-5 SHAP feature importances for a given
`transaction_id`. Fetches from `GET /explain/{transaction_id}`.

### 5. Drift Monitor (`/drift`)
Time-series chart of ADWIN drift alert count over time. Data sourced from
the `drift_alerts` PostgreSQL table via a thin REST endpoint.

---

## Scaffolding (Week 8)

Run inside `apps/ui/`:

```bash
npx create-next-app@latest . --typescript --tailwind --app
```

Then add dependencies:

```bash
npm install recharts @radix-ui/react-dialog lucide-react clsx tailwind-merge
npx shadcn-ui@latest init
```

---

## API Integration

All API calls target the GraphPulse FastAPI service. During local development:

```
NEXT_PUBLIC_API_URL=http://localhost:8000
```

Key endpoints used by the UI:

| Endpoint | UI Page |
|---|---|
| `GET /health` | Model Comparison |
| `POST /score` | Live Feed (manual test form) |
| `GET /explain/{id}` | SHAP Explanations |
| `GET /metrics` | Drift Monitor (raw Prometheus text) |

For SSE streaming, a thin `/api/stream` Next.js route handler will proxy
the Kafka scores topic via the consumer output.

---

## Development

```bash
cd apps/ui
npm run dev          # http://localhost:3000
npm run build        # Production build
npm run lint         # ESLint
```
