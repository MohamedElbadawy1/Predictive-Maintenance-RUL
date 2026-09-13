# Sprint 22 — FastAPI Service (`api/`)

---

# Goal

Wrap the `pipeline/` layer (Sprint 21) in a real, callable HTTP API — routes
deliberately separated into their own `routes/` folder rather than one flat
`main.py`, so prediction and training logic each have their own file as the
project grows.

Built **after** `pipeline/`, not before — every route here is a thin wrapper
calling into logic that already existed and was already tested (`MLflowInferencePipeline`,
`TrainingPipeline`). No business logic lives in a route handler.

---

# Structure

```
api/
├── main.py              # FastAPI app, health checks, router registration
├── schemas.py           # Pydantic request/response models
├── dependencies.py       # Cached inference pipeline + in-memory job store
└── routes/
    ├── predict.py        # POST /predict/, POST /predict/reload
    └── train.py           # POST /train/tuning, POST /train/best-params, GET /train/status/{id}
```

---

# Endpoints

| Method | Path | What it does |
|---|---|---|
| GET | `/health` | Liveness — always responds, doesn't touch the model |
| GET | `/health/ready` | Readiness — reports whether the model is actually loaded |
| POST | `/predict/` | Predict RUL for one or more engines |
| POST | `/predict/reload` | Force-reload the cached model from MLflow (call after a promotion) |
| POST | `/train/tuning` | Kick off a full hyperparameter search (background job) |
| POST | `/train/best-params` | Retrain with the champion's known params, no search (background job) |
| GET | `/train/status/{job_id}` | Poll a training job's status/result |

Interactive docs at `/docs` (Swagger) once the server is running — generated
automatically from the Pydantic schemas, verified to render (all 7 paths present in
the OpenAPI schema).

---

# Key Design Decisions

**Lazy-loaded, cached model — not loaded at startup.** `PipelineCache` loads
`MLflowInferencePipeline` on the *first* prediction request, not when the API
process starts. This means `/health` responds immediately even if MLflow isn't
reachable yet; the tradeoff is that a misconfiguration only surfaces on the first
real prediction, not at boot. `/health/ready` exists specifically to let external
tooling check "is the model actually loaded" separately from "is the process alive."

**Training runs as background jobs, not synchronous requests.** A tuning job takes
minutes (confirmed: even a lightweight fixed-params retrain took ~19 real seconds
in testing; a full Optuna search takes much longer). Blocking an HTTP request on
that would time out or tie up a worker for no reason. `POST /train/*` returns `202
Accepted` with a `job_id` immediately; `GET /train/status/{job_id}` polls for the
result. The job store is deliberately simple (in-memory, process-local) — stated
plainly as a scope-appropriate choice, not a claim of production-grade durability. A
real high-availability deployment would use a proper task queue (Celery + Redis or
similar) so jobs survive an API restart; this project isn't at the scale where that
tradeoff is worth the added complexity yet.

**A promoted model doesn't automatically update what's being served.** Training a
new champion and having the API actually use it are two different actions on
purpose — `_run_tuning_job`/`_run_fixed_params_job` call `pipeline_cache.reload()`
only when `result.promoted` is `True`, so a training run that doesn't beat the
champion changes nothing about what's currently being served.

---

# Real, End-to-End Verification

Every endpoint was tested with FastAPI's `TestClient` against the real MLflow
registry state — not just import-checked:

1. **Health checks**: `/health` returns `200` unconditionally; `/health/ready`
   correctly reports `model_loaded: False` before any request, `True` after.
2. **Real prediction**: sent a real engine's full cycle history (from the actual
   `test_FD004` file) through `POST /predict/` — got back `predicted_RUL:
   140.96971230949706`, an **exact match** to the same engine's value computed
   independently by `InferencePipeline` (Phase 1) and `MLflowInferencePipeline`
   (Sprint 21). Three more layers on top of the same verified core, same answer.
3. **Reload**: `POST /predict/reload` correctly reloads and reports the champion
   version; `/health/ready` correctly flips to `True` afterward.
4. **Real training job, full lifecycle**: `POST /train/best-params` → `202` with a
   `job_id` → polled `GET /train/status/{job_id}` → watched it transition from
   `"running"` to `"completed"` with the real structured result (fetched real
   champion params from MLflow, correctly cast types, retrained, correctly declined
   to promote a tied result) — all in the background, request returned immediately.
5. **Input validation**: a reading missing required sensor fields correctly returns
   `422` with a clear "Field required" message (Pydantic, not a custom check —
   validation happens before any pipeline code even runs). An empty `readings` list
   is also rejected at `422`. A nonexistent `job_id` returns a clear `404`, not a
   crash.

---

# Running It

```bash
pip install fastapi uvicorn
uvicorn api.main:app --reload
```

Then visit `http://localhost:8000/docs` for interactive Swagger docs, or:

```bash
curl -X POST http://localhost:8000/predict/ \
  -H "Content-Type: application/json" \
  -d '{"readings": [...]}'
```

---

# What's Next (Not Built This Sprint)

- **Containerization** (Dockerfile) — package the API + model artifacts +
  dependencies together.
- **A real task queue** if training job volume or durability-across-restarts ever
  matters — the in-memory job store is a stated, deliberate limitation, not an
  oversight.
- **Authentication** — this API currently has none; fine for local/internal use,
  not for anything exposed publicly.
- **Formal test suite** (`tests/test_api.py`) — the five verification scenarios
  above are the natural first test cases, currently one-off checks rather than a
  permanent, re-runnable suite.
