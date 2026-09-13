# ZENOVA Step 17: End-to-End Orchestration & Execution Tracing

## 1. Executive Summary & Objective

The **ZENOVA End-to-End Orchestration Subsystem** is the central nervous system of the platform. It unites all specialized sub-components—preprocessing, voice transcription, emotion classification, observational symptom signal extraction, crisis risk appraisal, personal baseline tracking, multimodal fusion, structured context aggregation, strategy planning, curated knowledge retrieval (RAG), strategy-controlled LLM response generation, independent safety gating, and longitudinal persistence—into an end-to-end clinical workflow.

Rather than an unwieldy, monolithic procedure, the orchestrator is implemented as a composable pipeline of decoupled stages. Each stage is independently testable, instrumented with fine-grained execution spans, and adheres to strict clinical fail-safe invariants.

```
USER INPUT (Text and/or Acoustic Waveform)
    ↓
1. Input Processing Stage (Validation, Sanitization, Voice Transcription Fallback)
    ↓
2. Analytical Layer Stage (Emotion, Symptoms, Risk, Voice, Behavior, Baseline, Multimodal Fusion)
    ↓
3. Context Aggregation Stage (Normalized Multimodal Context Snapshot, Cryptographic Context Hashing)
    ↓
4. Triage & Risk Decision Stage (Acute Risk Check & Preliminary Clinician Escalation)
    ├────────────────────────────┬────────────────────────────┐
    ▼                            ▼                            │
[HIGH / CRITICAL RISK]        [LOW / MODERATE RISK]           │
Crisis Bypass Branch          Normal Support Pathway          │
    │                            │                            │
    │                            ├─ Strategy Planner (ESConv) │
    │                            │     ↓                      │
    │                            ├─ Curated RAG Grounding     │
    │                            │     ↓                      │
    │                            ├─ Strategy-Controlled LLM   │
    │                            │     ↓                      │
    │                            └─ Independent Safety Gate   │
    │                                  │                      │
    └────────────────────────────┬─────┘                      │
                                 ▼                            │
5. Intervention Stage Output Selection                        │
    ↓                                                         │
6. Longitudinal State & Execution Trace Persistence Stage ◄───┘
    ↓
Therapeutic Response Delivered to User
```

---

## 2. Pipeline Stages Architecture

The pipeline decomposes the turn lifecycle into 6 modular stages inheriting from `BasePipelineStage` (`src/zenova/orchestration/stages/base.py`):

| Stage | Class | File | Core Responsibilities |
|---|---|---|---|
| **1. Input Processing** | `InputProcessingStage` | `input_processing.py` | Validates session and user IDs; analyzes acoustic voice features; executes audio transcription fallback if text is empty/placeholder. |
| **2. Analytical Layer** | `AnalyticalLayerStage` | `analytical_layer.py` | Dispatches analytical modules (emotion, symptoms, risk, voice, passive behavior, personal baseline, and multimodal fusion); captures individual telemetry spans. |
| **3. Context Engine** | `ContextAggregationStage` | `context_engine.py` | Assembles multimodal context blocks, embeds cross-modal discrepancy data, computes SHA-256 context hash, and snapshots state. |
| **4. Triage Decision** | `TriageDecisionStage` | `triage_decision.py` | Evaluates acute risk triggers (`HIGH`/`CRITICAL`) and clinician escalation thresholds; determines whether crisis bypass is warranted. |
| **5. Intervention** | `InterventionStage` | `intervention.py` | **Crisis branch**: Immediately activates human escalation alert and serves verified crisis de-escalation response. **Normal branch**: Predicts Hill's support strategy, performs curated RAG retrieval, invokes strategy-controlled LLM (with clinical template fallback), verifies safety via independent gate. |
| **6. Persistence** | `PersistenceStage` | `persistence.py` | Persists conversation turns, context snapshots, safety audit trails, and execution traces; replicates trace to in-memory circular buffer. |

---

## 3. Safe Failure Handling (Zero Silent Fabrication)

In mental health support, algorithmic components must **fail safely and transparently** rather than silently hallucinating clinical states or factual answers:

```
Component Failure           Safe Degraded Behavior                          Safety Impact
──────────────────────────────────────────────────────────────────────────────────────────
Emotion Model Down          Neutral baseline, confidence=0.0                Downstream prompts do not assume false emotional states.
Symptom Model Down          Empty signals, subclinical severity             Prevents fabricated psychiatric marker claims.
Risk Model Down             Conservative heuristic keyword scanner          Safety-critical: flags acute cues immediately; never assumes low risk silently.
LLM Generation Down         Deterministic, verified clinical templates      Patient receives helpful, clinically approved response matching predicted strategy.
RAG Retrieval Down          Graceful continuation without citations         Never invents fabricated sources or citations.
Database Down               In-memory circular buffer (200 traces)          Therapeutic response delivered without blocking or crashing.
```

### 3.1. Conservative Triage Posture for Risk Model Failure
When the statistical or neural risk classifier throws an exception or is unavailable, `get_safe_fallback_risk_result()` executes a deterministic regex scan over `CRITICAL_CRISIS_PATTERNS` and `HIGH_RISK_PATTERNS`. If explicit acute keywords (e.g. `suicide`, `end my life`, `want to die`, `overdose`) are detected, the system immediately flags `RiskLevel.CRITICAL` / `CrisisCategory.SUICIDAL_IDEATION`, bypassing generation and alerting clinicians.

---

## 4. Execution Tracing Subsystem

Every dialogue turn produces an immutable `PipelineTrace` composed of individual `TraceSpan` telemetry blocks.

### 4.1. In-Memory Ring Buffer (`InMemoryTraceBuffer`)
A thread-safe circular ring buffer (retaining the most recent 200 traces) guarantees that clinical supervisors and developers can inspect end-to-end execution traces even if database connections lock or fail.

### 4.2. Database Persistence (`TraceRepository`)
Stores execution traces in the `execution_traces` database table with indexed session and user keys, latency metrics, degraded module lists, and serialized spans.

---

## 5. High-Level REST API Reference

The orchestrator exposes high-level unified endpoints while maintaining individual module accessibility:

### 5.1. Process Turn
- **Endpoint**: `POST /api/v1/orchestrator/process`
- **Request Body**:
  ```json
  {
    "session_id": "sess_abc123",
    "user_id": "user_xyz789",
    "text": "I feel so overwhelmed by my workload lately.",
    "audio_base64": null,
    "metadata": {}
  }
  ```
- **Response**: `OrchestrationResult` schema including full response, strategy, emotion, symptoms, risk, safety verification, latency, and embedded execution `trace`.

### 5.2. Inspect Execution Trace
- **Endpoint**: `GET /api/v1/orchestrator/traces/{trace_id}`
- **Headers**: `X-User-Role: clinician`, `X-User-ID: dr_smith`
- **Response**: `PipelineTrace` object detailing all spans, timings, and component degradation states.

### 5.3. List Execution Traces
- **Endpoint**: `GET /api/v1/orchestrator/traces?session_id=...&status=nominal&limit=50`
- **Headers**: `X-User-Role: clinician`

### 5.4. Component Health
- **Endpoint**: `GET /api/v1/orchestrator/health`
- **Response**: Live health and degradation status of all 12 platform modules.
