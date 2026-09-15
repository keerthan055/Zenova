# ZENOVA: Clinical-Grade Conversational Wellbeing & Emotional Support AI

[![Tests](https://img.shields.io/badge/tests-391%20passed-brightgreen.svg)](tests/)
[![Coverage](https://img.shields.io/badge/coverage-86%25-green.svg)](tests/)
[![Status](https://img.shields.io/badge/status-production--ready-blue.svg)](ZENOVA_RELEASE.md)
[![License](https://img.shields.io/badge/license-Apache--2.0-lightgrey.svg)](LICENSE)

**ZENOVA** is a modular, multimodal, safety-gated platform engineered to provide evidence-grounded emotional support, psychiatric symptom signal tracking, dynamic baseline anomaly detection, and automated crisis escalation.

> [!IMPORTANT]
> **REGULATORY & CLINICAL BOUNDARY NOTICE**:  
> ZENOVA is an investigational AI research prototype and supportive companion. It is **NOT** a certified medical device (SaMD) and does **NOT** provide clinical diagnoses, psychiatric disorder classifications, or pharmaceutical treatments. Its clinical efficacy has not been established in randomized clinical trials. All outputs represent observational analytical signals.

---

## 1. System Architecture & Dual-Branch Execution

ZENOVA strictly separates analytical clinical risk evaluation from generative conversational synthesis, guaranteeing that crisis detection overrides the LLM with deterministic emergency workflows:

```
USER INPUT (Text, Audio, Passive Wearable Telemetry)
  ↓
INPUT PREPROCESSING (Sanitization, 12D MFCC Prosody Extractor)
  ↓
ANALYTICAL LAYER:
  • Emotion Classifier (GoEmotions Ekman)
  • Symptom Detector (PsySym DSM-5 Signal Detector)
  • Crisis Assessor (C-SSRS Risk Classifier)
  • Behavioral Sensing (Passive Sensor Deviations)
  ↓
MULTIMODAL GATED FUSION (GMU with Dynamic Modality Availability Masking)
  ↓
PERSONAL BASELINE (Longitudinal Z-Score Anomaly Tracking)
  ↓
CONTEXT ENGINE (10-Turn Sliding Window Multi-Turn Memory)
  ↓
RISK TRIAGE DECISION
┌────────────────────────────────────────┐
│ RISK >= HIGH / CRITICAL?               │
└───────────────────┬────────────────────┘
                    │
           YES      │      NO
            ↓       │       ↓
     SAFETY / HUMAN │ STRATEGY PLANNER (ESConv)
     ESCALATION     │       ↓
     (Alert ID,     │ RAG (Evidence-Based Grounding)
     988 Hotline)   │       ↓
            │       │ LLM GENERATOR (Compassionate Synthesis)
            │       │       ↓
            │       │ SAFETY GATE (12 Independent Boundary Rules)
            │       │       ↓
            └───────┴───────→ USER
                    │
                    ↓
     ENCRYPTED LONGITUDINAL PERSISTENCE
     (PostgreSQL / SQLite + Audit Trail)
                    │
                    ↓
     CLINICIAN WEB DASHBOARD
     (Explainability Cards & Risk Trajectory Telemetry)
```

---

## 2. Quickstart & Clean-Environment Reproduction

Follow these exact steps to reproduce the documented results from a clean clone:

### Step 1: Clone Repository & Virtual Environment
```powershell
git clone https://github.com/keerthan055/Zenova.git
cd Zenova

# Create and activate Python 3.11+ virtual environment
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### Step 2: Install Package Dependencies
```powershell
# Upgrade pip to latest
python -m pip install --upgrade pip

# Install package in editable development mode (includes SQLAlchemy, PyTorch, FastAPI, etc.)
pip install -e ".[dev]"
```

### Step 3: Initialize Database
```powershell
$env:PYTHONPATH="src"
python -c "import asyncio; from zenova.db.session import init_db; asyncio.run(init_db())"
```

### Step 4: Run the Complete Automated Test Suite (391 Tests)
```powershell
$env:PYTHONPATH="src"
python -m pytest tests/ -v --cov=src/zenova --cov-report=term
```
*Expected output: `391 passed, 0 failed, 86% coverage`.*

### Step 5: Provision Initial Administrator (Optional CLI)
```powershell
$env:PYTHONPATH="src"
python scripts/create_user.py --email admin@zenova.ai --password "AdminSecure2026!" --role admin --display-name "System Administrator"
```

### Step 6: Start the Application Server
```powershell
$env:PYTHONPATH="src"
python -m uvicorn zenova.api.main:app --host 0.0.0.0 --port 8000 --reload
```

---

## 3. Platform Portals & Endpoints

When the server is running on `http://localhost:8000`:

| Portal / Endpoint | URL | Description |
| :--- | :--- | :--- |
| **Authentication Portals** | `http://localhost:8000/login` <br> `http://localhost:8000/register` | Responsive authentication web pages with Argon2id hashing, password strength evaluation, and cookie sessions. |
| **User-Facing Application** | `http://localhost:8000/app` | Accessible, responsive Single Page App with chat, audio input, daily check-ins, and GDPR privacy controls. |
| **Clinician Dashboard** | `http://localhost:8000/dashboard` | Active escalation alerts, patient risk trajectories, and model explainability cards (requires `clinician` or `admin` role). |
| **OpenAPI / Swagger Docs** | `http://localhost:8000/docs` | Interactive documentation for all authentication, orchestrator, user, dashboard, and escalation endpoints. |
| **Auth REST API** | `POST /api/v1/auth/login` <br> `POST /api/v1/auth/register` | JWT token issuing, refresh rotation, password reset, and verification APIs. |
| **Orchestration API** | `POST /api/v1/orchestrator/process` | Central high-level entry point executing multimodal input processing through to storage. |
| **Liveness Health Probe** | `GET /health/live` | Kubernetes-compatible liveness health probe. |
| **Readiness Health Probe** | `GET /health/ready` | Verifies database, model weights, and cache readiness. |
| **Prometheus Metrics** | `GET /metrics` | Real-time telemetry: latencies, throughput, and safety interception counters. |

---

## 4. Docker Production Deployment

To run ZENOVA with containerized PostgreSQL, Redis, and Prometheus services:

```powershell
# Build and start multi-container cluster
docker-compose up -d --build

# Inspect container health
docker-compose ps

# View application logs
docker-compose logs -f api
```

---

## 5. Key Verified Metrics

| Dimension | Metric | Measured Value | Standard / SLA |
| :--- | :--- | :--- | :--- |
| **Crisis Sensitivity** | Recall on Acute Ideation | **`1.0000`** | Target: $1.0000$ (Zero false negatives) |
| **Crisis FNR** | False Negative Rate | **`0.0000`** | Target: $0.0000$ |
| **Symptom Detection** | Micro-F1 (10 DSM-5 signals) | **`0.9573`** | Target: $> 0.90$ |
| **Safety Interception** | Red-Team Adversarial Pass | **`100.0%`** (15/15 scenarios) | Target: $100\%$ |
| **Hallucination Rate** | Complete Pipeline | **`0.0%`** | Target: $< 2.0\%$ |
| **Inference Latency** | $p50$ Median Latency | **`38.93 ms`** | Target: $< 50\text{ ms}$ |
| **Crisis Bypass Latency**| $p50$ Emergency Response | **`39.62 ms`** | Target: $< 100\text{ ms}$ |
| **API Reliability** | Handled Request Rate | **`100.0%`** | Target: $> 99.9\%$ |
| **Automated Tests** | Pass Rate | **`391 / 391 (100%)`** | Target: $100\%$ |

---

## 6. Complete Documentation Index

- [`docs/authentication_and_security.md`](docs/authentication_and_security.md): Authentication architecture, Argon2id specifications, JWT/session lifecycle, IDOR defense, and RBAC matrix.
- [`ZENOVA_RELEASE.md`](ZENOVA_RELEASE.md): Release manifest, module inventory, model hashes, limitations, and future work.
- [`docs/research_and_technical_documentation.md`](docs/research_and_technical_documentation.md): Comprehensive publication-grade 27-section research whitepaper.
- [`docs/complete_system_integration.md`](docs/complete_system_integration.md): Step 22 system integration, dual-branch routing, and fault-tolerance resilience matrix.
- [`docs/complete_evaluation_report.md`](docs/complete_evaluation_report.md): In-depth model metrics, ablation studies (Configs A–E), and human-evaluation Likert protocols.
- [`docs/user_facing_application.md`](docs/user_facing_application.md): Specification of the `/app` interface, accessibility guidelines, and privacy controls.
- [`docs/clinician_dashboard.md`](docs/clinician_dashboard.md): Clinician dashboard, explainability cards, and role-based access controls.