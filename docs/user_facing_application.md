# ZENOVA User-Facing Application Architecture & Design Guide

## 1. Overview & Architectural Principles

The **ZENOVA User-Facing Application** provides an intuitive, accessible, privacy-preserving interface connecting users and patients to the ZENOVA supportive conversational AI pipeline.

### Core Architectural Mandates:
1. **Zero Client-Side ML Logic**:
   - The frontend is strictly a presentation and interaction layer.
   - All emotion analysis, symptom signal tracking, risk detection, strategy planning, safety gate verification, and escalation decisions are executed on the backend via high-level REST APIs (`/api/v1/orchestrator/process`, `/api/v1/user/...`).
2. **Clear Non-Medical Boundary**:
   - The application explicitly informs users that ZENOVA is an AI emotional support companion and **NOT** a doctor, therapist, or emergency medical service.
   - The UI never delivers medical diagnoses, clinical labels, or drug prescriptions.
   - Persistent disclaimers and an onboarding agreement modal safeguard informed consent.
3. **Seamless Safety & Crisis Integration**:
   - High-risk turns do not rely on fragile client-side regexes or heuristic guessing.
   - When the backend orchestrator detects crisis risk or triggers an escalation (`escalated_to_human: true` or `block_and_escalate`), the UI renders a prominent, unmissable crisis intervention card with 1-click 988 call and text links.
4. **User-Controlled Data Privacy & Right to be Forgotten**:
   - Full data ownership: Users can toggle conversation history saving, voice audio extraction, and wearable telemetry at any time.
   - One-click GDPR/HIPAA-compliant JSON data export.
   - Permanent one-click data purging honoring the Right to be Forgotten.

---

## 2. Component Architecture

```mermaid
flowchart TD
    subgraph Browser / Client Viewport
        UI[Single Page Web Application :8000/app]
        Chat[Conversational Chat Panel]
        Mic[Web Audio / MediaRecorder]
        Checkin[Wellbeing Check-in Widget]
        Resources[Support Resources Hub]
        Privacy[Privacy & Settings Controls]
        Disclaimer[Non-Medical Banner & Onboarding Modal]
    end

    subgraph ZENOVA API Perimeter
        API[FastAPI Gateway :8000]
        UserRoutes[/api/v1/user/*]
        OrchRoute[/api/v1/orchestrator/process]
        ConvRoute[/api/v1/conversation/*]
    end

    subgraph Backend Orchestration Engine
        Orch[ZenovaOrchestrator Pipeline]
        SafetyGate[Independent Safety Gate]
        Escalation[Clinician Escalation Engine]
        DB[(SQLite / PostgreSQL)]
    end

    UI --> Chat
    UI --> Checkin
    UI --> Resources
    UI --> Privacy
    UI --> Disclaimer

    Chat -->|POST JSON text/audio| OrchRoute
    Mic -->|Base64 Audio WebM/WAV| OrchRoute
    Checkin -->|POST checkin| UserRoutes
    Privacy -->|PUT preferences| UserRoutes
    Privacy -->|POST export / DELETE purge| UserRoutes
    Chat -->|GET session history| ConvRoute

    OrchRoute --> Orch
    Orch --> SafetyGate
    SafetyGate --> Escalation
    UserRoutes --> DB
    ConvRoute --> DB
```

---

## 3. Key Feature Modules

### 3.1. Conversational Interface (`GET /app`)
* **Real-Time Message Timeline**:
  * Clean conversational bubbles with speaker tags and timestamps.
  * Screen-reader friendly semantic structure (`role="log"`, `aria-live="polite"`).
  * Auto-resizing textarea with keyboard support (`Enter` to submit, `Shift+Enter` for newline).
* **Optional Voice Input**:
  * Utilizes browser-native `MediaRecorder` and Web Audio API.
  * Real-time visual recording pulse indicator.
  * Audio payload encoded as base64 and dispatched to `/api/v1/orchestrator/process`.
  * Graceful degradation: If microphone permission is denied, a non-intrusive alert directs the user to continue typing seamlessly.

### 3.2. Wellbeing Check-ins
* **Daily / On-Demand Check-in**:
  * Mood Valence Slider: Mapped from 1 ("Very Low") to 10 ("Flourishing") and continuous affective valence ($-1.0 \le v \le +1.0$).
  * Sleep Duration: Hours slept in the past 24 hours ($0 \le h \le 14$).
  * Subjective Stress Level: 5-point Likert scale (1: Very Calm to 5: Overwhelming).
  * Subjective Energy Level: 5-point Likert scale (1: Exhausted to 5: Vital).
  * Optional Reflective Journal Note.
* **Longitudinal Trend Analytics**:
  * `GET /api/v1/user/checkins/{user_id}` returns historical averages for mood, sleep, and stress.
  * Non-diagnostic supportive reflection messages reinforce emotional self-awareness without clinical labeling.

### 3.3. Support Resources & Grounding Hub
* **Verified Emergency Hotlines**:
  * 988 Suicide & Crisis Lifeline (Call/Text 988, 24/7/365, Free & Confidential).
  * Crisis Text Line (Text HOME to 741741).
  * The Trevor Project (1-866-488-7386 or Text START to 678-678).
  * Veterans Crisis Line (Dial 988, Press 1).
  * National Maternal Mental Health Hotline (1-833-852-6262).
  * Befrienders Worldwide / International Association for Suicide Prevention (IASP).
* **Interactive Self-Care Techniques**:
  * 5-4-3-2-1 Sensory Grounding: Step-by-step interactive checklist to anchor during acute anxiety.
  * Box Breathing (4-4-4-4): Guided visual cadence for autonomic regulation.

### 3.4. Privacy & Data Governance
* **User Preferences (`GET / PUT /api/v1/user/preferences/{user_id}`)**:
  * Toggle: `save_history` (Enable/disable session saving).
  * Toggle: `enable_voice` (Enable/disable acoustic speech feature analysis).
  * Toggle: `enable_wearables` (Enable/disable passive wearable integration).
  * Privacy Mode: `standard`, `anonymized`, or `ephemeral`.
  * Communication Style: `warm_empathic`, `solution_focused`, `reflective`, `gentle_minimal`.
* **Data Portability (`POST /api/v1/user/export/{user_id}`)**:
  * Aggregates all user sessions, turns, check-in history, and configuration preferences into an instant JSON download.
* **Right to be Forgotten (`DELETE /api/v1/user/data/{user_id}`)**:
  * Atomically purges all user sessions, turns, check-ins, baselines, and observations from the database.

---

## 4. Accessibility & Responsive Standards

* **WCAG 2.1 AA Compliance**:
  * Contrast ratios exceed $4.5:1$ across all text elements in both dark and light modes.
  * Semantic HTML5 elements (`<header>`, `<main>`, `<aside>`, `<nav>`, `<section>`, `<footer>`).
  * Keyboard navigable with visible focus states and skip links.
  * Explicit ARIA attributes: `role="tablist"`, `role="tab"`, `aria-selected`, `aria-label`.
* **Responsive Viewports**:
  * Mobile phone viewports ($< 640\text{px}$): Single-column view with collapsible session drawer.
  * Tablets ($640\text{px} - 1024\text{px}$): Adaptive split panes.
  * Desktop ($> 1024\text{px}$): Dual-pane layout with persistent session navigation and chat timeline.

---

## 5. Future Roadmap & Extensibility Hooks

1. **Mobile Application (PWA / Native Wrapper)**:
   * Architecture ready for Progressive Web App manifest (`manifest.json`) and service worker offline caching.
   * Responsive layout matches iOS and Android WebView containers (React Native / Flutter wrapper).
2. **Wearable & Sensor Telemetry Integration**:
   * Extensible schema support for Apple HealthKit, Google Health Connect, Fitbit, and Garmin APIs.
   * Heart rate variability (HRV), sleep staging, and circadian rhythm alignment feeding Step 6 behavioral analysis.
3. **Multilingual Support (i18n)**:
   * Language selector with dictionary hook for Spanish, French, German, and Mandarin.
4. **Clinician Care Team Communication**:
   * Consent-based export or direct clinician dashboard linking (`allow_clinician_sharing`), enabling licensed therapists to view longitudinal baseline summaries.
