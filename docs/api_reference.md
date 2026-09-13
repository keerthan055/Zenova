# ZENOVA API Reference

Version: 0.1.0 (Step 1 Foundation)  
Base URL: `http://localhost:8000`

---

## Health & Status Endpoints

### 1. `GET /health`
Liveness and readiness check.

**Response (200 OK):**
```json
{
  "status": "healthy",
  "service": "ZENOVA",
  "version": "0.1.0",
  "environment": "development",
  "safety_strict_mode": true,
  "models_loaded": true
}
```

### 2. `GET /status`
Architectural status endpoint displaying active module providers and placeholder flags.

**Response (200 OK):**
```json
{
  "status": "operational",
  "active_providers": {
    "emotion": "placeholder",
    "symptom": "placeholder",
    "risk": "placeholder",
    "baseline": "placeholder",
    "strategy": "placeholder",
    "generator": "placeholder",
    "safety": "placeholder"
  },
  "is_placeholder_mode": true,
  "message": "ZENOVA Step 1 Foundation active. ML models are configured via placeholders."
}
```

---

## Conversational Support Endpoints

### 3. `POST /api/v1/conversation/turn`
Processes a single user conversational turn through the modular pipeline (Emotion -> Symptoms -> Risk -> Baseline -> Crisis Check -> Strategy -> Generation -> Safety Gate -> Storage).

**Request Body (`UserInput`):**
```json
{
  "session_id": "session_abc123",
  "user_id": "user_xyz789",
  "text": "I have been feeling overwhelmed by my coursework lately.",
  "modality": "text"
}
```

**Response (200 OK - Normal Turn):**
```json
{
  "session_id": "session_abc123",
  "turn_id": 1,
  "user_input": "I have been feeling overwhelmed by my coursework lately.",
  "response": "Thank you for sharing that with me. Could you tell me more about what has been on your mind recently?",
  "escalated_to_human": false,
  "escalation_event_id": null,
  "emotion": {
    "is_placeholder": true,
    "module_version": "placeholder-v0.1.0",
    "primary_emotion": "neutral",
    "confidence": 0.5,
    "valence": 0.0,
    "arousal": 0.0,
    "dominance": 0.0
  },
  "symptoms": {
    "is_placeholder": true,
    "module_version": "placeholder-v0.1.0",
    "signals": [],
    "aggregate_severity": "none"
  },
  "risk": {
    "is_placeholder": true,
    "module_version": "placeholder-v0.1.0",
    "risk_level": "low",
    "requires_immediate_escalation": false
  },
  "strategy": {
    "is_placeholder": true,
    "module_version": "placeholder-v0.1.0",
    "selected_strategy": "Question",
    "stage": "Exploration"
  },
  "safety": {
    "is_placeholder": true,
    "module_version": "placeholder-v0.1.0",
    "is_safe": true,
    "action": "allow"
  },
  "latency_ms": 15.2,
  "timestamp": "2026-09-10T01:10:00Z"
}
```

**Response (200 OK - Crisis Bypass Triggered):**
```json
{
  "session_id": "session_crisis_99",
  "turn_id": 1,
  "user_input": "I want to hurt myself",
  "response": "I hear how much pain you are experiencing right now, and your safety is the most important thing. I am an AI wellbeing companion and cannot provide emergency care. Please connect with trained professionals who can help you immediately:\n• Call or text 988 (USA & Canada)\n• Text HOME to 741741\n• International: https://findahelpline.com/",
  "escalated_to_human": true,
  "escalation_event_id": "esc_4f9a12c8b0e1",
  "risk": {
    "is_placeholder": true,
    "risk_level": "critical",
    "crisis_category": "suicidal_ideation",
    "requires_immediate_escalation": true
  }
}
```

### 4. `GET /api/v1/conversation/{session_id}/history`
Fetches all stored conversation turns for a given session.

---

## Clinician Escalation Endpoints

### 5. `GET /api/v1/escalations`
Lists logged crisis escalation incidents requiring clinician review. Query parameter `status` (`pending`, `acknowledged`, `resolved`) supported.

### 6. `POST /api/v1/escalations/{event_id}/acknowledge`
A clinician marks an escalation event as acknowledged.

**Request Body:**
```json
{
  "clinician_id": "dr_patel_lic_4021",
  "notes": "Contacted user's designated support contact per emergency protocol."
}
```

---

## Model Registry & Configuration Endpoints

### 7. `GET /api/v1/models`
Inspects all registered model architectures, checkpoints, parameters, and active providers.

### 8. `POST /api/v1/models/switch`
Swaps the active provider for any ML task dynamically without modifying code or restarting the application.

**Request Body:**
```json
{
  "task": "strategy",
  "provider_name": "esconv_classifier"
}
```
