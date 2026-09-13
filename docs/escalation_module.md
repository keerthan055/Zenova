# ZENOVA Step 14: Human/Clinician Escalation Workflow

## 1. Overview & Clinical Scope

In any AI-mediated emotional and mental health support architecture, autonomous generation must yield immediately to human clinician oversight whenever a situation exceeds algorithmic competence or safety boundaries. 

The **ZENOVA Human/Clinician Escalation Engine** provides a deterministic, transparent, and auditable pipeline for triage, alert dispatch, clinician claim, active review, intervention recording, and case resolution.

---

## 2. Trigger Taxonomy & Decision Matrix

Escalation alerts are not triggered arbitrarily. Every alert must originate from one of five documented trigger classes:

| Trigger Source | Evaluation Criteria | Severity | Reason Code |
| :--- | :--- | :---: | :--- |
| **High/Critical Risk** | `RiskResult.risk_level in [HIGH, CRITICAL]` or `requires_immediate_escalation == True` | `CRITICAL` / `HIGH` | `CRITICAL_RISK_DETECTED`, `HIGH_RISK_DETECTED` |
| **Repeated Concerning Signals** | $\ge 3$ consecutive turns of acute negative affect (sadness, grief, fear, shame with valence $\le -0.50$ and confidence $\ge 0.70$) or $\ge 2$ consecutive turns of moderate/severe symptoms | `HIGH` | `REPEATED_CONCERNING_AFFECT`, `PERSISTENT_ACUTE_SYMPTOMS` |
| **Longitudinal Baseline Changes** | Deviation $z \ge 3.0\sigma$ on anxiety, depression, or distress score, or behavioral passive sensing anomaly across $\ge 3$ lifestyle domains | `HIGH` / `MEDIUM` | `LONGITUDINAL_BASELINE_SPIKE`, `BEHAVIORAL_ANOMALY_SPIKE` |
| **Safety Gate Events** | `SafetyResult.action == BLOCK_AND_ESCALATE` or critical policy violation (`harmful_instructions`, `crisis_mishandling`, `delusions`) | `CRITICAL` | `SAFETY_GATE_BLOCK_AND_ESCALATE`, `SAFETY_CRITICAL_VIOLATION` |
| **Configured Clinician Rules** | Explicit requests for human clinicians, suspected acute substance overdose, immediate domestic violence, or unaccompanied minor in crisis | `CRITICAL` / `HIGH` | `EXPLICIT_HUMAN_REQUEST`, `SUBSTANCE_OVERDOSE_SUSPICION`, `DOMESTIC_VIOLENCE_INTIMIDATION`, `PEDIATRIC_CRISIS` |

---

## 3. Alert Lifecycle & State Transitions

Alerts follow a strict state machine:

```
[ Inbound Event ] ──> PENDING ──> ACKNOWLEDGED ──> IN_REVIEW ──> RESOLVED
                         │                                           │
                         └─────────────> DISMISSED <─────────────────┘
```

1. **`PENDING`**: Initial state immediately upon trigger evaluation. Alert appears in the clinician triage queue sorted by severity (`CRITICAL` first) and timestamp.
2. **`ACKNOWLEDGED`**: A licensed clinician or triage supervisor claims ownership of the alert ticket.
3. **`IN_REVIEW`**: Clinician is actively inspecting the context snapshot, turn trajectory, symptom trends, and risk cues.
4. **`RESOLVED`**: Clinician completes human intervention and logs an action from the clinical action taxonomy along with mandatory resolution rationale.
5. **`DISMISSED`**: Clinician or supervisor concludes the situation was a benign false positive or no clinical action was warranted.

### Clinical Action Taxonomy (`EscalationActionType`)

- `emergency_services_contacted`: 911 / EMS dispatched for imminent physical peril.
- `hotline_warm_transfer`: Warm handoff to 988 Suicide & Crisis Lifeline or Crisis Text Line (741741).
- `user_outreach_call`: Clinician telephone or video check-in.
- `safety_plan_activated`: Stanley-Brown evidence-based suicide safety plan enacted.
- `appointment_scheduled`: Urgent outpatient psychiatric or psychological visit scheduled.
- `referred_to_external_care`: Outpatient community mental health referral provided.
- `false_positive_flagged`: Validated non-crisis or benign dialogue context.
- `no_action_required`: Low-risk situation handled satisfactorily without further intervention.

---

## 4. Role-Based Access Control (RBAC)

Patient confidentiality and HIPAA/GDPR data minimization principles require strict boundaries:

- **`clinician`**: Access to active queue, unmasked clinical context, symptom cues, acknowledge/resolve workflows.
- **`triage_supervisor`**: Full clinician privileges plus ticket reassignment, rule updates, and audit oversight.
- **`system_admin`**: Technical administration, system configuration, rule updates. Identifiable patient text and symptom cues are masked.
- **`auditor`**: Read-only compliance access to anonymized alert logs and transition histories.
- **`patient`**: Zero access to clinical queues, internal triage notes, or clinician attribution.

---

## 5. Audit Trail & Healthcare Compliance

Every alert transition generates an immutable record in `escalation_audit_logs`:
- `audit_id`: Cryptographic identifier
- `alert_id`: Foreign key to `escalation_events.event_id`
- `action`: `created`, `acknowledged`, `status_changed`, `resolved`, `dismissed`
- `actor_id`: ID of the clinician or system process
- `actor_role`: Clinician role at time of action
- `previous_status` & `new_status`
- `details`: Metadata snapshot and resolution notes
- `timestamp`: UTC timestamp
