# ZENOVA Authentication, Session Management, and Security Architecture

## 1. Executive Summary & Objective

The ZENOVA mental health conversational AI system manages sensitive, user-specific data including conversational turns, emotional distress trajectories, psychiatric symptom indicators, risk scores, longitudinal personal baselines, and safety escalation records. 

To ensure strict privacy, confidentiality, and regulatory compliance principles, the **Authentication and User Account Management Subsystem** provides a zero-trust, identity-aware security foundation. It decouples user credentials from clinical signals, prevents horizontal and vertical privilege escalation, resists brute-force and credential-stuffing attacks, and provides full cryptographic auditability.

---

## 2. Threat Model & Security Principles

```
  [ Attacker / Adversary ]
             │
             ▼
 ┌───────────────────────┐   1. Rate Limiting & Slow Brute Force Defense
 │  Cloudflare / Reverse │   2. Constant-Time User Lookup & Generic Errors
 │         Proxy         │   3. CSP / Strict-Transport-Security / Anti-Sniffing
 └───────────┬───────────┘
             │
             ▼
 ┌───────────────────────┐   4. Dual-Mode Token Transport (HttpOnly Cookies + Bearer)
 │   FastAPI Gateway     │   5. IDOR Defense: Token Identity Bound via `get_current_user`
 └───────────┬───────────┘
             │
             ▼
 ┌───────────────────────┐   6. Argon2id Password Hashing ($m=65536, t=3, p=4$)
 │    Auth Service &     │   7. Ephemeral Access Tokens (JWT, 30 min)
 │   Security Service    │   8. Rotating Refresh Tokens (SHA-256 in DB, 30 days)
 └───────────┬───────────┘   9. Single-Use Tokens (Verification & Reset)
             │
             ▼
 ┌───────────────────────┐  10. Decoupled Tables: `users` (credentials) vs
 │  SQLAlchemy ORM & DB  │      `user_auth_sessions`, `user_baselines`, `turns`
 └───────────────────────┘
```

### Threat Vectors Mitigated

| Threat Vector | Mitigation Mechanism | Verification Suite |
| :--- | :--- | :--- |
| **Credential Stuffing / Brute Force** | Argon2id memory-hard hashing ($m=64\text{MB}, t=3, p=4$); sliding window rate-limiting middleware (60 req/min). | `tests/unit/test_auth_security.py` |
| **Insecure Direct Object Reference (IDOR)** | Route identity derived strictly from validated JWT/cookie identity (`enforce_user_ownership`). User A cannot access User B's settings, baseline, check-ins, or data exports. | `tests/security/test_auth_security_controls.py` |
| **Privilege Escalation (Vertical RBAC)** | Strict role enforcement (`require_role("clinician", "admin")`). Superuser bypass restricted to `"admin"`. | `tests/security/test_auth_security_controls.py` |
| **Token Theft / XSS Token Stealing** | Session cookies configured with `HttpOnly=True`, `SameSite=Lax`, and conditional `Secure=True`. Access tokens are short-lived (30 min). | `tests/integration/test_auth_api.py` |
| **Token Replay / Hijacking** | Refresh token rotation: each refresh invalidates the old token and issues a new pair. Tokens stored as irreversible SHA-256 hashes in database. | `tests/unit/test_auth_service.py` |
| **Account Enumeration / Timing Attacks** | Generic error messages (`"Invalid email or password"`) for both non-existent users and incorrect passwords. Argon2 dummy verification performed to preserve timing consistency. | `tests/unit/test_auth_service.py` |
| **Password Reset / Verification Replay** | Password reset and email verification tokens are high-entropy (256-bit cryptographically secure random bytes), stored as SHA-256 hashes, single-use, and expire automatically. | `tests/security/test_auth_security_controls.py` |
| **Data Leakage in API & Logs** | Pydantic response models (`SafeUserResponse`) omit `password_hash`, `salt`, and auth tokens. Audit logging redacts sensitive fields. | `tests/security/test_auth_security_controls.py` |

---

## 3. Cryptographic Architecture & Specifications

### 3.1 Password Hashing (Argon2id)
ZENOVA employs the **Argon2id** algorithm via `argon2-cffi`, the gold standard winner of the Password Hashing Competition (PHC). Argon2id provides resistance against both GPU/ASIC side-channel attacks and cache-timing attacks.

* **Memory Cost ($m$):** 65,536 KiB (64 MiB)
* **Time Cost ($t$):** 3 iterations
* **Parallelism ($p$):** 4 parallel threads
* **Salt:** 16 cryptographically secure random bytes generated per password hash.
* **Timing Uniformity:** When an email address is not found during login, a dummy Argon2id hash is computed against standard parameters to ensure constant response times ($\sim 40\text{--}60\text{ ms}$), defeating user enumeration via response latency timing.

### 3.2 Token Architecture
1. **Access Token (JWT):**
   * **Algorithm:** HMAC-SHA256 (`HS256`)
   * **Claims:** `sub` (User ID), `email`, `role`, `iat` (issued at timestamp), `exp` (expiration timestamp).
   * **Lifetime:** 30 minutes (configurable via `SECURITY__TOKEN_EXPIRE_MINUTES`).
   * **Validation:** Explicitly requires `sub`, `iat`, and `exp` claims. Rejects expired or signature-mismatched tokens.
2. **Refresh Token (Opaque Hash):**
   * **Format:** 64-character hex string generated from 32 cryptographically secure random bytes (`secrets.token_hex(32)`).
   * **Storage:** Only the SHA-256 digest (`hashlib.sha256(raw_token.encode()).hexdigest()`) is stored in the database table `user_auth_sessions`.
   * **Rotation:** Every call to `/api/v1/auth/refresh` marks the submitted token as `revoked=True` and issues a brand-new token pair.
   * **Lifetime:** 30 days (configurable via `SECURITY__REFRESH_TOKEN_EXPIRE_DAYS`).
3. **Password Reset & Email Verification Tokens:**
   * **Entropy:** 256 bits (`secrets.token_urlsafe(32)`).
   * **Storage:** Irreversible SHA-256 hash stored in `password_reset_tokens` and `email_verification_tokens`.
   * **Single Use:** Flagged as `used=True` immediately upon redemption; subsequent attempts are rejected with HTTP 400 Bad Request.

---

## 4. Role-Based Access Control (RBAC) Taxonomy

| Role | Description | Accessible Endpoints & Modules |
| :--- | :--- | :--- |
| `user` | Standard authenticated client / patient. | Conversation turns (`/api/v1/conversation/turn`), own preferences (`/api/v1/user/preferences/{user_id}`), own check-ins, own data export and deletion, self-profile management (`/api/v1/auth/me`). |
| `clinician` | Licensed mental health professional or supervisor. | All `user` endpoints plus Clinician Dashboard (`/api/v1/dashboard/overview`, `/api/v1/dashboard/modules`, `/api/v1/dashboard/patients`, `/api/v1/dashboard/timeline`), escalation triage, and cross-patient clinical summaries. |
| `admin` | System administrator and compliance officer. | Full system access: Clinician Dashboard, system health metrics, backup execution, user role elevation, and audit trails (`/api/v1/auth/audits`). |

---

## 5. Database Schema & Models

```sql
-- Core User Account Table
CREATE TABLE users (
    id VARCHAR(36) PRIMARY KEY,
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    display_name VARCHAR(100),
    role VARCHAR(32) NOT NULL DEFAULT 'user',
    is_active BOOLEAN NOT NULL DEFAULT 1,
    is_verified BOOLEAN NOT NULL DEFAULT 0,
    created_at TIMESTAMP NOT NULL,
    updated_at TIMESTAMP NOT NULL
);

-- Stateful Session & Refresh Token Table
CREATE TABLE user_auth_sessions (
    id VARCHAR(36) PRIMARY KEY,
    user_id VARCHAR(36) NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    refresh_token_hash VARCHAR(64) UNIQUE NOT NULL,
    ip_address VARCHAR(45),
    user_agent TEXT,
    expires_at TIMESTAMP NOT NULL,
    revoked BOOLEAN NOT NULL DEFAULT 0,
    created_at TIMESTAMP NOT NULL
);

-- Single-Use Password Reset Tokens
CREATE TABLE password_reset_tokens (
    id VARCHAR(36) PRIMARY KEY,
    user_id VARCHAR(36) NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    token_hash VARCHAR(64) UNIQUE NOT NULL,
    expires_at TIMESTAMP NOT NULL,
    used BOOLEAN NOT NULL DEFAULT 0,
    created_at TIMESTAMP NOT NULL
);

-- Single-Use Email Verification Tokens
CREATE TABLE email_verification_tokens (
    id VARCHAR(36) PRIMARY KEY,
    user_id VARCHAR(36) NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    token_hash VARCHAR(64) UNIQUE NOT NULL,
    expires_at TIMESTAMP NOT NULL,
    used BOOLEAN NOT NULL DEFAULT 0,
    created_at TIMESTAMP NOT NULL
);

-- Security Audit Log
CREATE TABLE auth_audit_logs (
    id VARCHAR(36) PRIMARY KEY,
    user_id VARCHAR(36) REFERENCES users(id) ON DELETE SET NULL,
    email VARCHAR(255),
    event_type VARCHAR(64) NOT NULL,
    ip_address VARCHAR(45),
    user_agent TEXT,
    success BOOLEAN NOT NULL DEFAULT 1,
    metadata_json TEXT,
    created_at TIMESTAMP NOT NULL
);
```

---

## 6. REST API Endpoints

All endpoints are prefixed with `/api/v1/auth`.

| Method | Endpoint | Description | Auth Required |
| :--- | :--- | :--- | :--- |
| `POST` | `/register` | Register new account with Argon2id hash. Dispatches verification token. | None |
| `POST` | `/login` | Authenticate credentials; returns access token + sets HttpOnly cookies. | None |
| `POST` | `/refresh` | Rotate refresh token and issue new token pair. | Refresh Token |
| `POST` | `/logout` | Revoke session and clear HttpOnly browser cookies. | Optional Token |
| `GET` | `/me` | Retrieve currently authenticated user profile. | Bearer / Cookie |
| `PUT` | `/profile` | Update user display name and profile attributes. | Bearer / Cookie |
| `POST` | `/change-password` | Update account password (requires old password verification). | Bearer / Cookie |
| `POST` | `/forgot-password` | Request password reset email (constant-time response). | None |
| `POST` | `/reset-password` | Confirm password reset using single-use token. | None |
| `POST` | `/verify-email` | Validate account email using single-use token. | None |
| `POST` | `/resend-verification` | Request a fresh email verification link. | None |
| `GET` | `/audits` | View system security and authentication audit events. | Admin Only |

---

## 7. Web UI Pages

ZENOVA provides responsive, accessible HTML pages styled with modern CSS variables, glassmorphic accents, and client-side form validation:

* `/login` - Secure user login with "Remember me" support and password toggle.
* `/register` - Account registration with interactive password strength checker.
* `/forgot-password` - Self-service password recovery initiation.
* `/reset-password?token=...` - Single-use token password resetting.
* `/verify-email?token=...` - One-click email verification.

---

## 8. CLI User Management

Administrators can provision and elevate users directly via the CLI without bypassing Argon2id validation:

```bash
# Create an admin user
python scripts/create_user.py --email admin@zenova.ai --password "AdminMasterPassword2026!" --role admin --display-name "System Administrator"

# Create a clinician user
python scripts/create_user.py --email dr.smith@zenova.ai --password "ClinicianPass2026!" --role clinician --display-name "Dr. Jane Smith"
```

---

## 9. Limitations & Production Readiness Notice

> [!WARNING]
> **Research & Prototyping Disclaimer**:
> 1. While the authentication implementation adheres to OWASP, NIST SP 800-63B, and cryptographic best practices (Argon2id, rotating hashed tokens, IDOR defenses, CSRF/XSS cookie mitigations), this software is currently a local research prototype.
> 2. It is **NOT** formally certified under SOC2 Type II, ISO 27001, or HIPAA.
> 3. Production deployments must configure real SMTP credentials (`EMAIL__PROVIDER=smtp`), an enterprise Postgres database (`DATABASE_URL=postgresql+asyncpg://...`), and run behind a TLS-terminating load balancer with HTTP Strict Transport Security (`HSTS`).
