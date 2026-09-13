# ZENOVA Enterprise Deployment & Operations Guide

## 1. Architecture Overview

ZENOVA is containerized as an asynchronous, stateful, and observable microservice stack:
- **Application Engine**: Python 3.11 / FastAPI running with `uvicorn` (multi-worker support).
- **Relational Storage**: PostgreSQL 15+ (production) / SQLite (development).
- **In-Memory Cache & Rate Limiting**: Redis 7+ (session buffer, sliding token bucket rate limiter).
- **Observability**: Prometheus metric scraping (`/metrics`), structured JSON log outputs with PII/PHI redaction, and optional Sentry error tracking.

---

## 2. Environment Configurations

ZENOVA strictly segregates operational environments via the `ZENOVA_ENV` variable:

| Tier | `ZENOVA_ENV` | Database | Logging | Security Restrictions |
|---|---|---|---|---|
| **Development** | `development` | SQLite / Local Postgres | Human-readable text | Permissive local defaults, CORS wildcard allowed |
| **Testing** | `testing` | In-memory / Isolated SQLite | Minimal text | Fast iteration, mock providers supported |
| **Staging** | `staging` | Managed PostgreSQL | Structured JSON | Mirror of production with staging domain CORS |
| **Production** | `production` | High-Availability PostgreSQL | Structured JSON with PII masking | **Zero wildcard CORS**, mandatory high-entropy secret key ($\ge 32$ chars), `debug=False` enforced |

---

## 3. Docker & Docker Compose Deployment

### 3.1. Local & Staging Deployment
```bash
# Copy and edit local environment variables
cp .env.example .env

# Build and start all services (API, Postgres, Redis, Prometheus)
docker-compose up -d --build

# View container logs
docker-compose logs -f api

# Verify health status
curl -f http://localhost:8000/health/ready
```

### 3.2. Hardened Production Deployment
```bash
# Ensure production environment variables are exported or placed in cloud secrets
# Run with production override (applies resource limits, restart: always, and log rotation)
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d
```

---

## 4. Kubernetes Deployment Manifests

For cloud orchestration on Kubernetes (EKS, GKE, AKS, or bare-metal k8s), deploy using the following manifests:

### 4.1. Namespace & ConfigMap
```yaml
apiVersion: v1
kind: Namespace
metadata:
  name: zenova
---
apiVersion: v1
kind: ConfigMap
metadata:
  name: zenova-config
  namespace: zenova
data:
  ZENOVA_ENV: "production"
  ZENOVA_LOG_LEVEL: "INFO"
  ZENOVA_STRUCTURED_LOGGING: "true"
  RATE_LIMIT_PER_MINUTE: "120"
  RATE_LIMIT_BURST: "30"
```

### 4.2. Kubernetes Secret (Managed via External Secrets Operator / Vault)
```yaml
apiVersion: v1
kind: Secret
metadata:
  name: zenova-secrets
  namespace: zenova
type: Opaque
stringData:
  ZENOVA_SECRET_KEY: "replace-with-secure-high-entropy-production-secret-key"
  DATABASE_URL: "postgresql+asyncpg://zenova_app:password@postgres.zenova:5432/zenova_prod"
  DATABASE_URL_SYNC: "postgresql://zenova_app:password@postgres.zenova:5432/zenova_prod"
  REDIS_URL: "redis://:redis_password@redis.zenova:6379/0"
```

### 4.3. Deployment Manifest
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: zenova-api
  namespace: zenova
  labels:
    app: zenova-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: zenova-api
  template:
    metadata:
      labels:
        app: zenova-api
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/path: "/metrics"
        prometheus.io/port: "8000"
    spec:
      securityContext:
        runAsUser: 10001
        runAsGroup: 10001
        fsGroup: 10001
      containers:
      - name: api
        image: your-registry.io/zenova:latest
        imagePullPolicy: IfNotPresent
        ports:
        - containerPort: 8000
        envFrom:
        - configMapRef:
            name: zenova-config
        - secretRef:
            name: zenova-secrets
        resources:
          requests:
            cpu: "500m"
            memory: "1Gi"
          limits:
            cpu: "2000m"
            memory: "4Gi"
        livenessProbe:
          httpGet:
            path: /health/live
            port: 8000
          initialDelaySeconds: 15
          periodSeconds: 20
          timeoutSeconds: 5
        readinessProbe:
          httpGet:
            path: /health/ready
            port: 8000
          initialDelaySeconds: 20
          periodSeconds: 15
          timeoutSeconds: 5
```

---

## 5. Cloud Provider Deployment Recipes

### 5.1. Google Cloud (Cloud Run + Cloud SQL)
1. **Cloud SQL**: Provision a private PostgreSQL 15 instance.
2. **Secret Manager**: Store `DATABASE_URL` and `ZENOVA_SECRET_KEY`.
3. **Artifact Registry**: Push Docker image.
4. **Cloud Run**:
   ```bash
   gcloud run deploy zenova-api \
     --image=gcr.io/your-project/zenova:latest \
     --platform=managed \
     --region=us-central1 \
     --set-env-vars=ZENOVA_ENV=production,ZENOVA_STRUCTURED_LOGGING=true \
     --set-secrets=DATABASE_URL=zenova-db-url:latest,ZENOVA_SECRET_KEY=zenova-secret:latest \
     --min-instances=2 \
     --max-instances=10 \
     --memory=2Gi \
     --cpu=2
   ```

### 5.2. Amazon Web Services (ECS Fargate + RDS Aurora)
1. **RDS Aurora PostgreSQL**: Multi-AZ cluster.
2. **AWS Secrets Manager**: Encrypt credentials with KMS.
3. **ECR**: Push container image.
4. **ECS Task Definition**: Define Fargate task pointing to ECR image with `awsvpc` networking.

---

## 6. Database Migrations in Production

Run database schema migrations automatically during container initialization or as a Kubernetes pre-install/pre-upgrade job:

```bash
# Execute migrations via Python CLI runner
python scripts/run_migrations.py head
```

---

## 7. Operational Runbook & Health Checks

- **Liveness**: `GET /health/live` returns HTTP 200 if process is running.
- **Readiness**: `GET /health/ready` returns HTTP 200 if DB, models, vector store, and disk space are all healthy; returns HTTP 503 if any critical dependency is down.
- **Prometheus Metrics**: `GET /metrics` outputs request rates, durations, active crisis escalations, and safety gate actions.
- **Interactive Documentation**: Available at `/docs` (Swagger UI) and `/redoc` (ReDoc). Static schema exported at `docs/openapi.json`.
