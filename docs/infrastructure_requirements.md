# ZENOVA Infrastructure Sizing & Operational Requirements

## 1. Hardware & Compute Specifications

ZENOVA utilizes lightweight transformer architectures optimized for CPU inference alongside local dense embedding retrievers:

### 1.1. Sizing Matrix by Environment Tier

| Resource | Development | Staging | Production (Per Node) | Production Cluster (3 Replicas) |
|---|---|---|---|---|
| **vCPU** | 2 cores | 4 cores | 4 cores | 12 cores total |
| **RAM** | 4 GB | 8 GB | 8 GB | 24 GB total |
| **Storage (App/OS)** | 20 GB SSD | 50 GB SSD | 50 GB NVMe SSD | 150 GB NVMe SSD |
| **Storage (Database)** | 10 GB SSD | 50 GB SSD | 200 GB Managed SSD | Multi-AZ High Availability |
| **GPU Acceleration** | Optional | Optional | Optional (Recommended for >100 QPS) | NVIDIA T4 / A10G (optional) |
| **Network Bandwidth** | Standard | 1 Gbps | 1 Gbps | 10 Gbps redundant |

---

## 2. Network & Ingress Requirements

- **Inbound Ports**:
  - `443/TCP`: HTTPS TLS Termination (Ingress / Application Load Balancer).
  - `80/TCP`: HTTP Redirect to HTTPS.
  - `8000/TCP`: Internal container port (mapped from ALB).
- **Internal Ports**:
  - `5432/TCP`: PostgreSQL Database.
  - `6379/TCP`: Redis Cache.
  - `9090/TCP`: Prometheus Metrics Collector.
- **Outbound Ports**:
  - `443/TCP`: Outbound HTTPS for third-party LLM APIs (OpenAI, Anthropic, Google Vertex AI) and Sentry telemetry.

---

## 3. Storage & IOPS Requirements

- **Database Storage**:
  - Provision minimum **3,000 IOPS** (e.g. AWS gp3 or GCP Balanced Persistent Disk) to support concurrent turn write throughput and non-blocking execution trace persistence.
- **Log Retention**:
  - Retain local log files with rolling 100MB caps (max 5 archives = 500MB max local log consumption per node).
  - Forward logs to managed cloud storage (CloudWatch, Cloud Logging) with a **90-day hot retention** and **1-year cold archive** policy.

---

## 4. Scalability & High Availability Limits

- **Horizontal Pod Autoscaler (HPA)**:
  - Target CPU utilization: **70%**.
  - Target memory utilization: **75%**.
  - Minimum replicas: **2** (across distinct Availability Zones).
  - Maximum replicas: **10** (for peak traffic handling).
- **Graceful Shutdown**:
  - Configured termination grace period: **30 seconds** to drain ongoing conversational turns and persist execution traces before SIGKILL.
