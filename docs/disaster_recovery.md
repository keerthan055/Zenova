# ZENOVA Disaster Recovery & Business Continuity Plan

## 1. Objectives & Service Level Agreements (SLAs)

In critical mental-health and conversational wellbeing operations, continuous availability and data preservation are paramount:

- **Recovery Point Objective (RPO)**: **< 1 Hour**  
  Maximum allowable period of data loss in catastrophic events.
- **Recovery Time Objective (RTO)**: **< 15 Minutes**  
  Maximum allowable system downtime before service restoration.
- **Data Durability**: **99.999999999% (11 9's)** via cloud object storage replication.

---

## 2. Automated Backup Strategy

### 2.1. Backup Types & Frequencies

1. **Continuous WAL Archiving (Point-in-Time Recovery)**:
   - PostgreSQL Write-Ahead Logs (WAL) streamed continuously to multi-region cloud object storage (S3 / GCS).
   - Enables point-in-time recovery down to the second within a 35-day sliding window.
2. **Daily Snapshot Backups**:
   - Automated daily snapshots generated at 02:00 UTC using `scripts/backup_db.py`.
   - Compressed with Gzip (`.db.gz`), encrypted with AES-256 (KMS-managed keys), and checksum-verified with SHA-256.
   - Retention policy: **Last 7 daily snapshots maintained on-disk**, older backups archived to cold storage.

---

## 3. Disaster Recovery Scenarios & Playbooks

### Scenario A: Primary Database Node Failure
1. **Detection**: Health probe `/health/ready` reports HTTP 503; CloudWatch/Stackdriver emits `DatabaseUnreachable` alarm.
2. **Automated Failover**: Managed cloud database (AWS RDS / Cloud SQL) initiates automatic standby promotion in Secondary AZ (< 60 seconds).
3. **Application Reconnection**: SQLAlchemy connection pool recycles idle connections and reconnects to the newly promoted primary endpoint without container restart.

### Scenario B: Cloud Region Outage
1. **DNS Failover**: Route53 / Cloud DNS health check redirects traffic to the Warm Standby Region.
2. **Database Promotion**: Promote read-replica in the secondary region to primary write mode.
3. **Container Scale-up**: Increase HPA replica count in the secondary region from warm minimum (2) to operational production capacity (6).
4. **Verification**: Execute `GET /health/ready` across all ingress endpoints.

### Scenario C: Accidental Table Corruption / Malicious Modification
1. **Isolate**: Place application into maintenance mode (`ZENOVA_ENV=maintenance` or redirect ingress).
2. **Locate Target Restore Point**: Identify corruption timestamp from `execution_traces` and `safety_audit_logs`.
3. **Execute Point-in-Time Restore**:
   ```bash
   # Restore snapshot to temporary staging database
   python scripts/backup_db.py restore --timestamp 2026-09-14T01:45:00Z
   ```
4. **Data Verification**: Validate user session counts and audit integrity.
5. **Switch Traffic**: Repoint production connection string to verified restored database.

---

## 4. Disaster Recovery Testing & Drills

- **Quarterly Table Restore Drills**: Dry-run restoration of backup archives into an isolated sandbox environment.
- **Annual Chaos Engineering Exercise**: Simulated Availability Zone failure to verify zero data loss and automated failover within SLA limits.
